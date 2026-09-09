"""
Visualization utilities for EDGAR.
Centralizes image generation for LLM feedback and dashboard fit panels.
"""

from __future__ import annotations

import os
import warnings
import multiprocessing as mp
import cloudpickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..evolution.program import Program
from ..jax.utils import _to_jax

if TYPE_CHECKING:
    from ..io.task_spec import TaskSpec


def _feedback_image_worker(
    queue: mp.Queue,
    spec_bytes: bytes,
    data: dict,
    parents_bytes: bytes,
    program_bytes: bytes,
) -> None:
    try:
        data = _to_jax(data)
        spec = cloudpickle.loads(spec_bytes)
        parents = cloudpickle.loads(parents_bytes)
        program = cloudpickle.loads(program_bytes)

        b = program.birth
        img_path = os.path.join(
            spec.output_dir,
            "image_feedback",
            f"gen_{b.generation:03d}",
            f"island_{b.island:03d}",
            f"batch_{b.batch_index:03d}",
            "image.png",
        )
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        spec.plot_fn(data, parents, save_path=img_path, rng=spec.rng)

        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            queue.put((img_bytes, img_path))
        else:
            queue.put((None, None))
    except Exception as e:
        import traceback

        warnings.warn(
            f"[plotting subprocess worker] failed: {e}\n{traceback.format_exc()}"
        )
        queue.put((None, None))


def _program_fits_worker(
    queue: mp.Queue, spec_bytes: bytes, data: dict, programs_bytes: bytes
) -> None:
    try:
        data = _to_jax(data)
        spec = cloudpickle.loads(spec_bytes)
        programs = cloudpickle.loads(programs_bytes)

        plot_dir = Path(spec.output_dir) / "image_fits"
        plot_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for p in programs:
            if p.params_init is None or p.params is None:
                continue

            save_path = plot_dir / f"P{p.idx:04d}.png"
            try:
                spec.plot_fn(
                    data,
                    [p, p],
                    save_path=str(save_path),
                    losses=[
                        p.program_losses.discover.init,
                        p.program_losses.discover.final,
                    ],
                    sample_losses=[p.sample_losses_init, p.sample_losses],
                    program_names=[f"{p.name} (Init)", f"{p.name} (Final)"],
                    params=[p.params_init, p.params],
                    rng=spec.rng,
                )
                results.append((p.idx, str(save_path)))
            except Exception as e:
                import traceback

                warnings.warn(
                    f"[plotting subprocess worker] failed for P#{p.idx}: {e}\n{traceback.format_exc()}"
                )
        queue.put(results)
    except Exception as e:
        import traceback

        warnings.warn(
            f"[plotting subprocess worker] failed: {e}\n{traceback.format_exc()}"
        )
        queue.put([])


def generate_feedback_image(
    spec: TaskSpec, data: dict, parents: list[Program], program: Program
) -> bytes | None:
    """Renders a model-fit image for image-feedback prompts.

    This function uses the project-specific `plot_fn` defined in the `TaskSpec`
    to generate a visualization of a program's model fit, often comparing it
    to parent programs or ground truth data. The generated image serves as
    multimodal feedback for Large Language Models during program generation.
    The image is saved to a structured directory within the run output.

    Args:
        spec: The `TaskSpec` object containing configuration and callable
            functions, including the `plot_fn`.
        data: The input data (e.g., `X_discover`) required by the `plot_fn`
            to render the model's performance.
        parents: A list of `Program` objects that served as parents for the
            current `program`. These are often included in the plot for
            contextual feedback to the LLM.
        program: The `Program` object for which the feedback image is being
            generated.

    Returns:
        The raw bytes of the generated image if successful, otherwise `None`.
    """
    if spec is None or spec.plot_fn is None or data is None:
        return None

    ctx = mp.get_context(os.environ.get("EDGAR_MP_START_METHOD", "spawn"))
    queue = ctx.Queue()
    spec_bytes = cloudpickle.dumps(spec)
    parents_bytes = cloudpickle.dumps(parents)
    program_bytes = cloudpickle.dumps(program)

    proc = ctx.Process(
        target=_feedback_image_worker,
        args=(queue, spec_bytes, data, parents_bytes, program_bytes),
    )
    proc.start()
    try:
        img_bytes, img_path = queue.get(timeout=120)
    except Exception as e:
        proc.kill()
        proc.join()
        warnings.warn(f"[plotting] feedback image subprocess timed out or failed: {e}")
        return None

    proc.join()
    if img_path is not None:
        program.image_path = img_path
    return img_bytes


def generate_program_fits(
    spec: TaskSpec, data: dict, programs: list[Program] | Any
) -> None:
    """Generates a comparison plot for each program showing initial and final model fits.

    For each provided program, this function uses the project-specific `plot_fn`
    to visualize the model's performance with its initial parameters (estimated
    by the parameter estimator) and its final, optimized parameters (after
    gradient descent). These plots are saved to the `image_fits` directory
    within the run output and are typically used for post-hoc analysis and
    dashboard display.

    Args:
        spec: The `TaskSpec` object containing configuration and callable
            functions, including the `plot_fn`.
        data: The input data (e.g., `X_discover`) required by the `plot_fn`
            to render the model's performance.
        programs: A list of `Program` objects for which comparison plots
            should be generated.
    """
    if spec.plot_fn is None:
        return

    ctx = mp.get_context(os.environ.get("EDGAR_MP_START_METHOD", "spawn"))
    queue = ctx.Queue()
    spec_bytes = cloudpickle.dumps(spec)
    programs_bytes = cloudpickle.dumps(programs)

    proc = ctx.Process(
        target=_program_fits_worker,
        args=(queue, spec_bytes, data, programs_bytes),
    )
    proc.start()
    try:
        results = queue.get(timeout=120)
    except Exception as e:
        proc.kill()
        proc.join()
        warnings.warn(f"[plotting] program fits subprocess timed out or failed: {e}")
        return

    proc.join()
    for p_idx, save_path in results:
        for p in programs:
            if p.idx == p_idx:
                p.fit_image_path = save_path
                break


def generate_trajectory_image(spec: TaskSpec, programs: list[Program] | Any) -> None:
    """Generates an optimization trajectory plot for each program.

    For each program, this function visualizes the loss over gradient descent
    steps for all parallel parameter estimations. The trajectory corresponding
    to the best-performing estimator (with the lowest final loss) is
    highlighted. The plots are saved to the `image_trajectories` directory
    within the run output and are used for post-hoc analysis and dashboard
    display. The y-axis (Training Loss) is displayed on a logarithmic scale.

    If a program does not have any trajectory data (e.g., because
    `save_trajectories` is disabled in the configuration), no plot is produced.

    Args:
        spec: The `TaskSpec` object containing configuration.
        programs: A list of `Program` objects for which trajectory plots
            should be generated.
    """
    import matplotlib.pyplot as plt

    plot_dir = Path(spec.output_dir) / "image_trajectories"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for p in programs:
        discover_losses = p.program_losses.discover
        if not discover_losses or discover_losses.trajectories is None:
            continue

        save_path = plot_dir / f"P{p.idx:04d}_traj.png"
        try:
            plt.figure(figsize=(6, 4))
            trajectories = discover_losses.trajectories

            # Find the best trajectory index based on the final loss step
            # trajectories is (n_opts, max_iter)
            best_estimator_idx = p.best_estimator_idx

            # Plot each trajectory
            for idx in range(trajectories.shape[0]):
                traj = trajectories[idx]
                is_best = idx == best_estimator_idx
                color = (
                    "#22c55e" if is_best else "#a1a1aa"
                )  # Green-500 if best, Zinc-400 if other
                alpha = 1.0 if is_best else 0.5
                linewidth = 2.0 if is_best else 1.0
                label = f"Estimator {idx} (Best)" if is_best else f"Estimator {idx}"

                plt.plot(
                    traj, color=color, alpha=alpha, linewidth=linewidth, label=label
                )

            plt.title(
                f"Optimization Trajectories - Program #{p.idx}",
                fontsize=11,
                fontweight="semibold",
            )
            plt.xlabel("Gradient Descent Step", fontsize=9)
            plt.ylabel("Training Loss", fontsize=9)
            plt.yscale("log")
            plt.grid(True, which="both", ls="-", alpha=0.15)
            plt.legend(loc="upper right", frameon=True, fontsize=8)
            plt.tight_layout()

            plt.savefig(save_path, dpi=150)
            plt.close()
            p.trajectory_image_path = str(save_path)
        except Exception as e:
            warnings.warn(
                f"[plotting] failed to generate trajectory plot for P#{p.idx}: {e}"
            )


def generate_program_images(
    spec: TaskSpec, data: dict, programs: list[Program] | Any
) -> None:
    """Generates both fit comparison plots and optimization trajectory plots for each program.

    Args:
        spec: The `TaskSpec` object containing configuration and callable
            functions, including the `plot_fn`.
        data: The input data (e.g., `X_discover`) required by the `plot_fn`
            to render the model's performance.
        programs: A list of `Program` objects for which comparison and
            trajectory plots should be generated.
    """
    generate_program_fits(spec, data, programs)
    generate_trajectory_image(spec, programs)
