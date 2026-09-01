"""
Visualization utilities for EDGAR.
Centralizes image generation for LLM feedback and dashboard fit panels.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..evolution.program import Program

if TYPE_CHECKING:
    from ..io.task_spec import TaskSpec


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
    try:
        spec.plot_fn(data, parents, save_path=img_path, rng=spec.rng)
        program.image_path = img_path
        return open(img_path, "rb").read()
    except Exception as e:
        warnings.warn(f"[plotting] plot_fn failed for program #{program.idx}: {e}")
        return None


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

    plot_dir = Path(spec.output_dir) / "image_fits"
    plot_dir.mkdir(parents=True, exist_ok=True)

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
            p.fit_image_path = str(save_path)
        except Exception as e:
            warnings.warn(f"[plotting] failed to generate fit plot for P#{p.idx}: {e}")


def generate_trajectory_image(spec: TaskSpec, programs: list[Program] | Any) -> None:
    """Generates an optimization trajectory plot for each program.

    Plots the loss over gradient descent steps for each parallel estimator.
    The trajectory with the lowest final loss is highlighted.
    """
    import matplotlib.pyplot as plt

    plot_dir = Path(spec.output_dir) / "image_trajectories"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for p in programs:
        discover_losses = p.program_losses.discover
        if not discover_losses or not discover_losses.trajectories:
            continue

        save_path = plot_dir / f"P{p.idx:04d}_traj.png"
        try:
            plt.figure(figsize=(6, 4))
            trajectories = discover_losses.trajectories

            # Find the best trajectory index based on the final loss step
            best_estimator_idx = -1
            best_final_loss = float("inf")
            for idx, traj in enumerate(trajectories):
                if traj and traj[-1] < best_final_loss:
                    best_final_loss = traj[-1]
                    best_estimator_idx = idx

            # Plot each trajectory
            for idx, traj in enumerate(trajectories):
                if not traj:
                    continue
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
