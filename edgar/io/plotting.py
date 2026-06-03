"""
Visualization utilities for EDGAR.
Centralizes image generation for LLM feedback and dashboard fit panels.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from ..evolution.program import Program

if TYPE_CHECKING:
    from ..io.task_spec import TaskSpec


def generate_feedback_image(
    spec: TaskSpec, data: dict, parents: list[Program], program: Program
) -> bytes | None:
    """
    Renders a model-fit image for image-feedback prompts using spec.plot_fn.
    Saves it to spec.output_dir/image_feedback/gen_NNN/island_NNN/batch_NNN/image.png.
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
        spec.plot_fn(data, parents, save_path=img_path)
        program.image_path = img_path
        return open(img_path, "rb").read()
    except Exception as e:
        warnings.warn(f"[plotting] plot_fn failed for program #{program.idx}: {e}")
        return None


def generate_program_fits(
    spec: TaskSpec, data: dict, programs: list[Program] | any
) -> None:
    """
    Generates a comparison plot for each program showing fit with params_init (from estimator)
    and final params (post-GD). Saves to spec.output_dir/image_fits/P{idx:04d}.png.
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
            )
            p.fit_image_path = str(save_path)
        except Exception as e:
            warnings.warn(f"[plotting] failed to generate fit plot for P#{p.idx}: {e}")
