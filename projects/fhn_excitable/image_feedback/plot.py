"""Diagnostic plots for the FHN state-space project.

Shows one-step predicted mean (from each program's apply_model output) over-
laid on the true observation for a few sample trajectories
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import jax.numpy as jnp

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


def plot_model_fits(
    data,
    programs,
    save_path="",
    losses=None,
    sample_losses=None,
    program_names=None,
    rng: np.random.Generator | None = np.random.default_rng(),
    params=None,
    max_show: int = 4,
    window: int = 600,
):
    if not save_path:
        raise ValueError("Please provide a save_path for the plot")

    save_p = Path(save_path).resolve()
    repo_root = save_p
    for _ in range(10):
        if (repo_root / "projects" / "fhn_excitable").is_dir():
            break
        if repo_root.parent == repo_root:
            raise RuntimeError(
                f"couldn't locate repo root walking up from {save_p}; "
                "expected projects/fhn_excitable/ somewhere above."
            )
        repo_root = repo_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from projects.fhn_excitable.data_loader.load_data import apply_model   # noqa: E402

    if losses is None:
        losses = [p.program_losses.discover.final for p in programs]
    if program_names is None:
        program_names = [p.name for p in programs]
    if params is None:
        params = [p.params for p in programs]

    y = np.asarray(data["y"])
    n_samples, T = y.shape
    n_show = min(max_show, n_samples)
    show_idx = np.linspace(0, n_samples - 1, n_show).astype(int)
    T_show = int(min(window, T - 1))

    model_fns = []
    for p in programs:
        try:
            model_fns.append(p.compile_model())
        except Exception:
            model_fns.append(None)

    colors = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    fig, axes = plt.subplots(n_show, 1, figsize=(11, 2.6 * n_show), squeeze=False)

    for row, s in enumerate(show_idx):
        ax = axes[row, 0]

        for j, model_fn in enumerate(model_fns):
            if model_fn is None or params[j] is None:
                continue
            try:
                sample_data = {"y": jnp.asarray(y[s:s + 1])}
                sample_params = {
                    k: jnp.asarray(np.asarray(v)[s:s + 1]) for k, v in params[j].items()
                }
                out = np.asarray(apply_model(model_fn, sample_data, sample_params))
                means = out[0, :T_show, 0]
                ax.plot(np.arange(1, T_show + 1), means,
                        color=colors[j % len(colors)], lw=0.8, label=program_names[j])
            except Exception:
                continue

        ax.plot(np.arange(T_show + 1), y[s, :T_show + 1], "k-", lw=0.6, alpha=0.5, label="true V")
        ax.set_ylabel(f"traj {s}")
        if row == 0:
            ax.legend(fontsize=7, loc="upper right")
            ax.set_title(f"FHN state-space one-step mean vs true (first {T_show} bins)")
        if row == n_show - 1:
            ax.set_xlabel("time bin")

    title_parts = []
    for j in range(len(programs)):
        loss_str = f"{losses[j]:.4f}" if losses[j] is not None else "n/a"
        title_parts.append(f"{program_names[j]}: loss={loss_str}")
    fig.suptitle("fhn_excitable fits\n" + "  |  ".join(title_parts), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
