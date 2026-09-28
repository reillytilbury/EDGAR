"""Full-rollout prediction comparison to observed values. These images are fed as diagnostic input to the LLM
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
    params=None,
    rng: np.random.Generator | None = None,
    max_show: int = 5,
    stim_index: int = 0,
    window: int = 0,
):
    """Simple full-rollout prediction comparison plot vs observed values.
        x-axis : time (in ms)
        y-axis : activity (E or I)
    """
    if not save_path:
        raise ValueError("Please provide a save_path for the plot")

    # plot.py is loaded via exec(), so __file__ is unavailable — walk up from the
    # save_path to find the repo root and import the project's apply_model: the same
    # free-rollout apply_model_fn the scorer uses (TaskSpec.apply_model_fn).
    save_p = Path(save_path).resolve()
    repo_root = save_p
    for _ in range(10):
        if (repo_root / "projects" / "wilson_cowan").is_dir():
            break
        if repo_root.parent == repo_root:
            raise RuntimeError(
                f"couldn't locate repo root walking up from {save_p}; "
                "expected projects/wilson_cowan/ somewhere above."
            )
        repo_root = repo_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from projects.wilson_cowan.data_loader.load_data import (  # noqa: E402
        apply_model as apply_model_fn,
    )

    if losses is None:
        losses = [p.program_losses.discover.final for p in programs]
    if program_names is None:
        program_names = [p.name for p in programs]
    if params is None:
        params = [p.params for p in programs]

    E = np.asarray(data["E"])                 # (n, n_stim, T)
    I = np.asarray(data["I"])
    n_samples, n_stim, T = E.shape
    si = int(min(max(stim_index, 0), n_stim - 1))

    n_show = min(max_show, n_samples)
    show_idx = np.linspace(0, n_samples - 1, n_show).astype(int)

    # Pick the best program that actually compiles and has params: lowest loss.
    candidates = []
    for j, p in enumerate(programs):
        if params[j] is None:
            continue
        try:
            fn = p.compile_model()
        except Exception:
            continue
        loss_j = losses[j] if (losses[j] is not None) else np.inf
        candidates.append((loss_j, j, fn))
    if not candidates:
        raise RuntimeError("no program could be compiled for the rollout plot")
    _, best_j, model_fn = min(candidates, key=lambda t: t[0])

    # Build true / free-rollout (E, I) for the chosen program at this stim condition.
    # apply_model_fn is the scorer's free-rollout scan: seeded from the true first
    # observation, then fed its own prediction; only the stimulus is teacher-forced.
    true_E = np.empty((n_show, T)); true_I = np.empty((n_show, T))
    pred_E = np.empty((n_show, T - 1)); pred_I = np.empty((n_show, T - 1))
    for row, s in enumerate(show_idx):
        sample_data = {
            "E": jnp.asarray(E[s:s + 1]),
            "I": jnp.asarray(I[s:s + 1]),
            "stim_E": jnp.asarray(np.asarray(data["stim_E"])[s:s + 1]),
            "stim_I": jnp.asarray(np.asarray(data["stim_I"])[s:s + 1]),
        }
        sample_params = {
            k: jnp.asarray(np.asarray(v)[s:s + 1]) for k, v in params[best_j].items()
        }
        out = np.asarray(apply_model_fn(model_fn, sample_data, sample_params))
        # out: (1, n_stim, T-1, >=2) -> free-rollout (E, I) at t = 1..T-1
        true_E[row] = E[s, si]; true_I[row] = I[s, si]
        pred_E[row] = out[0, si, :, 0]; pred_I[row] = out[0, si, :, 1]

    loss_str = f"{losses[best_j]:.4f}" if losses[best_j] is not None else "n/a"
    model_name = f"{program_names[best_j]}: loss={loss_str}"
    sample_labels = [f"sample {s}" for s in show_idx]

    T_show = T if window <= 0 else int(min(window, T))
    t_true = np.arange(T_show)
    t_pred = np.arange(1, T_show)

    # One row per shown sample; E panel (left) and I panel (right), data vs rollout.
    fig, axes = plt.subplots(
        n_show, 2, figsize=(11, 2.2 * n_show + 0.5), squeeze=False,
    )
    for row in range(n_show):
        for ci, (chan, obs, pred, mcolor) in enumerate([
            ("E", true_E, pred_E, "tab:red"),
            ("I", true_I, pred_I, "tab:blue"),
        ]):
            ax = axes[row, ci]
            ax.plot(t_true, obs[row, :T_show], color="0.35", lw=0.9, label="data")
            ax.plot(t_pred, pred[row, :T_show - 1], color=mcolor, lw=0.9,
                    alpha=0.9, label="model (rollout)")
            ax.set_title(f"{sample_labels[row]} — {chan}", fontsize=9)
            ax.tick_params(labelsize=7)
            y_max = float(np.max(obs[row, :T_show]))
            ax.set_ylim(-0.1, y_max * 1.1)
            if ci == 0:
                ax.set_ylabel("activity", fontsize=8)
            if row == n_show - 1:
                ax.set_xlabel("time bin", fontsize=8)
            if row == 0 and ci == 0:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"Data vs free-rollout — stim {si}"
        + (f"  |  {model_name}" if model_name else ""),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
