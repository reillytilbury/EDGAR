"""Residual cross-dependence diagnostic for the Wilson-Cowan discovery task.

A one-step-ahead, teacher-forced overlay of pred vs true is *not* discriminating:
predicting y[t] from y[t-1] one step ahead barely tests the dynamics, so a good
model and a badly-wrong one both sit almost on top of the true trace. The
informative object is the RESIDUAL r = y - yhat, which under teacher forcing is
approximately the sum of the true-model terms the candidate omitted.

This renders the "option 4" grid: each residual channel (r_E, r_I) plotted
against every lag-1 observable (E[t-1], I[t-1]) — the raw cross-dependence.
    * DIAGONAL  (r_E vs E, r_I vs I): a missing/wrong SELF term.
    * OFF-DIAG  (r_E vs I, r_I vs E): a missing CROSS-coupling term.
A structureless cloud around r~0 means that regressor is captured; a residual
that trends with the regressor means a term in that variable is missing. The
pooled Pearson r in each panel title is a scalar summary of that trend.

Points are coloured by TIME, with the colormap mapped to the detected stimulus-
transient window (baseline dwarfs it in raw time and saturates to the ends). A
colour-ordered loop => the residual depends on phase/history (a dynamic term),
not just the instantaneous regressor value.

Signature matches the other projects' `plot_model_fits` so the EDGAR image-
feedback hook can call it unchanged.
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
    max_show: int = 5,
    stim_index: int = 0,
    window: int = 0,
):
    """Residual cross-dependence grid for the best program, one stim condition."""
    if not save_path:
        raise ValueError("Please provide a save_path for the plot")

    # plot.py is loaded via exec(), so __file__ is unavailable — walk up from the
    # save_path to find the repo root and import the project's apply_model.
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
    from projects.wilson_cowan.data_loader.load_data import apply_model  # noqa: E402

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
        raise RuntimeError("no program could be compiled for the residual plot")
    _, best_j, model_fn = min(candidates, key=lambda t: t[0])

    # Build true / predicted (E, I) for the chosen program at this stim condition.
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
        out = np.asarray(apply_model(model_fn, sample_data, sample_params))
        # out: (1, n_stim, T-1, 2) -> predicted (E, I) at t = 1..T-1
        true_E[row] = E[s, si]; true_I[row] = I[s, si]
        pred_E[row] = out[0, si, :, 0]; pred_I[row] = out[0, si, :, 1]

    sample_labels = [f"sample {s}" for s in show_idx]
    loss_str = f"{losses[best_j]:.4f}" if losses[best_j] is not None else "n/a"
    model_name = f"{program_names[best_j]}: loss={loss_str}"

    # ── Option 4: raw residual cross-dependence grid ──────────────────────────
    rE = true_E[:, 1:] - pred_E          # (n_show, T-1)
    rI = true_I[:, 1:] - pred_I
    E_prev = true_E[:, :-1]
    I_prev = true_I[:, :-1]

    resids = [("r_E", rE), ("r_I", rI)]
    regs = [("E[t-1]", E_prev), ("I[t-1]", I_prev)]

    # Layout: a top strip of trajectory panels (data vs model, for two randomly
    # chosen samples — each split into an E and an I panel) sits above the 2x2
    # residual-diagnostic grid.
    fig = plt.figure(figsize=(12, 12.5))
    gs = fig.add_gridspec(
        3, 1, height_ratios=[1.0, 2.0, 2.0],
        hspace=0.35, top=0.9, bottom=0.06, left=0.08, right=0.97,
    )

    # ── Top strip: two random samples, each as an E (left) and I (right) panel ────
    n_traj = min(2, n_show)
    traj_rows = np.sort(np.random.default_rng().choice(n_show, n_traj, replace=False))
    T_full = true_E.shape[1]
    T_show = T_full if window <= 0 else int(min(window, T_full))
    t_true = np.arange(T_show)
    t_pred = np.arange(1, T_show)
    top_gs = gs[0].subgridspec(1, 2 * n_traj, wspace=0.35)
    for k, r in enumerate(traj_rows):
        for ci, (chan, obs, pred, mcolor) in enumerate([
            ("E", true_E, pred_E, "tab:red"),
            ("I", true_I, pred_I, "tab:blue"),
        ]):
            ax = fig.add_subplot(top_gs[0, 2 * k + ci])
            ax.plot(t_true, obs[r, :T_show], color="0.35", lw=0.8, label="data")
            ax.plot(t_pred, pred[r, :T_show - 1], color=mcolor, lw=0.8,
                    alpha=0.9, label="model")
            ax.set_title(f"{sample_labels[r]} — {chan}", fontsize=9)
            ax.set_xlabel("time bin", fontsize=8)
            ax.tick_params(labelsize=7)
            if ci == 0:
                ax.set_ylabel("activity", fontsize=8)
            if k == 0 and ci == 0:
                ax.legend(fontsize=6, loc="upper right")
            ax.set_ylim(-0.1, 3.5)

    # Colour the scatter by TIME so the hysteresis loops (rise vs fall of the pulse)
    # are legible. Baseline dwarfs the transient in raw time, so map the colormap to
    # the detected stimulus-transient window; pre/post baseline saturate to its ends.
    tvec = np.tile(np.arange(1, T_full), n_show)          # residual time index, pooled
    act = true_E + true_I
    base = np.median(act, axis=1, keepdims=True)
    thr = base + 0.05 * (act.max(axis=1, keepdims=True) - base)
    active_cols = np.where((act > thr).any(axis=0))[0]
    if active_cols.size:
        t0, t1 = int(active_cols.min()), int(active_cols.max())
    else:
        t0, t1 = 1, T_full - 1
    norm = plt.Normalize(vmin=max(t0, 1), vmax=max(t1, t0 + 1))

    # ── Bottom: 2x2 residual cross-dependence grid ───────────────────────────────
    grid_gs = gs[1:3].subgridspec(2, 2, hspace=0.3, wspace=0.22)
    axes = np.empty((2, 2), dtype=object)
    for ri in range(2):
        for ci in range(2):
            axes[ri, ci] = fig.add_subplot(grid_gs[ri, ci])
    sc = None
    for ri, (rname, R) in enumerate(resids):
        for ci, (xname, X) in enumerate(regs):
            ax = axes[ri, ci]
            ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
            sc = ax.scatter(X.ravel(), R.ravel(), c=tvec, cmap="turbo", norm=norm,
                            s=5, alpha=0.35, edgecolors="none", rasterized=True)
            corr = np.corrcoef(X.ravel(), R.ravel())[0, 1]   # pooled r (summary)
            diag = ri == ci
            ax.set_title(f"{rname} vs {xname}   (r={corr:+.2f})"
                         + ("  [self]" if diag else "  [CROSS]"),
                         fontsize=10, fontweight="normal" if diag else "bold",
                         color="tab:red" if (not diag and abs(corr) > 0.1) else "k")
            if ri == 1:
                ax.set_xlabel(xname)
            if ci == 0:
                ax.set_ylabel(f"residual {rname}")
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("time bin  (colour spans the stimulus transient)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        f"Residual cross-dependence — stim {si}"
        + (f"  |  {model_name}" if model_name else ""),
        fontsize=11, y=0.965,
    )
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
