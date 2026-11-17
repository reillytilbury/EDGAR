import numpy as np
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt


def plot_model_fits(
    data,
    programs,
    save_path="",
    losses=None,
    sample_losses=None,
    program_names=None,
    params=None,
    *,
    rng: np.random.Generator,
    n_cells_show: int = 9,
    n_bins: int = 30,
    window_deg: float = 30.0,
    **kwargs,
):
    """
    Plot data and model predictions for up to 9 random cells, every contrast in one panel.

    Layout: a 3 x 3 grid of cells. Each cell is a 2 x 2 block:
        top left:     data (test trials): binned means, sky / royal / navy blue for 25 / 50 / 100%,
                      dash-dot / dashed / solid (same styles in the model panels).
        top right:    v1 = the first (worse) program's predictions, lime / green / dark green, over the data.
        bottom left:  v2 = the second (better) program's predictions, orange / red / maroon, over the data.
        bottom right: 100% vs 25% response at directions within +-window_deg of the peak,
                      for the data (blue) and both models (green, red); dashed line is y = x.
    The three tuning panels in a block share their y-axis.

    Args:
        data: dict with keys 'stimulus', 'contrast', 'response', each of shape (n_cells, n_trials).
        programs: list of Program objects with .params and .compile_model() methods. EDGAR passes
            parents worst-first, so programs[0] is the worse model (green), programs[1] the better (red).
        save_path: file path to save the figure.
        losses: list of loss values for each program, if None defaults to program.program_losses.discover.final
        sample_losses: list of arrays of shape (n_cells,) for each program, if None defaults to program.sample_losses
        program_names: optional list of names for the programs, if None defaults to program.name
        params: list of parameter dictionaries for each program, if None defaults to program.params
        n_cells_show: number of cells to plot (at most 9).
        n_bins: number of direction bins for the binned-mean lines.
        window_deg: half-width of the window around the peak used for the scatter panel.
    """
    if not save_path:
        raise ValueError("Please provide a save path for the plot")

    if losses is None:
        losses = [program.program_losses.discover.final for program in programs]
    if sample_losses is None:
        sample_losses = [
            program.sample_losses if program.sample_losses is not None else None
            for program in programs
        ]
    if program_names is None:
        program_names = [program.name for program in programs]
    if params is None:
        params = [program.params for program in programs]

    stimulus = np.asarray(data["stimulus"])  # (n_cells, n_trials)
    contrast = np.asarray(data["contrast"])  # (n_cells, n_trials)
    response = np.asarray(data["response"])  # (n_cells, n_trials)
    n_cells = stimulus.shape[0]
    contrast_levels = np.unique(contrast)
    # one colour family per source, three distinct hues for 25% / 50% / 100% contrast, all solid:
    # data sky/royal/navy blue, v1 lime/green/dark green, v2 orange/red/maroon
    # (prompts.yaml describes the same scheme)
    data_colours = _contrast_palette(["#7fd3f7", "#1f6fd6", "#0b1a5c"], len(contrast_levels))
    model_shades = [_contrast_palette(["#b5e655", "#2ca02c", "#0b4d3a"], len(contrast_levels)),
                    _contrast_palette(["#ffb347", "#e0201b", "#5c0a1e"], len(contrast_levels))]
    # line style also marks contrast: 25% dash-dot, 50% dashed, 100% solid
    con_styles = ["-.", "--", "-"] if len(contrast_levels) == 3 else ["-"] * len(contrast_levels)

    # programs arrive worst-first (EDGAR sorts parents by descending loss), so the worse
    # model is green and the better model is red; prompts.yaml describes the same mapping
    model_colours = ["tab:green", "tab:red"]
    n_models = min(len(programs), 2)
    # short labels matching the prompt ("Model v1", "Model v2"); the fit-comparison images
    # EDGAR makes pass "<name> (Init)" / "<name> (Final)", so label those init / final instead
    if all(str(n).endswith(("(Init)", "(Final)")) for n in program_names[:n_models]):
        model_labels = [str(n).rsplit("(", 1)[1].rstrip(")").lower() for n in program_names[:n_models]]
    else:
        model_labels = [f"v{k + 1}" for k in range(n_models)]
    model_fns = [program.compile_model() for program in programs[:n_models]]

    n_show = min(n_cells_show, n_cells, 9)
    sample_indices = rng.choice(n_cells, size=n_show, replace=False)
    x_grid = np.linspace(0, 2 * np.pi, 240, endpoint=False)

    fig = plt.figure(figsize=(26, 24))
    outer = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.18)

    for b, s in enumerate(sample_indices):
        block = outer[b // 3, b % 3].subgridspec(2, 2, hspace=0.35, wspace=0.25)
        ax_data = fig.add_subplot(block[0, 0])
        ax_models = [fig.add_subplot(block[0, 1]), fig.add_subplot(block[1, 0])]
        ax_sc = fig.add_subplot(block[1, 1])
        lo, hi = np.inf, -np.inf

        # ── data panel ──
        dir_means = {}  # per contrast: (unique directions, mean over test repeats)
        data_curves = {}  # per contrast: binned-mean curve, redrawn under the model curves
        for j, c_level in enumerate(contrast_levels):
            mask = contrast[s] == c_level
            theta, y_obs = stimulus[s, mask], response[s, mask]
            u_dirs, inv = np.unique(theta, return_inverse=True)
            means = np.bincount(inv, weights=y_obs) / np.bincount(inv)
            dir_means[c_level] = (u_dirs, means)
            centres, bin_means = _binned_mean(theta, y_obs, n_bins)
            data_curves[c_level] = (centres, bin_means)
            ax_data.plot(centres, bin_means, color=data_colours[j], ls=con_styles[j], lw=2.5,
                         label=f"{100 * c_level:.0f}%")
            lo, hi = min(lo, bin_means.min()), max(hi, bin_means.max())
        ax_data.set_title(f"Cell {s}: data (test)", fontsize=12)
        ax_data.legend(fontsize=9, title="contrast", title_fontsize=9, loc="upper right",
                       framealpha=0.6)

        # ── window around the peak, from the data at the cell's preferred contrast ──
        u_dirs = dir_means[contrast_levels[0]][0]
        pref = max(contrast_levels, key=lambda c: np.linalg.norm(dir_means[c][1]))
        peak = u_dirs[np.argmax(dir_means[pref][1])]
        dd = np.abs(np.angle(np.exp(1j * (u_dirs - peak))))
        win = dd <= np.deg2rad(window_deg)
        c_lo, c_hi = contrast_levels[0], contrast_levels[-1]
        ax_data.axvspan(peak - np.deg2rad(window_deg), peak + np.deg2rad(window_deg),
                        color="0.9", zorder=0)

        # ── model panels ──
        all_sc = [dir_means[c_lo][1][win], dir_means[c_hi][1][win]]
        ax_sc.scatter(dir_means[c_lo][1][win], dir_means[c_hi][1][win], s=22,
                      color=data_colours[-1], alpha=0.8, label="data")
        for k in range(n_models):
            ax = ax_models[k]
            params_s = {key: np.asarray(v[s]) for key, v in params[k].items()}
            con_losses = []
            preds_win = {}
            for j, c_level in enumerate(contrast_levels):
                ax.plot(*data_curves[c_level], color=data_colours[j], ls=con_styles[j],
                        lw=1.8, alpha=0.8)
                mask = contrast[s] == c_level
                trial_data = {"stimulus": jnp.asarray(stimulus[s, mask]),
                              "contrast": jnp.asarray(contrast[s, mask])}
                y_trial = np.asarray(model_fns[k](trial_data, params_s)).flatten()
                # same scaling as loss_fn, restricted to this cell's trials at this contrast
                con_loss = 10 * np.mean((response[s, mask] - y_trial) ** 2)
                con_losses.append(con_loss)
                grid_data = {"stimulus": jnp.asarray(x_grid),
                             "contrast": jnp.full_like(jnp.asarray(x_grid), float(c_level))}
                y_pred = np.asarray(model_fns[k](grid_data, params_s)).flatten()
                ax.plot(x_grid, y_pred, color=model_shades[k][j], ls=con_styles[j], lw=2.5,
                        label=f"{100 * c_level:.0f}%: loss={con_loss:.2f}")
                lo, hi = min(lo, y_pred.min()), max(hi, y_pred.max())
                dir_data = {"stimulus": jnp.asarray(u_dirs[win]),
                            "contrast": jnp.full(int(win.sum()), float(c_level))}
                preds_win[c_level] = np.asarray(model_fns[k](dir_data, params_s)).flatten()
            cell_loss = sample_losses[k][s] if sample_losses[k] is not None else np.mean(con_losses)
            ax.set_title(f"{model_labels[k]} (cell loss={cell_loss:.2f})",
                         fontsize=12, color=model_colours[k], fontweight="bold")
            ax.legend(fontsize=9, loc="upper right", framealpha=0.6)
            for spine in ax.spines.values():
                spine.set_edgecolor(model_colours[k])
                spine.set_linewidth(2)
            order = np.argsort(preds_win[c_lo])
            ax_sc.plot(preds_win[c_lo][order], preds_win[c_hi][order], "o-",
                       color=model_colours[k], ms=4, lw=2, alpha=0.7,
                       label=model_labels[k])
            all_sc += [preds_win[c_lo], preds_win[c_hi]]
        for k in range(n_models, 2):
            ax_models[k].axis("off")

        # ── shared y-limits on the tuning panels (binned means + model curves, not single trials) ──
        pad = 0.1 * (hi - lo + 1e-9)
        for ax in [ax_data] + ax_models[:n_models]:
            ax.set_xlim(0, 2 * np.pi)
            ax.set_ylim(lo - pad, hi + 2 * pad)
            ax.set_xticks(np.arange(0, 2 * np.pi + 0.01, np.pi / 2))
            ax.set_xticklabels(["0", "90", "180", "270", "360"], fontsize=9)
            ax.tick_params(axis="y", labelsize=9)
        ax_data.set_ylabel("response", fontsize=10)

        # ── scatter panel ──
        vals = np.concatenate(all_sc)
        v_lo, v_hi = vals.min(), vals.max()
        ax_sc.plot([v_lo, v_hi], [v_lo, v_hi], ls="--", color="0.5", lw=1)
        ax_sc.set_xlabel(f"{100 * c_lo:.0f}% response", fontsize=10)
        ax_sc.set_ylabel(f"{100 * c_hi:.0f}% response", fontsize=10)
        ax_sc.set_title(f"{100 * c_hi:.0f}% vs {100 * c_lo:.0f}% (+-{window_deg:.0f} deg of peak)",
                        fontsize=12)
        ax_sc.tick_params(labelsize=9)
        ax_sc.legend(fontsize=9, loc="upper left", framealpha=0.6)

    loss_parts = [
        f"{model_labels[k]} ({'green' if k == 0 else 'red'}): loss="
        + (f"{losses[k]:.3f}" if losses[k] is not None else "n/a")
        for k in range(n_models)
    ]
    fig.subplots_adjust(top=0.94)
    fig.suptitle(
        "Model fits on held-out test trials (each block of 4 panels is one cell)\n"
        + "   |   ".join(loss_parts),
        fontsize=20, y=0.975,
    )
    plt.savefig(save_path, dpi=80.0, bbox_inches="tight")
    plt.close(fig)


def _contrast_palette(hexes, n):
    """The three given colours for three contrasts; otherwise interpolate between them."""
    if n == len(hexes):
        return list(hexes)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("c", hexes)
    return [cmap(t) for t in np.linspace(0, 1, n)]


def _binned_mean(theta, y, n_bins):
    """Mean of y in n_bins equal-width circular direction bins over [0, 2pi).
    Returns bin centres and means, with empty bins dropped."""
    edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    idx = np.clip(np.digitize(np.mod(theta, 2 * np.pi), edges) - 1, 0, n_bins - 1)
    sums = np.bincount(idx, weights=y, minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ok = counts > 0
    return centres[ok], sums[ok] / counts[ok]
