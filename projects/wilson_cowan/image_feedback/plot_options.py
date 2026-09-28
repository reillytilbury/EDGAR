"""Candidate residual-diagnostic plots for the Wilson-Cowan discovery task.

The problem these solve: a one-step-ahead, teacher-forced prediction is *easy*.
Any flexible recurrent model — and even a badly-wrong mechanistic one — sits
almost perfectly on top of the true trace when you overlay pred vs true, because
predicting y[t] from y[t-1] one step ahead barely tests the dynamics. The
overlay looks identical for a good model and a bad one, so it can't discriminate.

The fix is to look at the RESIDUAL r[t] = y[t] - yhat[t] and ask whether it still
contains structure the model failed to capture. If the model is right, the
residual is just observation noise: zero-mean, state-independent, temporally
white. Any departure from that is uncaptured signal.

Three independent options below, each taking precomputed numpy arrays so they are
easy to test and reuse. Shapes:
    true_E, true_I : (n_show, T)     observed E / I traces
    pred_E, pred_I : (n_show, T-1)   one-step predictions, aligned to t = 1..T-1
The residual is r = true[:, 1:] - pred; the "previous value" driving step t is
true[:, :-1].
"""
from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


def _sample_colors(n):
    """Stable per-sample color map (same color for a sample in every panel / channel).

    NB: a "sample" is a distinct mean-field Wilson-Cowan population (its own parameter
    set), NOT a single cell; E and I are that population's excitatory / inhibitory
    sub-populations.
    """
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def _binned_mean(x, y, edges, min_count: int = 20):
    """Mean of y in shared bins of x; returns (centers, means) for populated bins."""
    n_bins = len(edges) - 1
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    cx, cy = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() > min_count:
            cx.append(0.5 * (edges[b] + edges[b + 1])); cy.append(y[m].mean())
    return np.array(cx), np.array(cy)


# ── Option 1: trajectory with the residual overlaid on a second axis ──────────
def plot_trajectory_residual_overlay(
    true_E, true_I, pred_E, pred_I, sample_labels, save_path,
    stim_index: int = 0, model_name: str = "",
):
    """Full trajectory (faint) with the one-step residual on a twin axis (bold).

    Rows are samples (mean-field populations), columns are E / I. The trajectory is drawn faint on the left
    axis to give scale/context; the residual r = true - pred is drawn on a right
    axis zoomed to its own range, so its structure is visible even though it is
    ~100x smaller than the peak. A good model's residual is a flat noise band; a
    misspecified model shows a coherent excursion locked to the transient (e.g.
    the model consistently under/over-shoots the rising or falling edge).
    """
    true_E = np.asarray(true_E); true_I = np.asarray(true_I)
    pred_E = np.asarray(pred_E); pred_I = np.asarray(pred_I)
    n_show, T = true_E.shape
    t_traj = np.arange(T)
    t_res = np.arange(1, T)

    fig, axes = plt.subplots(n_show, 2, figsize=(13, 2.6 * n_show), squeeze=False)
    for row in range(n_show):
        for ci, (chan, obs, pred) in enumerate(
            [("E", true_E, pred_E), ("I", true_I, pred_I)]
        ):
            ax = axes[row, ci]
            ax.plot(t_traj, obs[row], color="0.6", lw=0.8, label="trajectory")
            ax.set_ylabel(f"{sample_labels[row]} — {chan}")
            axr = ax.twinx()
            resid = obs[row, 1:] - pred[row]
            axr.axhline(0.0, color="k", lw=0.5, alpha=0.3)
            axr.plot(t_res, resid, color="tab:red", lw=0.7, label="residual")
            axr.set_ylabel("residual", color="tab:red", fontsize=8)
            axr.tick_params(axis="y", labelcolor="tab:red", labelsize=7)
            if row == 0 and ci == 1:
                lines = ax.get_lines()[:1] + axr.get_lines()[1:]
                ax.legend(lines, [ln.get_label() for ln in lines],
                          fontsize=7, loc="upper right")
            if row == n_show - 1:
                ax.set_xlabel("time bin")
    fig.suptitle(
        f"trajectory (grey) vs one-step residual (red) — stim {stim_index}"
        + (f"  |  {model_name}" if model_name else ""),
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Option 2: residual as a function of the previous value ────────────────────
def plot_residual_vs_prev(
    true_E, true_I, pred_E, pred_I, sample_labels, save_path,
    stim_index: int = 0, model_name: str = "", n_bins: int = 25,
):
    """Residual r[t] scattered against the previous value y[t-1], per channel.

    If the model has captured the state-dependence of the dynamics, the residual
    is independent of the state: a structureless cloud around zero at every
    activity level. A systematic curve in the binned mean (black) is uncaptured
    signal — e.g. the model's gain is wrong at high activity, so it always
    under-predicts there. This is often the single most revealing diagnostic
    because it localizes the failure in STATE space, not time.
    """
    true_E = np.asarray(true_E); true_I = np.asarray(true_I)
    pred_E = np.asarray(pred_E); pred_I = np.asarray(pred_I)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=False)
    for ci, (chan, obs, pred) in enumerate(
        [("E", true_E, pred_E), ("I", true_I, pred_I)]
    ):
        ax = axes[0, ci]
        prev = obs[:, :-1].ravel()
        resid = (obs[:, 1:] - pred).ravel()
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
        ax.scatter(prev, resid, s=2, alpha=0.15, color="tab:blue",
                   edgecolors="none", rasterized=True)
        # Binned mean ± std reveals the systematic (state-dependent) component.
        lo, hi = np.percentile(prev, [0.5, 99.5])
        edges = np.linspace(lo, hi, n_bins + 1)
        idx = np.clip(np.digitize(prev, edges) - 1, 0, n_bins - 1)
        centers, means, stds = [], [], []
        for b in range(n_bins):
            m = idx == b
            if m.sum() > 20:
                centers.append(0.5 * (edges[b] + edges[b + 1]))
                means.append(resid[m].mean())
                stds.append(resid[m].std())
        centers = np.array(centers); means = np.array(means); stds = np.array(stds)
        ax.plot(centers, means, "k-", lw=1.6, label="binned mean")
        ax.fill_between(centers, means - stds, means + stds, color="k", alpha=0.15)
        ax.set_xlabel(f"previous value  {chan}[t-1]")
        ax.set_ylabel(f"residual  {chan}[t] - pred")
        ax.set_title(f"{chan}")
        if ci == 0:
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(
        f"residual vs previous value — stim {stim_index}"
        + (f"  |  {model_name}" if model_name else "")
        + "  (flat black line = state-dependence captured)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Option 4: residual cross-dependence grid ─────────────────────────────────
def plot_residual_cross_dependence(
    true_E, true_I, pred_E, pred_I, sample_labels, save_path,
    stim_index: int = 0, model_name: str = "", n_bins: int = 25,
):
    """Each residual channel vs every lag-1 observable — the CROSS-dependence.

    Under teacher forcing the residual r_c(t) = y_c(t) - yhat_c(t) is approximately
    the sum of the true-model terms the candidate omitted. So a residual that
    depends on a regressor the model already uses (the DIAGONAL, r_E vs E[t-1]) flags
    a missing/wrong self-term (nonlinearity, leak); a residual that depends on the
    OTHER population (the OFF-DIAGONAL, r_E vs I[t-1]) flags missing cross-coupling
    (here: the E<-I inhibition Model 1 drops). A flat black line with r~0 means that
    regressor's contribution is captured; a slope means a missing term in that
    variable. Pearson r is annotated per panel (see the CCF view for whether the
    dependence is instantaneous or spread over lags).
    """
    true_E = np.asarray(true_E); true_I = np.asarray(true_I)
    pred_E = np.asarray(pred_E); pred_I = np.asarray(pred_I)
    rE = true_E[:, 1:] - pred_E          # (n_samples, T-1) — kept 2-D for per-sample plotting
    rI = true_I[:, 1:] - pred_I
    E_prev = true_E[:, :-1]
    I_prev = true_I[:, :-1]
    n = rE.shape[0]
    colors = _sample_colors(n)

    resids = [("r_E", rE), ("r_I", rI)]
    regs = [("E[t-1]", E_prev), ("I[t-1]", I_prev)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    for ri, (rname, R) in enumerate(resids):
        for ci, (xname, X) in enumerate(regs):
            ax = axes[ri, ci]
            ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
            lo, hi = np.percentile(X.ravel(), [0.5, 99.5])
            edges = np.linspace(lo, hi, n_bins + 1)      # shared bins across samples
            for s in range(n):
                ax.scatter(X[s], R[s], s=2, alpha=0.12, color=colors[s],
                           edgecolors="none", rasterized=True)
            for s in range(n):                            # binned line per sample, on top
                cx, cy = _binned_mean(X[s], R[s], edges)
                ax.plot(cx, cy, color=colors[s], lw=1.6,
                        label=sample_labels[s] if (ri == 0 and ci == 0) else None)
            corr = np.corrcoef(X.ravel(), R.ravel())[0, 1]   # pooled r (summary)
            diag = ri == ci
            ax.set_title(f"{rname} vs {xname}   (pooled r={corr:+.2f})"
                         + ("  [self]" if diag else "  [CROSS]"),
                         fontsize=10, fontweight="bold" if not diag else "normal",
                         color="tab:red" if (not diag and abs(corr) > 0.1) else "k")
            if ri == 1:
                ax.set_xlabel(xname)
            if ci == 0:
                ax.set_ylabel(f"residual {rname}")
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7, loc="upper left", title="sample")
    fig.suptitle(
        f"residual cross-dependence — stim {stim_index}"
        + (f"  |  {model_name}" if model_name else "")
        + "\noff-diagonal r != 0  =>  missing cross-population coupling",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Option 6: PARTIAL residual cross-dependence grid ─────────────────────────
def plot_residual_partial_cross_dependence(
    true_E, true_I, pred_E, pred_I, sample_labels, save_path,
    stim_index: int = 0, model_name: str = "", n_bins: int = 25,
):
    """Like the cross-dependence grid, but each panel shows the PARTIAL dependence:
    the target regressor's UNIQUE contribution after projecting out the other
    regressor. This is an added-variable (partial-regression) plot — both axes have
    the other lag-1 observable regressed out — so co-active populations (E and I peak
    together) can no longer masquerade as coupling. A slope here means the residual
    genuinely needs THAT variable beyond the rest; a flat line means the raw
    correlation was collinearity. Compare panel-by-panel with the raw grid: an
    off-diagonal that is strong raw but flat here (e.g. r_E vs I) was a confound; one
    that survives (e.g. r_I vs E) is a real missing term. Partial r annotated.
    """
    true_E = np.asarray(true_E); true_I = np.asarray(true_I)
    pred_E = np.asarray(pred_E); pred_I = np.asarray(pred_I)
    rE = true_E[:, 1:] - pred_E          # (n_samples, T-1) — kept 2-D for per-sample plotting
    rI = true_I[:, 1:] - pred_I
    E_prev = true_E[:, :-1]
    I_prev = true_I[:, :-1]
    n = rE.shape[0]
    colors = _sample_colors(n)

    def project_out(y, c):
        """Residual of y after regressing on c (with intercept)."""
        A = np.c_[np.ones_like(c), c]
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return y - A @ beta

    resids = [("r_E", rE), ("r_I", rI)]
    regs = [("E[t-1]", E_prev), ("I[t-1]", I_prev)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    for ri, (rname, R) in enumerate(resids):
        for ci, (xname, X) in enumerate(regs):
            ax = axes[ri, ci]
            ctrl = regs[1 - ci][1]                 # the OTHER regressor (2-D)
            # Pooled projection (keeps the annotated partial r comparable), then
            # reshape back to (n_samples, T-1) so each sample can be drawn separately.
            r_perp = project_out(R.ravel(), ctrl.ravel()).reshape(n, -1)
            x_perp = project_out(X.ravel(), ctrl.ravel()).reshape(n, -1)
            ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
            lo, hi = np.percentile(x_perp.ravel(), [0.5, 99.5])
            edges = np.linspace(lo, hi, n_bins + 1)      # shared bins across samples
            for s in range(n):
                ax.scatter(x_perp[s], r_perp[s], s=2, alpha=0.12, color=colors[s],
                           edgecolors="none", rasterized=True)
            for s in range(n):                            # binned line per sample, on top
                cx, cy = _binned_mean(x_perp[s], r_perp[s], edges)
                ax.plot(cx, cy, color=colors[s], lw=1.6,
                        label=sample_labels[s] if (ri == 0 and ci == 0) else None)
            pcorr = np.corrcoef(x_perp.ravel(), r_perp.ravel())[0, 1]
            diag = ri == ci
            ax.set_title(f"{rname} vs {xname} | {regs[1-ci][0]}   (pooled partial r={pcorr:+.2f})"
                         + ("  [self]" if diag else "  [CROSS]"),
                         fontsize=10, fontweight="bold" if not diag else "normal",
                         color="tab:red" if (not diag and abs(pcorr) > 0.1) else "k")
            if ri == 1:
                ax.set_xlabel(f"{xname}  (⊥ {regs[1-ci][0]})")
            if ci == 0:
                ax.set_ylabel(f"residual {rname}  (⊥ {regs[1-ci][0]})")
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7, loc="upper left", title="sample")
    fig.suptitle(
        f"PARTIAL residual cross-dependence — stim {stim_index}"
        + (f"  |  {model_name}" if model_name else "")
        + "\neach panel controls for the OTHER regressor (added-variable plot)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Option 7: PARTIAL cross-dependence, as a compact heatmap ─────────────────
def plot_residual_partial_heatmap(
    true_E, true_I, pred_E, pred_I, sample_labels, save_path,
    stim_index: int = 0, model_name: str = "",
):
    """The same partial correlations as the opt6 grid, reduced to a 2x2 heatmap.

    Cell (row=residual channel, col=lag-1 regressor) is the partial correlation of that
    residual with that regressor, controlling for the OTHER regressor (so E<->I
    collinearity cannot fake a value — identical numbers to opt6). No scatter, no
    added-variable axes: just the signed strength of each leftover dependence.

    Colour/sign key: RED (+) the residual grows with that variable -> the model
    under-predicts there -> add an INCREASING (e.g. excitatory) term. BLUE (-) -> the
    model over-predicts -> add a DECREASING term (inhibition, self-decay, saturation).
    Near white (~0) -> already captured. Diagonal = self term, off-diagonal = cross
    coupling. Trades opt6's shape/nonlinearity cue for at-a-glance readability.
    """
    true_E = np.asarray(true_E); true_I = np.asarray(true_I)
    pred_E = np.asarray(pred_E); pred_I = np.asarray(pred_I)
    rE = (true_E[:, 1:] - pred_E).ravel()
    rI = (true_I[:, 1:] - pred_I).ravel()
    E_prev = true_E[:, :-1].ravel()
    I_prev = true_I[:, :-1].ravel()

    def project_out(y, c):
        A = np.c_[np.ones_like(c), c]
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return y - A @ beta

    resids = [("r_E", rE), ("r_I", rI)]
    regs = [("E[t-1]", E_prev), ("I[t-1]", I_prev)]
    M = np.zeros((2, 2))
    for i, (_rn, r) in enumerate(resids):
        for j, (_xn, x) in enumerate(regs):
            ctrl = regs[1 - j][1]
            M[i, j] = np.corrcoef(project_out(r, ctrl), project_out(x, ctrl))[0, 1]

    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks([0, 1]); ax.set_xticklabels([regs[0][0], regs[1][0]])
    ax.set_yticks([0, 1]); ax.set_yticklabels([resids[0][0], resids[1][0]])
    ax.set_xlabel("regressor at t-1  (other variable controlled for)")
    ax.set_ylabel("residual channel")
    ax.set_xticks(np.arange(-0.5, 2), minor=True)
    ax.set_yticks(np.arange(-0.5, 2), minor=True)
    ax.grid(which="minor", color="white", lw=3)
    ax.tick_params(which="minor", length=0)
    for i in range(2):
        for j in range(2):
            tag = "self" if i == j else "cross"
            ax.text(j, i, f"{M[i, j]:+.2f}\n[{tag}]", ha="center", va="center",
                    fontsize=15, fontweight="bold",
                    color="white" if abs(M[i, j]) > 0.45 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("partial correlation   (+ add increasing term · − add inhibition/saturation)",
                   fontsize=8)
    ax.set_title(
        f"partial residual dependence — stim {stim_index}"
        + (f"\n{model_name}" if model_name else ""),
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Option 8: latent slow-variable timescale scan ────────────────────────────
def plot_latent_timescale_scan(
    true_E, true_I, pred_E, pred_I, sample_labels, save_path,
    stim_index: int = 0, model_name: str = "", taus=None,
):
    """Scan for a MISSING HIDDEN SLOW VARIABLE and reveal its timescale.

    A model with full instantaneous E<->I coupling can still miss a slow latent
    variable — e.g. a slow adaptation / slow inhibition S that low-pass-filters one
    population and feeds back over a long timescale. Instantaneous regressors (opt4/6)
    cannot see this: the missing term depends on the HISTORY of activity, not on the
    single previous sample.

    Construction: for a grid of candidate timescales tau (in time bins), build a leaky
    integral S_tau(t) = a*S_tau(t-1) + (1-a)*x(t) with a = exp(-1/tau), for x = I and
    x = E. Then measure the PARTIAL correlation of each residual (r_E, r_I) with the
    lagged S_tau(t-1), CONTROLLING for the instantaneous E[t-1] and I[t-1] — so only the
    slow component the model does not already capture can register. Plotting this
    partial correlation vs tau gives a curve that is ~0 at small tau (there S_tau equals
    the instantaneous input, already controlled for), rises to a peak at tau* (the
    timescale of the true missing variable), then falls as tau -> inf (S_tau becomes DC).

    Read-out: a clear peak means "add a hidden state that integrates <that input> with
    time constant ~tau*"; the SIGN gives the feedback polarity (negative = slow
    inhibition). Solid line = latent integrates I; dashed = integrates E. Flat curves
    near zero (as for the true model) mean no missing slow variable — the control.
    """
    true_E = np.asarray(true_E); true_I = np.asarray(true_I)
    pred_E = np.asarray(pred_E); pred_I = np.asarray(pred_I)
    n, T = true_E.shape
    if taus is None:
        taus = np.unique(np.round(np.logspace(np.log10(5), np.log10(6000), 28)).astype(int))

    def leaky(sig2d, tau):
        """Causal leaky integral along time (per sample), timescale tau bins."""
        a = np.exp(-1.0 / tau)
        out = np.empty_like(sig2d)
        out[:, 0] = sig2d[:, 0]
        for t in range(1, sig2d.shape[1]):
            out[:, t] = a * out[:, t - 1] + (1.0 - a) * sig2d[:, t]
        return out

    E_prev = true_E[:, :-1].ravel()
    I_prev = true_I[:, :-1].ravel()
    C = np.c_[np.ones_like(E_prev), E_prev, I_prev]     # controls: instantaneous E,I

    def project_out(y):
        beta, *_ = np.linalg.lstsq(C, y, rcond=None)
        return y - C @ beta

    rE_perp = project_out((true_E[:, 1:] - pred_E).ravel())
    rI_perp = project_out((true_I[:, 1:] - pred_I).ravel())

    def curve(r_perp, src2d):
        out = []
        for tau in taus:
            s_prev = leaky(src2d, tau)[:, :-1].ravel()
            x_perp = project_out(s_prev)
            sx = np.std(x_perp)
            out.append(np.corrcoef(r_perp, x_perp)[0, 1] if sx > 1e-12 else 0.0)
        return np.array(out)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=False)
    for ci, (rname, r_perp) in enumerate([("r_E", rE_perp), ("r_I", rI_perp)]):
        ax = axes[0, ci]
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
        for src2d, label, ls in [(true_I, "latent integrates I", "-"),
                                 (true_E, "latent integrates E", "--")]:
            c = curve(r_perp, src2d)
            line, = ax.semilogx(taus, c, ls, lw=1.8, label=label)
            k = int(np.argmax(np.abs(c)))
            ax.plot(taus[k], c[k], "o", color=line.get_color(), ms=6)
            ax.annotate(f"tau*={taus[k]}", (taus[k], c[k]), fontsize=8,
                        textcoords="offset points", xytext=(4, 6),
                        color=line.get_color())
        ax.set_xlabel("latent integration timescale  tau  (time bins)")
        ax.set_ylabel(f"partial corr of {rname} with slow latent\n(instantaneous E,I controlled)")
        ax.set_title(rname)
        if ci == 1:
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        f"latent slow-variable timescale scan — stim {stim_index}"
        + (f"  |  {model_name}" if model_name else "")
        + "  (peak => missing hidden state; tau* = its timescale)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Option 5: residual-observation cross-correlation function ────────────────
def plot_residual_ccf(
    true_E, true_I, pred_E, pred_I, sample_labels, save_path,
    stim_index: int = 0, model_name: str = "", max_lag: int = 60,
):
    """Cross-correlation of each residual with the OTHER population over lags.

    corr(r_E(t), I(t-k)) and corr(r_I(t), E(t-k)) as a function of lag k. A sharp
    peak near k=1 means a missing INSTANTANEOUS coupling (the fast W_EI / W_IE terms);
    a broad, slowly-decaying tail means the missing coupling acts over a slow
    timescale (a mis-specified slow-inhibition variable S). This distinguishes
    "add a direct E<->I term" from "fix the slow hidden variable", which the lag-1
    scatter alone cannot.
    """
    true_E = np.asarray(true_E); true_I = np.asarray(true_I)
    pred_E = np.asarray(pred_E); pred_I = np.asarray(pred_I)

    def xcorr(r, x, k):
        r = r - r.mean(); x = x - x.mean()
        d = np.sqrt(np.sum(r * r) * np.sum(x * x))
        if d == 0:
            return np.zeros(k + 1)
        # corr(r(t), x(t-h)): align r[h:] with x[:-h]
        return np.array([np.sum(r[h:] * x[: len(x) - h]) / d if h > 0
                         else np.sum(r * x) / d for h in range(k + 1)])

    lags = np.arange(max_lag + 1)
    pairs = [("r_E(t) vs I(t-k)", true_E, pred_E, true_I),
             ("r_I(t) vs E(t-k)", true_I, pred_I, true_E)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=False)
    for ci, (title, obs, pred, other) in enumerate(pairs):
        ax = axes[0, ci]
        n_used = obs.shape[1] - 1
        band = 1.96 / np.sqrt(n_used)
        ax.axhspan(-band, band, color="0.8", alpha=0.6, label="95% band")
        ax.axhline(0.0, color="k", lw=0.5)
        for row in range(obs.shape[0]):
            r = obs[row, 1:] - pred[row]
            x = other[row, :-1]
            ax.plot(lags, xcorr(r, x, max_lag), lw=1.0, alpha=0.8,
                    label=sample_labels[row])
        ax.set_xlabel("lag k (time bins)")
        ax.set_ylabel("cross-correlation")
        ax.set_title(title)
        if ci == 1:
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(
        f"residual–other-population cross-correlation — stim {stim_index}"
        + (f"  |  {model_name}" if model_name else "")
        + "  (peak@k=1: fast coupling;  broad tail: slow variable)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Option 3: residual autocorrelation ────────────────────────────────────────
def plot_residual_acf(
    true_E, true_I, pred_E, pred_I, sample_labels, save_path,
    stim_index: int = 0, model_name: str = "", max_lag: int = 60,
):
    """Autocorrelation of the residual per channel, one line per sample.

    Observation noise is temporally white, so a correct model's residual ACF sits
    inside the ~95% white-noise band (grey) for every lag >= 1. Slowly-decaying
    positive autocorrelation means the residual carries temporal structure the
    one-step model missed — the classic signature of a dynamics term that is
    absent or mistimed (e.g. a missing slow adaptation variable).
    """
    true_E = np.asarray(true_E); true_I = np.asarray(true_I)
    pred_E = np.asarray(pred_E); pred_I = np.asarray(pred_I)

    def acf(x, k):
        x = x - x.mean()
        denom = np.sum(x * x)
        if denom == 0:
            return np.zeros(k + 1)
        return np.array([np.sum(x[: len(x) - h] * x[h:]) / denom
                         for h in range(k + 1)])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=False)
    lags = np.arange(max_lag + 1)
    for ci, (chan, obs, pred) in enumerate(
        [("E", true_E, pred_E), ("I", true_I, pred_I)]
    ):
        ax = axes[0, ci]
        n_used = obs.shape[1] - 1
        band = 1.96 / np.sqrt(n_used)
        ax.axhspan(-band, band, color="0.8", alpha=0.6, label="95% white-noise band")
        ax.axhline(0.0, color="k", lw=0.5)
        for row in range(obs.shape[0]):
            resid = obs[row, 1:] - pred[row]
            ax.plot(lags, acf(resid, max_lag), lw=1.0, alpha=0.8,
                    label=sample_labels[row])
        ax.set_xlabel("lag (time bins)")
        ax.set_ylabel("residual autocorrelation")
        ax.set_title(f"{chan}")
        ax.set_ylim(-0.3, 1.0)
        if ci == 1:
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(
        f"residual autocorrelation — stim {stim_index}"
        + (f"  |  {model_name}" if model_name else "")
        + "  (inside grey band = white = captured)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def plot_model_fits_stale(
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
    traj_rows = np.sort(rng.choice(n_show, n_traj, replace=False))
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