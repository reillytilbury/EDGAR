#!/usr/bin/env python
"""Plot fitted-parameter predictions against the real Wilson-Cowan data.

Reads the fitted params saved by ``wc_recover_params.py`` (the sibling
``*.npz``) and, for a couple of discover samples and both stim conditions,
overlays on the real (repeat-averaged) E/I traces:

  * one-step TEACHER-FORCED prediction (what the loss actually sees — feeds the
    real previous value at every step, so it hugs the data for essentially any
    params);
  * FREE-RUNNING simulation (feeds the model's own output back in — the honest
    test of whether the fitted params reproduce the dynamics), for both the
    fitted params and the true params.

Also prints an RMSE table so the numbers are inspectable without the figure.

Usage:
  python wc_plot_fit.py [--scheme-index I] [--samples a,b] [--fold PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

WC = Path(__file__).resolve().parents[1]        # projects/wilson_cowan/
REPO = WC.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(WC / "data_loader"))
sys.path.insert(0, str(WC / "seed_programs"))

from load_data import apply_model                # noqa: E402
from wilson_cowan import model_jax               # noqa: E402

import matplotlib                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402

PARAM_KEYS = [
    "tau_E", "tau_I", "W_EE", "W_IE", "W_EI", "W_II",
    "E_max", "I_max", "C_E", "C_I", "XE", "XI",
]
FOLD_DEFAULT = "/home/dabin/data/wc_simulations/wc_fold0.npz"
RESULTS_NPZ = WC / "scripts" / "wc_recover_params_results.npz"
DT_MS = 1 / 30  # ms per bin (see simulate_data.DT)
STIM_LABELS = ["excitatory pulse", "inhibitory pulse"]


def _simulate_free(params: dict, stim_E: np.ndarray, stim_I: np.ndarray,
                   E0: float, I0: float):
    """Free-running rollout: carry (E,I), predict next from own output.

    Uses the model's stim-at-(t-1) convention (matching apply_model). Returns
    E,I of length T with the given initial condition prepended.
    """
    xs = {"stim_E_prev": jnp.asarray(stim_E[:-1]),
          "stim_I_prev": jnp.asarray(stim_I[:-1])}

    def step(carry, s):
        E_prev, I_prev = carry
        y_prev = {"E_prev": E_prev, "I_prev": I_prev,
                  "stim_E_prev": s["stim_E_prev"], "stim_I_prev": s["stim_I_prev"]}
        _, (E, I) = model_jax({}, y_prev, params)
        return (E, I), (E, I)

    _, (Es, Is) = jax.lax.scan(step, (jnp.asarray(E0), jnp.asarray(I0)), xs)
    Es = np.concatenate([[E0], np.asarray(Es)])
    Is = np.concatenate([[I0], np.asarray(Is)])
    return Es, Is


def _params_for(mat_12: np.ndarray) -> dict:
    """(12,) vector in PARAM order -> scalar-param dict."""
    return {k: jnp.asarray(mat_12[j]) for j, k in enumerate(PARAM_KEYS)}


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default=FOLD_DEFAULT)
    ap.add_argument("--results", default=str(RESULTS_NPZ))
    ap.add_argument("--scheme-index", type=int, default=3,
                    help="row of fitted_params to plot (default 3 = wilson_cowan defaults)")
    ap.add_argument("--samples", default="0,2",
                    help="comma-separated discover-sample positions to plot")
    ap.add_argument("--outdir", default=str(WC / "scripts"))
    args = ap.parse_args()

    res = np.load(args.results)
    scheme_names = res["scheme_names"].tolist()
    disc_idx = res["disc_idx"]
    true_params = res["true_params"]              # (12, n_disc)
    fitted_params = res["fitted_params"]          # (n_schemes, 12, n_disc)
    scheme = scheme_names[args.scheme_index]

    raw = np.load(args.fold)
    train_data = np.asarray(raw["train_data"])    # (n_samples, 2, T, 2)  real (avg)
    stimuli = np.asarray(raw["stimuli"])          # (2, T, 2)
    T = train_data.shape[2]
    t_ms = np.arange(T) * DT_MS
    sample_pos = [int(x) for x in args.samples.split(",")]

    print(f"fold={args.fold}")
    print(f"fitted scheme [{args.scheme_index}] = {scheme!r}")
    print(f"discover rows: {disc_idx.tolist()}; plotting positions {sample_pos}\n")

    # One-step teacher-forced predictions via the real apply_model, for the
    # plotted samples only (build a small data dict of just those samples).
    def _tf_predict(param_mat_12xn, pos_list):
        n = len(pos_list)
        rows = disc_idx[pos_list]
        data = {
            "E": jnp.asarray(train_data[rows, :, :, 0]),
            "I": jnp.asarray(train_data[rows, :, :, 1]),
            "stim_E": jnp.asarray(np.broadcast_to(stimuli[None, :, :, 0], (n, 2, T))),
            "stim_I": jnp.asarray(np.broadcast_to(stimuli[None, :, :, 1], (n, 2, T))),
        }
        params = {k: jnp.asarray(param_mat_12xn[j, pos_list]) for j, k in enumerate(PARAM_KEYS)}
        out = np.asarray(apply_model(model_jax, data, params))  # (n, 2, T-1, 2)
        return out

    tf_fit = _tf_predict(fitted_params[args.scheme_index], sample_pos)
    tf_true = _tf_predict(true_params, sample_pos)

    print(f"{'sample':<8}{'stim':<20}{'chan':<5}"
          f"{'freerun_fit':>12}{'freerun_true':>13}{'onestep_fit':>12}")
    for pi, pos in enumerate(sample_pos):
        row = int(disc_idx[pos])
        p_fit = _params_for(fitted_params[args.scheme_index, :, pos])
        p_true = _params_for(true_params[:, pos])

        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        fig.suptitle(f"discover sample pos {pos} (params row {row}) — "
                     f"fitted: {scheme}", fontsize=11)

        for si in range(2):  # stim condition
            E_real = train_data[row, si, :, 0]
            I_real = train_data[row, si, :, 1]
            sE, sI = stimuli[si, :, 0], stimuli[si, :, 1]

            Ef, If = _simulate_free(p_fit, sE, sI, E_real[0], I_real[0])
            Et, It = _simulate_free(p_true, sE, sI, E_real[0], I_real[0])

            for ci, (real, free_f, free_t, tf, chan) in enumerate([
                (E_real, Ef, Et, tf_fit[pi, si, :, 0], "E"),
                (I_real, If, It, tf_fit[pi, si, :, 1], "I"),
            ]):
                ax = axes[ci, si]
                ax.scatter(t_ms, real, color="k", s=3, alpha=0.2, label="real data")
                ax.plot(t_ms, free_t, color="tab:green", lw=1.2, ls="--",
                        label="free-run (true params)")
                # ax.plot(t_ms, free_f, color="tab:blue", lw=1.2, ls="--",
                #         label="free-run (fitted params)")
                ax.plot(t_ms[1:], tf, color="tab:red", lw=0.8, alpha=0.7,
                        label="one-step (fitted)")
                if ci == 0:
                    ax.set_title(STIM_LABELS[si], fontsize=10)
                ax.set_ylabel(f"{chan} activity")
                if ci == 1:
                    ax.set_xlabel("time (ms)")
                if ci == 0 and si == 0:
                    ax.legend(fontsize=7, loc="upper right")

                # ax.set_xlim(0, 200)
                # max_y_val = max(np.max(real), np.max(free_f), np.max(free_t), np.max(tf))
                # ax.set_ylim(0.0, np.maximum(max_y_val * 0.4, 5.0))  # zoom in on the first part of the trace

                print(f"{pos:<8}{STIM_LABELS[si]:<20}{chan:<5}"
                      f"{_rmse(real, free_f):>12.4f}{_rmse(real, free_t):>13.4f}"
                      f"{_rmse(real[1:], tf):>12.4f}")

        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out_png = Path(args.outdir) / f"wc_fit_sample{pos}_scheme{args.scheme_index}.png"
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        print(f"  -> saved {out_png}\n")


if __name__ == "__main__":
    main()
