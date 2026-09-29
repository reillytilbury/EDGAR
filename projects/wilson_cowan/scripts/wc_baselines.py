"""Data-only interpretability anchors for the Wilson-Cowan NLL.

Wilson-Cowan targets REAL data, so there is no ground-truth model — an *oracle*
NLL (the true one-step predictor's loss, as in ``fhn_excitable``) is not
computable here. Instead we provide two anchors that need only the data:

  * PERSISTENCE baseline — ``mean(E[t], I[t]) = (E[t-1], I[t-1])``, scored under
    WC's exact heteroscedastic loss (with the noise coefficient fit in closed
    form, per sample). This is the floor an evolved model must beat; without it
    the heteroscedastic NLL numbers are uninterpretable.

  * Empirical NOISE FLOOR — an approximate irreducible one-step NLL, estimated
    from the across-repeat variance of the RAW per-repeat dataset (the observation
    noise you cannot predict away). This is the honest real-data substitute for an
    oracle: it bounds only the *noise* part, not model misspecification, and needs
    the un-averaged repeats (the fold files are already repeat-averaged).

Both are reported on the WC loss scale (nat/bin, summed over the E and I channels),
so they line up with the per-sample losses printed by the scorer.

Run:  python projects/wilson_cowan/scripts/wc_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from projects.wilson_cowan.data_loader.load_data import (  # noqa: E402
    load_data, loss_fn, WARMUP_STEPS, EPS_MEAN,
)


def persistence_nll(data: dict) -> np.ndarray:
    """Per-sample persistence NLL under WC's exact heteroscedastic loss.

    The persistence predictor is ``E_hat[t] = E[t-1]`` (likewise I), i.e. the
    ``apply_model`` means are just the ``[:-1]`` slice of the observed trace. The
    single per-sample noise coefficient ``phi = exp(log_noise_coef)`` is fit in
    closed form: for fixed means, ``d/dphi`` of WC's NLL gives the optimum
    ``phi* = <r^2 / max(mean, EPS_MEAN)>`` averaged over every scored (E and I,
    stim, post-warmup time) point — exactly the maximiser the gradient-descent
    fit would find. We then hand the stacked ``(means, log(phi*))`` to the real
    ``loss_fn`` so the warmup skip and the sum-over-channels / mean-over-(stim,time)
    reduction match the scorer bit-for-bit. Returns shape ``(n_samples,)``.
    """
    E = np.asarray(data["E"], dtype=np.float64)  # (n, n_stim, T)
    I = np.asarray(data["I"], dtype=np.float64)
    n = E.shape[0]

    means_E = E[:, :, :-1]  # persistence: predict y[t] with y[t-1]  → (n, n_stim, T-1)
    means_I = I[:, :, :-1]

    # Scored points = post-warmup, aligned exactly as loss_fn slices them.
    mE = np.maximum(means_E[:, :, WARMUP_STEPS:], EPS_MEAN)
    mI = np.maximum(means_I[:, :, WARMUP_STEPS:], EPS_MEAN)
    rE = E[:, :, 1 + WARMUP_STEPS:] - means_E[:, :, WARMUP_STEPS:]
    rI = I[:, :, 1 + WARMUP_STEPS:] - means_I[:, :, WARMUP_STEPS:]

    # Per-sample optimal phi, pooling E and I (phi is shared across channels).
    wE = (rE ** 2 / mE).reshape(n, -1)
    wI = (rI ** 2 / mI).reshape(n, -1)
    phi = np.concatenate([wE, wI], axis=1).mean(axis=1)  # (n,)
    log_nc = np.log(np.maximum(phi, 1e-12))

    # Rebuild the full (T-1) model_output and let loss_fn do the reduction.
    log_nc_b = np.broadcast_to(log_nc[:, None, None], means_E.shape)
    model_output = jnp.asarray(np.stack([means_E, means_I, log_nc_b], axis=-1))
    out = loss_fn(model_output, {"E": jnp.asarray(E), "I": jnp.asarray(I)})
    return np.asarray(out)


def noise_floor_nll(raw_path: str, r_eff: int = 4) -> tuple[np.ndarray, dict] | None:
    """Approximate irreducible one-step NLL from the RAW per-repeat dataset.

    ``raw_path`` must point at the un-averaged raw npz whose ``data`` array is
    ``(n_samples, n_stim, n_repeats, T, 2)`` (last axis = E, I) — the fold files
    are already repeat-averaged and cannot give this. The true signal is estimated
    as the across-repeat mean; the observation variance as the across-repeat
    variance. The scorer fits on a fold's TEST split, which is an ``r_eff``-repeat
    average (``n_repeats / n_folds`` = 4 by default), so the noise on the target the
    model actually sees is ``var_single / r_eff``.

    The floor is the loss of a PERFECT-mean predictor under WC's noise model:
    ``phi`` is fit per sample (closed form, pooling E and I), then the per-sample
    NLL is ``0.5·(<log(phi·max(mean,EPS))> + 1)`` summed over channels — the ``+1``
    is ``E[r^2/var]`` at the optimum. Returns ``(per_sample_floor, info)`` or
    ``None`` if the raw file is absent or not per-repeat.

    Caveats: assumes Gaussian noise and a perfect mean (so it bounds only the
    noise, not model misspecification); the on-disk raw and fold files may come
    from different generations (e.g. different T), so treat it as a scale
    reference, not a fold-exact bound.
    """
    p = Path(raw_path)
    if not p.exists():
        return None
    raw = np.load(p)
    if "data" not in raw or raw["data"].ndim != 5:
        return None
    d = np.asarray(raw["data"], dtype=np.float64)  # (n, n_stim, R, T, 2)
    R = d.shape[2]

    mu = d.mean(axis=2)                    # true signal      (n, n_stim, T, 2)
    var_single = d.var(axis=2, ddof=1)     # per-point obs var (n, n_stim, T, 2)
    resid_var = var_single / r_eff         # noise on an r_eff-averaged target

    # One-step alignment + warmup: predict y[t] (t>=1) with the perfect mean mu[t].
    m = np.maximum(mu[:, :, 1 + WARMUP_STEPS:, :], EPS_MEAN)   # (n, n_stim, T', 2)
    rv = resid_var[:, :, 1 + WARMUP_STEPS:, :]
    n = d.shape[0]

    phi = (rv / m).reshape(n, -1).mean(axis=1)                 # per-sample optimal phi
    phi_b = phi[:, None, None, None]
    nll = 0.5 * (np.log(phi_b * m) + rv / (phi_b * m))         # (n, n_stim, T', 2)
    floor = (nll[..., 0] + nll[..., 1]).mean(axis=(1, 2))      # sum E,I; mean stim,time
    info = {"raw_path": str(p), "n_repeats": int(R), "r_eff": int(r_eff),
            "T_raw": int(d.shape[3])}
    return floor, info


def _default_raw_path(fold_path: str) -> str:
    """Guess the raw per-repeat npz that a ``*_fold*.npz`` was derived from.

    ``save_kfold_splits`` writes ``{prefix}_fold{f}.npz`` next to the raw file; the
    WCS raw is ``wilson_cowan_slow.npz``. Falls back to the fold's directory.
    """
    d = Path(fold_path).parent
    return str(d / "wilson_cowan_slow.npz")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    data_path = cfg["io"]["data_path"]
    pp = dict(cfg.get("project_params", {}))

    (Xd_tr, Xd_te), (Xv_tr, Xv_te), _ = load_data(data_path=data_path, **pp)

    print(f"\n[wc_baselines] WARMUP_STEPS={WARMUP_STEPS}, EPS_MEAN={EPS_MEAN}")
    print("\n== persistence baseline (mean = y_prev), WC heteroscedastic NLL ==")
    for name, (tr, te) in [("discover", (Xd_tr, Xd_te)), ("validate", (Xv_tr, Xv_te))]:
        p_tr = persistence_nll(tr)
        p_te = persistence_nll(te)
        print(f"  {name}: train NLL {p_tr.mean():+.4f} (per-sample {np.round(p_tr, 3)})")
        print(f"  {name}: test  NLL {p_te.mean():+.4f} (per-sample {np.round(p_te, 3)})")

    print("\n== empirical noise floor (from raw per-repeat data) ==")
    fold_T = int(Xd_te["E"].shape[-1])
    raw_path = _default_raw_path(data_path)
    res = noise_floor_nll(raw_path)
    if res is None:
        print(f"  skipped: no per-repeat raw dataset at {raw_path}")
        print("  (fold files are repeat-averaged; point this at the raw (n,2,R,T,2) npz)")
    else:
        floor, info = res
        print(f"  raw: {info['raw_path']}  (R={info['n_repeats']} repeats, "
              f"r_eff={info['r_eff']}, T_raw={info['T_raw']})")
        if info["T_raw"] != fold_T:
            # A valid floor must come from the SAME generation as the fold. The raw on
            # disk here does not match (different T), so its estimate is not comparable
            # to the fold's losses (it can even land above persistence) — suppress the
            # number rather than print something misleading.
            print(f"  SKIPPED number: raw T ({info['T_raw']}) != fold T ({fold_T}); the raw "
                  "on disk is a different generation from this fold, so the floor is not "
                  "comparable. Regenerate a matching raw+folds (save_data → "
                  "save_kfold_splits from one run) to enable a valid floor.")
        else:
            print(f"  approx irreducible NLL floor: {floor.mean():+.4f} "
                  f"(per-sample {np.round(floor, 3)})")
            print("  NOTE approximate: assumes a perfect mean + Gaussian noise, so it bounds "
                  "only the observation noise, not model misspecification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
