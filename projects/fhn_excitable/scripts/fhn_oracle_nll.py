"""Compute the ORACLE Gaussian NLL floor for the FHN testbed.

The oracle knows the true dynamics (a, b, ε, I0) and — for each trajectory —
the true (V_raw, w_raw) sequence and the affine (shift, scale) that took raw
V to normalised y-space. Its one-step prediction of y[t+1] is the
deterministic Euler step of the true FHN SDE in raw scale, then affine-mapped
back to y-scale:

    E[V_raw[t+1]] = V_raw[t] + dt · (V_raw[t] - V_raw[t]³/3 - w_raw[t] + I0)
    E[y[t+1]]    = (E[V_raw[t+1]] - y_shift) / y_scale

σ is fit per-trajectory by MLE on the residuals. Any evolved model's post-
optimisation NLL is bounded below by this value.

Run:
    python projects/fhn_excitable/scripts/fhn_oracle_nll.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.fhn_excitable.data_loader.load_data import (   # noqa: E402
    load_data, TEST_WARMUP_STEPS,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    pp = dict(cfg.get("project_params", {}))
    (Xd_tr, Xd_te), _, _ = load_data(**pp)

    y = np.asarray(Xd_te["y"], dtype=np.float64)
    V_true = np.asarray(Xd_te["_V_true"], dtype=np.float64)
    w_true = np.asarray(Xd_te["_w_true"], dtype=np.float64)
    y_shift = np.asarray(Xd_te["_y_shift"], dtype=np.float64)[:, None]     # (n, 1)
    y_scale = np.asarray(Xd_te["_y_scale"], dtype=np.float64)[:, None]
    dt, a, b, eps, I0 = pp["dt"], pp["a"], pp["b"], pp["eps"], pp["I0"]

    # One-step FHN Euler in raw scale, using true V and true w.
    V_next_raw = V_true[:, :-1] + dt * (
        V_true[:, :-1] - V_true[:, :-1] ** 3 / 3.0 - w_true[:, :-1] + I0
    )
    V_next_y = (V_next_raw - y_shift) / y_scale                             # (n, T-1)

    resid = y[:, 1:] - V_next_y
    resid_pw = resid[:, TEST_WARMUP_STEPS:]
    sigma_mle = np.maximum(resid_pw.std(axis=1), 1e-6)
    nll_per_traj = np.log(sigma_mle) + 0.5
    nll_oracle = float(nll_per_traj.mean())

    pers = float(np.asarray(Xd_te["_persistence_nll"]).mean())

    print(f"\n=== FHN oracle NLL (discovery test window) ===")
    print(f"  post-warmup residual std      : min={sigma_mle.min():.4f}  "
          f"mean={sigma_mle.mean():.4f}  max={sigma_mle.max():.4f}")
    print(f"  ORACLE NLL floor              : {nll_oracle:+.4f} nat / bin")
    print(f"  PERSISTENCE baseline NLL      : {pers:+.4f} nat / bin")
    print(f"  Discovery budget for evolution: {pers - nll_oracle:+.4f} nat / bin")

    if nll_oracle > pers:
        print("\n  ⚠ WARNING: oracle NLL > persistence NLL. Either the oracle "
              "computation is wrong, or the noise is small enough that "
              "persistence is already close to the theoretical floor. "
              "Consider lowering proc_noise_std for a wider discovery budget.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
