"""Structural anti-leakage self-test for the FHN state-space DSL.

Same invariants as oscillator_ss/leakage_check.py, plus a different baseline
sanity: for FHN, the four 1-D seeds must NOT already match the oracle floor
— if they did, evolution would have no room to discover the recovery
variable. We verify:

  1. ISOLATION — perturbing y[t:] leaves every pred[<t] bit-exact.
  2. SHAPE — validate_step passes for every seed.
  3. NLL sanity — seeds produce finite O(1) losses.
  4. Seed-vs-oracle gap — no seed reaches within 0.03 nat of the oracle,
     confirming that a hand-written 1-D seed cannot solve FHN and any
     evolved program that does is discovering the hidden variable.

Run:  python projects/fhn_excitable/leakage_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml   # noqa: E402

from projects.fhn_excitable.data_loader.load_data import (   # noqa: E402
    load_data, apply_model, loss_fn, validate_step, WARMUP_STEPS,
)
from edgar.scoring.scoring import _optimize, _eval_loss   # noqa: E402


# ── seed-program loading (mirrors what the pipeline does) ──


def _numpy_to_jax_source(src: str) -> str:
    out = src.replace("import numpy as np", "import jax.numpy as jnp")
    out = out.replace("np.", "jnp.")
    if "import jax.numpy as jnp" not in out:
        out = "import jax.numpy as jnp\n" + out
    return out


def _load_seed(seed_num: int) -> tuple:
    seed_dir = Path(__file__).parent / "seed_programs"
    model_src = (seed_dir / f"model{seed_num}.py").read_text()
    param_est_src = (seed_dir / f"param_est{seed_num}.py").read_text()

    ns_m: dict = {}
    exec(_numpy_to_jax_source(model_src), ns_m)
    model_fn = ns_m["model"]
    default_params = dict(model_fn.DEFAULT_PARAMS)

    ns_p: dict = {}
    exec(param_est_src, ns_p)
    param_est_fn = ns_p["parameter_estimator"]

    return model_fn, param_est_fn, default_params, model_src


# ── invariants ──


def check_isolation(model_fn, data_train, params) -> tuple[bool, dict]:
    """Perturbing y[t:] must not change any pred[<t]."""
    y = np.asarray(data_train["y"])
    t_perturb = y.shape[1] // 2
    y_pert = y.copy()
    y_pert[:, t_perturb:] += 100.0

    out_orig = np.asarray(apply_model(model_fn, {"y": jnp.asarray(y)}, params))
    out_pert = np.asarray(apply_model(model_fn, {"y": jnp.asarray(y_pert)}, params))

    means_orig = out_orig[..., 0]
    means_pert = out_pert[..., 0]

    prefix_bitexact = np.array_equal(means_orig[:, :t_perturb], means_pert[:, :t_perturb])
    suffix_differs  = not np.array_equal(means_orig[:, t_perturb:], means_pert[:, t_perturb:])

    return (prefix_bitexact and suffix_differs), {
        "prefix_bitexact (must be True)": prefix_bitexact,
        "suffix_differs (must be True)":  suffix_differs,
        "t_perturb": t_perturb,
    }


def check_shape(model_fn, default_params, source) -> tuple[bool, dict]:
    try:
        validate_step(model_fn, default_params, program_code=source)
        return True, {"validate_step": "OK"}
    except AssertionError as e:
        return False, {"validate_step": f"FAILED: {e}"}


def check_scoring(model_fn, param_est_fn, default_params, data_train, data_test):
    n = data_train["y"].shape[0]
    y_np = np.asarray(data_train["y"])
    per_sample = [param_est_fn({"y": y_np[i]}) for i in range(n)]
    params_init = {
        k: jnp.stack([jnp.asarray(s[k]) for s in per_sample]) for k in per_sample[0]
    }
    L0 = _eval_loss(model_fn, loss_fn, params_init, data_test, apply_model)
    params = _optimize(
        model_fn, loss_fn, params_init, data_train,
        gd_config={"max_iter": 100, "learning_rate": 0.005, "gradient_clip_norm": 5.0},
        apply_model_fn=apply_model,
    )
    Lf = _eval_loss(model_fn, loss_fn, params, data_test, apply_model)
    finite = bool(np.isfinite(L0) and np.isfinite(Lf))
    improved = bool(Lf <= L0 + 1e-3)
    reasonable = bool(abs(Lf) < 100.0)
    return (finite and improved and reasonable), {
        "L_init": float(L0), "L_final": float(Lf),
        "improved": improved, "reasonable": reasonable,
    }


def _oracle_nll(pp: dict, Xd_te) -> float:
    """Compute oracle NLL floor (same math as scripts/fhn_oracle_nll.py)."""
    y = np.asarray(Xd_te["y"], dtype=np.float64)
    V = np.asarray(Xd_te["_V_true"], dtype=np.float64)
    w = np.asarray(Xd_te["_w_true"], dtype=np.float64)
    y_shift = np.asarray(Xd_te["_y_shift"], dtype=np.float64)[:, None]
    y_scale = np.asarray(Xd_te["_y_scale"], dtype=np.float64)[:, None]
    dt, I0 = pp["dt"], pp["I0"]
    V_next_raw = V[:, :-1] + dt * (V[:, :-1] - V[:, :-1] ** 3 / 3.0 - w[:, :-1] + I0)
    V_next_y = (V_next_raw - y_shift) / y_scale
    resid = (y[:, 1:] - V_next_y)[:, WARMUP_STEPS:]
    sigma_mle = np.maximum(resid.std(axis=1), 1e-6)
    return float((np.log(sigma_mle) + 0.5).mean())


def _self_test() -> int:
    root = Path(__file__).parent
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    pp = dict(cfg.get("project_params", {}))

    (Xd_tr, Xd_te), _, _ = load_data(**pp)
    print(f"[leakage_check] loaded discover train/test: {Xd_tr['y'].shape} / {Xd_te['y'].shape}")

    L_oracle = _oracle_nll(pp, Xd_te)
    print(f"[leakage_check] oracle NLL floor: {L_oracle:+.4f}")

    ok_all = True

    print("\n== structural invariants ==")
    for i in [1, 2, 3, 4]:
        model_fn, _, default_params, source = _load_seed(i)
        n = Xd_tr["y"].shape[0]
        params_stacked = {k: jnp.full((n,), float(v)) for k, v in default_params.items()}
        shp_ok, _ = check_shape(model_fn, default_params, source)
        iso_ok, iso = check_isolation(model_fn, Xd_tr, params_stacked)
        print(f"  seed {i}: shape={'OK' if shp_ok else 'BAD'}  "
              f"isolation={'OK' if iso_ok else 'LEAK'} {iso}")
        ok_all &= (shp_ok and iso_ok)

    print("\n== scoring (load → optimize → loss) ==")
    seed_losses = {}
    for i in [1, 2, 3, 4]:
        model_fn, pe_fn, default_params, source = _load_seed(i)
        ok, info = check_scoring(model_fn, pe_fn, default_params, Xd_tr, Xd_te)
        seed_losses[i] = info["L_final"]
        print(f"  seed {i}: L_init={info['L_init']:+.4f}  L_final={info['L_final']:+.4f}  "
              f"{'OK' if ok else 'FAIL'}")
        ok_all &= ok

    print("\n== seed-vs-oracle gap (must be non-trivial) ==")
    best_seed = min(seed_losses.values())
    gap = best_seed - L_oracle
    gap_ok = gap >= 0.03
    print(f"  best seed L_final: {best_seed:+.4f}  |  oracle: {L_oracle:+.4f}  |  "
          f"gap: {gap:+.4f} nat  {'OK' if gap_ok else 'TOO SMALL'}")
    if not gap_ok:
        print(f"  Any 1-D seed within 0.03 nat of oracle would mean linear seeds already "
              f"solve FHN — seed ladder needs rethinking.")
    ok_all &= gap_ok

    print("\nSelf-test:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
