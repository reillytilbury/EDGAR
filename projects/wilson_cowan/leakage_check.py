"""Structural anti-leakage self-test for the Wilson-Cowan state-space DSL.

WC's premise is cheat-proof one-step-ahead autoregression: the model sees only the
PREVIOUS observation (bundled with the previous stimulus) and must predict the next.
Because ``y_prev`` here is a *dict* carrying the stimulus, there is more surface for
an accidental future-index or wrong slice than in the scalar fhn task — so a
structural regression guard matters. We verify:

  1. ISOLATION — perturbing the observed E/I at ``t:`` leaves every prediction at
     ``<t`` bit-exact (and the suffix does change). Passes because ``y[>=t]`` is
     never in scope at step ``t``.
  2. SHAPE — each seed's ``model(state, y_prev, params)`` returns ``(new_state, (E, I))``
     with a state pytree matching the ``s0_*`` init and finite scalar outputs. (An
     inline eager check; WC has no separate ``validate_step`` entry point yet.)
  3. NLL sanity — each seed loads, optimises, and produces a finite, ~O(1) loss that
     does not increase.
  4. Persistence gap (DIAGNOSTIC, non-gating) — how the best seed compares to the
     persistence baseline (there is no ground-truth oracle on the real-data target;
     persistence is the trivial floor an evolved model has to clear). Reported and
     warned on, but it does not gate PASS/FAIL: the structural guarantee (1-3) is what
     this file protects, and a strong one-step persistence baseline is a property of the
     metric/data, not a leak. See scripts/wc_baselines.py.

Run:  python projects/wilson_cowan/leakage_check.py
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

import yaml  # noqa: E402

from projects.wilson_cowan.data_loader.load_data import (  # noqa: E402
    load_data, apply_model, loss_fn, _split_params_s0,
)
from projects.wilson_cowan.scripts.wc_baselines import persistence_nll  # noqa: E402
from edgar.scoring.scoring import _optimize, _eval_loss  # noqa: E402


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


def check_isolation(model_fn, data, params) -> tuple[bool, dict]:
    """Perturbing the observed E/I at ``t:`` must not change any prediction at ``<t``."""
    E = np.asarray(data["E"])
    I = np.asarray(data["I"])
    t = E.shape[-1] // 2  # perturb the second half of the observed trace

    E_pert = E.copy(); E_pert[:, :, t:] += 100.0
    I_pert = I.copy(); I_pert[:, :, t:] += 100.0

    base = {"E": jnp.asarray(E), "I": jnp.asarray(I),
            "stim_E": data["stim_E"], "stim_I": data["stim_I"]}
    pert = {**base, "E": jnp.asarray(E_pert), "I": jnp.asarray(I_pert)}

    means_orig = np.asarray(apply_model(model_fn, base, params))[..., :2]  # (n, n_stim, T-1, 2)
    means_pert = np.asarray(apply_model(model_fn, pert, params))[..., :2]

    # Output index j is computed from the observation at index j; perturbing at t
    # leaves predictions j<t bit-exact and must change j>=t.
    prefix_bitexact = np.array_equal(means_orig[:, :, :t], means_pert[:, :, :t])
    suffix_differs = not np.array_equal(means_orig[:, :, t:], means_pert[:, :, t:])

    return (prefix_bitexact and suffix_differs), {
        "prefix_bitexact (must be True)": prefix_bitexact,
        "suffix_differs (must be True)": suffix_differs,
        "t_perturb": t,
    }


def check_shape(model_fn, default_params) -> tuple[bool, dict]:
    """Eager single-step check: returns ``(new_state, (E, I))``, state pytree stable, finite."""
    init_state, dyn_params = _split_params_s0(default_params)
    init_j = jax.tree_util.tree_map(jnp.asarray, init_state)
    dyn_j = jax.tree_util.tree_map(jnp.asarray, dyn_params)
    y_prev = {
        "E_prev": jnp.asarray(0.5), "I_prev": jnp.asarray(0.5),
        "stim_E_prev": jnp.asarray(0.0), "stim_I_prev": jnp.asarray(0.0),
    }
    try:
        new_state, mean = model_fn(init_j, y_prev, dyn_j)
    except Exception as e:  # noqa: BLE001
        return False, {"error": f"{type(e).__name__}: {e}"}

    same_state = (
        jax.tree_util.tree_structure(new_state)
        == jax.tree_util.tree_structure(init_j)
    )
    is_pair = isinstance(mean, tuple) and len(mean) == 2
    finite = is_pair and bool(jnp.all(jnp.isfinite(jnp.asarray(mean))))
    ok = same_state and is_pair and finite
    return ok, {
        "state_pytree_matches_s0": same_state,
        "mean_is_(E,I)_pair": is_pair,
        "finite": finite,
    }


def check_scoring(model_fn, param_est_fn, data_train, data_test) -> tuple[bool, dict]:
    keys = ("E", "I", "stim_E", "stim_I")
    n = data_train["E"].shape[0]
    per_sample = [
        param_est_fn({k: np.asarray(data_train[k][i]) for k in keys}) for i in range(n)
    ]
    params_init = {
        k: jnp.stack([jnp.asarray(float(s[k])) for s in per_sample]) for k in per_sample[0]
    }
    L0 = _eval_loss(model_fn, loss_fn, params_init, data_test, apply_model)
    # _optimize returns (list_of_optimized_param_sets, loss_trajectories); we passed a
    # single per-sample-stacked set, so take element 0.
    opt_params, _ = _optimize(
        model_fn, loss_fn, params_init, data_train,
        gd_config={"max_iter": 100, "learning_rate": 0.001, "gradient_clip_norm": 5.0},
        apply_model_fn=apply_model,
    )
    params = opt_params[0]
    Lf = _eval_loss(model_fn, loss_fn, params, data_test, apply_model)
    finite = bool(np.isfinite(L0) and np.isfinite(Lf))
    improved = bool(Lf <= L0 + 1e-3)
    reasonable = bool(abs(Lf) < 100.0)
    return (finite and improved and reasonable), {
        "L_init": float(L0), "L_final": float(Lf),
        "improved": improved, "reasonable": reasonable,
    }


def _self_test() -> int:
    root = Path(__file__).parent
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    data_path = cfg["io"]["data_path"]
    pp = dict(cfg.get("project_params", {}))

    (Xd_tr, Xd_te), _, _ = load_data(data_path=data_path, **pp)
    print(f"[leakage_check] discover train/test E: {Xd_tr['E'].shape} / {Xd_te['E'].shape}")

    L_pers = float(persistence_nll(Xd_te).mean())
    print(f"[leakage_check] persistence baseline (discover test): {L_pers:+.4f}")

    seeds = [1, 2]
    ok_all = True

    print("\n== structural invariants ==")
    for i in seeds:
        model_fn, _, default_params, _ = _load_seed(i)
        n = Xd_tr["E"].shape[0]
        params_stacked = {k: jnp.full((n,), float(v)) for k, v in default_params.items()}
        shp_ok, shp = check_shape(model_fn, default_params)
        iso_ok, iso = check_isolation(model_fn, Xd_tr, params_stacked)
        print(f"  seed {i}: shape={'OK' if shp_ok else 'BAD'} {shp}  "
              f"isolation={'OK' if iso_ok else 'LEAK'} {iso}")
        ok_all &= (shp_ok and iso_ok)

    print("\n== scoring (load → optimize → loss) ==")
    seed_losses = {}
    for i in seeds:
        model_fn, pe_fn, _, _ = _load_seed(i)
        ok, info = check_scoring(model_fn, pe_fn, Xd_tr, Xd_te)
        seed_losses[i] = info["L_final"]
        print(f"  seed {i}: L_init={info['L_init']:+.4f}  L_final={info['L_final']:+.4f}  "
              f"{'OK' if ok else 'FAIL'}")
        ok_all &= ok

    # The structural anti-leakage guarantee (isolation + shape + finite scoring) is what
    # gates PASS/FAIL — that is what this file exists to protect. The persistence gap
    # below is a QUALITY diagnostic, not a leakage failure, so it is reported (and warned
    # on) but does not flip the self-test.
    print("\nSelf-test (structural):", "PASS" if ok_all else "FAIL")

    print("\n== DIAGNOSTIC: seed vs persistence (quality, non-gating) ==")
    best_seed = min(seed_losses.values())
    gap = L_pers - best_seed  # positive → seed is better (lower NLL) than persistence
    print(f"  best seed L_final: {best_seed:+.4f}  |  persistence: {L_pers:+.4f}  |  "
          f"gap: {gap:+.4f} nat")
    if gap <= 0.0:
        print("  WARNING: no seed beats persistence. On finely-sampled data the one-step "
              "'predict the last value' baseline is very strong, so this may reflect the "
              "metric (one-step NLL) rather than the seeds — a free-running rollout metric "
              "is the natural next signal. Worth investigating before a full run.")
    else:
        print("  OK: best seed beats persistence.")

    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
