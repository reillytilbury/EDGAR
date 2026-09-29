#!/usr/bin/env python
"""Throwaway sanity check: can GRADIENT DESCENT ALONE recover the true
Wilson-Cowan parameters from the simulated data, and how sensitive is the
search to the initial parameters?

We reuse the *real* scoring path:
  - the committed jax model  model_jax  from
    projects/wilson_cowan/seed_programs/wilson_cowan.py,
  - the project's load_data / apply_model / loss_fn,
  - edgar.scoring.scoring._optimize  (the Adam loop the scorer actually runs).

Ground truth: the npz stores `params` (n_samples, 12) in PARAM order. load_data
splits the 8 samples 50/50 into discover/validate with sample_split_seed; we
reproduce that permutation so each discover sample lines up with its true params.

For each initialisation scheme we report:
  train loss, test loss, and per-(param,sample) relative recovery error
  |fit-true|/|true|  (median / p90 over the 12 params x n_discover samples),
against the NOISE FLOOR = loss evaluated at the true params.

Usage: python wc_recover_params.py [--max-iter N] [--lr LR] [--fold PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

WC = Path(__file__).resolve().parents[1]        # projects/wilson_cowan/
REPO = WC.parents[1]                             # repo root
# Make `edgar` and the project's sibling modules importable by name.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(WC / "data_loader"))
sys.path.insert(0, str(WC / "seed_programs"))

import importlib  # noqa: E402

from edgar.scoring.scoring import _optimize, _eval_loss  # noqa: E402
from load_data import load_data, apply_model, loss_fn     # noqa: E402

SAMPLE_SPLIT_SEED = 42

# Per-model wiring: the jax model module (seed_programs), the defaults module whose
# PARAM_MEDIAN key order defines the npz `params` column order, and the default fold.
MODELS = {
    "wilson_cowan": {
        "module": "wilson_cowan",
        "defaults": "wilson_cowan_defaults",
        "fold": "/home/dabin/data/wc_simulations/wc_fold0.npz",
    },
    "wilson_cowan_slow": {
        "module": "wilson_cowan_slow",
        "defaults": "wilson_cowan_slow_defaults",
        "fold": "/home/dabin/data/wc_simulations/wcs_fold0.npz",
    },
}

# Column order of the npz `params` array == PARAM_MEDIAN.keys() in simulate_data.
# Set from the selected model's defaults module in main().
PARAM_KEYS: list[str] = []


def _discover_idx(n_samples: int, seed: int) -> np.ndarray:
    perm = np.random.default_rng(seed).permutation(n_samples)
    return np.sort(perm[: n_samples // 2])


def _params_dict(mat: np.ndarray) -> dict:
    """(n, 12) rows in PARAM order -> {key: (n,) jnp array}."""
    return {k: jnp.asarray(mat[:, j]) for j, k in enumerate(PARAM_KEYS)}


def _stack_params(pd: dict) -> np.ndarray:
    """{key: (n,)} -> (12, n) in PARAM order, as numpy."""
    return np.stack([np.asarray(pd[k]) for k in PARAM_KEYS], axis=0)


def _rel_err(fit: dict, true_mat: np.ndarray) -> np.ndarray:
    """(12, n) relative error |fit-true|/|true|.  true_mat is (n,12)."""
    fit_a = _stack_params(fit)            # (12, n)
    true_a = true_mat.T                    # (12, n)
    return np.abs(fit_a - true_a) / np.maximum(np.abs(true_a), 1e-12)


def _sample_losses(model_fn, params, data, apply_model_fn) -> np.ndarray:
    """Per-discover-sample loss (n,) for the given params on `data`."""
    return np.asarray(loss_fn(apply_model_fn(model_fn, data, params), data))


def _detail_block(name, fit, true_disc, disc_idx, train_l, test_l) -> str:
    """Return a text block: per-sample train/test loss + true-vs-optimised table.

    Columns are the discover (train) samples; rows are the 12 parameters. For
    each sample we show the true value, the optimised value, and the signed
    relative error (fit-true)/|true|.
    """
    fit_a = _stack_params(fit)            # (12, n)
    true_a = true_disc.T                   # (12, n)
    n = true_a.shape[1]
    lines = [f"=== {name} : true vs optimised parameters ==="]

    # Per-sample train/test loss (the held-out cross-validation comparison).
    lines.append(f"  {'sample':<8}{'disc_row':>9}{'train_loss':>12}{'test_loss':>12}")
    for j in range(n):
        lines.append(f"  s{j:<7}{int(disc_idx[j]):>9}{train_l[j]:>12.5f}{test_l[j]:>12.5f}")
    lines.append(f"  {'MEAN':<8}{'':>9}{train_l.mean():>12.5f}{test_l.mean():>12.5f}")

    # Parameter table: for each sample, true / fit / signed relative error.
    head = f"\n  {'param':<7}"
    for j in range(n):
        head += f"| s{j} true     fit      relerr "
    lines.append(head)
    for pi, k in enumerate(PARAM_KEYS):
        row = f"  {k:<7}"
        for j in range(n):
            t, f = true_a[pi, j], fit_a[pi, j]
            rel = (f - t) / max(abs(t), 1e-12)
            row += f"| {t:9.4g} {f:9.4g} {rel:+7.2f} "
        lines.append(row)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wilson_cowan", choices=list(MODELS),
                    help="which dynamical-system model to recover")
    ap.add_argument("--fold", default=None,
                    help="fold npz; defaults to the selected model's fold0")
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--quick", action="store_true",
                    help="only 3 schemes (true / default / one jitter) for a fast check")
    ap.add_argument("--out", default=None,
                    help="Markdown file to write the full results to (also saves a "
                         "sibling .npz of the fitted params). Defaults to a "
                         "model-named file under scripts/. Pass '' to skip.")
    args = ap.parse_args()

    cfg = MODELS[args.model]
    if args.fold is None:
        args.fold = cfg["fold"]
    if args.out is None:
        args.out = str(WC / "scripts" / f"{args.model}_recover_params_results.md")

    # Column order of the npz `params` == the defaults module's PARAM_MEDIAN keys.
    global PARAM_KEYS
    PARAM_KEYS = list(importlib.import_module(cfg["defaults"]).PARAM_MEDIAN.keys())

    model_jax = importlib.import_module(cfg["module"]).model_jax
    model_fn = model_jax
    default_params = model_jax.DEFAULT_PARAMS
    apply_model_fn = apply_model

    print(f"model={args.model}  n_params={len(PARAM_KEYS)}")

    # Ground-truth params + the exact discover-sample -> param-row map.
    raw = np.load(args.fold)
    params_all = np.asarray(raw["params"])            # (8, 12), PARAM order
    n_samples = params_all.shape[0]
    disc_idx = _discover_idx(n_samples, SAMPLE_SPLIT_SEED)
    true_disc = params_all[disc_idx]                  # (n_disc, 12)
    n_disc = len(disc_idx)

    # Discover train/test dicts, built exactly as the real scorer sees them.
    (X_disc_train, X_disc_test), _, _ = load_data(
        args.fold, sample_split_seed=SAMPLE_SPLIT_SEED
    )

    gd = {"learning_rate": args.lr, "max_iter": args.max_iter,
          "gradient_clip_norm": args.clip}

    print(f"fold={args.fold}")
    print(f"discover sample rows (into params): {disc_idx.tolist()}  (n={n_disc})")
    print(f"gd: lr={args.lr}  max_iter={args.max_iter}  clip={args.clip}\n")

    # Noise floor: loss at the TRUE params (no optimisation).
    true_pd = _params_dict(true_disc)
    tr0 = _eval_loss(model_fn, loss_fn, true_pd, X_disc_train, apply_model_fn)
    te0 = _eval_loss(model_fn, loss_fn, true_pd, X_disc_test, apply_model_fn)
    print(f"[NOISE FLOOR @ true params]  train={tr0:.5f}  test={te0:.5f}\n")

    # ── Initialisation schemes ──────────────────────────────────────────────
    # Each entry: (name, init_params, print_detail?). All inits are truth-AGNOSTIC
    # (the realistic EDGAR setting) except the "true" drift-check control, which
    # verifies the true params are a fixed point / minimum of the GD landscape.
    def const_init(value: float) -> dict:
        return {k: jnp.full((n_disc,), float(value)) for k in PARAM_KEYS}

    default_vec = np.array([float(default_params[k]) for k in PARAM_KEYS])  # (12,)

    schemes: list[tuple[str, dict, bool]] = []
    schemes.append(("true (drift check)", _params_dict(true_disc), True))
    schemes.append(("all zeros", const_init(0.0), True))
    schemes.append(("small const (0.01)", const_init(0.01), True))
    schemes.append(("wilson_cowan defaults", _params_dict(np.tile(default_vec, (n_disc, 1))), True))
    # Random restarts: default * 10^U(-1,1) per param (a decade around the generic
    # defaults, sampled on a LOG scale because params span orders of magnitude).
    n_restart = 1 if args.quick else 3
    for s in range(n_restart):
        rng = np.random.default_rng(100 + s)
        factors = 10.0 ** rng.uniform(-1.0, 1.0, size=(n_disc, len(PARAM_KEYS)))
        mat = default_vec[None, :] * factors
        schemes.append((f"random restart (10^U[-1,1]) seed={s}", _params_dict(mat), False))

    # ── Summary: aggregate train/test loss + recovery error per scheme ──────
    hdr = f"{'scheme':<32} {'train':>9} {'test':>9} {'relerr_med':>11} {'relerr_p90':>11}"
    print(hdr)
    print("-" * len(hdr))
    fits: dict[str, dict] = {}
    losses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    summary_rows = []
    for name, p_init, _detail in schemes:
        fit = _optimize(model_fn, loss_fn, p_init, X_disc_train, gd, apply_model_fn)
        fits[name] = fit
        train_l = _sample_losses(model_fn, fit, X_disc_train, apply_model_fn)  # (n,)
        test_l = _sample_losses(model_fn, fit, X_disc_test, apply_model_fn)
        losses[name] = (train_l, test_l)
        re = _rel_err(fit, true_disc)
        row = (f"{name:<32} {train_l.mean():>9.5f} {test_l.mean():>9.5f} "
               f"{np.median(re):>11.3f} {np.percentile(re, 90):>11.3f}")
        summary_rows.append(row)
        print(row)

    # ── Detail: true vs optimised parameter tables ──────────────────────────
    # Print only the flagged inits to the terminal; the saved file gets ALL of
    # them (so the random restarts' fitted params are inspectable too).
    detail_blocks = {
        name: _detail_block(name, fits[name], true_disc, disc_idx, *losses[name])
        for name, _p, _d in schemes
    }
    for name, _p, detail in schemes:
        if detail:
            print("\n" + detail_blocks[name])

    # ── Persist results (Markdown + npz of fitted params) ───────────────────
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        md = [
            "# Wilson-Cowan parameter recovery by gradient descent",
            "",
            f"- fold: `{args.fold}`",
            f"- discover sample rows (into params): {disc_idx.tolist()} (n={n_disc})",
            f"- gd: lr={args.lr}, max_iter={args.max_iter}, clip={args.clip}",
            f"- **noise floor @ true params**: train={tr0:.5f}, test={te0:.5f}",
            "",
            "## Summary (aggregate over discover samples)",
            "",
            "```",
            hdr,
            "-" * len(hdr),
            *summary_rows,
            "```",
            "",
            "## Per-scheme detail (true vs optimised, per discover sample)",
            "",
        ]
        for name, _p, _d in schemes:
            md += ["```", detail_blocks[name], "```", ""]
        out_path.write_text("\n".join(md))

        # Machine-readable fitted params: (n_schemes, 12, n_disc) + labels.
        npz_path = out_path.with_suffix(".npz")
        fitted = np.stack([_stack_params(fits[name]) for name, _p, _d in schemes])
        np.savez(
            npz_path,
            scheme_names=np.array([name for name, _p, _d in schemes]),
            param_keys=np.array(PARAM_KEYS),
            disc_idx=disc_idx,
            true_params=true_disc.T,            # (12, n_disc)
            fitted_params=fitted,               # (n_schemes, 12, n_disc)
        )
        print(f"\n[saved] {out_path}\n[saved] {npz_path}")


if __name__ == "__main__":
    main()
