"""EDGAR study commands: offline baseline, OpenCode Go preflight, search runs and locked-cell rescoring."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from edgar_study.datasets import (
    ACTIVITY_THRESHOLD,
    CONCENTRATION_THRESHOLD,
    SPLIT_SEED,
    TRIAL_SEED,
    load_dataset,
    prepare,
    synthetic_recording,
)

from edgar_study.upstream import MSE_SCALE, PENALTY, UPSTREAM_COMMIT, import_upstream


def run_baseline(args):
    import jax
    import jax.numpy as jnp
    import numpy as np

    if jax.default_backend() != "cpu":
        raise ValueError("The baseline requires a CPU JAX backend.")
    upstream = import_upstream(args.upstream.resolve())
    objective = upstream["hypothesis_engine"].objective
    quadratic_loss = upstream["loss_functions"].quadratic_loss
    seeds, utils = upstream["seed_programs"], upstream["utils"]
    if args.synthetic:
        response, angles = synthetic_recording()
        sha256 = None
    else:
        response, angles, sha256 = load_dataset(args.dataset, utils.extract_stimulus_related_response)
    x, y, groups, indices = prepare(response, angles, args.protocol)

    metrics = {}
    for partition in ("discovery", "selection"):
        rows = groups[partition]
        metrics[partition] = {}
        for label, model, estimator in (
            ("single_gaussian", seeds.neuron_model_1_jax, seeds.parameter_estimator_1),
            ("double_gaussian", seeds.neuron_model_2_jax, seeds.parameter_estimator_2),
        ):
            # EDGAR samples cells for JAX compatibility checks using NumPy's RNG.
            np.random.seed(0)
            initial_score, _, score, params = objective(
                model, estimator, quadratic_loss, x[rows], y[rows],
                random_seed=TRIAL_SEED, max_iter=1000, tol=1e-6,
                param_penalty_weight=PENALTY,
            )
            columns = indices["score_trial_columns"]
            predictions = jax.vmap(lambda stimuli, p: model(stimuli, *p))(x[rows][:, columns], params)
            mse = float(jnp.mean(jnp.square(predictions - y[rows][:, columns])))
            count = params.shape[1]
            if not np.isfinite([initial_score, score, mse]).all():
                raise ValueError(f"Nonfinite baseline metrics for {partition}/{label}.")
            if not np.isclose(score, MSE_SCALE * mse + PENALTY * count, rtol=1e-6, atol=1e-6):
                raise ValueError(f"Upstream score differs from the recorded MSE for {partition}/{label}.")
            metrics[partition][label] = {
                "initial_score": initial_score, "score": score, "mse": mse, "parameters": count,
            }

    manifest = {
        "dataset": "synthetic" if args.synthetic else args.dataset.name,
        "dataset_sha256": sha256,
        "upstream_commit": UPSTREAM_COMMIT,
        "protocol": args.protocol,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "activity_threshold": ACTIVITY_THRESHOLD,
        "concentration_threshold": CONCENTRATION_THRESHOLD,
        "split_seed": SPLIT_SEED,
        "trial_seed": TRIAL_SEED,
        "penalty": PENALTY,
        "cells_in_recording": response.shape[0],
        "trials_in_recording": response.shape[1],
        "trials_per_eligible_cell": x.shape[1],
        "partition_counts": {key: len(rows) for key, rows in groups.items()},
        "metrics": metrics,
    }
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}")
    with tempfile.TemporaryDirectory(dir=target.parent) as temp:
        staging = Path(temp) / "run"
        staging.mkdir()
        np.savez_compressed(staging / "splits.npz", **indices)
        (staging / "baseline.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.rename(staging, target)
    print(f"Wrote {target / 'baseline.json'} and {target / 'splits.npz'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    upstream = argparse.ArgumentParser(add_help=False)
    upstream.add_argument("--upstream", type=Path, default=Path("."))
    fitter = argparse.ArgumentParser(add_help=False)
    fitter.add_argument("--fitter", choices=("auto", "upstream", "adam", "lm"), default=None,
                        help="Per-cell parameter fitting. auto (run default): compiled Adam split across "
                             "CPU cores when JAX has only a CPU, upstream's loop otherwise. lm is "
                             "Levenberg-Marquardt, faster but scores differently. rescore defaults to "
                             "the run's own fitter.")
    dataset = argparse.ArgumentParser(add_help=False)
    dataset.add_argument("--dataset", type=Path, required=True)
    models = argparse.ArgumentParser(add_help=False)
    models.add_argument("--model", default="hy4-flash-free", help="Translator, proposer and estimator model.")
    models.add_argument("--image-model", default=None,
                        help="Optional vision model that describes diagnostic images for --model.")
    models.add_argument("--allow-paid", action="store_true",
                        help="Permit models with a nonzero price (spend cap still applies).")
    models.add_argument("--results", type=Path, default=Path("results"))
    models.add_argument("--max-tokens", type=int, default=32_000)
    models.add_argument("--reasoning-effort", default="none",
                        help="Effort for translator, routine proposals and estimators "
                             "(upstream's non-thinking Gemini 2.0 tiers).")

    baseline = commands.add_parser("baseline", parents=[upstream], help="Fit and score the two Gaussian seeds.")
    source = baseline.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument("--synthetic", action="store_true")
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--protocol", choices=("paper-fidelity", "strict-holdout-v1"),
                          default="strict-holdout-v1")
    baseline.set_defaults(func=run_baseline)

    preflight = commands.add_parser("preflight", parents=[models],
                                    help="Check model availability, text and image calls.")
    preflight.set_defaults(func=lambda args: _load("preflight", "preflight")(args))

    run = commands.add_parser("run", parents=[models, dataset, upstream, fitter],
                              help="Run the upstream EDGAR search with OpenCode Go models.")
    run.add_argument("--iterations", type=int, default=12)
    run.add_argument("--time-limit", type=float, default=600, help="Minutes.")
    run.add_argument("--large-model", default=None,
                     help="Model for every third generation (upstream's thinking tier). "
                          "Defaults to --model with --large-reasoning-effort.")
    run.add_argument("--large-reasoning-effort", default="high")
    run.add_argument("--temperature-scale", type=float, default=0.5,
                     help="Multiplies upstream's temperatures, which run from 1 to 2.")
    run.add_argument("--concurrency", type=int, default=16)
    run.add_argument("--no-images", action="store_true", help="Text-only feedback condition.")
    run.add_argument("--islands", type=int, default=8)
    run.add_argument("--batch-size", type=int, default=6, help="Proposals per island per generation.")
    run.add_argument("--cell-fraction", type=float, default=1.0,
                     help="Search on this random fraction of the discovery cells only (rescore still "
                          "uses the full selection and locked-test cells).")
    run.set_defaults(func=lambda args: _load("engine", "run")(args))

    rescore = commands.add_parser("rescore", parents=[dataset, upstream, fitter],
                                  help="Select on validation cells, score once on locked cells.")
    rescore.add_argument("--run", type=Path, required=True)
    rescore.add_argument("--top", type=int, default=10)
    rescore.set_defaults(func=lambda args: _load("rescore", "rescore")(args))

    report = commands.add_parser("report", help="Summarise a run and plot best-so-far discovery loss.")
    report.add_argument("--run", type=Path, required=True)
    report.set_defaults(func=lambda args: _load("report", "report")(args))

    args = parser.parse_args()
    args.func(args)


def _load(module, name):
    """Import subcommands lazily so the offline baseline needs no network client."""
    import importlib
    return getattr(importlib.import_module(f"edgar_study.{module}"), name)


if __name__ == "__main__":
    main()
