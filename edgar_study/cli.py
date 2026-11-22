"""Fit EDGAR's original Gaussian seeds without contacting a model provider."""

import argparse
import json
import os
from pathlib import Path
import subprocess
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

UPSTREAM_COMMIT = "0943922dcb73b12dbee8e440b3e1ff700cb49c10"
UPSTREAM_MODULES = (
    "diagnostic", "genetic_helpers", "hypothesis_engine", "loss_functions", "seed_programs", "utils",
)
PENALTY = 0.01


def import_upstream(checkout):
    # Check the actual working-tree files, including partially checked-out repos.
    for name in UPSTREAM_MODULES:
        path = checkout / f"{name}.py"
        expected = subprocess.check_output(
            ["git", "-C", str(checkout), "show", f"{UPSTREAM_COMMIT}:{path.name}"]
        )
        if path.read_bytes() != expected:
            raise ValueError(f"{path.name} differs from pinned upstream.")
    sys.path.insert(0, str(checkout))
    import hypothesis_engine
    import loss_functions
    import seed_programs
    import utils

    return hypothesis_engine.objective, loss_functions.quadratic_loss, seed_programs, utils


def run_baseline(args):
    import jax
    import jax.numpy as jnp
    import numpy as np

    if jax.default_backend() != "cpu":
        raise ValueError("The baseline requires a CPU JAX backend.")
    objective, quadratic_loss, seeds, utils = import_upstream(args.upstream.resolve())
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
            if not np.isclose(score, 10 * mse + PENALTY * count, rtol=1e-6, atol=1e-6):
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
    baseline = commands.add_parser("baseline", help="Fit and score the two Gaussian seeds.")
    source = baseline.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument("--synthetic", action="store_true")
    baseline.add_argument("--upstream", type=Path, default=Path("."))
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--protocol", choices=("paper-fidelity", "strict-holdout-v1"),
                          default="strict-holdout-v1")
    args = parser.parse_args()
    run_baseline(args)


if __name__ == "__main__":
    main()
