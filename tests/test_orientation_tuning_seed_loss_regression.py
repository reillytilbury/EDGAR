import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.orientation_tuning import data_parser, seed_programs
from src.data_structures import ensure_inputs, ensure_outputs
from src.hypothesis_engine import objective


BASELINE_PATH = Path(__file__).parent / "baselines" / "orientation_tuning_seed_losses_test_mode.json"


def _neuron_model_1_jax(X, theta_pref=0.0, baseline=0.0, amplitude=1.0, tuning_width=1.0):
    theta = X[0]
    theta_pref = jnp.clip(theta_pref, 0, 2 * jnp.pi)
    baseline = jnp.clip(baseline, 0, None)
    amplitude = jnp.clip(amplitude, 0, None)
    tuning_width = jnp.clip(tuning_width, 0.01, None)
    dist = jnp.abs(jnp.arctan2(jnp.sin(theta - theta_pref), jnp.cos(theta - theta_pref)))
    return baseline + amplitude * jnp.exp(-0.5 * (dist / tuning_width) ** 2)


def _neuron_model_2_jax(X, theta_pref=0.0, baseline=0.0, amplitude_1=1.0, amplitude_2=0.0, tuning_width=1.0):
    theta = X[0]
    theta_pref = jnp.clip(theta_pref, 0, 2 * jnp.pi)
    baseline = jnp.clip(baseline, 0, None)
    amplitude_1 = jnp.clip(amplitude_1, 0, None)
    amplitude_2 = jnp.clip(amplitude_2, 0, None)
    tuning_width = jnp.clip(tuning_width, 0.01, None)

    dist_1 = jnp.abs(jnp.arctan2(jnp.sin(theta - theta_pref), jnp.cos(theta - theta_pref)))
    theta_pref_2 = (theta_pref + jnp.pi) % (2 * jnp.pi)
    dist_2 = jnp.abs(jnp.arctan2(jnp.sin(theta - theta_pref_2), jnp.cos(theta - theta_pref_2)))
    return baseline + amplitude_1 * jnp.exp(-0.5 * (dist_1 / tuning_width) ** 2) + amplitude_2 * jnp.exp(
        -0.5 * (dist_2 / tuning_width) ** 2
    )


def _compute_seed_losses(settings):
    data = data_parser.load_and_process_data(
        data_path=settings["data_path"],
        activity_threshold=settings["activity_threshold"],
        conc_threshold=settings["conc_threshold"],
    )

    inputs = ensure_inputs(data["inputs"]).to_tensor()
    outputs = ensure_outputs(data["outputs"]).to_tensor()

    n_samples = int(inputs.shape[0])
    train_idx, test_idx = data_parser.create_train_test_sample_split(
        n_samples,
        training_sample_ratio=settings["training_sample_ratio"],
        random_seed=settings["random_seed"],
    )
    inputs_train = inputs[train_idx]
    outputs_train = outputs[train_idx]
    inputs_test = inputs[test_idx]
    outputs_test = outputs[test_idx]

    models = [_neuron_model_1_jax, _neuron_model_2_jax]
    estimators = [seed_programs.parameter_estimator_1, seed_programs.parameter_estimator_2]

    results = []
    for i, (model, estimator) in enumerate(zip(models, estimators), start=1):
        _, _, train_loss, _ = objective(
            model=model,
            param_estimator=estimator,
            x=inputs_train,
            y=outputs_train,
            create_train_test_trial_split_fn=data_parser.create_train_test_trial_split,
            fit_params=settings["fit_params"],
            max_iter=settings["max_iter"],
            param_penalty_weight=settings["param_penalty_weight"],
            tol=settings["tol"],
            learning_rate=settings["learning_rate"],
            use_param_estimator=True,
            trial_batch_size=settings["trial_batch_size"],
            random_seed=settings["random_seed"],
        )
        _, _, test_loss, _ = objective(
            model=model,
            param_estimator=estimator,
            x=inputs_test,
            y=outputs_test,
            create_train_test_trial_split_fn=data_parser.create_train_test_trial_split,
            fit_params=settings["fit_params"],
            max_iter=settings["max_iter"],
            param_penalty_weight=settings["param_penalty_weight"],
            tol=settings["tol"],
            learning_rate=settings["learning_rate"],
            use_param_estimator=True,
            trial_batch_size=settings["trial_batch_size"],
            random_seed=settings["random_seed"],
        )
        results.append({"seed": i, "train_loss": float(train_loss), "test_loss": float(test_loss)})
    return results


def test_orientation_tuning_seed_losses_match_baseline():
    if not BASELINE_PATH.exists():
        pytest.skip(f"Missing baseline file: {BASELINE_PATH}")

    baseline = json.loads(BASELINE_PATH.read_text())
    settings = baseline["settings"]

    data_path = Path(settings["data_path"])
    if not data_path.exists():
        pytest.skip(f"Orientation tuning dataset not found at {data_path}")

    actual = _compute_seed_losses(settings)
    expected = baseline["results"]

    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got["seed"] == want["seed"]
        assert np.isclose(got["train_loss"], want["train_loss"], rtol=1e-5, atol=1e-5), (
            f"Seed {got['seed']} train_loss changed: {got['train_loss']} vs {want['train_loss']}"
        )
        assert np.isclose(got["test_loss"], want["test_loss"], rtol=1e-5, atol=1e-5), (
            f"Seed {got['seed']} test_loss changed: {got['test_loss']} vs {want['test_loss']}"
        )
