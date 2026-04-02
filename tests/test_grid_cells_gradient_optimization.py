import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hypothesis_engine import objective
from src import utils


def _jax_grid_model(data, params):
    x = jnp.asarray(data["pos_x"])
    y = jnp.asarray(data["pos_y"])

    lam = jnp.clip(params["lam"], 0.2, 1.5)
    theta = params["theta"]
    baseline = params["baseline"]
    amplitude = params["amplitude"]

    ct = jnp.cos(theta)
    st = jnp.sin(theta)
    xr = ct * x + st * y
    yr = -st * x + ct * y

    phase_x = 2.0 * jnp.pi * xr / lam
    phase_y = 2.0 * jnp.pi * yr / lam
    return baseline + amplitude * (jnp.cos(phase_x) + jnp.cos(phase_y))


_jax_grid_model.DEFAULT_PARAMS = {
    "lam": 0.5,
    "theta": 0.0,
    "baseline": 0.0,
    "amplitude": 0.5,
}


def _param_estimator(data):
    response = np.asarray(data["response"])
    return {
        "lam": 0.8,
        "theta": 0.25,
        "baseline": float(np.mean(response) - 0.25),
        "amplitude": max(float(np.std(response)), 0.1),
    }


def _loss_fn(model_output, data):
    return (jnp.asarray(data["response"]) - model_output) ** 2


def _make_dataset(n_samples: int = 3, n_trials: int = 120) -> dict[str, np.ndarray]:
    grid = np.linspace(-1.0, 1.0, n_trials)
    pos_x = np.broadcast_to(grid[None, :], (n_samples, n_trials))
    pos_y = np.broadcast_to(np.roll(grid, 9)[None, :], (n_samples, n_trials))

    true_params = [
        {"lam": 0.58, "theta": 0.05, "baseline": 0.20, "amplitude": 0.90},
        {"lam": 0.64, "theta": 0.12, "baseline": 0.15, "amplitude": 0.75},
        {"lam": 0.72, "theta": -0.08, "baseline": 0.10, "amplitude": 1.05},
    ]
    response = np.stack(
        [
            np.asarray(_jax_grid_model({"pos_x": pos_x[i], "pos_y": pos_y[i]}, true_params[i]))
            for i in range(n_samples)
        ],
        axis=0,
    )

    return {
        "pos_x": pos_x,
        "pos_y": pos_y,
        "response": response,
    }


def test_objective_improves_grid_cell_loss_with_dict_data():
    data = _make_dataset()
    trial_idx = np.arange(utils.data_n_trials(data))
    train = utils.slice_data_trials(data, trial_idx[::2])
    test = utils.slice_data_trials(data, trial_idx[1::2])

    initial_loss, initial_params, final_loss, final_params = objective(
        model=_jax_grid_model,
        param_estimator=_param_estimator,
        data=[train, test],
        loss_fn=_loss_fn,
        param_penalty_weight=0.0,
        fit_params=True,
        max_iter=200,
        learning_rate=0.05,
        grad_descent_batch_size=32,
    )

    assert final_loss < initial_loss

    for params in (initial_params, final_params):
        assert set(params) == {"lam", "theta", "baseline", "amplitude"}
        for key, value in params.items():
            arr = np.asarray(value)
            assert arr.shape == (3,)
            assert np.isfinite(arr).all(), key


def test_objective_accepts_single_sample_dicts_after_trial_splitting():
    data = _make_dataset(n_samples=2, n_trials=60)
    train = utils.slice_data_trials(data, slice(None, None, 2))
    sample = utils.get_data_sample(train, 0)

    params = _param_estimator(sample)
    prediction = _jax_grid_model(sample, params)

    assert sample.keys() == {"pos_x", "pos_y", "response"}
    assert np.asarray(sample["pos_x"]).ndim == 1
    assert np.asarray(prediction).shape == np.asarray(sample["response"]).shape


def main(argv: list[str] | None = None) -> int:
    import pytest

    if argv is None:
        argv = sys.argv[1:]
    pytest_args = list(argv) if argv else ["-q"]
    pytest_args.append(__file__)
    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
