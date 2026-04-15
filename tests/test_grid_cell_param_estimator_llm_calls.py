import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.grid_cells import spec
from src.hypothesis_engine import compute_initial_params
from src import utils


def _make_grid_cell_batch(n_samples: int = 3, n_trials: int = 144) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    pos_x = rng.uniform(-1.0, 1.0, size=(n_samples, n_trials))
    pos_y = rng.uniform(-1.0, 1.0, size=(n_samples, n_trials))

    params_per_sample = [
        {
            "lam": 0.60 + 0.03 * i,
            "theta": 0.10 * i,
            "phi_x": -0.15 + 0.08 * i,
            "phi_y": 0.12 - 0.05 * i,
            "baseline": 0.15 + 0.02 * i,
            "amplitude": 0.90 + 0.10 * i,
            "sigma": 0.12 + 0.01 * i,
        }
        for i in range(n_samples)
    ]

    response = np.stack(
        [
            spec.model_v1({"pos_x": pos_x[i], "pos_y": pos_y[i]}, params_per_sample[i])
            for i in range(n_samples)
        ],
        axis=0,
    )
    response += rng.normal(scale=0.01, size=response.shape)
    response = np.clip(response, 0.0, None)

    return {
        "pos_x": pos_x,
        "pos_y": pos_y,
        "response": response,
    }


def test_grid_cell_param_estimators_accept_single_sample_dict_data():
    data = _make_grid_cell_batch(n_samples=1, n_trials=196)
    sample = utils.slice_data_samples(data, 0)

    for estimator, expected_keys in (
        (spec.param_est_v1, set(spec.model_v1.DEFAULT_PARAMS)),
        (spec.param_est_v2, set(spec.model_v2.DEFAULT_PARAMS)),
    ):
        params = estimator(sample)

        assert set(params) == expected_keys
        for key, value in params.items():
            arr = np.asarray(value)
            assert arr.shape == ()
            assert np.isfinite(arr).all(), key


def test_compute_initial_params_batches_grid_cell_param_dicts():
    data = _make_grid_cell_batch(n_samples=4, n_trials=169)

    params = compute_initial_params(spec.param_est_v1, spec.model_v1, data)

    assert set(params) == set(spec.model_v1.DEFAULT_PARAMS)
    for key in spec.model_v1.DEFAULT_PARAMS:
        values = np.asarray(params[key])
        assert values.shape == (4,)
        assert np.isfinite(values).all(), key


def test_compute_initial_params_fallback_uses_model_default_params():
    data = _make_grid_cell_batch(n_samples=2, n_trials=100)

    def failing_estimator(_data):
        raise RuntimeError("estimator failure")

    params = compute_initial_params(failing_estimator, spec.model_v1, data)

    for key, default_value in spec.model_v1.DEFAULT_PARAMS.items():
        values = np.asarray(params[key])
        assert values.shape == (2,)
        assert np.allclose(values, default_value)


def main(argv: list[str] | None = None) -> int:
    import pytest

    if argv is None:
        argv = sys.argv[1:]
    pytest_args = list(argv) if argv else ["-q"]
    pytest_args.append(__file__)
    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
