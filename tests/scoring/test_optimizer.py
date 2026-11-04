import jax.numpy as jnp
import numpy as np
import pytest
from edgar.scoring.optimizer import AdamOptimizer, ScipyOptimizer


# Simple test model and data
def model_fn(data, params):
    return params["w"] * data["x"]


def loss_fn(output, data):
    return (output - data["y"]) ** 2


@pytest.fixture
def optimizer():
    x = jnp.array([1.0, 2.0, 3.0])
    data_train = {
        "x": jnp.stack([x, x]),
        "y": jnp.stack([x, 2 * x]),
    }  # data of shape (2, 3) = (n_samples, n_x)
    gd_config = {"max_iter": 100, "learning_rate": 0.1}
    return AdamOptimizer(model_fn, loss_fn, data_train, gd_config)


def test_optimizer_parallel_convergence(optimizer):
    # Test with 5 parallel optimizations starting from different values
    n_opts = 5
    n_samples = 2
    initial_parameters = [
        {"w": jnp.array(n_samples * [0.1])},
        {"w": jnp.array(n_samples * [0.5])},
        {"w": jnp.array(n_samples * [1.5])},
        {"w": jnp.array(n_samples * [2.0])},
        {"w": jnp.array(n_samples * [5.0])},
    ]
    # Check flattening of parameters
    flat_all, opt_state = optimizer.flatten_and_init_params(initial_parameters)
    assert flat_all.shape == (n_opts, n_samples)
    expected_flat_all = jnp.stack(
        [
            jnp.array(n_samples * [0.1]),
            jnp.array(n_samples * [0.5]),
            jnp.array(n_samples * [1.5]),
            jnp.array(n_samples * [2.0]),
            jnp.array(n_samples * [5.0]),
        ]
    )
    assert jnp.allclose(flat_all, expected_flat_all)
    # Check optimization
    optimized_params, loss_trajectories = optimizer.run_optimization(
        flat_all, opt_state
    )
    # Check optimized parameters of correct shape and value
    assert len(optimized_params) == n_opts
    for p in optimized_params:
        assert p["w"].shape == (n_samples,)
        assert jnp.allclose(
            p["w"], jnp.array([1.0, 2.0]), atol=0.05
        )  # Should converge to the true values

    # Check loss trajectories of correct shape and that optimization improved them
    assert loss_trajectories.shape == (optimizer.gd_config["max_iter"], n_opts)
    initial_losses = loss_trajectories[0, :]
    final_losses = loss_trajectories[-1, :]
    assert jnp.all(final_losses < initial_losses)


def test_scipy_optimizer_independently_fits_multiple_structured_samples():
    x = jnp.array([[1.0], [2.0]])  # (2, 1) two samples
    data = {"x": x, "y": jnp.array([[2.0], [3.0]]) * x}  # (2, 1)

    def scalar_loss(output, target):
        return jnp.mean((output - target["y"]) ** 2, axis=-1)

    # Structured params are handled by ravel/unravel and SciPy finite differences.
    # Each sample has its own target and should get its own fitted parameter.
    def model(data, params):
        return params["nested"]["w"] * data["x"]

    optimizer = ScipyOptimizer(
        model, scalar_loss, data, {"method": "Powell", "options": {"maxiter": 30}}
    )
    starts = [{"nested": {"w": jnp.full((2,), w)}} for w in (0.0, 8.0)]
    params, traces = optimizer.optimize(starts)
    assert len(params) == 2
    assert all(np.allclose(p["nested"]["w"], [2.0, 3.0], atol=1e-4) for p in params)
    assert traces.shape[0] == 2
    assert np.all(np.isfinite(traces))
    assert np.all(np.isfinite(traces[:, 0]))


def test_scipy_loss_is_traced_once_for_samples_of_same_shape():
    traces = []

    def model(data, params):
        # Called while tracing the jitted scalar loss, not on every SciPy call.
        traces.append(1)
        return params["w"] * data["x"]

    data = {
        "x": jnp.ones((2, 3)),
        "y": jnp.asarray([[2.0] * 3, [3.0] * 3]),
    }

    def loss(output, target):
        return jnp.mean((output - target["y"]) ** 2, axis=-1)

    optimizer = ScipyOptimizer(
        model, loss, data, {"method": "Nelder-Mead", "options": {"maxiter": 40}}
    )
    params, trajectories = optimizer.optimize([{"w": jnp.zeros(2)}])
    assert len(traces) == 1
    assert np.allclose(params[0]["w"], [2.0, 3.0], atol=0.1)
    assert trajectories.shape[0] == 1
    assert np.isfinite(trajectories).all()
