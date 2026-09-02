import jax.numpy as jnp
import pytest
from edgar.scoring.optimizer import Optimizer


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
    return Optimizer(model_fn, loss_fn, data_train, gd_config)


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
