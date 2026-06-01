from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def _to_jax(d):
    return {k: jnp.array(v) if k != "_sample_indices" else v for k, v in d.items()}


def target_function(x, a, b, c, k, phi_0):
    """
    Synthetic ground-truth function.

    f(x) = (a*x^2 + b*x + c) * sin(k*x + phi_0)
    """
    return (a * x**2 + b * x + c) * np.sin(k * x + phi_0)


def load_data(
    data_path: str = "",
    n_samples: int = 20,
    n_trials: int = 50,
    n_eval_samples: int = 4,
    **kwargs,
):
    """
    Minimal noiseless synthetic data for system tests.
    Data-generating process: y = (a*x^2 + b*x + c) * sin(k*x + phi_0).
    No use of random generation to fix behaviour.
    """
    a = np.linspace(0.1, 1, n_samples)
    b = np.linspace(0.1, 1, n_samples)
    c = np.linspace(0.1, 1, n_samples)
    k = np.full(n_samples, 6)
    phi_0 = np.zeros((n_samples,))
    x = np.linspace(-1.0, 1.0, n_trials)
    y = np.array(
        [target_function(x, a[i], b[i], c[i], k[i], phi_0[i]) for i in range(n_samples)]
    )
    X = {"x": np.tile(x, (n_samples, 1)), "y": y}

    disc_idx = np.arange(n_samples // 2)
    val_idx = np.arange(n_samples // 2, n_samples)

    train_trials = np.arange(n_trials)[::2]
    test_trials = np.arange(n_trials)[1::2]

    x_disc = X["x"][disc_idx]
    y_disc = X["y"][disc_idx]
    x_val = X["x"][val_idx]
    y_val = X["y"][val_idx]

    X_disc_train = {"x": x_disc[:, train_trials], "y": y_disc[:, train_trials]}
    X_disc_test = {"x": x_disc[:, test_trials], "y": y_disc[:, test_trials]}
    X_val_train = {"x": x_val[:, train_trials], "y": y_val[:, train_trials]}
    X_val_test = {"x": x_val[:, test_trials], "y": y_val[:, test_trials]}

    # take the first n_eval_samples from the discovery set as eval samples
    eval_pos = np.arange(len(disc_idx))[:n_eval_samples]
    X_eval = {
        "x": x_disc[eval_pos][:, train_trials],
        "y": y_disc[eval_pos][:, train_trials],
    }
    X_eval["_sample_indices"] = eval_pos

    return (
        (_to_jax(X_disc_train), _to_jax(X_disc_test)),
        (_to_jax(X_val_train), _to_jax(X_val_test)),
        _to_jax(X_eval),
    )


def loss_fn(model_output, data):
    """Scaled squared error loss."""
    return 10 * jnp.mean((data["y"] - model_output) ** 2, axis=-1)
