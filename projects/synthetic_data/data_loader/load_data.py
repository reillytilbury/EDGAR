import numpy as np
from typing import Dict, Tuple

from src import utils

# Configuration constants from project config
SEED = 42
N_SAMPLES = 1000
N_TRIALS = 2000
NOISE_STD = 0.1


def load_and_process_data(
    data_path: str = "",
    SEED: int = SEED,
    n_samples: int = N_SAMPLES,
    n_trials: int = N_TRIALS,
    noise_std: float = NOISE_STD,
) -> Dict[str, np.ndarray]:
    """
    Simulate synthetic single-input regression data.

    Data-generating process:
    y = (a*x^2 + b*x + c) * sin(k*x + phi_0) + epsilon,
    where epsilon ~ N(0, noise_std^2), and each sample has its own parameters.

    Returns
    -------
    dict[str, np.ndarray]
        - 'x': shape (n_samples, n_trials)
        - 'y': shape (n_samples, n_trials)
    """
    rng = np.random.default_rng(SEED)

    a = rng.uniform(-1.0, 1.0, n_samples)
    b = rng.uniform(-1.0, 1.0, n_samples)
    c = rng.uniform(-1.0, 1.0, n_samples)
    k = rng.uniform(1.0, 5.0, n_samples)
    phi_0 = rng.uniform(0.0, 2 * np.pi, n_samples)

    x = rng.uniform(-1.0, 1.0, n_trials)
    noise = rng.normal(0.0, noise_std, (n_samples, n_trials))

    y = np.array([
        _target_function(x, a[i], b[i], c[i], k[i], phi_0[i]) for i in range(n_samples)
    ]) + noise

    return {'x': np.tile(x, (n_samples, 1)), 'y': y}


def train_test_split(
    X: Dict[str, np.ndarray],
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = utils.data_n_samples(X)
    n_trials = utils.data_n_trials(X)
    assert n_samples >= 2
    assert n_trials >= 2

    rng = np.random.default_rng(random_seed)
    train_samples = rng.choice(np.arange(n_samples), n_samples // 2, replace=False)
    train_trials = rng.choice(np.arange(n_trials), n_trials // 2, replace=False)
    return train_samples, train_trials


def loss_fn(model_output, data):
    """Scaled squared error loss."""
    y_true = data['y']
    return jnp.mean(10 * (y_true - model_output) ** 2)


def _target_function(x, a, b, c, k, phi_0):
    """Synthetic ground-truth: f(x) = (a*x^2 + b*x + c) * sin(k*x + phi_0)"""
    return (a * x**2 + b * x + c) * np.sin(k * x + phi_0)
