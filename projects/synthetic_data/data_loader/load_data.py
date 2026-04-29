from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from typing import Dict, Tuple


def load_data(
    data_path: str = "",
    seed: int = 42,
    n_samples: int = 1000,
    n_trials: int = 2000,
    noise_std: float = 0.1,
    n_eval_trials: int = 100,
) -> Tuple[Tuple[Dict, Dict], Tuple[Dict, Dict], Dict]:
    """
    Simulate synthetic single-input regression data.

    Data-generating process:
    y = (a*x^2 + b*x + c) * sin(k*x + phi_0) + epsilon,
    where epsilon ~ N(0, noise_std^2), and each sample has its own parameters.

    Returns
    -------
    X_discover, X_validate, X_eval
        X_discover = (train, test) dicts split by trials for use in the LLM loop.
        X_validate = (train, test) dicts held out for final evaluation.
        X_eval = small fixed trial subset from X_discover for fingerprinting.
    """
    rng = np.random.default_rng(seed)

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

    X = {'x': np.tile(x, (n_samples, 1)), 'y': y}

    return _split(X, seed, n_eval_trials)


def loss_fn(model_output, data):
    """Scaled squared error loss."""
    y_true = data['y']
    return jnp.mean(10 * (y_true - model_output) ** 2)


# ── internal helpers ──

def _split(
    X: Dict[str, np.ndarray],
    seed: int,
    n_eval_trials: int,
) -> Tuple[Tuple[Dict, Dict], Tuple[Dict, Dict], Dict]:
    """Random sample split + random trial split → (X_discover, X_validate, X_eval)."""
    n_samples = next(iter(X.values())).shape[0]
    n_trials = next(iter(X.values())).shape[-1]
    rng = np.random.default_rng(seed + 1)  # offset to decouple from data generation seed

    perm_s = rng.permutation(n_samples)
    disc_idx = np.sort(perm_s[:n_samples // 2])
    val_idx = np.sort(perm_s[n_samples // 2:])

    perm_t = rng.permutation(n_trials)
    train_trials = np.sort(perm_t[:n_trials // 2])
    test_trials = np.sort(perm_t[n_trials // 2:])

    def _sel(sidx, tidx):
        return {k: v[sidx][..., tidx] for k, v in X.items()}

    X_disc_train = _sel(disc_idx, train_trials)
    X_disc_test = _sel(disc_idx, test_trials)
    X_val_train = _sel(val_idx, train_trials)
    X_val_test = _sel(val_idx, test_trials)

    n_eval = min(n_eval_trials, len(train_trials))
    eval_trials = np.sort(rng.choice(train_trials, n_eval, replace=False))
    X_eval = _sel(disc_idx, eval_trials)

    return (X_disc_train, X_disc_test), (X_val_train, X_val_test), X_eval


def _target_function(x, a, b, c, k, phi_0):
    """Synthetic ground-truth: f(x) = (a*x^2 + b*x + c) * sin(k*x + phi_0)"""
    return (a * x**2 + b * x + c) * np.sin(k * x + phi_0)
