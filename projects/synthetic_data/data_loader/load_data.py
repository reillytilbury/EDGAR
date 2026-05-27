from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def _to_jax(d):
    return {k: jnp.array(v) if k != "_sample_indices" else v for k, v in d.items()}


def load_data(
    data_path: str = "",
    seed: int = 42,
    n_samples: int = 1000,
    n_trials: int = 2000,
    noise_std: float = 0.1,
    n_eval_samples: int = 10,
):
    """
    Simulate synthetic single-input regression data.

    Data-generating process:
    y = (a*x^2 + b*x + c) * sin(k*x + phi_0) + epsilon,
    where epsilon ~ N(0, noise_std^2), and each sample has its own parameters.

    Returns
    -------
    X_discover, X_validate, X_eval
        Samples split 50/50, trials split 50/50 within discover/validate.

        X_discover = (train, test)
            X_disc_train: (n_samples//2, n_trials//2) - used by LLM loop.
            X_disc_test:  (n_samples//2, n_trials//2) - test within discovery phase.

        X_validate = (train, test)
            X_val_train: (n_samples//2, n_trials//2) - held out, unseen by LLM.
            X_val_test:  (n_samples//2, n_trials//2) - held out final evaluation.

        X_eval
            Small fingerprint subset from discover train trials, used for deduplication.
            Shape: (n_eval_samples, n_trials//2).
    """
    rng = np.random.default_rng(seed)

    # ── generate per-sample parameters and observations ──
    a     = rng.uniform(-1.0, 1.0, n_samples)
    b     = rng.uniform(-1.0, 1.0, n_samples)
    c     = rng.uniform(-1.0, 1.0, n_samples)
    k     = rng.uniform(1.0, 5.0, n_samples)
    phi_0 = rng.uniform(0.0, 2 * np.pi, n_samples)

    x     = rng.uniform(-1.0, 1.0, n_trials)
    noise = rng.normal(0.0, noise_std, (n_samples, n_trials))
    y     = np.array([_target_function(x, a[i], b[i], c[i], k[i], phi_0[i])
                      for i in range(n_samples)]) + noise
    X     = {'x': np.tile(x, (n_samples, 1)), 'y': y}

    # ── split samples 50/50 into discover / validate ──
    # Use seed+1 to decouple splitting from data generation
    rng = np.random.default_rng(seed + 1)

    perm_s    = rng.permutation(n_samples)
    disc_idx  = np.sort(perm_s[:n_samples // 2])
    val_idx   = np.sort(perm_s[n_samples // 2:])

    # ── split trials 50/50 into train / test ──
    perm_t       = rng.permutation(n_trials)
    train_trials = np.sort(perm_t[:n_trials // 2])
    test_trials  = np.sort(perm_t[n_trials // 2:])

    x_disc = X['x'][disc_idx]
    y_disc = X['y'][disc_idx]
    x_val  = X['x'][val_idx]
    y_val  = X['y'][val_idx]

    X_disc_train = {'x': x_disc[:, train_trials], 'y': y_disc[:, train_trials]}
    X_disc_test  = {'x': x_disc[:, test_trials],  'y': y_disc[:, test_trials]}
    X_val_train  = {'x': x_val[:,  train_trials],  'y': y_val[:,  train_trials]}
    X_val_test   = {'x': x_val[:,  test_trials],   'y': y_val[:,  test_trials]}

    # ── build X_eval: small subset of discover train for fingerprinting ──
    eval_samples = np.sort(rng.choice(disc_idx, min(n_eval_samples, len(disc_idx)), replace=False))
    eval_pos     = np.searchsorted(disc_idx, eval_samples)
    X_eval       = {'x': x_disc[eval_pos][:, train_trials], 'y': y_disc[eval_pos][:, train_trials]}

    # Position of each eval sample within disc_idx, for param matching in scoring
    X_eval['_sample_indices'] = eval_pos

    return (_to_jax(X_disc_train), _to_jax(X_disc_test)), (_to_jax(X_val_train), _to_jax(X_val_test)), _to_jax(X_eval)


def loss_fn(model_output, data):
    """Scaled squared error loss."""
    return 10 * jnp.mean((data['y'] - model_output) ** 2, axis=-1)


def _target_function(x, a, b, c, k, phi_0):
    """Synthetic ground-truth: f(x) = (a*x^2 + b*x + c) * sin(k*x + phi_0)"""
    return (a * x**2 + b * x + c) * np.sin(k * x + phi_0)
