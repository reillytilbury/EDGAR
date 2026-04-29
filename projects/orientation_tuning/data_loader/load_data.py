from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from typing import Dict, Tuple


def load_data(
    data_path: str,
    activity_threshold: float,
    conc_threshold: float,
    random_seed: int = 42,
    n_eval_trials: int = 100,
    n_eval_samples: int = 10,
) -> Tuple[Tuple[Dict, Dict], Tuple[Dict, Dict], Dict]:
    """
    Load and preprocess orientation-tuning neural data.

    Returns
    -------
    X_discover, X_validate, X_eval
        X_discover = (train, test) dicts split by trials for use in the LLM loop.
        X_validate = (train, test) dicts held out for final evaluation.
        X_eval = small fixed trial subset from X_discover for fingerprinting.
    """
    neural_data = np.load(data_path, allow_pickle=True).item()
    response = _extract_stimulus_related_response(neural_data, n_pcs=0)

    angles = neural_data['istim']
    n_trials_raw = response.shape[1]
    n_trials_small = int(n_trials_raw * activity_threshold)

    active = (response > 0).astype(np.float32)
    firing_probs = np.mean(active, axis=1)
    conc = np.abs(
        np.sum(np.exp(2j * angles)[np.newaxis, :] * response, axis=1)
        / np.sum(response, axis=1)
    )
    good_cells = np.where((firing_probs > activity_threshold) & (conc > conc_threshold))[0]
    n_good_cells = len(good_cells)

    response_cropped = np.zeros((n_good_cells, n_trials_small))
    angles_cropped = np.zeros((n_good_cells, n_trials_small))
    for i, cell in enumerate(good_cells):
        active_trials_idx = np.where(response[cell] > 0)[0][:n_trials_small]
        response_cropped[i] = response[cell, active_trials_idx]
        angles_cropped[i] = angles[active_trials_idx]

    response_cropped = _normalize_response(response_cropped)
    X = {
        'stimulus': angles_cropped,
        'response': response_cropped,
    }

    return _split(X, random_seed, n_eval_trials, n_eval_samples)


def loss_fn(model_output, data):
    """Scaled squared error loss."""
    return jnp.mean(10 * (data['response'] - model_output) ** 2)


# ── internal helpers ──

def _split(
    X: Dict[str, np.ndarray],
    seed: int,
    n_eval_trials: int,
    n_eval_samples: int,
) -> Tuple[Tuple[Dict, Dict], Tuple[Dict, Dict], Dict]:
    """Random sample split + random trial split → (X_discover, X_validate, X_eval)."""
    n_samples = next(iter(X.values())).shape[0]
    n_trials = next(iter(X.values())).shape[-1]
    rng = np.random.default_rng(seed)

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
    eval_samples = np.sort(rng.choice(disc_idx, min(n_eval_samples, len(disc_idx)), replace=False))
    X_eval = _sel(eval_samples, eval_trials)

    # Store which position each eval sample occupies in disc_idx for param matching in scoring
    eval_sample_positions = np.searchsorted(disc_idx, eval_samples)
    X_eval['_sample_indices'] = eval_sample_positions

    return (X_disc_train, X_disc_test), (X_val_train, X_val_test), X_eval


def _extract_stimulus_related_response(
    data: dict,
    n_pcs: int = 8,
    z_score: bool = False,
    spont_mean_removal: bool = False,
) -> np.ndarray:
    sresp = np.asarray(data['sresp'])
    if spont_mean_removal:
        sresp = sresp - np.asarray(data['mean_spont'])[:, np.newaxis]
    if n_pcs > 0:
        u_spont = np.asarray(data['u_spont'])
        sresp = sresp - u_spont[:, :n_pcs] @ (u_spont[:, :n_pcs].T @ sresp)
    if z_score:
        sresp = (sresp - np.mean(sresp, axis=1, keepdims=True)) / np.std(sresp, axis=1, keepdims=True)
    return sresp


def _normalize_response(response: np.ndarray) -> np.ndarray:
    return 100 * response / np.linalg.norm(response, axis=1, keepdims=True)
