from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def _to_jax(d):
    return {k: jnp.array(v) if k != "_sample_indices" else v for k, v in d.items()}


def load_data(
    data_path: str,
    activity_threshold: float,
    conc_threshold: float,
    random_seed: int = 42,
    n_eval_samples: int = 10,
):
    """
    Load and preprocess orientation-tuning neural data.

    Returns
    -------
    X_discover, X_validate, X_eval
        Samples split 50/50, trials split 50/50 within discover/validate.

        X_discover = (train, test)
            X_disc_train: (n_cells//2, n_trials//2) - used by LLM loop.
            X_disc_test:  (n_cells//2, n_trials//2) - test within discovery phase.

        X_validate = (train, test)
            X_val_train: (n_cells//2, n_trials//2) - held out, unseen by LLM.
            X_val_test:  (n_cells//2, n_trials//2) - held out final evaluation.

        X_eval
            Small fingerprint subset from discover train trials, used for deduplication.
            Shape: (n_eval_samples, n_trials//2).
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
    angles_cropped   = np.zeros((n_good_cells, n_trials_small))
    for i, cell in enumerate(good_cells):
        active_trials_idx = np.where(response[cell] > 0)[0][:n_trials_small]
        response_cropped[i] = response[cell, active_trials_idx]
        angles_cropped[i]   = angles[active_trials_idx]

    response_cropped = _normalize_response(response_cropped)
    X_stimulus = angles_cropped
    X_response = response_cropped

    n_samples, n_trials = X_stimulus.shape

    # ── split samples 50/50 into discover / validate ──
    rng = np.random.default_rng(random_seed)

    perm_s   = rng.permutation(n_samples)
    disc_idx = np.sort(perm_s[:n_samples // 2])
    val_idx  = np.sort(perm_s[n_samples // 2:])

    # ── split trials 50/50 into train / test ──
    perm_t       = rng.permutation(n_trials)
    train_trials = np.sort(perm_t[:n_trials // 2])
    test_trials  = np.sort(perm_t[n_trials // 2:])

    stimulus_disc = X_stimulus[disc_idx]
    response_disc = X_response[disc_idx]
    stimulus_val  = X_stimulus[val_idx]
    response_val  = X_response[val_idx]

    X_disc_train = {'stimulus': stimulus_disc[:, train_trials], 'response': response_disc[:, train_trials]}
    X_disc_test  = {'stimulus': stimulus_disc[:, test_trials],  'response': response_disc[:, test_trials]}
    X_val_train  = {'stimulus': stimulus_val[:,  train_trials], 'response': response_val[:,  train_trials]}
    X_val_test   = {'stimulus': stimulus_val[:,  test_trials],  'response': response_val[:,  test_trials]}

    # ── build X_eval: small subset of discover train for fingerprinting ──
    eval_samples = np.sort(rng.choice(disc_idx, min(n_eval_samples, len(disc_idx)), replace=False))
    eval_pos     = np.searchsorted(disc_idx, eval_samples)
    X_eval = {
        'stimulus': stimulus_disc[eval_pos][:, train_trials],
        'response': response_disc[eval_pos][:, train_trials],
        '_sample_indices': eval_pos,
    }

    return (_to_jax(X_disc_train), _to_jax(X_disc_test)), (_to_jax(X_val_train), _to_jax(X_val_test)), _to_jax(X_eval)


def loss_fn(model_output, data):
    """Scaled squared error loss."""
    return 10 * jnp.mean((data['response'] - model_output) ** 2, axis=-1)


# ── internal helpers ──

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
