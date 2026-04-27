import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple

from src import utils


def load_and_process_data(
    data_path: str,
    activity_threshold: float,
    conc_threshold: float,
) -> Dict[str, np.ndarray]:
    """
    Load and preprocess orientation-tuning neural data.

    Returns
    -------
    dict with keys:
        'stimulus': shape (n_samples, n_trials), stimulus angles (radians)
        'response': shape (n_samples, n_trials), neural responses
    """
    neural_data = np.load(data_path, allow_pickle=True).item()
    response = _extract_stimulus_related_response(neural_data, n_pcs=0)

    angles = neural_data['istim']
    n_trials = response.shape[1]
    n_trials_small = int(n_trials * activity_threshold)

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
    return {
        'stimulus': angles_cropped,
        'response': response_cropped,
    }


def train_test_split(
    X: Dict[str, np.ndarray],
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = utils.data_n_samples(X)
    n_trials = utils.data_n_trials(X)
    assert n_samples >= 2
    assert n_trials >= 2

    sample_key = jax.random.PRNGKey(random_seed)
    shuffled_samples = jax.random.permutation(sample_key, jnp.arange(n_samples))
    train_samples = np.asarray(shuffled_samples[:n_samples // 2], dtype=np.int64)

    trial_key = jax.random.PRNGKey(0)
    shuffled_trials = jax.random.permutation(trial_key, jnp.arange(n_trials))
    train_trials = np.asarray(shuffled_trials[:n_trials // 2], dtype=np.int64)

    return train_samples, train_trials


def loss_fn(model_output, data):
    """Scaled squared error loss."""
    return jnp.mean(10 * (data['response'] - model_output) ** 2)


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
