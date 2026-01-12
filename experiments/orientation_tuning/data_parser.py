import os
import asyncio
import ast
import jax
import time
import numpy as np
import scipy as sc
import asyncio
import pickle
import jax.numpy as jnp
import pandas as pd
from dotenv import load_dotenv
from typing import Callable, Dict, Any, Optional, Sequence, Union, Tuple, List
# gemini client
from google import genai
from google.genai import types


def vmap_over_cells(model_fn):
    """Return a version of `model_fn` that accepts
       (theta, params_matrix) and runs one row per cell."""
    def _wrapped(theta, params_row):
        # params_row shape: (k,)  ← one cell’s parameters
        return model_fn(theta, *params_row)   # unpack to scalars
    return jax.vmap(_wrapped, in_axes=(None, 0))   # x shared, params batched

def circular_distance_rad_np(angle1, angle2) -> np.ndarray:
    """Shortest distance between two angles (radians) on a circle.
    Args:
        angle1: First angle (radians). (float or np.ndarray)
        angle2: Second angle (radians). (float or np.ndarray)
    Returns:
        diff: absolute smalles distance between the two angles (radians). (float or np.ndarray)
    """
    diff = angle1 - angle2
    diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi  # Normalize to [-pi, pi]
    return np.abs(diff)

def circular_distance_rad_jax(angle1, angle2) -> jnp.ndarray:
    """Shortest distance between two angles (radians) on a circle.
    Args:
        angle1: First angle (radians). (float or jnp.ndarray)
        angle2: Second angle (radians). (float or jnp.ndarray)
    Returns:
        diff: absolute smalles distance between the two angles (radians). (float or jnp.ndarray)
    """
    diff = angle1 - angle2
    diff = jnp.mod(diff + jnp.pi, 2 * jnp.pi) - jnp.pi  # Normalize to [-pi, pi] # Changed np to jnp
    return jnp.abs(diff) # Changed np to jnp

def extract_stimulus_related_response(data: dict, n_pcs: int = 8, z_score: bool = False, spont_mean_removal: bool = False) -> np.ndarray:
    """
    Extracts the stimulus-related response from the data. Copy pasted with small modifications from https://github.com/MouseLand/stringer-et-al-2019/blob/master/utils.py#L98
    Args:
        data (dict): The data dictionary containing the stimulus-related response and other information. Values expected to be convertible to JAX arrays.
        n_pcs (int): The number of spointaneous PCs to remove from the response.
        z_score (bool): Whether to z-score the response.
    Returns:
        stim_related_response (jnp.ndarray): The stimulus-related response matrix.
    """
    # Convert relevant data parts to JAX arrays explicitly if needed, JAX often handles np arrays implicitly
    sresp = np.asarray(data['sresp'])

    if spont_mean_removal:
        mean_spont = np.asarray(data['mean_spont'])
        sresp = sresp - mean_spont[:, np.newaxis]

    if n_pcs > 0:
        u_spont = np.asarray(data['u_spont'])
        sresp = sresp - u_spont[:, :n_pcs] @ (u_spont[:, :n_pcs].T @ sresp)

    if z_score:
        sresp = (sresp - np.mean(sresp, axis=1, keepdims=True)) / np.std(sresp, axis=1, keepdims=True)

    return sresp

def unbiased_signal_fraction(R, min_repeats=2):
    """
    Compute unbiased fraction of stimulus-related variance (Sahani & Linden, 2003)
    using explicit repeats per angle (no binning).

    Parameters
    ----------
    R : array, shape (n_repeats, n_cells, n_angles)
        Neural responses: repeats × cells × unique angles.
    min_repeats : int
        Minimum repeats per stimulus (default: 2).

    Returns
    -------
    signal_fraction : array, shape (n_cells,)
        Fraction of total variance explained by the stimulus.
    components : dict
        Contains:
          - 'S2': stimulus-related variance per cell
          - 'V2': noise variance per cell
          - 'mu_angles': mean response per angle (n_cells × n_angles)
          - 'var_angles': within-angle variance (n_cells × n_angles)
    """

    n_repeats, n_cells, n_angles = R.shape
    if n_repeats < min_repeats:
        raise ValueError(f"Need at least {min_repeats} repeats per angle, got {n_repeats}.")

    # ----------------------------------------------------------------
    # Per-angle means and within-angle variances
    # ----------------------------------------------------------------
    mu_angles = np.mean(R, axis=0)        # (n_cells, n_angles)
    var_angles = np.var(R, axis=0, ddof=1)  # (n_cells, n_angles)

    # ----------------------------------------------------------------
    # Unbiased stimulus-related and noise variances
    # ----------------------------------------------------------------
    N = n_angles
    R_s = np.full(N, n_repeats, dtype=float)  # repeats per stimulus, constant

    # Global mean across all stimuli
    fbar_dot = np.mean(mu_angles, axis=1)  # (n_cells,)

    # Across-stimulus variance (stimulus-related term)
    term1 = np.mean((mu_angles - fbar_dot[:, None])**2, axis=1)

    # Bias correction term (Eq. from Sahani & Linden, 2003)
    term2 = ((N - 1) / N**2) * np.sum(var_angles / R_s[None, :], axis=1)

    S2 = term1 - term2  # unbiased stimulus-related variance
    V2 = np.sum(var_angles / R_s[None, :], axis=1) / N  # average noise variance

    # Fraction of total variance due to the stimulus
    signal_fraction = S2 / (S2 + V2)
    signal_fraction = np.clip(signal_fraction, 0, 1)

    return signal_fraction, {
        "S2": S2,
        "V2": V2,
        "mu_angles": mu_angles,
        "var_angles": var_angles,
    }

def load_data(data_dir: Union[str, List[List[str]]],
              data_type: str = 'stringer',
              shuffle: bool = False,
              conc_thresh: float = 0.4, 
              activity_thresh: float = 0.0, 
              signal_fraction_thresh: float = 0.0,
              n_pcs: int = 0,
              n_bins: int = 256, 
              min_repeats: int = 6) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Load and preprocess neural data from a specified directory.

    Parameters
    ----------
    data_dir : str
        Path to the .npy file containing neural data (if 'stringer' or 'ali') or [[data_paths], [metadata_paths]] (if 'jacob').
    data_type : str
        Type of data to load ('stringer' or 'jacob' or 'ali')
    shuffle : bool
        Whether to shuffle the repeats for each trial. Only relevant if we have exact repeats (i.e., Jacob's data).
    conc_thresh : float
        Concentration threshold for filtering neurons.
    activity_thresh : float
        Activity threshold for filtering neurons.
    signal_fraction_thresh : float
        Signal fraction threshold for filtering neurons.
    n_pcs : int
        Number of spontaneous principal components to use for spontaneous activity subtraction. (only for 'stringer' data)
    n_bins : int
        Number of bins for response averaging.
    min_repeats : int
        Minimum number of repeats for response averaging.

    Returns 
    -------
    response : jnp.ndarray
        Preprocessed neural response. (n_repeats, n_cells, n_bins)
    angles : jnp.ndarray
        Preprocessed stimulus angles. (n_bins,)
    """
    assert data_type in ['stringer', 'jacob', 'ali'], "data_type must be either 'stringer', 'jacob', or 'ali'"
    if n_pcs > 0:
        assert data_type == 'stringer', "n_pcs can only be used with 'stringer' data_type"

    # load data matrix (n_cells, n_trials) and angles (n_trials,)
    if data_type == 'stringer':
        neural_data = np.load(data_dir, allow_pickle=True).item()
        response = extract_stimulus_related_response(neural_data, n_pcs=n_pcs)
        angles = neural_data['istim']
        if shuffle:
            # shuffle responses for each trial
            n_trials = angles.shape[0]
            perm = np.random.permutation(n_trials)
            response = response[:, perm]
            angles = angles[perm]

    elif data_type == 'jacob':
        assert isinstance(data_dir, list) and len(data_dir) == 2, "For 'jacob' data_type, data_dir must be a list of two lists: [[data_paths], [metadata_paths]]"
        data_dirs, metadata_dirs = data_dir
        responses = []
        for data_dir in data_dirs:
            response = np.load(data_dir).T
            responses.append(response)
        angles = []
        for metadata_dir in metadata_dirs:
            mat_data = sc.io.loadmat(metadata_dir, simplify_cells=True)
            # in the single block case the first and last angles should be removed
            if 'BZ016' in metadata_dir:
                angles.append(np.array([entry['gratingOrient'] for entry in mat_data['block']['paramsValues']])[1:-1])
            else: 
                angles.append(np.array([entry['gratingOrient'] for entry in mat_data['block']['paramsValues']]))
        # remove responses where angle = 1
        for i in range(len(responses)):
            responses[i] = responses[i][:, angles[i] != 1]
            angles[i] = angles[i][angles[i] != 1]
            angles[i] = np.deg2rad(angles[i])
        # for each repeat, reorder angles and responses
        for i in range(len(responses)):
            responses[i] = responses[i][:, np.argsort(angles[i])]
            angles[i] = np.sort(angles[i])
        # now turn responses into an array and replace angles with any of its entries
        response = np.array(responses)
        n_blocks = response.shape[0]
        angles = angles[0]
        # optionally shuffle repeats for each trial
        if shuffle:
            for trial in range(len(angles)):
                perm = np.random.permutation(n_blocks)
                response[:, :, trial] = response[perm, :, trial]
        # Jacob's data included 0 as well as 2pi, so shift any angles starting with 6.2831 to 2pi - small epsilon
        angles[angles >= 6.2831] = 2 * np.pi - 1e-5
        response_flat = np.transpose(response, (1, 2, 0))  # n_cells x n_trials x n_blocks
        response_flat = response_flat.reshape(response_flat.shape[0], -1)  # n_cells x (n_trials*n_blocks)
        angles_flat = np.repeat(angles, n_blocks)  # now angles is (n_trials*n_blocks)
        response, angles = response_flat, angles_flat
    
    else:  # 'ali' data
        # with open(data_dir, 'rb') as f:
        #     neural_data = pickle.load(f)
        # response = neural_data['resps'].T    # shape (n_cells, n_trials)
        # angles = neural_data['stims'].astype(float) 
        # angles = angles % 360  # ensure angles are in [0, 360)
        # angles = np.deg2rad(angles)  # convert to radians
        # angles = angles % (2 * np.pi)  # ensure angles are in [0, 2pi)
        # # set each cell's 1th percentile response to 0
        # response = response - np.percentile(response, 1, axis=1, keepdims=True)
        # response[response < 0] = 0
        angles = np.load(data_dir[0])
        angles = np.deg2rad(angles)  # convert to radians
        response = np.load(data_dir[1])
        response = response.mean(axis=-1)
        response = response.T  # shape (n_cells, n_trials)
        # optionally shuffle responses for each trial
        if shuffle:
            n_trials = angles.shape[0]
            perm = np.random.permutation(n_trials)
            response = response[:, perm]
            angles = angles[perm]

    # Activity, concentration filtering
    active = (response > 0).astype(np.float32)
    firing_probs = np.mean(active, axis=1)
    conc = np.abs(np.sum(np.exp(2j * angles)[np.newaxis, :] * response, axis=1) / np.sum(response, axis=1))
    good_cells = np.where((firing_probs > activity_thresh) & (conc > conc_thresh))[0]
    n_good_cells = len(good_cells)
    print(f"Selected {n_good_cells} / {response.shape[0]} cells with activity > {activity_thresh} and concentration > {conc_thresh}.")

    # Keep only good cells
    conc = conc[good_cells]
    firing_probs = firing_probs[good_cells]
    response = response[good_cells, :]

    # bin responses
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # digitize angles
    bin_indices = np.digitize(angles, bin_edges) - 1
    # raise error if any bin has fewer than min_repeats
    min_bin_counts = np.min(np.bincount(bin_indices, minlength=n_bins))
    if min_bin_counts < min_repeats:
        raise ValueError(f"Not enough repeats in some bins. Minimum repeats: {min_bin_counts}, required: {min_repeats}")
    response_binned = np.zeros((min_repeats, n_good_cells, n_bins))
    for b in range(n_bins):
        relevant_indices = np.where(bin_indices == b)[0]
        n_responses = len(relevant_indices)
        pool_size = n_responses // min_repeats
        # take average of each pool
        mean_responses = []
        for r in range(min_repeats):
            # this choice of pool indices mixes blocks
            # pool_indices = relevant_indices[r * pool_size:(r + 1) * pool_size]
            # this choice of indices keeps blocks separate
            pool_indices = relevant_indices[r::min_repeats][:pool_size]
            mean_responses.append(np.mean(response[:, pool_indices], axis=1))
        response_binned[:, :, b] = np.array(mean_responses)
    angles = bin_centers
    response = response_binned

    # Convert to JAX arrays
    response, angles = jnp.asarray(response), jnp.asarray(angles)

    # Normalize responses so that for each cell and repeat, the RMS across bins is 1
    activity_norms = jnp.linalg.norm(response, axis=-1)
    normalization_factors = activity_norms / jnp.sqrt(n_bins)
    response = response / normalization_factors[:, :, None]
    # compute signal fraction for each cell
    signal_fraction = unbiased_signal_fraction(np.array(response))[0]
    reliable_cells = jnp.where(signal_fraction > signal_fraction_thresh)[0]
    n_reliable_cells = len(reliable_cells)
    print(f"Selected {n_reliable_cells} / {n_good_cells} cells with signal fraction > {signal_fraction_thresh}.")
    # keep only reliable cells
    response = response[:, reliable_cells, :]
    return response, angles

