import numpy as np
import jax.numpy as jnp

from projects.minimodel_discovery.primitives import (
    gabor_bank16,
    local_gaussian_readout,
    pairwise_orientation_terms,
    quadrature_energy,
)
from projects.minimodel_discovery.seed_programs.param_est1 import (
    parameter_estimator as _param_est1,
    _normalized_to_pixel,
    _extract_patch,
    _pooled_energy_features,
    _positive_correlation_weights,
    _response_summary_stats,
)


def _whitened_stc_components(
    image: np.ndarray,
    response: np.ndarray,
    x0: float,
    y0: float,
    patch_radius: int = 4,
) -> tuple[np.ndarray, np.ndarray, float]:
    height, width, n_trials = image.shape
    center_y = _normalized_to_pixel(y0, height)
    center_x = _normalized_to_pixel(x0, width)
    patch_size = 2 * patch_radius + 1

    patches = np.stack(
        [
            _extract_patch(image[..., trial_idx], center_y, center_x, patch_size=patch_size).reshape(-1)
            for trial_idx in range(n_trials)
        ],
        axis=0,
    ).astype(np.float32)

    weights = np.clip(np.asarray(response, dtype=np.float32), a_min=0.0, a_max=None)
    if float(np.sum(weights)) <= 0.0 or patches.shape[0] < 4:
        empty = np.zeros((height, width), dtype=np.float32)
        return empty, empty, 0.0

    weights = weights / (np.sum(weights) + 1e-6)
    mean_patch = np.mean(patches, axis=0, keepdims=True)
    patches_centered = patches - mean_patch
    cov = (patches_centered.T @ patches_centered) / max(patches_centered.shape[0] - 1, 1)
    eps = 1e-3 * float(np.trace(cov) / max(cov.shape[0], 1) + 1e-6)
    eigvals, eigvecs = np.linalg.eigh(cov + eps * np.eye(cov.shape[0], dtype=np.float32))
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, eps))) @ eigvecs.T
    white = patches_centered @ inv_sqrt

    weighted_mean = np.sum(weights[:, None] * white, axis=0, keepdims=True)
    centered_white = white - weighted_mean
    stc = (centered_white.T * weights) @ centered_white
    delta = stc - np.eye(stc.shape[0], dtype=np.float32)
    delta_vals, delta_vecs = np.linalg.eigh(delta)

    pos_patch = delta_vecs[:, -1].reshape(patch_size, patch_size)
    neg_patch = delta_vecs[:, 0].reshape(patch_size, patch_size)
    strength = float(max(abs(delta_vals[-1]), abs(delta_vals[0])))

    pos_canvas = np.zeros((height, width), dtype=np.float32)
    neg_canvas = np.zeros((height, width), dtype=np.float32)

    y_start = max(center_y - patch_radius, 0)
    y_stop = min(center_y + patch_radius + 1, height)
    x_start = max(center_x - patch_radius, 0)
    x_stop = min(center_x + patch_radius + 1, width)

    patch_y_start = patch_radius - (center_y - y_start)
    patch_y_stop = patch_y_start + (y_stop - y_start)
    patch_x_start = patch_radius - (center_x - x_start)
    patch_x_stop = patch_x_start + (x_stop - x_start)

    pos_canvas[y_start:y_stop, x_start:x_stop] = pos_patch[patch_y_start:patch_y_stop, patch_x_start:patch_x_stop]
    neg_canvas[y_start:y_stop, x_start:x_stop] = neg_patch[patch_y_start:patch_y_stop, patch_x_start:patch_x_stop]
    return pos_canvas, neg_canvas, strength


def parameter_estimator(data):
    """
    STA + STC-based estimator with pairwise orientation weights and pool mix.

    Extends param_est1 by estimating pairwise orientation interaction weights
    via cross-correlation, using whitened STC to set divisive normalization
    strength, and adaptively mixing pooled vs. non-pooled energy features.

    data keys: 'image' (H, W, T), 'response' (T,)

    Returns:
        dict with x0, y0, sigma_x, sigma_y, baseline, gain, pool_mix,
        norm_strength, norm_bias, channel_weights, pairwise_weights
    """
    image = np.asarray(data["image"], dtype=np.float32)
    response = np.asarray(data["response"], dtype=np.float32)

    base_params = _param_est1(data)
    x0 = base_params["x0"]
    y0 = base_params["y0"]
    sigma_x = base_params["sigma_x"]
    sigma_y = base_params["sigma_y"]

    features_no_pool = _pooled_energy_features(image, x0, y0, sigma_x, sigma_y, pool=False)
    features_pool = _pooled_energy_features(image, x0, y0, sigma_x, sigma_y, pool=True)
    pairwise = np.asarray(pairwise_orientation_terms(jnp.asarray(features_no_pool)), dtype=np.float32)
    pairwise_centered = pairwise - np.mean(pairwise, axis=1, keepdims=True)
    response_centered = response - np.mean(response)
    pairwise_weights = (pairwise_centered @ response_centered) / (pairwise.shape[1] + 1e-6)
    if np.max(np.abs(pairwise_weights)) > 0:
        pairwise_weights = pairwise_weights / (np.max(np.abs(pairwise_weights)) + 1e-6)

    _, _, stc_strength = _whitened_stc_components(image, response, x0, y0)
    channel_weights = _positive_correlation_weights(0.5 * (features_no_pool + features_pool), response)
    baseline, amplitude = _response_summary_stats(response)

    mixed_features = 0.5 * (features_no_pool + features_pool)
    drive = np.sum(channel_weights[:, None] * mixed_features, axis=0)
    gain = amplitude / (np.std(drive) + 1e-3)

    pool_var = np.std(features_pool)
    no_pool_var = np.std(features_no_pool) + 1e-6
    pool_mix = float(np.clip(pool_var / (pool_var + no_pool_var), 0.05, 0.95))
    norm_strength = float(np.clip(0.25 + stc_strength, 0.05, 3.0))
    norm_bias = float(np.clip(np.mean(np.abs(mixed_features)), 0.1, 5.0))

    return {
        "x0": float(x0),
        "y0": float(y0),
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "baseline": float(baseline),
        "gain": float(np.clip(gain, 0.05, 10.0)),
        "pool_mix": pool_mix,
        "norm_strength": norm_strength,
        "norm_bias": norm_bias,
        "channel_weights": channel_weights.astype(np.float32),
        "pairwise_weights": pairwise_weights.astype(np.float32),
    }
