import numpy as np
import jax.numpy as jnp

from projects.minimodel_discovery.primitives import (
    gabor_bank16,
    local_gaussian_readout,
    make_gabor_kernels,
    quadrature_energy,
)


def _normalized_to_pixel(value: float, size: int) -> int:
    return int(np.clip(round((float(value) + 1.0) * 0.5 * (size - 1)), 0, size - 1))


def _extract_patch(image_2d: np.ndarray, center_y: int, center_x: int, patch_size: int) -> np.ndarray:
    radius = patch_size // 2
    padded = np.pad(image_2d, ((radius, radius), (radius, radius)), mode="reflect")
    y0 = center_y + radius
    x0 = center_x + radius
    patch = padded[y0 - radius:y0 + radius + 1, x0 - radius:x0 + radius + 1]
    return np.asarray(patch, dtype=np.float32)


def _compute_sta(image: np.ndarray, response: np.ndarray) -> np.ndarray:
    weights = np.clip(np.asarray(response, dtype=np.float32), a_min=0.0, a_max=None)
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones_like(weights, dtype=np.float32)
    sta = np.tensordot(image, weights, axes=([2], [0])) / (np.sum(weights) + 1e-6)
    return np.asarray(sta, dtype=np.float32)


def _fit_gaussian_from_sta(sta: np.ndarray) -> tuple[float, float, float, float]:
    weight = np.abs(np.asarray(sta, dtype=np.float32))
    if float(np.sum(weight)) <= 0.0:
        return 0.0, 0.0, 0.25, 0.25

    height, width = weight.shape
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    total = np.sum(weight) + 1e-6

    x0 = float(np.sum(weight * xx) / total)
    y0 = float(np.sum(weight * yy) / total)
    sigma_x = float(np.sqrt(np.sum(weight * (xx - x0) ** 2) / total))
    sigma_y = float(np.sqrt(np.sum(weight * (yy - y0) ** 2) / total))
    return x0, y0, max(sigma_x, 0.08), max(sigma_y, 0.08)


def _masked_sta_channel_weights(
    sta: np.ndarray,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
) -> np.ndarray:
    height, width = sta.shape
    center_y = _normalized_to_pixel(y0, height)
    center_x = _normalized_to_pixel(x0, width)
    patch = _extract_patch(sta, center_y, center_x, patch_size=25)
    kernels = np.asarray(make_gabor_kernels(), dtype=np.float32)[:, 0]
    simple_scores = np.tensordot(kernels, patch, axes=([1, 2], [0, 1]))
    pair_scores = []
    for idx in range(0, simple_scores.shape[0], 2):
        pair_scores.append(np.sqrt(simple_scores[idx] ** 2 + simple_scores[idx + 1] ** 2))
    weights = np.asarray(pair_scores, dtype=np.float32)
    weights = np.maximum(weights, 0.0)
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones_like(weights, dtype=np.float32)
    return (weights / (np.sum(weights) + 1e-6)).astype(np.float32)


def _pooled_energy_features(
    image: np.ndarray,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    *,
    pool: bool,
) -> np.ndarray:
    simple_maps = gabor_bank16(jnp.asarray(image, dtype=jnp.float32))
    energy_maps = quadrature_energy(simple_maps, pool=pool)
    pooled = local_gaussian_readout(
        energy_maps,
        x0=x0,
        y0=y0,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
    )
    return np.asarray(pooled, dtype=np.float32)


def _positive_correlation_weights(features: np.ndarray, response: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    response = np.asarray(response, dtype=np.float32)
    response_centered = response - np.mean(response)
    feats_centered = features - np.mean(features, axis=1, keepdims=True)
    weights = feats_centered @ response_centered
    weights = np.maximum(weights, 0.0)
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones_like(weights, dtype=np.float32)
    return (weights / (np.sum(weights) + 1e-6)).astype(np.float32)


def _response_summary_stats(response: np.ndarray) -> tuple[float, float]:
    baseline = float(np.percentile(response, 15))
    amplitude = float(np.percentile(response, 95) - baseline)
    return baseline, max(amplitude, 1e-3)


def parameter_estimator(data):
    """
    STA-based estimator for the energy model.

    Computes a spike-triggered average (STA) to localise the receptive field,
    fits a Gaussian to determine (x0, y0, sigma_x, sigma_y), then selects
    channel weights from a Gabor-patch analysis of the STA combined with
    positive-correlation weights from the pooled energy features.

    data keys: 'image' (H, W, T), 'response' (T,)

    Returns:
        dict with x0, y0, sigma_x, sigma_y, baseline, gain, channel_weights
    """
    image = np.asarray(data["image"], dtype=np.float32)
    response = np.asarray(data["response"], dtype=np.float32)

    sta = _compute_sta(image, response)
    x0, y0, sigma_x, sigma_y = _fit_gaussian_from_sta(sta)
    channel_weights = _masked_sta_channel_weights(sta, x0, y0, sigma_x, sigma_y)
    energy_features = _pooled_energy_features(image, x0, y0, sigma_x, sigma_y, pool=False)
    channel_weights = 0.5 * channel_weights + 0.5 * _positive_correlation_weights(energy_features, response)
    channel_weights = channel_weights / (np.sum(channel_weights) + 1e-6)

    baseline, amplitude = _response_summary_stats(response)
    drive = np.sum(channel_weights[:, None] * energy_features, axis=0)
    gain = amplitude / (np.std(drive) + 1e-3)
    return {
        "x0": float(x0),
        "y0": float(y0),
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "baseline": float(baseline),
        "gain": float(np.clip(gain, 0.05, 10.0)),
        "channel_weights": channel_weights.astype(np.float32),
    }
