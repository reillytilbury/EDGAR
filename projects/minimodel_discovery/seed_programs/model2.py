import jax.numpy as jnp
import numpy as np

from projects.minimodel_discovery.primitives import (
    divisive_normalization,
    gabor_bank16,
    local_gaussian_readout,
    pairwise_orientation_terms,
    positive_rate,
    quadrature_energy,
)


def _match_vector_length(values: jnp.ndarray, length: int) -> jnp.ndarray:
    arr = jnp.asarray(values, dtype=jnp.float32).reshape(-1)
    if int(arr.shape[0]) == length:
        return arr
    if int(arr.shape[0]) > length:
        return arr[:length]
    pad = jnp.zeros((length - int(arr.shape[0]),), dtype=jnp.float32)
    return jnp.concatenate([arr, pad], axis=0)


def model(data, params):
    """
    Energy model with pairwise orientation terms and divisive normalization.

    Combines pooled and un-pooled energy readouts with a learnable mix ratio,
    appends pairwise orientation interaction features, applies divisive
    normalization, then passes through a positive-rate output nonlinearity.

    data keys: 'image'  # shape (H, W, T)
    params: x0, y0, sigma_x, sigma_y, baseline, gain, pool_mix,
            norm_strength, norm_bias, channel_weights (8,), pairwise_weights (12,)

    Returns:
        jnp.ndarray: Predicted response, shape (T,)
    """
    image = jnp.asarray(data["image"], dtype=jnp.float32)
    x0 = jnp.clip(jnp.asarray(params["x0"], dtype=jnp.float32), -1.0, 1.0)
    y0 = jnp.clip(jnp.asarray(params["y0"], dtype=jnp.float32), -1.0, 1.0)
    sigma_x = jnp.clip(jnp.asarray(params["sigma_x"], dtype=jnp.float32), 0.03, 1.0)
    sigma_y = jnp.clip(jnp.asarray(params["sigma_y"], dtype=jnp.float32), 0.03, 1.0)
    baseline = jnp.asarray(params["baseline"], dtype=jnp.float32)
    gain = jnp.clip(jnp.asarray(params["gain"], dtype=jnp.float32), 0.0, 10.0)
    pool_mix = jnp.clip(jnp.asarray(params["pool_mix"], dtype=jnp.float32), 0.0, 1.0)
    norm_strength = jnp.clip(jnp.asarray(params["norm_strength"], dtype=jnp.float32), 0.0, 4.0)
    norm_bias = jnp.clip(jnp.asarray(params["norm_bias"], dtype=jnp.float32), 1e-3, 5.0)
    channel_weights = jnp.clip(_match_vector_length(params["channel_weights"], 8), 0.0, 5.0)

    simple_maps = gabor_bank16(image)
    energy_no_pool = quadrature_energy(simple_maps, pool=False)
    energy_pool = quadrature_energy(simple_maps, pool=True)
    readout_no_pool = local_gaussian_readout(energy_no_pool, x0=x0, y0=y0, sigma_x=sigma_x, sigma_y=sigma_y)
    readout_pool = local_gaussian_readout(energy_pool, x0=x0, y0=y0, sigma_x=sigma_x, sigma_y=sigma_y)
    base_features = (1.0 - pool_mix) * readout_no_pool + pool_mix * readout_pool

    pairwise = pairwise_orientation_terms(base_features)
    pairwise_weights = jnp.clip(_match_vector_length(params["pairwise_weights"], int(pairwise.shape[0])), -2.0, 2.0)
    weighted_base = channel_weights[:, None] * base_features
    weighted_pairwise = pairwise_weights[:, None] * pairwise
    feature_stack = jnp.concatenate([weighted_base, weighted_pairwise], axis=0)
    normalized = divisive_normalization(feature_stack, strength=norm_strength, bias=norm_bias)
    drive = gain * jnp.sum(normalized, axis=0)
    return positive_rate(baseline + drive)


model.DEFAULT_PARAMS = {
    "x0": 0.0,
    "y0": 0.0,
    "sigma_x": 0.25,
    "sigma_y": 0.25,
    "baseline": 0.1,
    "gain": 1.0,
    "pool_mix": 0.5,
    "norm_strength": 0.5,
    "norm_bias": 0.5,
    "channel_weights": np.full((8,), 1.0 / 8.0, dtype=np.float32),
    "pairwise_weights": np.zeros((12,), dtype=np.float32),
}
