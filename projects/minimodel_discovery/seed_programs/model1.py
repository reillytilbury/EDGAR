import jax.numpy as jnp
import numpy as np

from projects.minimodel_discovery.primitives import (
    gabor_bank16,
    local_gaussian_readout,
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
    Energy model with local Gaussian readout.

    Applies a 16-channel Gabor filter bank, computes quadrature energy maps,
    pools spatially with a Gaussian at (x0, y0), then takes a weighted sum
    of the 8 orientation channels followed by a positive-rate nonlinearity.

    data keys: 'image'  # shape (H, W, T) — greyscale image sequence
    params: x0, y0, sigma_x, sigma_y, baseline, gain, channel_weights (8,)

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
    channel_weights = jnp.clip(_match_vector_length(params["channel_weights"], 8), 0.0, 5.0)

    simple_maps = gabor_bank16(image)
    energy_maps = quadrature_energy(simple_maps, pool=False)
    pooled = local_gaussian_readout(energy_maps, x0=x0, y0=y0, sigma_x=sigma_x, sigma_y=sigma_y)
    drive = gain * jnp.einsum("c,ct->t", channel_weights, pooled)
    return positive_rate(baseline + drive)


model.DEFAULT_PARAMS = {
    "x0": 0.0,
    "y0": 0.0,
    "sigma_x": 0.25,
    "sigma_y": 0.25,
    "baseline": 0.1,
    "gain": 1.0,
    "channel_weights": np.full((8,), 1.0 / 8.0, dtype=np.float32),
}
