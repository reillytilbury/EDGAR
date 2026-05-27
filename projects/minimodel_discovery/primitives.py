import math

import jax.numpy as jnp
from jax import lax


_GABOR_KERNELS = None


def _coordinate_grid(size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    axis = jnp.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(axis, axis, indexing="ij")
    return yy, xx


def get_gabor_specs() -> list[dict[str, float]]:
    orientations = [0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0]
    scales = [(2.5, 0.34), (4.5, 0.18)]
    specs = []
    for sigma, frequency in scales:
        for theta in orientations:
            specs.append(
                {
                    "sigma": float(sigma),
                    "frequency": float(frequency),
                    "theta": float(theta),
                    "phase": 0.0,
                    "aspect": 0.65,
                }
            )
            specs.append(
                {
                    "sigma": float(sigma),
                    "frequency": float(frequency),
                    "theta": float(theta),
                    "phase": math.pi / 2.0,
                    "aspect": 0.65,
                }
            )
    return specs


def make_gabor_kernels(kernel_size: int = 25) -> jnp.ndarray:
    global _GABOR_KERNELS
    if _GABOR_KERNELS is not None and _GABOR_KERNELS.shape[-1] == kernel_size:
        return _GABOR_KERNELS

    yy, xx = _coordinate_grid(kernel_size)
    kernels = []
    for spec in get_gabor_specs():
        theta = spec["theta"]
        sigma = spec["sigma"]
        frequency = spec["frequency"]
        phase = spec["phase"]
        aspect = spec["aspect"]

        x_rot = xx * math.cos(theta) + yy * math.sin(theta)
        y_rot = -xx * math.sin(theta) + yy * math.cos(theta)

        gaussian = jnp.exp(-0.5 * ((x_rot / sigma) ** 2 + (aspect * y_rot / sigma) ** 2))
        carrier = jnp.cos(2.0 * math.pi * frequency * x_rot + phase)
        kernel = gaussian * carrier
        kernel = kernel - jnp.mean(kernel)
        kernel = kernel / (jnp.linalg.norm(kernel) + 1e-6)
        kernels.append(kernel)

    stacked = jnp.stack(kernels, axis=0)[:, None, :, :]
    _GABOR_KERNELS = stacked.astype(jnp.float32)
    return _GABOR_KERNELS


def gabor_bank16(image: jnp.ndarray, kernel_size: int = 25) -> jnp.ndarray:
    image = jnp.asarray(image, dtype=jnp.float32)
    if image.ndim != 3:
        raise ValueError(f"gabor_bank16 expects image with shape (H, W, T), got {image.shape}")

    kernels = make_gabor_kernels(kernel_size=kernel_size)
    batch = jnp.transpose(image, (2, 0, 1))[:, None, :, :]
    response = lax.conv_general_dilated(
        lhs=batch,
        rhs=kernels,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return jnp.transpose(response, (1, 2, 3, 0))


def _max_pool2x2(feature_maps: jnp.ndarray) -> jnp.ndarray:
    if feature_maps.ndim != 4:
        raise ValueError(
            f"_max_pool2x2 expects feature maps with shape (C, H, W, T), got {feature_maps.shape}"
        )
    batch = jnp.transpose(feature_maps, (3, 0, 1, 2))
    pooled = lax.reduce_window(
        batch,
        init_value=-jnp.inf,
        computation=lax.max,
        window_dimensions=(1, 1, 2, 2),
        window_strides=(1, 1, 2, 2),
        padding="VALID",
    )
    return jnp.transpose(pooled, (1, 2, 3, 0))


def quadrature_energy(
    simple_maps: jnp.ndarray,
    pool: bool = False,
    eps: float = 1e-4,
) -> jnp.ndarray:
    maps = _max_pool2x2(simple_maps) if pool else simple_maps
    n_simple = maps.shape[0]
    if n_simple % 2 != 0:
        raise ValueError(f"quadrature_energy expects an even number of channels, got {n_simple}")

    even = maps[0::2]
    odd = maps[1::2]
    return jnp.sqrt(jnp.square(even) + jnp.square(odd) + eps)


def _spatial_coordinate_maps(height: int, width: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    y = jnp.linspace(-1.0, 1.0, height, dtype=jnp.float32)
    x = jnp.linspace(-1.0, 1.0, width, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(y, x, indexing="ij")
    return yy, xx


def local_gaussian_readout(
    feature_maps: jnp.ndarray,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
) -> jnp.ndarray:
    maps = jnp.asarray(feature_maps, dtype=jnp.float32)
    if maps.ndim != 4:
        raise ValueError(
            f"local_gaussian_readout expects feature maps with shape (C, H, W, T), got {maps.shape}"
        )

    sigma_x = jnp.clip(jnp.asarray(sigma_x, dtype=jnp.float32), 0.03, 1.5)
    sigma_y = jnp.clip(jnp.asarray(sigma_y, dtype=jnp.float32), 0.03, 1.5)
    x0 = jnp.clip(jnp.asarray(x0, dtype=jnp.float32), -1.0, 1.0)
    y0 = jnp.clip(jnp.asarray(y0, dtype=jnp.float32), -1.0, 1.0)

    yy, xx = _spatial_coordinate_maps(maps.shape[1], maps.shape[2])
    wx = jnp.exp(-0.5 * ((xx - x0) / sigma_x) ** 2)
    wy = jnp.exp(-0.5 * ((yy - y0) / sigma_y) ** 2)
    weights = wy * wx
    weights = weights / (jnp.sum(weights) + 1e-6)
    return jnp.sum(maps * weights[None, :, :, None], axis=(1, 2))


def pairwise_orientation_terms(features: jnp.ndarray) -> jnp.ndarray:
    feats = jnp.asarray(features, dtype=jnp.float32)
    if feats.ndim != 2:
        raise ValueError(
            f"pairwise_orientation_terms expects shape (C, T), got {feats.shape}"
        )

    n_channels = int(feats.shape[0])
    terms = []
    n_groups = max(1, n_channels // 4)

    for group_idx in range(n_groups):
        start = 4 * group_idx
        stop = min(start + 4, n_channels)
        group = feats[start:stop]
        if group.shape[0] == 4:
            for offset in range(4):
                terms.append(group[offset] * group[(offset + 1) % 4])

    if n_channels >= 8:
        for ori in range(4):
            terms.append(feats[ori] * feats[ori + 4])

    if not terms:
        return jnp.zeros((1, feats.shape[1]), dtype=jnp.float32)
    return jnp.stack(terms, axis=0)


def divisive_normalization(
    features: jnp.ndarray,
    strength: float,
    bias: float,
) -> jnp.ndarray:
    feats = jnp.asarray(features, dtype=jnp.float32)
    strength = jnp.clip(jnp.asarray(strength, dtype=jnp.float32), 0.0, 10.0)
    bias = jnp.clip(jnp.asarray(bias, dtype=jnp.float32), 1e-3, 10.0)
    normalizer = bias + strength * jnp.mean(jnp.abs(feats), axis=0, keepdims=True)
    return feats / normalizer


def positive_rate(x: jnp.ndarray, floor: float = 1e-3) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=jnp.float32)
    floor = jnp.clip(jnp.asarray(floor, dtype=jnp.float32), 1e-6, 1.0)
    return floor + jnp.log1p(jnp.exp(x))
