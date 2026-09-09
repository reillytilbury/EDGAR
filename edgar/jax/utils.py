"""Utility functions for JAX operations and conversions in EDGAR."""

from typing import Any
import numpy as np
import jax


def _to_jax(x: Any) -> Any:
    """Recursively convert numpy arrays in a dictionary, list, or tuple to JAX arrays."""
    if isinstance(x, dict):
        return {k: _to_jax(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [_to_jax(v) for v in x]
    elif isinstance(x, tuple):
        return tuple(_to_jax(v) for v in x)
    elif isinstance(x, np.ndarray):
        return jax.device_put(x)
    return x


def _to_numpy(x: Any) -> Any:
    """Recursively convert JAX arrays in a dictionary, list, or tuple to NumPy arrays."""
    try:
        is_jax_array = isinstance(x, jax.Array)
    except Exception:
        is_jax_array = False

    if isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [_to_numpy(v) for v in x]
    elif isinstance(x, tuple):
        return tuple(_to_numpy(v) for v in x)
    elif is_jax_array or isinstance(x, np.ndarray):
        return np.asarray(x)
    return x


def _has_jax(x: Any) -> bool:
    """Recursively checks if any dictionary, list, tuple, or value contains JAX device arrays."""
    try:
        is_jax_array = isinstance(x, jax.Array)
    except AttributeError:
        is_jax_array = False

    if is_jax_array:
        return True
    elif isinstance(x, dict):
        return any(_has_jax(v) for v in x.values())
    elif isinstance(x, (list, tuple)):
        return any(_has_jax(v) for v in x)
    return False
