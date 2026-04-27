"""
Project-facing helpers for working with the EDGAR data dict and per-sample
parameter pytrees.

A data dict has shape ``{key: array}`` where every array shares a leading
``n_samples`` axis and a trailing ``n_trials`` axis. Params are pytrees whose
leaves can be either per-sample (with leading sample axis) or scalar/shared.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np

# Quiet noisy third-party loggers.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.ERROR)


# ── data dict helpers ──

def data_n_samples(X: Dict[str, np.ndarray]) -> int:
    return int(next(iter(X.values())).shape[0])


def data_n_trials(X: Dict[str, np.ndarray]) -> int:
    return int(next(iter(X.values())).shape[-1])


def slice_data_samples(X: Dict[str, np.ndarray], indices) -> Dict[str, np.ndarray]:
    """Slice the sample axis (dim 0) of every array in the data dict."""
    return {k: v[indices] for k, v in X.items()}


def slice_data_trials(X: Dict[str, np.ndarray], indices) -> Dict[str, np.ndarray]:
    """Slice the trial axis (last dim) of every array in the data dict."""
    return {k: v[..., indices] for k, v in X.items()}


def slice_data(X: Dict[str, np.ndarray], sample_indices, trial_indices) -> Dict[str, np.ndarray]:
    """Slice both sample and trial axes of every array in the data dict."""
    return slice_data_trials(slice_data_samples(X, sample_indices), trial_indices)


def get_data_sample(X: Dict[str, np.ndarray], idx: int) -> Dict[str, np.ndarray]:
    """Extract one sample from the data dict, dropping the sample axis."""
    return {k: v[idx] for k, v in X.items()}


def data_as_jax(X: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
    return {k: jnp.asarray(v) for k, v in X.items()}


def data_as_numpy(X: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in X.items()}


# ── params helpers ──

def tree_to_jax(params):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), params)


def stack_params(params_list):
    """Stack a list of per-sample param pytrees into a batched pytree."""
    if not params_list:
        return None
    return jax.tree_util.tree_map(
        lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], axis=0),
        *params_list,
    )


def broadcast_params(params, n_samples: int):
    """Ensure params have a leading sample axis, broadcasting as needed."""
    params = tree_to_jax(params)

    def _broadcast(arr):
        arr = jnp.asarray(arr)
        if arr.ndim == 0:
            return jnp.broadcast_to(arr, (n_samples,))
        if arr.shape[0] == n_samples:
            return arr
        if arr.shape[0] == 1:
            return jnp.broadcast_to(arr, (n_samples,) + arr.shape[1:])
        orig_shape = arr.shape
        arr = arr[None, ...]
        return jnp.broadcast_to(arr, (n_samples,) + orig_shape)

    return jax.tree_util.tree_map(_broadcast, params)


def slice_params(params, idx: int):
    """Slice a batched params pytree at the given sample index."""
    return jax.tree_util.tree_map(lambda x: x if jnp.ndim(x) == 0 else x[idx], params)


# ── model evaluation ──

def vmap_over_samples(model_fn: Callable):
    """Vmap a single-sample model fn over the leading sample axis of data and params."""
    def _wrapped(data_i, params_i):
        return model_fn(data_i, params_i)
    return jax.vmap(_wrapped, in_axes=(0, 0))


def call_model(model_fn, data, params, prefer_jax: bool = True):
    """Invoke a single-sample model with data/params converted to JAX when possible."""
    if prefer_jax:
        try:
            data_jax = data_as_jax(data) if isinstance(data, dict) else jnp.asarray(data)
            params_jax = tree_to_jax(params)
            return model_fn(data_jax, params_jax)
        except Exception as jax_exc:
            try:
                data_np = data_as_numpy(data) if isinstance(data, dict) else np.asarray(data)
                return model_fn(data_np, params)
            except Exception:
                raise jax_exc
    data_np = data_as_numpy(data) if isinstance(data, dict) else np.asarray(data)
    return model_fn(data_np, params)


def compute_evaluation_matrix(program, params, eval_points):
    """Evaluate ``program`` on an ``eval_points`` grid (data dict with sample axis).

    Args:
        program: Single-sample model function.
        params: Param pytree (broadcast to n_samples if needed).
        eval_points: Eval data dict with leading sample axis.

    Returns:
        jnp.ndarray: vmapped model output on the grid.
    """
    if eval_points is None:
        raise ValueError("eval_points must be provided.")
    eval_data = data_as_jax(eval_points)
    n_samples = data_n_samples(eval_data)
    params_arr = broadcast_params(tree_to_jax(params), n_samples)
    return vmap_over_samples(program)(eval_data, params_arr)
