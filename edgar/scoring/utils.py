import math
from typing import Any, Callable

import jax
import jax.numpy as jnp


def _safe_loss(val: Any) -> float:
    """Returns a float representation of a loss value, mapping None, nan,
    and non-finite values to float("inf"), while letting invalid types raise.
    """
    if val is None:
        return float("inf")
    val_float = float(val)
    if not math.isfinite(val_float):
        return float("inf")
    return val_float


def _evaluate_model_output(
    model_fn: Callable[[Any, Any], jax.Array],
    params: dict[str, Any],
    data: dict[str, Any],
) -> jax.Array:
    """Evaluates the model output for a given sample of parameters and data.

    Args:
        model_fn: A (compiled) JAX model function of signature (data, params) -> output, where data, params are of unbatched shape, (n_x, ...) and output is of shape (output_shape,)
        params: PyTree of model parameters, where each leaf has shape (n_samples, n_x, ...)
        data: PyTree of data, where each leaf has shape (n_samples, n_x, ...)
    Returns:
        A JAX array of shape (n_samples, output_shape) containing the model output for each sample.
    """
    return jax.vmap(model_fn, in_axes=(0, 0))(data, params)


def _evaluate_sample_losses(
    model_fn: Callable[[Any, Any], jax.Array],
    loss_fn: Callable[[jax.Array, Any], jax.Array],
    params: dict[str, Any],
    data: dict[str, Any],
) -> jax.Array:
    """Computes the per-sample loss, where data and params are batched pytrees with leaves of leading dimension (n_samples, ...).

    `model_fn` maps data of shape (n_x, ...) -> output_shape.
    This method computes the model output for all samples using a vmap, yielding a batched model output of shape (n_samples, output_shape).
    The loss_fn is then applied to the batched model output and the data -> yielding an (n_samples,) array of losses.

    Args:
        model_fn: A (compiled) JAX model function of signature (data, params) -> output, where data, params are of unbatched shape, (n_x, ...) and output is of shape (output_shape,)
        loss_fn: A JAX-compatible loss function of signature (batched_model_output, data) -> loss, where batched_model_output is of shape (n_samples, output_shape) and data is of shape (n_samples, n_x, ...)
        params: PyTree of batched model parameters, where each leaf has shape (n_samples, n_x, ...)
        data: PyTree of batched data, where each leaf has shape (n_samples, n_x, ...)
    Returns:
        A JAX array of shape (n_samples,) containing the loss for each sample.
    """
    output = _evaluate_model_output(model_fn, params, data)
    return loss_fn(output, data)


def _evaluate_scalar_loss(
    model_fn: Callable[[Any, Any], jax.Array],
    loss_fn: Callable[[jax.Array, Any], jax.Array],
    params: dict[str, Any],
    data: dict[str, Any],
) -> jax.Array:
    """Computes the mean loss over all samples of the input data and params."""
    return jnp.mean(_evaluate_sample_losses(model_fn, loss_fn, params, data))
