import math
from typing import Any, Callable

import jax
import jax.numpy as jnp


def _safe_loss(val: Any) -> float:
    """Returns a float representation of a loss value.

    Maps `None`, `NaN`, and non-finite values to `float("inf")`, while allowing
    invalid types to raise an error.

    Args:
        val: The loss value to process. Can be `None`, a numeric type, or other types.

    Returns:
        A float representation of the loss value. Returns `float("inf")` if the input
        is `None`, `NaN`, or non-finite.
    """
    if val is None:
        return float("inf")
    val_float = float(val)
    if not math.isfinite(val_float):
        return float("inf")
    return val_float


def _evaluate_model_output(
    model_fn: Callable[[dict, dict], jax.Array],
    params: dict[str, Any],
    data: dict[str, Any],
) -> jax.Array:
    """Evaluates the model output by vmapping over the leading axis of data and params leaves.

    This function applies the `model_fn` to each sample in the input `data` and `params`
    pytrees, effectively parallelizing the model evaluation across samples.

    Args:
        model_fn: A (compiled) JAX model function with signature `(data_unbatched, params_unbatched) -> output_unbatched`.
            `data_unbatched` and `params_unbatched` are PyTrees of unbatched shapes,
            and `output_unbatched` is a JAX array of shape `(output_shape,)`.
        params: PyTree of model parameters, where each leaf has a leading dimension
            corresponding to the number of samples, i.e., shape `(n_samples, ...)`.
        data: PyTree of input data, where each leaf has a leading dimension
            corresponding to the number of samples, i.e., shape `(n_samples, ...)`.

    Returns:
        A JAX array of shape `(n_samples, output_shape)` containing the model output
        for each sample.
    """
    return jax.vmap(model_fn, in_axes=(0, 0))(data, params)


def _evaluate_sample_losses(
    model_fn: Callable[[dict, dict], jax.Array],
    loss_fn: Callable[[jax.Array, dict], jax.Array],
    params: dict[str, Any],
    data: dict[str, Any],
) -> jax.Array:
    """Computes the per-sample loss for batched data and parameters.

    This function first evaluates the `model_fn` for all samples using `jax.vmap`
    to produce a batched model output. It then applies the `loss_fn` to this
    batched output and the input data to calculate the loss for each individual sample.

    Args:
        model_fn: A (compiled) JAX model function with signature `(data_unbatched, params_unbatched) -> output_unbatched`.
            `data_unbatched` and `params_unbatched` are PyTrees of unbatched shapes,
            and `output_unbatched` is a JAX array of shape `(output_shape,)`.
        loss_fn: A JAX-compatible loss function with signature `(batched_model_output, batched_data) -> loss`.
            `batched_model_output` is a JAX array of shape `(n_samples, output_shape)`,
            and `batched_data` is a PyTree where each leaf has a leading dimension
            corresponding to the number of samples, i.e., shape `(n_samples, ...)`.
        params: PyTree of batched model parameters, where each leaf has a leading
            dimension corresponding to the number of samples, i.e., shape `(n_samples, ...)`.
        data: PyTree of batched input data, where each leaf has a leading dimension
            corresponding to the number of samples, i.e., shape `(n_samples, ...)`.

    Returns:
        A JAX array of shape `(n_samples,)` containing the loss for each sample.
    """
    output = _evaluate_model_output(model_fn, params, data)
    return loss_fn(output, data)


def _evaluate_scalar_loss(
    model_fn: Callable[[dict, dict], jax.Array],
    loss_fn: Callable[[jax.Array, dict], jax.Array],
    params: dict[str, Any],
    data: dict[str, Any],
) -> jax.Array:
    """Computes the mean scalar loss over all samples.

    This function calculates the per-sample losses using `_evaluate_sample_losses`
    and then computes the mean of these losses, resulting in a single scalar loss
    value for the entire dataset.

    Args:
        model_fn: A (compiled) JAX model function with signature `(data_unbatched, params_unbatched) -> output_unbatched`.
            `data_unbatched` and `params_unbatched` are PyTrees of unbatched shapes,
            and `output_unbatched` is a JAX array of shape `(output_shape,)`.
        loss_fn: A JAX-compatible loss function with signature `(batched_model_output, batched_data) -> loss`.
            `batched_model_output` is a JAX array of shape `(n_samples, output_shape)`,
            and `batched_data` is a PyTree where each leaf has a leading dimension
            corresponding to the number of samples, i.e., shape `(n_samples, ...)`.
        params: PyTree of batched model parameters, where each leaf has a leading
            dimension corresponding to the number of samples, i.e., shape `(n_samples, ...)`.
        data: PyTree of batched input data, where each leaf has a leading dimension
            corresponding to the number of samples, i.e., shape `(n_samples, ...)`.

    Returns:
        A scalar JAX array representing the mean loss across all samples.
    """
    return jnp.mean(_evaluate_sample_losses(model_fn, loss_fn, params, data))