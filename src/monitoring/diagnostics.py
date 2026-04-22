import logging

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .. import utils
from ..scoring.objective import _clear_jax_runtime_cache


def _programs_df_to_programs_list(programs_df: pd.DataFrame,
                                    loss_func: callable,
                                    data: dict,
                                    complexity_penalty: float = 0.0,
                                    penalty_denominator: int = 1) -> list[dict]:
    """
    Convert a programs dataframe to the canonical programs_list plotting payload.
    Compute per sample losses for each program using the provided loss function,
    and include them in the programs_list dicts under the key 'losses'.

    Args:
        programs_df (pd.DataFrame): DataFrame with columns 'program'/'model' and 'params'.
        loss_func (callable): Loss function ``loss_func(model_output, data_i)``.
        data (dict[str, np.ndarray]): Data dict with sample axis at dim 0.
        complexity_penalty (float): Additive complexity penalty multiplier.
        penalty_denominator (int): Denominator for normalizing param count.
    """
    programs_list = []
    if programs_df is None or len(programs_df) == 0:
        return programs_list
    data_jax = utils.data_as_jax(data)
    n_samples = utils.data_n_samples(data_jax)

    if loss_func is None:
        raise ValueError("_programs_df_to_programs_list requires a loss_func; none was provided.")

    def _broadcast_params_cpu(params_in, n: int):
        def _b(arr):
            arr = np.asarray(arr)
            if arr.ndim == 0:
                return np.full((n,), arr, dtype=arr.dtype)
            if arr.shape[0] == n:
                return arr
            if arr.shape[0] == 1:
                return np.broadcast_to(arr, (n,) + arr.shape[1:])
            arr = arr[None, ...]
            return np.broadcast_to(arr, (n,) + arr.shape)
        return jax.tree_util.tree_map(_b, params_in)

    def _slice_params_cpu(params_in, idx: int):
        return jax.tree_util.tree_map(
            lambda arr: arr if np.ndim(arr) == 0 else np.asarray(arr)[idx],
            params_in,
        )

    for _, row in programs_df.iterrows():
        model = row.get('program', row.get('model'))
        params = row.get('params')
        if model is None or params is None:
            continue
        params_tree = utils.broadcast_params(params, n_samples)
        n_free_params_raw = utils.params_numel_per_sample(params_tree, n_samples=n_samples)
        n_free_params = n_free_params_raw / max(1, penalty_denominator)

        def _sample_loss(params_i, data_i):
            model_output = model(data_i, params_i)
            raw = jnp.asarray(loss_func(model_output, data_i))
            return jnp.mean(raw) if raw.ndim > 0 else raw

        try:
            losses = jax.vmap(_sample_loss, in_axes=(0, 0))(params_tree, data_jax)
        except Exception as exc:
            logging.info(
                "_programs_df_to_programs_list: vectorized loss computation failed "
                "(falling back to per-sample evaluation): %s",
                exc,
            )
            params_cpu = _broadcast_params_cpu(params, n_samples)
            data_np = utils.data_as_numpy(data)
            losses_np = []
            for sample_idx in range(n_samples):
                data_i = utils.get_data_sample(data_np, sample_idx)
                params_i = _slice_params_cpu(params_cpu, sample_idx)
                model_output = np.asarray(utils.call_model(model, data_i, params_i, prefer_jax=False))
                raw = np.asarray(loss_func(model_output, data_i))
                losses_np.append(float(np.mean(raw) if raw.ndim > 0 else raw))
            losses = jnp.asarray(losses_np, dtype=jnp.float32)

        penalty_term = float(complexity_penalty) * n_free_params
        losses = losses + penalty_term
        programs_list.append({
            'model': model,
            'params': params,
            'losses': losses,
        })
        _clear_jax_runtime_cache()

    return programs_list


def _align_eval_grid(X_eval, n_samples: int) -> dict:
    """Align evaluation grid sample dimension to *n_samples* by tiling if needed.

    Args:
        X_eval (dict[str, np.ndarray]): Evaluation data dict with sample axis
            at dim 0.
        n_samples (int): Desired number of samples.

    Returns:
        dict[str, np.ndarray]: Eval data dict with first dim equal to *n_samples*.
    """
    current_n = utils.data_n_samples(X_eval)
    if current_n == n_samples:
        return X_eval
    if current_n == 1:
        return {k: np.broadcast_to(v, (n_samples,) + v.shape[1:]) for k, v in X_eval.items()}
    idx = np.arange(n_samples, dtype=np.int64) % current_n
    return utils.slice_data_samples(X_eval, idx)
