import multiprocessing as mp

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree

from ..evolution.program import Program


def _worker(queue, program, data, loss_fn, config):
    try:
        result = score_program(program, data, loss_fn, config)
    except Exception:
        result = float("inf")
    queue.put(result)


def score_with_timeout(program, data, loss_fn, config) -> float:
    """Run score_program in a subprocess, returning inf if it exceeds timeout.

    Args:
        program: Candidate program to score.
        data: Tuple of ``(data_train, data_test)``, each a JAX data dict.
        loss_fn: Loss function, signature ``(output, data) -> scalar`` over the full batch.
        config: Scoring config dict with ``timeout_s`` and ``gradient_descent`` subsections.

    Returns:
        Scalar loss as a Python float, or inf if the process timed out or failed.
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(queue, program, data, loss_fn, config))
    proc.start()
    proc.join(timeout=config["timeout_s"])
    if proc.is_alive():
        proc.kill()
        proc.join()
        return float("inf")
    return queue.get()


def _get_params(param_est_fn, default_params, data_train):
    """Estimate initial parameters for all samples via vmapped param estimator.

    Args:
        param_est_fn: JAX-compatible parameter estimator, signature ``(data_i) -> params``.
        default_params: Single-sample fallback pytree (e.g. ``model_fn.DEFAULT_PARAMS``).
        data_train: Training data dict with leading sample axis.

    Returns:
        Batched parameter pytree with leading sample axis.
    """
    try:
        return jax.vmap(param_est_fn)(data_train)
    except Exception:
        n = next(iter(data_train.values())).shape[0]
        return jax.tree_util.tree_map(lambda x: jnp.stack([x] * n), default_params)


def _optimize(model_fn, loss_fn, params_init, data_train, gd_config):
    """Fit parameters on data_train using Adam, returning the best params found.

    Args:
        model_fn: JAX model, signature ``(data_i, params) -> output``.
        loss_fn: Loss function, signature ``(output, data) -> scalar`` over the full batch.
        params_init: Initial batched parameter pytree with leading sample axis.
        data_train: Training data dict with leading sample axis.
        gd_config: Dict with keys ``learning_rate`` and ``max_iter``.

    Returns:
        Optimized parameter pytree with the same structure as ``params_init``.
    """
    flat, unflatten = ravel_pytree(params_init)

    def total_loss(flat_p):
        p = unflatten(flat_p)
        output = jax.vmap(model_fn, in_axes=(0, 0))(data_train, p)
        return loss_fn(output, data_train)

    loss_and_grad = jax.jit(jax.value_and_grad(total_loss))
    opt = optax.adam(gd_config["learning_rate"])
    opt_state = opt.init(flat)
    best_loss, best_flat = float("inf"), flat

    for step in range(gd_config["max_iter"]):
        loss_val, grad = loss_and_grad(flat)
        if not jnp.isfinite(loss_val):
            break
        if float(loss_val) < best_loss:
            best_loss, best_flat = float(loss_val), flat.copy()
        updates, opt_state = opt.update(grad, opt_state, flat)
        flat = optax.apply_updates(flat, updates)
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss_val:.4f}")

    return unflatten(best_flat)


def _eval_loss(model_fn, loss_fn, params, data_test):
    """Compute mean loss over all samples in data_test.

    Args:
        model_fn: JAX model, signature ``(data_i, params) -> output``.
        loss_fn: Loss function, signature ``(output, data) -> scalar`` over the full batch.
        params: Batched parameter pytree with leading sample axis.
        data_test: Test data dict with leading sample axis.

    Returns:
        Scalar mean loss as a Python float.
    """
    output = jax.vmap(model_fn, in_axes=(0, 0))(data_test, params)
    return float(loss_fn(output, data_test))


def score_program(program: Program, data: tuple, loss_fn, config: dict) -> float:
    """Fit params on data[0] and return mean loss on data[1].

    Intended to be called inside a subprocess — the process timeout is the
    caller's responsibility.

    Args:
        program: Candidate program to score.
        data: Tuple of ``(data_train, data_test)``, each a JAX data dict.
        loss_fn: Loss function, signature ``(output, data) -> scalar`` over the full batch.
        config: Scoring config dict with ``gradient_descent`` and ``param_penalty_weight``.

    Returns:
        Scalar loss on data_test as a Python float.
    """
    data_train, data_test = data
    model_fn, param_est_fn = program.compile()

    params = _get_params(param_est_fn, model_fn.DEFAULT_PARAMS, data_train)
    params = _optimize(model_fn, loss_fn, params, data_train, config["gradient_descent"])

    complexity_penalty = config["param_penalty_weight"] * program.n_params
    return _eval_loss(model_fn, loss_fn, params, data_test) + complexity_penalty
