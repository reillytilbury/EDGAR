import multiprocessing as mp

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree

from ..evolution.program import Program


# ── helpers ──

def _worker(queue, program, data, loss_fn, config, X_eval):
    try:
        result = score_program(program, data, loss_fn, config, X_eval)
    except Exception:
        result = (float("inf"), float("inf"), None)
    queue.put(result)


def _get_params(param_est_fn, default_params, data_train):
    """Estimate initial parameters for all samples via vmapped param estimator."""
    try:
        return jax.vmap(param_est_fn)(data_train)
    except Exception:
        n = next(iter(data_train.values())).shape[0]
        return jax.tree_util.tree_map(lambda x: jnp.stack([x] * n), default_params)


def _optimize(model_fn, loss_fn, params_init, data_train, gd_config):
    """Fit parameters on data_train using Adam, returning the best params found."""
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
    output = jax.vmap(model_fn, in_axes=(0, 0))(data_test, params)
    return float(loss_fn(output, data_test))


def _eval_fingerprint(model_fn, params, X_eval):
    return jax.vmap(model_fn, in_axes=(0, 0))(X_eval, params)


# ── public API ──

def score_program(program: Program, data: tuple, loss_fn, config: dict, X_eval=None) -> tuple[float, float, jnp.ndarray | None]:
    """Score a program: fit params on data_train, evaluate on data_test.

    Intended to be called inside a subprocess — the process timeout is the
    caller's responsibility (see ``score_with_timeout``).

    Returns (final_loss, initial_loss, eval_fingerprint).
    """
    data_train, data_test = data
    model_fn, param_est_fn = program.compile()

    complexity_penalty = config["param_penalty_weight"] * program.n_params
    params_init = _get_params(param_est_fn, model_fn.DEFAULT_PARAMS, data_train)
    initial_loss = _eval_loss(model_fn, loss_fn, params_init, data_train) + complexity_penalty

    params = _optimize(model_fn, loss_fn, params_init, data_train, config["gradient_descent"])

    final_loss = _eval_loss(model_fn, loss_fn, params, data_test) + complexity_penalty
    fingerprint = _eval_fingerprint(model_fn, params, X_eval) if X_eval is not None else None
    return (final_loss, initial_loss, fingerprint)


def score_with_timeout(program, data, loss_fn, config, X_eval=None) -> tuple[float, float, jnp.ndarray | None]:
    """Run score_program in a spawn subprocess; kill on timeout.

    Uses spawn (not fork) so JAX initialises cleanly in the child process.
    Returns (inf, inf, None) on timeout or any failure inside the worker.
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(queue, program, data, loss_fn, config, X_eval))
    proc.start()
    proc.join(timeout=config["timeout_s"])
    if proc.is_alive():
        proc.kill()
        proc.join()
        return (float("inf"), float("inf"), None)
    return queue.get()

# TODO: Need to add the score func called by run.py that doesn't yet exist.