"""
Scoring. Mirrors the generate pattern in src/llm/generate.py:

    _score_one_model(program, ...)   # single program, timeout baked in
    score(population, ...)           # finds programs needing scoring, fills losses

Per-program work runs in a spawn subprocess so JAX initialises cleanly and
runaway models can be killed on timeout.
"""
from __future__ import annotations

import multiprocessing as mp

import cloudpickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree

from ..evolution.program import Program
from ..evolution.population import Population


# ── helpers ──

def _get_params(param_est_fn, default_params, data_train):
    try:
        return jax.vmap(param_est_fn)(data_train)
    except Exception:
        n = next(iter(data_train.values())).shape[0]
        return jax.tree_util.tree_map(lambda x: jnp.stack([x] * n), default_params)


def _optimize(model_fn, loss_fn, params_init, data_train, gd_config):
    flat, unflatten = ravel_pytree(params_init)

    def total_loss(flat_p):
        p = unflatten(flat_p)
        output = jax.vmap(model_fn, in_axes=(0, 0))(data_train, p)
        return jnp.mean(loss_fn(output, data_train))

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
    return float(jnp.mean(loss_fn(output, data_test)))


def _eval_sample_losses(model_fn, loss_fn, params, data_test):
    output = jax.vmap(model_fn, in_axes=(0, 0))(data_test, params)
    return np.asarray(loss_fn(output, data_test))


def _eval_fingerprint(model_fn, params, X_eval):
    sample_indices = X_eval.pop('_sample_indices')
    params_matched = jax.tree_map(lambda p: p[sample_indices], params)
    return jax.vmap(model_fn, in_axes=(0, 0))(X_eval, params_matched)


def _worker(queue, program, data, loss_fn_bytes, config, X_eval):
    """Score one program inside a subprocess. Always puts a 3-tuple on the queue."""
    try:
        loss_fn = cloudpickle.loads(loss_fn_bytes)
        data_train, data_test = data
        model_fn, param_est_fn = program.compile()
        penalty = config["param_penalty_weight"] * program.n_params
        params_init = _get_params(param_est_fn, model_fn.DEFAULT_PARAMS, data_train)
        initial_loss = _eval_loss(model_fn, loss_fn, params_init, data_train) + penalty
        params = _optimize(model_fn, loss_fn, params_init, data_train, config["gradient_descent"])
        final_loss = _eval_loss(model_fn, loss_fn, params, data_test) + penalty
        fingerprint = _eval_fingerprint(model_fn, params, X_eval) if X_eval is not None else None
        sample_losses = _eval_sample_losses(model_fn, loss_fn, params, data_test)
        result = (final_loss, initial_loss, fingerprint, params, sample_losses)
    except Exception:
        result = (float("inf"), float("inf"), None, None, None)
    queue.put(result)


# ── per-program ──

def _score_one_model(
    program: Program,
    data: tuple,
    loss_fn,
    config: dict,
    X_eval=None,
) -> tuple[float, float, jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """Score one program in a spawn subprocess; kill on timeout.

    Auto-populates program.n_params via count_params() if not set.
    Returns (final_loss, initial_loss, eval_fingerprint, params).
    """
    if program.n_params is None:
        try:
            program.count_params()
        except Exception:
            return (float("inf"), float("inf"), None, None, None)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    loss_fn_bytes = cloudpickle.dumps(loss_fn)
    proc = ctx.Process(target=_worker, args=(queue, program, data, loss_fn_bytes, config, X_eval))
    proc.start()
    proc.join(timeout=config["timeout_s"])
    if proc.is_alive():
        proc.kill()
        proc.join()
        return (float("inf"), float("inf"), None, None, None)
    return queue.get()


# ── population-level ──

def _has_jax_code(program: Program) -> bool:
    """Does this program have jax code that can be scored?"""
    has_model = program.code_jax.model
    has_param_est = program.code_jax.param_est
    return bool(has_model and has_param_est)


def _needs_scoring(population: Population, split: str) -> list[Program]:
    """Programs with jax code whose `split` final loss hasn't been written yet.

    Initialization behavior:
    - discover: initialized to None, so all programs with JAX code are scored
    - validate: initialized to inf, only set to None for programs alive at loop end 
      (via population.prepare_validation_scoring), allowing selective validation 
      scoring.

    A program with a scalar/inf loss is treated as already scored —
    inf means scoring genuinely failed and shouldn't be retried.
    """
    return [
        population[i] for i in range(len(population))
        if _has_jax_code(population[i])
        and getattr(population[i].program_losses, split).final is None
    ]


def score(
    population: Population,
    X_split: tuple,
    X_eval,
    config: dict,
    loss_fn,
    split: str,
) -> None:
    """Score every program needing scoring on the given split.

    Mutates: program.program_losses.<split>.{init, final}, program.eval_fingerprint,
    program.sample_losses, program.n_params.

    Pass X_eval=None to skip fingerprint computation (e.g. on validate scoring,
    so the discover-derived fingerprint isn't overwritten).
    """
    for program in _needs_scoring(population, split):
        final_loss, initial_loss, fingerprint, params, sample_losses = _score_one_model(
            program, X_split, loss_fn, config, X_eval
        )
        loss_pair = getattr(program.program_losses, split)
        loss_pair.init = initial_loss
        loss_pair.final = final_loss
        if fingerprint is not None:
            program.eval_fingerprint = fingerprint
        if params is not None:
            program.params = params
        if sample_losses is not None and split == "discover":
            program.sample_losses = sample_losses
