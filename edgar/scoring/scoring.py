"""
Scoring. Mirrors the generate pattern in edgar/llm/generate.py:

    _score_one_model(program, ...)   # single program, timeout baked in
    score(population, ...)           # finds programs needing scoring, fills losses

Per-program work runs in a subprocess so JAX initialises cleanly and runaway
models can be killed on timeout. The subprocess start method defaults to
"spawn" (safe with the macOS Objective-C runtime / JAX) but can be overridden
to "fork" via the EDGAR_MP_START_METHOD env var. Fork is useful for scripts
that don't use an `if __name__ == "__main__":` guard (e.g. interactive
notebooks, tutorial scripts) because it doesn't re-import the parent module
in each child.
"""
from __future__ import annotations

import multiprocessing as mp
import os

import cloudpickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree
import warnings

from ..evolution.program import NotValidated, Program
from ..evolution.population import Population


# ── helpers ──

def _get_params(param_est_fn, default_params, data_train):
    try:
        n = next(iter(data_train.values())).shape[0]
        data_np = {k: np.asarray(v) for k, v in data_train.items()}
        per_sample = [param_est_fn({k: v[i] for k, v in data_np.items()}) for i in range(n)]
        return {k: jnp.stack([jnp.asarray(s[k]) for s in per_sample]) for k in per_sample[0]}
    except Exception as e:
        warnings.warn(f"[scoring] param_est_fn failed, falling back to default params: {e}")
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

    for step in range(1, gd_config["max_iter"] + 1):
        loss_val, grad = loss_and_grad(flat) #loss_i, grad_i for parameters_i
        if not jnp.isfinite(loss_val):
            break
        if float(loss_val) < best_loss:
            best_loss, best_flat = float(loss_val), flat.copy() #store loss_i, parameters_i
        updates, opt_state = opt.update(grad, opt_state, flat)
        flat = optax.apply_updates(flat, updates) #update parameters to parameters_{i+1}
        # if step % 200 == 0 or step == gd_config["max_iter"]:
        #     print(f"step {step:4d}  loss {loss_val:.4f}")

    return unflatten(best_flat)


def _eval_loss(model_fn, loss_fn, params, data_test):
    if params is None:
        return float("inf")
    output = jax.vmap(model_fn, in_axes=(0, 0))(data_test, params)
    return float(jnp.mean(loss_fn(output, data_test)))


def _eval_sample_losses(model_fn, loss_fn, params, data_test):
    output = jax.vmap(model_fn, in_axes=(0, 0))(data_test, params)
    return np.asarray(loss_fn(output, data_test))


def _eval_fingerprint(model_fn, params, X_eval):
    sample_indices = X_eval['_sample_indices']
    params_matched = jax.tree_util.tree_map(lambda p: p[sample_indices], params)
    return jax.vmap(model_fn, in_axes=(0, 0))(X_eval, params_matched)


def _worker(queue, program, data, loss_fn_bytes, config, X_eval):
    """Score one program inside a subprocess. Always puts a 3-tuple on the queue."""
    try:
        loss_fn = cloudpickle.loads(loss_fn_bytes)
        data_train, data_test = data
        model_fn, param_est_fn = program.compile()
        penalty = config["param_penalty_weight"] * program.n_params
        params_init = _get_params(param_est_fn, program.default_params, data_train)
        initial_loss = _eval_loss(model_fn, loss_fn, params_init, data_test) + penalty
        params = _optimize(model_fn, loss_fn, params_init, data_train, config["gradient_descent"])
        final_loss = _eval_loss(model_fn, loss_fn, params, data_test) + penalty
    except Exception as e:
        import traceback
        print(f"[scoring] program #{program.idx} failed during compile/optimize/eval: {e}")
        print(f"[scoring] traceback:\n{traceback.format_exc()}")
        print(f"[scoring] code.model_jax:\n{program.code.model_jax}")
        print(f"[scoring] code.param_est:\n{program.code.param_est}")
        queue.put((float("inf"), float("inf"), None, None, None))
        return

    # Fingerprint and sample losses are non-critical: failures here don't poison the loss.
    try:
        fingerprint = _eval_fingerprint(model_fn, params, X_eval) if X_eval is not None else None
    except Exception as e:
        print(f"[scoring] program #{program.idx} fingerprint failed (ignored): {e}")
        fingerprint = None

    try:
        sample_losses = _eval_sample_losses(model_fn, loss_fn, params, data_test)
    except Exception as e:
        print(f"[scoring] program #{program.idx} sample_losses failed (ignored): {e}")
        sample_losses = None

    queue.put((final_loss, initial_loss, fingerprint, params, sample_losses))


# ── per-program ──

def _score_one_model(
    program: Program,
    data: tuple,
    loss_fn,
    config: dict,
    X_eval=None,
) -> tuple[float, float, jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """Score one program in a spawn subprocess; kill on timeout.

    Returns (final_loss, initial_loss, eval_fingerprint, params).
    """
    if program.n_params is None:
        warnings.warn(f"Program #{program.idx} has n_params=None, applying infinite loss, verify that its default_params were set prior to scoring")
        return (float("inf"), float("inf"), None, None, None)

    ctx = mp.get_context(os.environ.get("EDGAR_MP_START_METHOD", "spawn"))
    queue = ctx.Queue()
    loss_fn_bytes = cloudpickle.dumps(loss_fn)
    proc = ctx.Process(target=_worker, args=(queue, program, data, loss_fn_bytes, config, X_eval))
    proc.start()
    try:
        result = queue.get(timeout=config["timeout_s"])
    except mp.queues.Empty: #if subproces doesn't respond in time config["timeout_s"]
        proc.kill()
        proc.join()
        return (float("inf"), float("inf"), None, None, None)
    proc.join()
    return result


# ── population-level ──

def _has_jax_code(program: Program) -> bool:
    """Does this program have jax model code and a numpy param_est that can be scored?"""
    return bool(program.code.model_jax and program.code.param_est)


def _needs_scoring(population: Population, split: str) -> list[Program]:
    """Programs with jax code whose `split` final loss hasn't been written yet.

    Initialization behavior:
    - discover: initialized to None, so all programs with JAX code are scored
    - validate: initialized to NotValidated, only set to None for programs alive at loop end 
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

def rank(
        population: Population,
) -> None:
    """ 
        Rank programs surviving at the end of the evolution by validate.final loss.
        Only ranks those which have validate.final != NotValidated, which should be the ones alive at the end of the evolution.
    """
    validated_program_indices = [i for i in range(len(population)) if not isinstance(population[i].program_losses.validate.final, NotValidated)]
    validated_program_indices.sort(key=lambda i: population[i].program_losses.validate.final or float("inf"))
    print("Ranking of programs by validate.final loss:")
    for rank, program in enumerate([population[i] for i in validated_program_indices], start=1):
        program.rank = rank
        if program.program_losses.validate.final is None:
            loss_display = float("inf")
        else:
            loss_display = program.program_losses.validate.final
        print(f"Rank {rank}: Program #{program.idx} ({program.name}): {loss_display:.4f}")