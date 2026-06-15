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
import time

import traceback
import cloudpickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree
import warnings

from ..evolution.program import (
    ModelLoadingError,
    NotValidated,
    ParamEstLoadingError,
    Program,
)
from ..evolution.population import Population
from ..io.metrics import get_active_metrics, stream_line


# ── helpers ──


def _get_params(param_est_fn, default_params, data_train):
    try:
        n = next(iter(data_train.values())).shape[0]
        data_np = {k: np.asarray(v) for k, v in data_train.items()}
        per_sample = [
            param_est_fn({k: v[i] for k, v in data_np.items()}) for i in range(n)
        ]
        return {
            k: jnp.stack([jnp.asarray(s[k]) for s in per_sample]) for k in per_sample[0]
        }
    except Exception as e:
        warnings.warn(
            f"[scoring] param_est_fn failed at runtime, falling back to default params: {e}"
        )
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
        loss_val, grad = loss_and_grad(flat)  # loss_i, grad_i for parameters_i
        if not jnp.isfinite(loss_val):
            break
        if float(loss_val) < best_loss:
            best_loss, best_flat = (
                float(loss_val),
                flat.copy(),
            )  # store loss_i, parameters_i
        updates, opt_state = opt.update(grad, opt_state, flat)
        flat = optax.apply_updates(
            flat, updates
        )  # update parameters to parameters_{i+1}
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
    sample_indices = X_eval["_sample_indices"]
    params_matched = jax.tree_util.tree_map(lambda p: p[sample_indices], params)
    return jax.vmap(model_fn, in_axes=(0, 0))(X_eval, params_matched)


def _worker(queue, program_bytes, data, loss_fn_bytes, config, X_eval, split):
    """Score one program inside a subprocess. Always puts a 7-tuple on the queue."""
    program = cloudpickle.loads(program_bytes)
    loss_fn = cloudpickle.loads(loss_fn_bytes)
    data_train, data_test = data

    try:
        model_fn = program.compile_model()
    except ModelLoadingError as e:
        print(f"[scoring] program #{program.idx} model failed to load: {e}")
        queue.put((float("inf"), float("inf"), None, None, None, None, None))
        return

    try:
        param_est_fn = program.compile_param_est()
    except ParamEstLoadingError as e:
        warnings.warn(
            f"[scoring] program #{program.idx} param_est failed to load, falling back to default_params: {e}"
        )
        param_est_fn = None

    try:
        penalty = config["param_penalty_weight"] * program.n_params
        params_init = _get_params(param_est_fn, program.default_params, data_train)
        initial_loss = _eval_loss(model_fn, loss_fn, params_init, data_test) + penalty
        params = _optimize(
            model_fn, loss_fn, params_init, data_train, config["gradient_descent"]
        )
        final_loss = _eval_loss(model_fn, loss_fn, params, data_test) + penalty
    except Exception as e:
        print(f"[scoring] program #{program.idx} failed during optimize/eval: {e}")
        print(f"[scoring] traceback:\n{traceback.format_exc()}")
        print(f"[scoring] code.model_jax:\n{program.code.model_jax}")
        queue.put((float("inf"), float("inf"), None, None, None, None, None))
        return

    # Fingerprint and sample losses are non-critical: failures here don't poison the loss.
    try:
        fingerprint = (
            _eval_fingerprint(model_fn, params, X_eval) if X_eval is not None else None
        )
    except Exception as e:
        print(f"[scoring] program #{program.idx} fingerprint failed (ignored): {e}")
        fingerprint = None

    try:
        sample_losses = _eval_sample_losses(model_fn, loss_fn, params, data_test)
    except Exception as e:
        print(f"[scoring] program #{program.idx} sample_losses failed (ignored): {e}")
        sample_losses = None

    try:
        sample_losses_init = (
            _eval_sample_losses(model_fn, loss_fn, params_init, data_test)
            if split == "discover"
            else None
        )
    except Exception as e:
        print(
            f"[scoring] program #{program.idx} sample_losses_init failed (ignored): {e}"
        )
        sample_losses_init = None

    queue.put(
        (
            final_loss,
            initial_loss,
            fingerprint,
            params,
            sample_losses,
            params_init,
            sample_losses_init,
        )
    )


# ── per-program ──


def _score_one_model(
    program: Program,
    data: tuple,
    loss_fn,
    config: dict,
    X_eval=None,
    split: str = "discover",
) -> tuple[
    float,
    float,
    jnp.ndarray,
    dict | None,
    jnp.ndarray | None,
    dict | None,
    jnp.ndarray | None,
]:
    """Score one program in a spawn subprocess; kill on timeout.

    Returns ``(final_loss, initial_loss, eval_fingerprint, params, sample_losses,
    params_init, sample_losses_init)``. On timeout or worker exception, returns
    the all-inf 7-tuple. ``score()`` uses ``_score_one_with_outcome`` directly
    to recover the precise outcome (``timeout`` vs ``inf``) for metrics.
    """
    result = _score_one_with_outcome(program, data, loss_fn, config, X_eval, split)
    return result[:7]


def _score_one_with_outcome(
    program: Program,
    data: tuple,
    loss_fn,
    config: dict,
    X_eval=None,
    split: str = "discover",
) -> tuple[
    float,
    float,
    jnp.ndarray,
    dict | None,
    jnp.ndarray | None,
    dict | None,
    jnp.ndarray | None,
    str,
]:
    """Same as ``_score_one_model`` but also returns an outcome label:
    ``"ok"`` (finite loss), ``"timeout"`` (subprocess killed), ``"inf"``
    (worker raised or program has no params).
    """
    if program.n_params is None:
        warnings.warn(
            f"Program #{program.idx} has n_params=None, applying infinite loss, "
            "verify that its default_params were set prior to scoring"
        )
        return (float("inf"), float("inf"), None, None, None, None, None, "inf")

    ctx = mp.get_context(os.environ.get("EDGAR_MP_START_METHOD", "spawn"))
    queue = ctx.Queue()
    loss_fn_bytes = cloudpickle.dumps(loss_fn)
    program_bytes = cloudpickle.dumps(program)
    proc = ctx.Process(
        target=_worker,
        args=(queue, program_bytes, data, loss_fn_bytes, config, X_eval, split),
    )
    proc.start()
    try:
        result = queue.get(timeout=config["timeout_s"])
    except mp.queues.Empty:
        proc.kill()
        proc.join()
        return (
            float("inf"),
            float("inf"),
            None,
            None,
            None,
            None,
            None,
            "timeout",
        )
    proc.join()
    final, init, fp, params, samples, params_init, samples_init = result
    # Worker puts (inf, inf, None, None, None, None, None) on its own exception
    # path. Treat that as "inf" rather than "timeout" so metrics distinguish them.
    outcome = "ok" if np.isfinite(final) else "inf"
    return (final, init, fp, params, samples, params_init, samples_init, outcome)


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
        population[i]
        for i in range(len(population))
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

    Streams per-program tick lines to ``run.log`` and updates the active
    ``RunMetrics`` (if any) so the dashboard can show ``score (k/n)`` live.
    """
    queue = _needs_scoring(population, split)
    n_total = len(queue)
    metrics = get_active_metrics()
    counters = {"ok": 0, "timeout": 0, "inf": 0}
    latencies_ms: list[float] = []

    for k, program in enumerate(queue, start=1):
        t0 = time.monotonic()
        (
            final_loss,
            initial_loss,
            fingerprint,
            params,
            sample_losses,
            params_init,
            sample_losses_init,
            outcome,
        ) = _score_one_with_outcome(program, X_split, loss_fn, config, X_eval, split)
        latency_ms = (time.monotonic() - t0) * 1000.0
        latencies_ms.append(latency_ms)
        counters[outcome] += 1

        loss_pair = getattr(program.program_losses, split)
        loss_pair.init = initial_loss
        loss_pair.final = final_loss
        if fingerprint is not None:
            program.eval_fingerprint = fingerprint
        if params is not None:
            program.params = params
        if params_init is not None:
            program.params_init = params_init
        if sample_losses is not None and split == "discover":
            program.sample_losses = sample_losses
        if sample_losses_init is not None and split == "discover":
            program.sample_losses_init = sample_losses_init

        if metrics is not None:
            metrics.record_score_result(program.idx, latency_ms, outcome)

        # Cheap progress line so the user sees movement during the slow stage.
        # Print every program for low n_total, every 4 for larger sweeps.
        tick_every = 1 if n_total <= 12 else 4
        if metrics is not None and (k == n_total or k % tick_every == 0):
            avg_s = (sum(latencies_ms) / len(latencies_ms)) / 1000.0
            stream_line(
                metrics,
                f"  [score {split}] {k}/{n_total}  "
                f"(avg {avg_s:.1f}s, {counters['ok']} ok, "
                f"{counters['timeout']} timeout, {counters['inf']} inf)",
            )


def rank(
    population: Population,
) -> None:
    """
    Rank programs surviving at the end of the evolution by validate.final loss.
    Only ranks those which have validate.final != NotValidated, which should be the ones alive at the end of the evolution.
    """
    validated_program_indices = [
        i
        for i in range(len(population))
        if not isinstance(population[i].program_losses.validate.final, NotValidated)
    ]
    validated_program_indices.sort(
        key=lambda i: population[i].program_losses.validate.final or float("inf")
    )
    print("Ranking of programs by validate.final loss:")
    for rank, program in enumerate(
        [population[i] for i in validated_program_indices], start=1
    ):
        program.rank = rank
        if program.program_losses.validate.final is None:
            loss_display = float("inf")
        else:
            loss_display = program.program_losses.validate.final
        print(
            f"Rank {rank}: Program #{program.idx} ({program.name}): {loss_display:.4f}"
        )
