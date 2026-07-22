"""
Scoring module for EDGAR.

This module provides functionalities for evaluating the performance of generated
models, mirroring the asynchronous generation pattern found in `edgar/llm/generate.py`.
It handles the scoring of individual programs within isolated subprocesses
and orchestrates population-level scoring and ranking.

Key functionalities include:
- `_score_one_model`: Scores a single program in a dedicated subprocess, ensuring
  clean JAX initialization and robust timeout enforcement.
- `score`: Identifies programs requiring evaluation for a specific data split
  (e.g., 'discover' or 'validate') and updates their performance metrics.

Per-program work runs in a subprocess so JAX initialises cleanly and runaway
models can be killed on timeout. The subprocess start method defaults to
"spawn" (safe with the macOS Objective-C runtime / JAX) but can be overridden
to "fork" via the EDGAR_MP_START_METHOD environment variable. Fork is useful for
scripts that don't use an `if __name__ == "__main__":` guard (e.g., interactive
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
from .utils import _safe_loss


# ── helpers ──


def _get_params(param_est_fn, default_params, data_train):
    """Estimates initial parameters for a model, falling back to defaults if the parameter estimator fails.

    This function attempts to use the provided `param_est_fn` to derive initial
    parameters for each sample in the training data. If the `param_est_fn` is
    not provided or fails during execution, it falls back to stacking the
    `default_params` for each sample.

    Args:
        param_est_fn: The parameter estimator function (callable) from the program.
            Expected to take a single data sample (dict) and return a dict of parameters.
        default_params: A dictionary of default parameters for the model.
        data_train: A dictionary of training data, where keys are feature names
            and values are JAX arrays. Assumes the first dimension is the batch size.

    Returns:
        A JAX pytree (dictionary of JAX arrays) representing the initial parameters
        for each sample in `data_train`.
    """
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
    """Performs gradient descent to optimize model parameters.

    This function uses the Optax library and JAX to perform gradient-based
    optimization of a model's parameters. It minimizes the `loss_fn` using
    the Adam optimizer, tracking the best parameters found during the
    optimization process.

    Args:
        model_fn: The JAX-compiled model function (callable). Expected to take
            data and parameters, and return predictions.
        loss_fn: The loss function (callable). Expected to take predictions
            and data, and return per-sample losses.
        params_init: The initial parameters for the model (JAX pytree).
        data_train: A dictionary of training data, where keys are feature names
            and values are JAX arrays. Assumes the first dimension is the batch size.
        gd_config: Configuration dictionary for gradient descent,
            e.g., `{"learning_rate": 1e-3, "max_iter": 1000}`.

    Returns:
        A JAX pytree representing the optimized parameters.
    """
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
            print(f"  [optimize] Step {step}: Non-finite loss {loss_val}, breaking.")
            break

        if not jnp.all(jnp.isfinite(grad)):
            print(f"  [optimize] Step {step}: Non-finite gradient detected!")
            # We don't break here yet to see if loss becomes NaN in next step

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
    """Computes the overall scalar loss for a model on a given dataset.

    This function calculates the mean of the per-sample losses from the `loss_fn`
    after running the `model_fn` with the provided parameters on the test data.

    Args:
        model_fn: The JAX-compiled model function (callable).
        loss_fn: The loss function (callable) that returns per-sample losses.
        params: The model parameters (JAX pytree).
        data_test: A dictionary of test data.

    Returns:
        The scalar mean loss as a float. Returns `float("inf")` if parameters are `None`.
    """
    if params is None:
        return float("inf")
    output = jax.vmap(model_fn, in_axes=(0, 0))(data_test, params)
    return float(jnp.mean(loss_fn(output, data_test)))


def _eval_sample_losses(model_fn, loss_fn, params, data_test):
    """Computes individual losses for each sample in a dataset.

    This function applies the model and loss function to each sample in the
    test data to get an array of per-sample loss values, without any
    complexity penalty.

    Args:
        model_fn: The JAX-compiled model function (callable).
        loss_fn: The loss function (callable) that returns per-sample losses.
        params: The model parameters (JAX pytree).
        data_test: A dictionary of test data.

    Returns:
        A NumPy array of per-sample loss values.
    """
    output = jax.vmap(model_fn, in_axes=(0, 0))(data_test, params)
    return np.asarray(loss_fn(output, data_test))


def _eval_fingerprint(model_fn, params, X_eval):
    """Generates a low-dimensional "fingerprint" of model outputs for deduplication.

    This fingerprint is used to compare models and identify functionally
    identical or very similar programs, even if their code differs. It applies
    the model to a small, fixed subset of the evaluation data (`X_eval`).

    Args:
        model_fn: The JAX-compiled model function (callable).
        params: The model parameters (JAX pytree).
        X_eval: A dictionary of evaluation data, containing a `_sample_indices`
            key to select a subset of samples for fingerprinting.

    Returns:
        A JAX array representing the model's output fingerprint.
    """
    sample_indices = X_eval["_sample_indices"]
    params_matched = jax.tree_util.tree_map(lambda p: p[sample_indices], params)
    return jax.vmap(model_fn, in_axes=(0, 0))(X_eval, params_matched)


def _worker(queue, program_bytes, data, loss_fn_bytes, config, X_eval, split):
    """Scores one program inside a subprocess.

    This function deserializes a `Program` and `loss_fn`, compiles the program's
    JAX model and parameter estimator, performs parameter estimation,
    optimization, calculates various losses, and generates a fingerprint and
    sample-specific losses. All results are placed onto a multiprocessing queue.
    It includes robust error handling for model loading, optimization, and
    evaluation steps.

    Args:
        queue: A multiprocessing Queue to put the results on.
        program_bytes: A `cloudpickle`-serialized `Program` object.
        data: A tuple `(data_train, data_test)` containing dictionaries of JAX arrays
            for training and testing.
        loss_fn_bytes: A `cloudpickle`-serialized loss function.
        config: A dictionary containing scoring configuration, e.g.,
            `param_penalty_weight` and `gradient_descent` settings.
        X_eval: A dictionary of evaluation data for fingerprinting, or `None`.
        split: A string indicating the current scoring split (e.g., "discover" or "validate").

    Returns:
        None. Results are placed on the `queue` as a 7-tuple:
        `(final_loss, initial_loss, fingerprint, params, sample_losses, params_init, sample_losses_init)`.
        If any critical failure occurs (model loading, optimization), infinite
        losses and `None` for other results are returned.
    """
    program = cloudpickle.loads(program_bytes)
    loss_fn = cloudpickle.loads(loss_fn_bytes)
    if isinstance(loss_fn, tuple):
        loss_fn_train, loss_fn_test = loss_fn
    else:
        loss_fn_train = loss_fn_test = loss_fn

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
        initial_loss = (
            _eval_loss(model_fn, loss_fn_test, params_init, data_test) + penalty
        )
        params = _optimize(
            model_fn, loss_fn_train, params_init, data_train, config["gradient_descent"]
        )
        final_loss = _eval_loss(model_fn, loss_fn_test, params, data_test) + penalty
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
        sample_losses = _eval_sample_losses(model_fn, loss_fn_test, params, data_test)
    except Exception as e:
        print(f"[scoring] program #{program.idx} sample_losses failed (ignored): {e}")
        sample_losses = None

    try:
        sample_losses_init = (
            _eval_sample_losses(model_fn, loss_fn_test, params_init, data_test)
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
    """Scores a single program in a dedicated subprocess, enforcing a timeout.

    This function delegates to `_score_one_with_outcome` to execute the
    scoring in a separate process, allowing for robust timeout handling.
    If the worker process does not return a result within `config["timeout_s"]`,
    it is killed, and the program is assigned an infinite loss.

    Args:
        program: The `Program` object to be scored.
        data: A tuple `(data_train, data_test)` containing dictionaries of JAX arrays
            for training and testing.
        loss_fn: The loss function (callable).
        config: A dictionary containing scoring configuration, including `timeout_s`.
        X_eval: Optional. A dictionary of evaluation data for fingerprinting.
            If `None`, fingerprint computation is skipped.
        split: The scoring split (e.g., "discover" or "validate").

    Returns:
        A 7-tuple: `(final_loss, initial_loss, eval_fingerprint, params, sample_losses, params_init, sample_losses_init)`.
        If a timeout occurs or `program.n_params` is `None`, infinite losses and `None`
        for other results are returned.
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
    """Checks if a program has both JAX model code and a NumPy parameter estimator.

    A program is considered ready for scoring if it has successfully had its
    numpy model translated to JAX and has a parameter estimator.

    Args:
        program: The `Program` object to check.

    Returns:
        True if the program has both `model_jax` and `param_est` code, False otherwise.
    """
    return bool(program.code.model_jax and program.code.param_est)


def _needs_scoring(population: Population, split: str) -> list[Program]:
    """Identifies programs in the population that need to be scored for a given split.

    Programs are considered to need scoring if they have JAX model code and their
    `final` loss for the specified `split` has not yet been set (i.e., it is `None`).
    For the 'validate' split, programs initialized with `NotValidated` are also
    considered unscored until `population.prepare_validation_scoring` is called.
    A program with a scalar or infinite loss is treated as already scored and
    will not be rescored.

    Args:
        population: The `Population` object containing all programs.
        split: The name of the scoring split (e.g., "discover" or "validate").

    Returns:
        A list of `Program` objects that require scoring for the specified split.
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
    """Scores every program needing scoring on the given split.

    This function iterates through the `population`, identifies programs that
    require scoring for the specified `split`, and then calls `_score_one_model`
    for each. The results (losses, fingerprint, parameters) are then used to
    mutate the corresponding `Program` object's attributes in place.

    Pass X_eval=None to skip fingerprint computation (e.g. on validate scoring,
    so the discover-derived fingerprint isn't overwritten).

    Streams per-program tick lines to ``run.log`` and updates the active
    ``RunMetrics`` (if any) so the dashboard can show ``score (k/n)`` live.

    Args:
        population: The `Population` object whose programs are to be scored.
            This object will be mutated in place.
        X_split: A tuple `(data_train, data_test)` containing dictionaries of JAX arrays
            for training and testing data specific to the current split.
        X_eval: A dictionary of evaluation data for fingerprint computation.
            Pass `None` to skip fingerprint computation (e.g., when scoring on
            the 'validate' split to avoid overwriting the 'discover'-derived fingerprint).
        config: A dictionary containing scoring configuration parameters.
        loss_fn: The loss function (callable) to be used for scoring.
        split: The name of the scoring split (e.g., "discover" or "validate").

    Returns:
        None. The `population` object is mutated in place, updating
        `program.program_losses.<split>.{init, final}`, `program.eval_fingerprint`,
        `program.params`, `program.params_init`, `program.sample_losses`,
        and `program.sample_losses_init`.
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
    """Ranks programs based on their final validation loss.

    This function assigns a numerical rank to programs that have successfully
    undergone validation scoring (i.e., their `validate.final` loss is not
    `NotValidated`). Programs are sorted by their `validate.final` loss in
    ascending order.

    Args:
        population: The `Population` object containing all programs.
            The `rank` attribute of individual `Program` objects will be
            mutated in place.

    Returns:
        None. The `rank` attribute of `Program` objects within the population
        is updated, and the ranking information is printed to the console.
    """
    validated_program_indices = [
        i
        for i in range(len(population))
        if not isinstance(population[i].program_losses.validate.final, NotValidated)
    ]
    validated_program_indices.sort(
        key=lambda i: _safe_loss(population[i].program_losses.validate.final)
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
