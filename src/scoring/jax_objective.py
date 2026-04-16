import gc
import inspect
import logging
import multiprocessing as mp
import textwrap
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import timeout_decorator
from jax.flatten_util import ravel_pytree

from .. import utils
from ..timeout_worker import run_estimator_from_source


class ObjectiveTimeout(Exception):
    """Raised when objective exceeds wall-clock timeout."""

class ProcessTimeoutUnavailable(RuntimeError):
    """Raised when process-based timeout backend cannot be used."""


def _get_callable_source(func) -> tuple[str | None, str | None]:
    """
    Best-effort extraction of callable source for spawn-safe subprocess execution.
    """
    code = getattr(func, "__source_code__", None)
    name = getattr(func, "__function_name__", None) or getattr(func, "__name__", None)
    if isinstance(code, str) and code.strip() and isinstance(name, str) and name:
        return code, name
    try:
        source = textwrap.dedent(inspect.getsource(func)).strip()
        if not source:
            return None, None
        inferred_name = getattr(func, "__name__", None)
        if not isinstance(inferred_name, str) or not inferred_name:
            return None, None
        return source, inferred_name
    except Exception:
        return None, None

def _clear_jax_runtime_cache():
    """
    Best-effort cache cleanup to reduce long-run GPU memory pressure.
    """
    try:
        jax.clear_caches()
    except Exception:
        pass
    gc.collect()

def _run_param_estimator_with_timeout(param_estimator, data_i, timeout_s: float):
    """
    Robust timeout wrapper for potentially hanging estimators.

    Uses a dedicated ``spawn`` subprocess so estimator execution can be force-
    terminated without ``fork`` conflicts in JAX multithreaded runtimes.
    """
    source_code, function_name = _get_callable_source(param_estimator)
    if not source_code or not function_name:
        raise ProcessTimeoutUnavailable(
            "Unable to extract source for estimator; process timeout unavailable."
        )

    start_methods = mp.get_all_start_methods()
    if "spawn" not in start_methods:
        raise ProcessTimeoutUnavailable("Process timeout requires 'spawn' start method.")

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=run_estimator_from_source,
        args=(source_code, function_name, utils.data_as_numpy(data_i), child_conn),
        daemon=True,
    )
    proc.start()
    child_conn.close()
    try:
        if not parent_conn.poll(timeout_s):
            raise TimeoutError(f"param_estimator timed out after {timeout_s:.2f}s")
        status, payload, detail = parent_conn.recv()
        if status == "ok":
            return payload
        raise RuntimeError(f"param_estimator failed: {payload} ({detail})")
    finally:
        parent_conn.close()
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=0.2)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=0.2)

def compute_initial_params(
    param_estimator,
    model,
    data,
    timeout_s: float | None = 5.0,
    deadline_s: float | None = None,
):
    """
    Estimate per-sample initial model parameters with safe fallbacks.

    Uses per-sample timeouts when provided and clamps each estimate to the
    remaining wall-clock time before ``deadline_s``.

    Args:
        param_estimator (callable): Parameter initializer with signature
            ``param_estimator(data_i) -> params_i`` for a single sample.
        model (callable): Model function used only for fallback default parameter
            inference when estimation fails.
        data (dict[str, np.ndarray]): Data dict where all arrays have shape
            ``(n_samples, ..., n_trials)``.
        timeout_s (float | None): Per-sample timeout in seconds. If ``None`` or
            non-positive, no explicit per-sample timeout is applied.
        deadline_s (float | None): Absolute wall-clock deadline expressed in
            ``time.time()`` seconds. If exceeded, raises ``ObjectiveTimeout``.

    Returns:
        pytree | None: Batched parameter pytree with leading sample axis for each
            leaf, or ``None`` if no fallback parameters are available.
    """
    defaults = compute_default_params(model)
    fallback = defaults
    if defaults is None:
        logging.info(
            "Default parameters unavailable; will fall back to the first successful estimate."
        )

    timeout_s_val = float(timeout_s) if timeout_s is not None and timeout_s > 0 else None
    n_samples = utils.data_n_samples(data)
    params_list = []

    def _remaining_time(sample_idx: int) -> float | None:
        if deadline_s is None:
            return None
        remaining_s = float(deadline_s) - time.time()
        if remaining_s <= 0:
            raise ObjectiveTimeout(
                f"objective timed out during parameter initialization at sample {sample_idx}"
            )
        return remaining_s

    def _effective_timeout(sample_idx: int) -> float | None:
        remaining_s = _remaining_time(sample_idx)
        if timeout_s_val is None:
            return remaining_s
        if remaining_s is None:
            return timeout_s_val
        return min(timeout_s_val, remaining_s)

    def _estimate_once(data_i, timeout_limit: float | None):
        if timeout_limit is None:
            return param_estimator(data_i)
        try:
            wrapped_estimator = timeout_decorator.timeout(
                timeout_limit, use_signals=True
            )(param_estimator)
        except Exception as exc:
            logging.info(
                "Signal timeout wrapper unavailable (%s); using process timeout backend.",
                exc,
            )
            return _run_param_estimator_with_timeout(param_estimator, data_i, timeout_limit)
        return wrapped_estimator(data_i)

    def _normalize_params(params_i):
        return jax.tree_util.tree_map(
            lambda value: (
                np.asarray(value)[0]
                if np.asarray(value).ndim >= 1 and np.asarray(value).shape[0] == 1
                else value
            ),
            params_i,
        )

    for sample_idx in range(n_samples):
        data_i = utils.data_as_numpy(utils.get_data_sample(data, sample_idx))
        timeout_limit = _effective_timeout(sample_idx)

        try:
            params_i = _estimate_once(data_i, timeout_limit)
            if params_i is None:
                raise ValueError("param_estimator returned None")
            params_i = _normalize_params(params_i)
        except ObjectiveTimeout:
            raise
        except (TimeoutError, timeout_decorator.TimeoutError):
            timeout_desc = (
                f"{timeout_limit:.2f}s" if timeout_limit is not None else "unknown timeout"
            )
            logging.info(
                "param_estimator timed out for sample %s after %s; using fallback.",
                sample_idx,
                timeout_desc,
            )
            params_i = fallback
        except ProcessTimeoutUnavailable as exc:
            logging.info(
                "param_estimator timeout backend unavailable for sample %s: %s",
                sample_idx,
                exc,
            )
            params_i = fallback
        except Exception as exc:
            logging.info(
                "param_estimator failed for sample %s: %s; using fallback.",
                sample_idx,
                exc,
            )
            params_i = fallback

        if params_i is None:
            logging.info(
                "Unable to initialize parameters for sample %s: no defaults or prior successful estimate available.",
                sample_idx,
            )
            return None

        fallback = params_i
        params_list.append(params_i)

    return utils.stack_params(params_list)

def compute_default_params(model):
    """
    Build default parameters from model metadata or signature.

    Preferred sources:
    1) ``model.default_params`` (callable or value)
    2) ``model.DEFAULT_PARAMS`` (value)
    3) Signature default for ``params`` when using ``model(X, params=...)``
    4) Legacy positional defaults for ``model(X, *params)``

    Returns:
        pytree | None: Default parameter pytree for a single sample, or ``None`` if
            defaults cannot be determined.
    """
    try:
        default_attr = getattr(model, "default_params", None)
        if default_attr is not None:
            return default_attr() if callable(default_attr) else default_attr
        default_attr = getattr(model, "DEFAULT_PARAMS", None)
        if default_attr is not None:
            return default_attr

        sig = inspect.signature(model)
        param_names = list(sig.parameters.keys())
        if len(param_names) >= 2 and param_names[1] == "params":
            params_param = sig.parameters["params"]
            if params_param.default is not inspect._empty:
                return params_param.default
            return None

        # Legacy path: positional parameters after X/theta.
        if len(param_names) <= 1:
            return None
        defaults = [
            sig.parameters[n].default if sig.parameters[n].default is not inspect._empty else 0.0
            for n in param_names[1:]
        ]
        default_arr = jnp.array(defaults, dtype=np.float32)
        return default_arr.reshape(1, -1)
    except Exception as e:
        logging.info(f"Error while generating default parameters: {e}")
        return None

def validate_model_output(
    output: jnp.ndarray,
    expected_n_trials: int,
    expected_n_targets: int = 1,
    allow_1d_for_single_target: bool = True,
) -> tuple[bool, str]:
    """
    Validate model output shape against expected trial/target dimensions.

    For scalar outputs (``expected_n_targets == 1``), 1D output is preferred and
    optional ``(1, n_trials)`` 2D output can be accepted.

    Args:
        output (jnp.ndarray): Model prediction for one sample.
        expected_n_trials (int): Expected number of trial points.
        expected_n_targets (int): Expected number of target channels.
        allow_1d_for_single_target (bool): Whether ``(n_trials,)`` and
            ``(1, n_trials)`` are both valid when ``expected_n_targets == 1``.

    Returns:
        tuple[bool, str]: ``(is_valid, error_message)`` where ``error_message``
            is empty when validation succeeds.
    """
    if expected_n_targets == 1:
        # Scalar output: prefer 1D array of shape (n_trials,)
        if output.ndim == 1:
            if output.shape[0] != expected_n_trials:
                return False, f"Model output n_trials={output.shape[0]} does not match expected {expected_n_trials}"
            return True, ""
        elif output.ndim == 2 and allow_1d_for_single_target:
            # Also accept 2D (1, n_trials) for single target
            if output.shape[0] != 1:
                return False, f"For n_targets=1, 2D output should have shape (1, n_trials), got {output.shape}"
            if output.shape[1] != expected_n_trials:
                return False, f"Model output n_trials={output.shape[1]} does not match expected {expected_n_trials}"
            return True, ""
        else:
            return False, f"Scalar model output should be 1D (n_trials,), got {output.ndim}D with shape {output.shape}"
    else:
        # Vectorized output: expect 2D array of shape (n_targets, n_trials)
        if output.ndim != 2:
            return False, f"Vectorized model output should be 2D (n_targets, n_trials), got {output.ndim}D with shape {output.shape}"
        if output.shape[0] != expected_n_targets:
            return False, f"Model output n_targets={output.shape[0]} does not match expected {expected_n_targets}"
        if output.shape[1] != expected_n_trials:
            return False, f"Model output n_trials={output.shape[1]} does not match expected {expected_n_trials}"
        return True, ""

def validate_model_execution(
    model,
    data: dict,
    initial_params,
    n_samples: int,
    n_validation_samples: int = 10,
) -> tuple[bool, str]:
    """
    Smoke-test model execution with JAX tracing.

    A random subset of samples is evaluated to ensure the model:
    1) executes without runtime errors,
    2) is JIT/trace compatible,
    3) returns finite outputs.

    Args:
        model (callable): Candidate model function ``model(data_i, params) -> output``.
        data (dict[str, np.ndarray]): Data dict with sample axis at dim 0.
        initial_params (pytree): Batched parameter pytree (leading sample axis).
        n_samples (int): Number of samples in the data.
        n_validation_samples (int): Maximum number of random samples to test.

    Returns:
        tuple[bool, str]: ``(is_valid, error_message)`` where ``error_message``
            captures the first failure reason.
    """
    try:
        model_jit = jax.jit(model)

        for sample_idx in np.random.choice(n_samples, size=min(n_validation_samples, n_samples), replace=False):
            data_i = utils.get_data_sample(data, sample_idx)
            data_i_jax = utils.data_as_jax(data_i)
            params_i = utils.slice_params(initial_params, sample_idx)
            output = model_jit(data_i_jax, params_i)

            # Basic validation: output should be finite
            if not jnp.all(jnp.isfinite(output)):
                return False, f"Model output contains non-finite values at sample {sample_idx}"

            # Validate with abstract tracer values
            jax.eval_shape(model_jit, data_i_jax, params_i)

        return True, ""
    except Exception as e:
        return False, f"Model failed to run or is incompatible with JAX tracing: {e}"

def objective(model, param_estimator, data,
              loss_fn=None, param_penalty_weight=0.1, penalty_denominator=1,
              fit_params=True,
              FAILED_PROGRAM_COST=jnp.inf, max_iter=1_000, learning_rate=3e-3,
              use_param_estimator=True, trial_batch_size=None,
              timeout_s: float | None = 5.0,
              objective_timeout_s: float | None = None) -> tuple[float, jnp.ndarray, float, jnp.ndarray]:
    """
    Evaluate a model by fitting parameters on train data and scoring on test data.

    Args:
        model (function): Model function for a single sample.
                          Signature: model(data_i, params) -> output
                          where data_i is a dict of arrays (no sample axis).
        param_estimator (function): Function to estimate initial parameters.
                          Signature: param_estimator(data_i) -> params
                          where data_i is a dict for one sample.
        data: Trial-split data. Expected length-2 container:
           - data[0]: train-trial data dict
           - data[1]: test-trial data dict
           Each element is a dict[str, np.ndarray] with shape (n_samples, ..., n_trials).
        loss_fn (function): Per-sample loss function.
                          Signature: loss_fn(model_output, data_i) -> scalar/array.
                          Required; no default.
        param_penalty_weight (float): Legacy complexity-penalty knob.
            Reported/saved losses are always raw data-fit losses (no penalty term).
        fit_params (bool): Whether to fit the parameters of the model. Default is True.
        FAILED_PROGRAM_COST (float): Cost assigned to failed models. Default is np.inf.
        max_iter (int): Maximum number of iterations for optimization. Default is 1_000.
        use_param_estimator (bool): Whether to use the parameter estimator to compute initial parameters. Default is True.
        trial_batch_size (int | None): Ignored. Trial batching is disabled.
        timeout_s (float | None): Per-sample timeout (seconds) for
            ``param_estimator`` calls during initialization.
            Set ``None`` or ``<=0`` to disable per-sample estimator timeout.
        objective_timeout_s (float | None): Hard wall-clock timeout (seconds) for
            the entire objective call (initialization + optimization + evaluation).
            Set ``None`` or ``<=0`` to disable.

    Returns:
        tuple[float, pytree, float, pytree]:
            (initial_loss, initial_params, final_loss, optimized_params)
    """
    t_start = time.time()
    deadline_s = (
        t_start + float(objective_timeout_s)
        if (objective_timeout_s is not None and objective_timeout_s > 0)
        else None
    )
    initial_params = None
    if loss_fn is None:
        raise ValueError("objective requires a loss_fn; none was provided.")

    def _check_timeout(stage: str):
        if deadline_s is not None and time.time() > deadline_s:
            raise ObjectiveTimeout(
                f"objective timed out after {float(objective_timeout_s):.2f}s during {stage}"
            )

    if not (isinstance(data, (list, tuple, np.ndarray)) and len(data) == 2):
        raise ValueError("objective expects data as length-2 container: [data_train, data_test].")

    data_train = utils.data_as_jax(data[0])
    data_test = utils.data_as_jax(data[1])

    n_samples = utils.data_n_samples(data_train)

    try:
        _check_timeout("parameter initialization")
        # Compute initial parameters
        # param_estimator receives y as (n_targets, n_trials) for each sample
        if use_param_estimator:
            initial_params = compute_initial_params(
                param_estimator,
                model,
                utils.data_as_numpy(data[0]),
                timeout_s=timeout_s,
                deadline_s=deadline_s,
            )
        else:
            initial_params = compute_default_params(model)
        _check_timeout("parameter initialization")
    except ObjectiveTimeout as e:
        logging.info(str(e))
        params_out = initial_params if initial_params is not None else {}
        return FAILED_PROGRAM_COST, params_out, FAILED_PROGRAM_COST, params_out
    except Exception as e:
        logging.info(f"Error during parameter initialization: {e}")
        empty_params = {}
        return FAILED_PROGRAM_COST, empty_params, FAILED_PROGRAM_COST, empty_params

    if initial_params is None:
        logging.info("Error: initial_params unavailable.")
        print("Program failed: initial_params unavailable.")
        empty_params = {}
        return FAILED_PROGRAM_COST, empty_params, FAILED_PROGRAM_COST, empty_params

    initial_params = utils.broadcast_params(initial_params, n_samples)

    n_params_raw = utils.params_numel_per_sample(initial_params, n_samples=n_samples)
    n_params = n_params_raw / max(1, penalty_denominator)
    # Memory guard: reject candidates with very large per-sample parameter payloads.
    try:
        params_single = utils.slice_params(initial_params, 0)
        param_bytes = int(
            sum(np.asarray(leaf).nbytes for leaf in jax.tree_util.tree_leaves(params_single))
        )
        max_param_bytes = 64 * 1024 * 1024  # 64 MiB per sample
        if param_bytes > max_param_bytes:
            logging.info(
                "Error: parameter payload too large (%d bytes > %d bytes).",
                param_bytes,
                max_param_bytes,
            )
            print(
                "Program failed: parameter payload exceeds memory guard "
                f"({param_bytes} bytes)."
            )
            return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params
    except Exception:
        # Keep objective robust even if size introspection fails.
        pass

    # Fail immediately if fit_params is True and non-numeric params
    if not utils.params_all_finite(initial_params):
        logging.info("Error: Parameters contain non-numeric or non-finite values.")
        print("Program failed: parameters contain non-numeric or non-finite values.")
        return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params
    if fit_params and not utils.params_all_inexact(initial_params):
        logging.info(
            "Error: Parameters contain int/bool leaves; rejecting before GD.\n%s",
            utils.params_tree_summary(initial_params, n_samples=n_samples),
        )
        print("Program failed: parameters must be floating-point for GD (found int/bool dtype).")
        return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params
    _check_timeout("model validation")
    
    # Validate model execution and output shape
    is_valid, error_msg = validate_model_execution(
        model, data_train, initial_params, n_samples,
        n_validation_samples=10
    )
    if not is_valid:
        logging.info(f"Model validation failed: {error_msg}")
        print(f"Program failed: model validation failed ({error_msg}).")
        return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params
    try:
        _check_timeout("post-validation")
    except ObjectiveTimeout as e:
        logging.info(str(e))
        return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params

    flat_init, unflatten = ravel_pytree(initial_params)

    # Per-sample loss function: model receives per-sample data dict, loss_fn receives model output + data
    def loss_single_sample(params, data_i):
        model_output = model(data_i, params)
        sample_loss = jnp.asarray(loss_fn(model_output, data_i))
        if sample_loss.ndim == 0:
            return sample_loss
        return jnp.mean(sample_loss)

    # Vectorize over samples. Dict is a native JAX pytree, so in_axes=0
    # maps over axis 0 of every leaf array.
    loss_total = jax.vmap(loss_single_sample, in_axes=(0, 0), out_axes=0)

    n_train_trials = utils.data_n_trials(data_train)
    effective_trial_batch_size = n_train_trials if trial_batch_size is None else int(trial_batch_size)

    @jax.jit
    def loss_single_batch(params_tree, data_batch):
        """Compute sum of losses for one batch (JIT-compiled)."""
        batch_losses = loss_total(params_tree, data_batch)  # (n_samples,)
        return jnp.sum(batch_losses)

    loss_and_grad_single_batch = jax.jit(jax.value_and_grad(loss_single_batch))

    @jax.jit
    def eval_single_batch(params_tree, data_batch):
        """Compute nansum of losses for one batch (JIT-compiled, no grad)."""
        batch_losses = loss_total(params_tree, data_batch)
        return jnp.nansum(batch_losses)

    def loss_and_grad_batched(flat_params):
        """Compute loss and gradient by accumulating over trial batches."""
        params_tree = unflatten(flat_params)
        total_loss = 0.0
        total_grad = jnp.zeros_like(flat_params)

        for start_idx in range(0, n_train_trials, effective_trial_batch_size):
            _check_timeout("loss/grad computation")
            end_idx = min(start_idx + effective_trial_batch_size, n_train_trials)
            batch_weight = (end_idx - start_idx) / n_train_trials
            data_batch = utils.slice_data_trials(data_train, slice(start_idx, end_idx))

            batch_loss, batch_grad_tree = loss_and_grad_single_batch(params_tree, data_batch)
            batch_grad_flat, _ = ravel_pytree(batch_grad_tree)

            total_loss += batch_loss * batch_weight
            total_grad += batch_grad_flat * batch_weight

        return total_loss / n_samples, total_grad / n_samples

    def _optimize_params(flat_params):
        if not fit_params:
            return initial_params, False

        learning_rate_local = float(learning_rate)
        opt = optax.adam(learning_rate_local, b1=0.9, b2=0.999, eps=1e-8)
        opt_state = opt.init(flat_params)

        def train_step(params, opt_state):
            loss, grad = loss_and_grad_batched(params)
            updates, new_opt_state = opt.update(grad, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        print_every = 50
        params = flat_params
        initial_loss, _ = loss_and_grad_batched(params)

        CATASTROPHIC_LOSS_THRESHOLD = 1e6
        if initial_loss > CATASTROPHIC_LOSS_THRESHOLD:
            print(f"Initial loss {initial_loss:.2e} exceeds threshold. Skipping optimization.")
            logging.info(f"Skipping optimization: initial loss {initial_loss:.2e} > {CATASTROPHIC_LOSS_THRESHOLD:.0e}")
            return initial_params, True

        best_loss, best_params = initial_loss.copy(), params.copy()
        for step in range(1, max_iter + 1):
            _check_timeout("optimization")
            params, opt_state, loss_val = train_step(params, opt_state)
            _check_timeout("optimization")
            if jnp.isnan(loss_val) or jnp.isinf(loss_val) or jnp.any(jnp.isnan(params)) or jnp.any(jnp.isinf(params)):
                logging.info(f"Loss is NaN or Inf at step {step}. Stopping optimization.")
                print(f"Final loss: {loss_val:.4f} at step {step}")
                break
            if loss_val > CATASTROPHIC_LOSS_THRESHOLD:
                logging.info(f"Loss exploded to {loss_val:.2e} at step {step}. Stopping optimization.")
                print(f"Loss exploded to {loss_val:.2e}. Stopping optimization.")
                return initial_params, True
            if loss_val < best_loss:
                best_loss = loss_val.copy()
                best_params = params.copy()
            if step % print_every == 0:
                print(f"step {step:4d}  loss {loss_val:.4f}")
        params = unflatten(best_params)
        print(f"params optimized. Loss: {best_loss:.4f}")
        return params, False

    # Compute final loss on test set
    def eval_loss_batched(params_tree, data_eval):
        """Compute loss by iterating over trial batches."""
        n_eval_trials = utils.data_n_trials(data_eval)
        weighted_sum = 0.0
        for start_idx in range(0, n_eval_trials, effective_trial_batch_size):
            _check_timeout("final loss evaluation")
            end_idx = min(start_idx + effective_trial_batch_size, n_eval_trials)
            batch_size = end_idx - start_idx
            data_batch = utils.slice_data_trials(data_eval, slice(start_idx, end_idx))
            weighted_sum += eval_single_batch(params_tree, data_batch) * (batch_size / n_eval_trials)
        return weighted_sum / n_samples

    try:
        params, failed_opt = _optimize_params(flat_init)
        if failed_opt:
            return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params

        def _eval_loss(params_tree, data_eval, label: str):
            _check_timeout(f"{label} loss evaluation")
            loss_val = eval_loss_batched(params_tree, data_eval)
            # loss_val = loss_val + param_penalty_weight * n_params
            n_nans = jnp.sum(jnp.isnan(loss_val))
            if n_nans > 0:
                print(f"Warning: {label} loss contains {n_nans} NaNs.")
            return jnp.nan_to_num(loss_val, nan=FAILED_PROGRAM_COST, posinf=FAILED_PROGRAM_COST, neginf=FAILED_PROGRAM_COST)

        initial_loss = _eval_loss(initial_params, data_test, "initial")
        final_loss = _eval_loss(params, data_test, "final")
    except ObjectiveTimeout as e:
        logging.info(str(e))
        params_out = initial_params if initial_params is not None else {}
        return FAILED_PROGRAM_COST, params_out, FAILED_PROGRAM_COST, params_out

    t_end = time.time()
    print(f"Time taken for optimization: {t_end - t_start:.4f} seconds")
    return float(initial_loss), initial_params, float(final_loss), params

def objective_simple(
    model,
    param_estimator,
    data,
    loss_fn,
):
    """
    Small reference implementation of objective for testing and debugging, without JIT or mini-batching.

    Args:
        model: function with signature model(data_i, params) -> output for a single sample.
        param_estimator: function with signature param_estimator(data_i) -> params for single sample.
        data: length-2 container of train/test data dicts.
        loss_fn: function with signature loss_fn(model_output, data_i) -> scalar or array.
    """
    print("Running objective_simple (no JIT, no mini-batching)...")
    data_train = data[0]
    data_test = data[1]

    n_samples = utils.data_n_samples(data_train)
    params_list = []
    for i in range(n_samples):
        data_i = utils.get_data_sample(data_train, i)
        params_list.append(param_estimator(data_i))

    losses = []
    for i in range(n_samples):
        params_i = params_list[i]
        data_i = utils.get_data_sample(data_test, i)
        model_output = np.asarray(model(data_i, params_i))
        losses.append(float(np.asarray(loss_fn(model_output, data_i)).mean()))
    print(f"Losses for each sample: {losses}")

    params = utils.stack_params(params_list)
    final_loss = float(np.mean(losses))
    print(f"Final loss (objective_simple): {final_loss:.4f}")
    return final_loss, params, final_loss, params

def _call_objective(use_simple_objective: bool, **kwargs):
    if use_simple_objective:
        return objective_simple(
            model=kwargs["model"],
            param_estimator=kwargs["param_estimator"],
            data=kwargs["data"],
            loss_fn=kwargs["loss_fn"],
        )
    return objective(**kwargs)
