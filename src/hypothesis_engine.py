import inspect
import json
import re
import os
import io
import tokenize
import gc
import multiprocessing as mp
import logging
import webbrowser
import asyncio
import textwrap
import numpy as np
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from dataclasses import dataclass, field
import timeout_decorator
import optax
import pandas as pd
from pathlib import Path
from typing import Any, Callable
from . import utils, llm_helper
from . import genetic_helpers_v2 as genetic_helpers  # Using v2 with compatibility API
from .timeout_worker import run_estimator_from_source
from .evolution_diagnostics import plot_train_vs_test_loss as plot_train_vs_test_loss_shared
from .monitoring import create_family_tree, create_dynamic_progress_update
from tqdm import tqdm
from google import genai
from dotenv import load_dotenv
import warnings
import time
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*"
)

class ObjectiveTimeout(Exception):
    """Raised when objective exceeds wall-clock timeout."""


class ProcessTimeoutUnavailable(RuntimeError):
    """Raised when process-based timeout backend cannot be used."""


@dataclass(slots=True)
class ModelGenerationResult:
    numpy_code: str | None
    prompt: str | None
    llm_response: str | None
    jax_code: str | None = None
    jax_callable: Callable | None = None
    jax_prompt: str | None = None
    jax_raw_response: str | None = None


@dataclass(slots=True)
class ParamEstimatorGenerationResult:
    code: str | None
    callable_obj: Callable | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateGenerationResult:
    model: ModelGenerationResult
    param_estimator: ParamEstimatorGenerationResult


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(relativeCreated)dms | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

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


def _normalize_generated_model_code(
    code_string: str | None,
    model_name: str,
    expected_version: int | None = None,
) -> str | None:
    """
    Accept common generated model function names and normalize to ``def {model_name}(...):``.

    Accepted top-level names:
    - ``{model_name}_v{expected_version}``
    - ``{model_name}_v<number>``
    - ``{model_name}``
    - ``model_v<number>``
    - ``model``
    """
    if code_string is None:
        return None
    code = textwrap.dedent(str(code_string)).strip()
    if not code:
        return code

    patterns = []
    if expected_version is not None:
        patterns.append(rf"^\s*def\s+{re.escape(model_name)}_v{int(expected_version)}\s*\(")
    patterns.extend(
        [
            rf"^\s*def\s+{re.escape(model_name)}_v\d+\s*\(",
            rf"^\s*def\s+{re.escape(model_name)}\s*\(",
            r"^\s*def\s+model_v\d+\s*\(",
            r"^\s*def\s+model\s*\(",
        ]
    )

    for pat in patterns:
        if re.search(pat, code, flags=re.MULTILINE):
            return re.sub(pat, f"def {model_name}(", code, count=1, flags=re.MULTILINE)
    return code


def _strip_strings_and_comments(code_string: str) -> str:
    """
    Remove Python string literals and comments for safer token scanning.
    """
    if not isinstance(code_string, str) or not code_string:
        return ""
    try:
        tokens = []
        reader = io.StringIO(code_string).readline
        for tok_type, tok_str, *_ in tokenize.generate_tokens(reader):
            if tok_type in (tokenize.STRING, tokenize.COMMENT):
                continue
            tokens.append(tok_str)
        return " ".join(tokens)
    except Exception:
        # Best effort: if tokenization fails, fall back to raw text.
        return code_string


def _find_banned_token(code_string: str, swear_words) -> str | None:
    """
    Return the first banned token found in executable code, or None.
    """
    if not swear_words:
        return None
    scan_text = _strip_strings_and_comments(code_string)
    for raw_word in swear_words:
        if not isinstance(raw_word, str) or not raw_word.strip():
            continue
        word = raw_word.strip()
        if "." in word and re.fullmatch(r"[A-Za-z0-9_.]+", word):
            # Match dotted paths with optional whitespace around dots.
            parts = [re.escape(p) for p in word.split(".") if p]
            if not parts:
                continue
            pattern = r"\b" + r"\s*\.\s*".join(parts) + r"\b"
        elif re.fullmatch(r"[A-Za-z0-9_]+", word):
            pattern = rf"\b{re.escape(word)}\b"
        else:
            pattern = re.escape(word)
        if re.search(pattern, scan_text, flags=re.IGNORECASE):
            return raw_word
    return None


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

        # Compute per-sample losses by vmapping the loss function
        def _sample_loss(params_i, data_i):
            model_output = model(data_i, params_i)
            raw = jnp.asarray(loss_func(model_output, data_i))
            return jnp.mean(raw) if raw.ndim > 0 else raw

        losses = jax.vmap(_sample_loss, in_axes=(0, 0))(params_tree, data_jax)

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


async def generate_new_model(current_island, llm_name, client,
                            data, x_eval, prompt_manager,
                            mode='explore', k_max=2, temp=1,
                            thinking_budget=1, img_dir=None,
                            plot_model_fits=None,
                            island_chat_manager=None, island_id: int = None,
                            batch_id: int = 0,
                            loss_fn=None,
                            loss_data=None,
                            complexity_penalty: float = 0.0,
                            use_large_model: bool = True):
    """
    Propose a new model program by querying the LLM from island context.

    Args:
        current_island (pd.DataFrame): Program population for one island.
        llm_name (str): Model name for legacy stateless LLM calls.
        client: LLM client handle used by helper wrappers.
        data (dict[str, np.ndarray]): Data dict forwarded to plotting.
        x_eval: Evaluation grid used for consistent plotting across projects.
        prompt_manager: PromptManager instance used to build prompts.
        mode (str): Search mode (typically ``"explore"`` or ``"exploit"``).
        k_max (int): Number of parent programs to include in prompt context.
        temp (float): Sampling temperature for LLM decoding.
        thinking_budget (float): Relative budget forwarded to LLM helper.
        img_dir (str | None): Base output path used for refinement-round
            feedback images.
        plot_model_fits (callable | None): Optional plotting callback.
        island_chat_manager (IslandChatManager | None): Optional chat-session manager.
        island_id (int | None): Island id for chat mode.
        batch_id (int): Batch id for chat mode.
        loss_fn (callable | None): Loss function used for per-sample diagnostics
            when building `programs_list` for plot feedback.
        loss_data (dict | None): Optional data dict to use specifically for
            diagnostics loss computation. Defaults to ``data``.
        complexity_penalty (float): Complexity-penalty multiplier used when
            computing diagnostics losses.
        use_large_model (bool): Whether chat mode should use large model path.

    Returns:
        tuple[str | None, str | None, str | None, tuple]:
            ``(code_string, prompt, llm_output, parent_ids)``.
            ``code_string`` is ``None`` when no valid code block is produced.
    """
    k = min(k_max, len(current_island))
    random_programs = current_island.sample(k, replace=False).reset_index(drop=True)
    random_programs = random_programs.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
    # save parent1_id and parent2_id. These are strings of the form "(iteration_number)_(birth_island)_(batch_index)"
    parent1_id = (random_programs['iteration_number'][0], 
                  random_programs['birth_island'][0], 
                  random_programs['batch_index'][0])
    parent2_id = (random_programs['iteration_number'][1],
                  random_programs['birth_island'][1], 
                  random_programs['batch_index'][1])
    use_image = (
        img_dir is not None
        and plot_model_fits is not None
    )
    use_chat_mode = island_chat_manager is not None and island_id is not None
    model_name = prompt_manager.get_model_name()
    
    # Use appropriate prompt function based on mode
    if use_chat_mode:
        program_prompt = prompt_manager.get_program_prompt(random_programs, mode=mode, use_image=use_image)
    else:
        program_prompt = prompt_manager.get_program_prompt_legacy(random_programs, mode=mode, use_image=use_image)

    if use_image:
        try:
            data_for_loss = data if loss_data is None else loss_data
            programs_list = _programs_df_to_programs_list(
                random_programs,
                loss_func=loss_fn,
                data=data_for_loss,
                complexity_penalty=complexity_penalty,
            )
            plot_model_fits(
                data=data,
                programs_list=programs_list,
                X_eval=x_eval,
                save_path=img_dir,
                labels=[f"v_{i+1}" for i in range(len(random_programs))],
            )
            
            img_path = Path(img_dir)
            with img_path.open("rb") as f:
                img_bytes = f.read()
        except Exception as e:
            logging.info(f"Error generating image for neuron model prompt: {e}")
            img_bytes = None
            # if we can't generate an image, we will just use the text prompt without image
            use_image = False
    else:
        img_bytes = None
    
    # Use chat-based or legacy LLM call
    if island_chat_manager is not None and island_id is not None:
        llm_output = await island_chat_manager.ask_island(
            island_id, program_prompt,
            batch_id=batch_id,
            mode=mode, 
            use_large_model=use_large_model,
            png_img=img_bytes
        )
    else:
        # Legacy: independent query
        llm_output = await llm_helper.call_llm_async(program_prompt, model_name=llm_name, client=client, temperature=temp, 
                                                thinking_budget=thinking_budget, img_bytes=img_bytes)
    
    code_string = utils.extract_code_block(llm_output)
    if code_string is None:
        return None, program_prompt, llm_output, (parent1_id, parent2_id)
    code_string = _normalize_generated_model_code(
        code_string,
        model_name=model_name,
        expected_version=k + 1,
    )

    return code_string, program_prompt, llm_output, (parent1_id, parent2_id)


async def generate_new_parameter_estimator(current_island,
                                           model_code_string: str,
                                           model_fn,
                                           llm_name, client,
                                           data,
                                           prompt_manager,
                                           mode='explore', k_max=1, temp=1,
                                           param_estimator_max_lines=100,
                                           swear_words=None,
                                           refine_rounds: int = 0,
                                           param_penalty_weight: float = 0.1,
                                           random_seed: int | None = None,
                                           island_chat_manager=None, island_id: int = None,
                                           batch_id: int = 0,
                                           iteration: int | None = None,
                                           use_simple_objective: bool = False,
                                           loss_fn=None,
                                           plot_model_fits=None,
                                           x_eval=None,
                                           image_refinement_dir=None,
                                           param_estimator_timeout_s: float | None = 5.0,
                                           objective_timeout_s: float | None = None,):
    """
    Generate and optionally refine a parameter-estimator function via LLM.

    This function prompts the LLM for a ``parameter_estimator`` implementation,
    validates/parses returned code, and can run iterative refinement rounds where
    each round is scored with ``objective(..., fit_params=False)``.

    Args:
        current_island (pd.DataFrame): Program population for one island.
        model_code_string (str): NumPy model source used in estimator prompt context.
        model_fn (callable): Executable model function used for scoring refinements.
        llm_name (str): Model name for legacy stateless LLM calls.
        client: LLM client handle used by helper wrappers.
        data: Length-2 container ``[data_train_trials, data_test_trials]`` of
            data dicts passed to ``objective`` during scoring.
        prompt_manager: PromptManager instance used to build prompts.
        mode (str): Search mode (typically ``"explore"`` or ``"exploit"``).
        k_max (int): Number of parent programs to include in prompt context.
        temp (float): Sampling temperature for LLM decoding.
        param_estimator_max_lines (int): Soft budget for generated estimator length.
        swear_words (list[str] | None): Token blacklist for generated code.
        refine_rounds (int): Number of iterative refinement rounds.
        param_penalty_weight (float): Parameter-count penalty used during scoring.
        random_seed (int | None): Base RNG seed for parent-program sampling.
        island_chat_manager (IslandChatManager | None): Optional chat-session manager.
        island_id (int | None): Island id for chat mode.
        batch_id (int): Batch id for chat mode.
        loss_fn (callable | None): Loss function forwarded to ``objective``.
        use_simple_objective (bool): Use the minimal objective implementation for scoring.
        param_estimator_timeout_s (float | None): Per-sample timeout (seconds)
            for estimator evaluation during refinement scoring.
        objective_timeout_s (float | None): Hard timeout (seconds) for each
            refinement objective call.

    Returns:
        tuple[str | None, callable | None, dict]: Best estimator code string, parsed
            callable, and metadata dict with prompt/response info.
            Returns ``(None, None, pe_metadata)`` when generation/validation fails.
    """
    pe_metadata = {
        "initial_prompt": None,
        "initial_response": None,
        "refinement_prompts": [],
        "refinement_responses": [],
        "refinement_codes": [],
        "status": None,
    }
    if model_code_string is None:
        pe_metadata["status"] = "missing_model_code"
        return None, None, pe_metadata
    if not (isinstance(data, (list, tuple, np.ndarray)) and len(data) == 2):
        logging.info("Parameter estimator generation expects data split as [train_trials, test_trials].")
        return None, None, pe_metadata

    k = min(k_max, len(current_island))
    sample_seed = None
    if random_seed is not None:
        island_offset = 0 if island_id is None else int(island_id)
        sample_seed = int(random_seed) + 10_000 * island_offset + int(batch_id)
    random_programs = current_island.sample(k, replace=False, random_state=sample_seed).reset_index(drop=True)
    # sort from worst to best (loss descending)
    random_programs = random_programs.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
    # Chat mode is not supported for parameter estimator generation/refinement.
    prompt = prompt_manager.get_parameter_estimator_prompt(
        random_programs,
        model_code_string=model_code_string,
        max_lines=param_estimator_max_lines,
    )
    if swear_words:
        banned_list = "\n".join(f"- {word}" for word in swear_words)
        prompt = (
            f"{prompt}\n\n"
            "**Banned tokens (do not use in code):**\n"
            f"{banned_list}\n"
        )
    
    llm_output = await llm_helper.call_llm_async(
        prompt,
        model_name=llm_name,
        client=client,
        temperature=temp,
        thinking_budget=0.25,
        img_bytes=None,
    )
    pe_metadata["initial_prompt"] = prompt
    pe_metadata["initial_response"] = llm_output
    # extract the code block from the LLM output
    code_string = utils.extract_code_block(llm_output)
    if code_string is None:
        pe_metadata["status"] = "no_code_block"
        return None, None, pe_metadata
    swear_word = _find_banned_token(code_string, swear_words)
    if swear_word is not None:
        pe_metadata["status"] = f"banned_token:{swear_word}"
        return None, None, pe_metadata
    code_string = re.sub(r"def\s+parameter_estimator_v\d+\s*\(", "def parameter_estimator(", code_string)
    code_string = re.sub(r"def\s+parameter_estimator_prev\s*\(", "def parameter_estimator(", code_string)
    func = utils.str_to_func(code_string, 'parameter_estimator')

    if func is None:
        pe_metadata["status"] = "parse_failed"
        return None, None, pe_metadata

    if refine_rounds <= 0 or model_fn is None:
        return code_string, func, pe_metadata

    iter_label = "?" if iteration is None else str(iteration)
    print(
        f"Param-est refinement start (iter={iter_label}, island={island_id}, "
        f"batch={batch_id}, rounds={refine_rounds}).",
        flush=True,
    )

    best_code = code_string
    best_func = func
    best_loss = float(jnp.inf)

    current_code = code_string
    current_func = func
    current_loss, current_params, _, _ = _call_objective(
        use_simple_objective,
        model=model_fn,
        param_estimator=current_func,
        data=data,
        loss_fn=loss_fn,
        fit_params=False,  # Don't fit parameters during refinement evaluation
        param_penalty_weight=param_penalty_weight,
        timeout_s=param_estimator_timeout_s,
        objective_timeout_s=objective_timeout_s,
    )

    if current_loss < best_loss:
        best_loss = current_loss
        best_code = current_code
        best_func = current_func

    for r in range(refine_rounds):
        print(
            f"Param-est refinement round {r+1}/{refine_rounds} "
            f"(iter={iter_label}, island={island_id}, batch={batch_id}).",
            flush=True,
        )
        if plot_model_fits is None or x_eval is None or image_refinement_dir is None:
            raise ValueError(
                "Parameter estimator refinement requires image feedback. "
                "Missing plot_model_fits/x_eval/image_refinement_dir."
            )
        img_bytes = None
        try:
            img_path = os.path.join(
                image_refinement_dir,
                f"param_est_refine_island_{island_id}_batch_{batch_id}_r{r+1}.png",
            )
            plot_model_fits(
                data=data[0],
                programs_list=[
                    {
                        "model": model_fn,
                        "params": current_params,
                        "losses": np.full(utils.data_n_samples(data[0]), float(current_loss)),
                    }
                ],
                X_eval=x_eval,
                save_path=img_path,
                labels=['PE'],
            )
            with open(img_path, "rb") as f:
                img_bytes = f.read()
        except Exception as e:
            logging.info(f"Error generating image for parameter estimator refinement: {e}")
            logging.info(f"Model code string was:\n{model_code_string}")
            raise RuntimeError(f"Param-estimator image generation failed: {e}") from e

        # Build refinement prompt using current estimator as the only parent
        refinement_df = pd.DataFrame({
            'train_loss': [current_loss],
            'program_code_string': [model_code_string],
            'parameter_estimator_code_string': [current_code],
        })

        refine_prompt = prompt_manager.get_parameter_estimator_refinement_prompt_legacy(
            refinement_df,
            model_code_string=model_code_string,
            max_lines=param_estimator_max_lines,
            refine_round=r + 1,
            refine_rounds=refine_rounds,
            current_loss=current_loss,
        )
        if swear_words:
            banned_list = "\n".join(f"- {word}" for word in swear_words)
            refine_prompt = (
                f"{refine_prompt}\n\n"
                "**Banned tokens (do not use in code):**\n"
                f"{banned_list}\n"
            )
        # Call LLM for refinement
        llm_output = await llm_helper.call_llm_async(
            refine_prompt,
            model_name=llm_name,
            client=client,
            temperature=temp,
            thinking_budget=0.25,
            img_bytes=img_bytes,
        )
        pe_metadata["refinement_prompts"].append(refine_prompt)
        pe_metadata["refinement_responses"].append(llm_output)

        new_code = utils.extract_code_block(llm_output)
        if new_code is None:
            pe_metadata["refinement_codes"].append(None)
            continue
        if _find_banned_token(new_code, swear_words) is not None:
            pe_metadata["refinement_codes"].append(None)
            continue

        new_code = re.sub(r"def\s+parameter_estimator_v\d+\s*\(", "def parameter_estimator(", new_code)
        new_code = re.sub(r"def\s+parameter_estimator_prev\s*\(", "def parameter_estimator(", new_code)
        pe_metadata["refinement_codes"].append(new_code)
        new_func = utils.str_to_func(new_code, 'parameter_estimator')
        if new_func is None:
            continue

        new_loss, _, _, _ = _call_objective(
            use_simple_objective,
            model=model_fn,
            param_estimator=new_func,
            data=data,
            loss_fn=loss_fn,
            fit_params=False,  # Don't fit parameters during refinement evaluation
            param_penalty_weight=param_penalty_weight,
            timeout_s=param_estimator_timeout_s,
            objective_timeout_s=objective_timeout_s,
        )

        print(
            f"Param-est refinement eval (iter={iter_label}, island={island_id}, "
            f"batch={batch_id}, round={r+1}): loss={new_loss:.6g}.",
            flush=True,
        )

        if new_loss < current_loss:
            current_code = new_code
            current_func = new_func
            current_loss = new_loss
            if new_loss < best_loss:
                best_loss = new_loss
                best_code = new_code
                best_func = new_func
        else:
            pass

    return best_code, best_func, pe_metadata


async def translate_to_jax(
    code_string: str,
    client,
    prompt_manager,
    llm_name='gemini-2.0-flash-lite',
    max_retries: int = 2,
    retry_delay_s: float = 2.0,
) -> tuple[str, callable, str | None, str | None]:
    """
    Translate a model code string to a JAX-compatible implementation via LLM.

    Args:
        code_string (str): Source code containing the NumPy model definition.
        client: The LLM client.
        prompt_manager: PromptManager used to build translation prompts.
        llm_name (str): LLM model name for translation.

    Returns:
        tuple[str | None, callable | None, str | None, str | None]:
            ``(jax_code_string, jax_callable, prompt, raw_response)``.
            Returns ``(None, None)`` when translation cannot be produced/parsed.
    """
    if code_string is None:
        return None, None, None, None
    
    prompt = prompt_manager.get_jax_translator_prompt(code_string)
    if prompt is None:
        return None, None, None, None
    
    # TODO rtilbury: Why is this necessary? 
    raw_response = None
    for attempt in range(max_retries + 1):
        raw_response = await llm_helper.call_llm_async(
            prompt,
            client=client,
            model_name=llm_name,
            temperature=0,
        )
        if isinstance(raw_response, str) and raw_response.strip():
            break
        if attempt < max_retries:
            sleep_s = float(retry_delay_s) * (2 ** attempt)
            logging.warning(
                "JAX translation attempt %d/%d failed for model %s; retrying in %.1fs.",
                attempt + 1,
                max_retries + 1,
                llm_name,
                sleep_s,
            )
            await asyncio.sleep(sleep_s)

    if not (isinstance(raw_response, str) and raw_response.strip()):
        logging.error(
            "JAX translation failed after %d attempts for model %s (empty/None response).",
            max_retries + 1,
            llm_name,
        )
        return None, None, prompt, raw_response

    jax_code_string = utils.extract_code_block(raw_response)
    if not (isinstance(jax_code_string, str) and jax_code_string.strip()):
        logging.error("JAX translation response did not contain an extractable code block.")
        return None, None, prompt, raw_response

    model_name = prompt_manager.get_model_name()
    func = utils.str_to_func(jax_code_string, model_name)
    if not callable(func):
        logging.error("Translated JAX code parsed but did not produce callable function %s.", model_name)
    return jax_code_string, func, prompt, raw_response


def _run_translation_check_on_eval(
    np_func,
    jax_func,
    param_estimator,
    data_train_trials,
    x_eval,
    max_samples: int = 3,
    max_eval_trials: int = 32,
):
    """
    Validate NumPy/JAX numerical agreement on a subset of evaluation points.

    Parameters are first estimated from observed train-trial data. The resulting
    parameter vectors are then used to compare NumPy vs JAX predictions on
    a small data subset.

    Args:
        np_func (callable): Original NumPy model.
        jax_func (callable): Translated JAX model.
        param_estimator (callable): Parameter estimator with signature
            ``param_estimator(data_i) -> params`` for a single sample.
        data_train_trials (dict[str, np.ndarray]): Train-trial data dict with
            sample axis at dim 0.
        x_eval: Evaluation grid (currently unused, kept for API compatibility).
        max_samples (int): Maximum number of samples to check.
        max_eval_trials (int): Maximum eval trials per sample to compare.

    Returns:
        None: Raises on mismatch; otherwise completes silently.
    """
    n_samples = utils.data_n_samples(data_train_trials)
    if n_samples <= 0:
        raise ValueError("No samples available for translation check.")

    n_trials = utils.data_n_trials(data_train_trials)
    n_eval_trials = min(5, int(max_eval_trials), n_trials)
    rng = np.random.default_rng(0)
    if n_eval_trials <= 0:
        raise ValueError("No trials available for translation check.")
    if n_eval_trials == n_trials:
        trial_idx = np.arange(n_trials)
    else:
        trial_idx = rng.choice(n_trials, size=n_eval_trials, replace=False)
    data_subset = utils.slice_data_trials(data_train_trials, trial_idx)

    n_check = min(max_samples, n_samples)
    sample_idx = np.linspace(0, n_samples - 1, num=n_check, dtype=int)
    data_subset = utils.slice_data_samples(data_subset, sample_idx)

    params_subset = compute_initial_params(
        param_estimator,
        np_func,
        data_subset,
    )
    if params_subset is None:
        raise ValueError("Failed to compute parameters for translation check.")

    utils.check_jax_translation(
        np_func=np_func,
        jax_func=jax_func,
        data=data_subset,
        params=params_subset,
        max_eval_trials=max_eval_trials,
    )


def _append_generation_record(filepath, record):
    """Append a single generation record as a JSON line."""
    with open(filepath, 'a') as f:
        f.write(json.dumps(record, default=str) + '\n')


def _drop_nonfinite_train_loss_rows(df: pd.DataFrame, context: str) -> tuple[pd.DataFrame, int]:
    """
    Remove rows whose train_loss is NaN/Inf/non-numeric.

    Returns:
        (clean_df, n_removed)
    """
    if df is None or len(df) == 0:
        return df, 0
    if 'train_loss' not in df.columns:
        return df, 0

    train_loss_num = pd.to_numeric(df['train_loss'], errors='coerce')
    finite_mask = np.isfinite(train_loss_num.to_numpy(dtype=float))
    n_removed = int((~finite_mask).sum())
    if n_removed > 0:
        logging.info(
            "%s: dropped %d programs with non-finite train_loss.",
            context,
            n_removed,
        )
        print(f"{context}: dropped {n_removed} programs with non-finite train_loss.", flush=True)
    clean_df = df.loc[finite_mask].reset_index(drop=True)
    return clean_df, n_removed


def _drop_nonfinite_train_loss_from_islands(islands: list[pd.DataFrame], context: str) -> list[pd.DataFrame]:
    """
    Apply non-finite train_loss filtering to every island.
    """
    cleaned = []
    total_removed = 0
    for island_idx, island_df in enumerate(islands):
        island_clean, removed = _drop_nonfinite_train_loss_rows(
            island_df,
            context=f"{context} (island={island_idx})",
        )
        cleaned.append(island_clean)
        total_removed += removed
    if total_removed > 0:
        logging.info("%s: total dropped non-finite-loss programs=%d", context, total_removed)
        print(f"{context}: total dropped non-finite-loss programs={total_removed}", flush=True)
    return cleaned

def _update_generation_log_records(filepath, updates_by_key):
    """Patch existing JSONL records in-place by candidate UID."""
    if not updates_by_key or not os.path.isfile(filepath):
        return

    normalized_updates = {
        (int(key[0]), int(key[1]), int(key[2])): value
        for key, value in updates_by_key.items()
    }

    with open(filepath, 'r') as f:
        lines = f.readlines()
    with open(filepath, 'w') as f:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec['iteration_number'], rec['birth_island'], rec['batch_index'])
            if key in normalized_updates:
                rec.update(normalized_updates[key])
            f.write(json.dumps(rec, default=str) + '\n')


def _apply_removal_reasons_to_log(filepath, removal_events):
    """Batch-update the JSONL log with removal_reason fields.

    Reads the log, adds ``removal_reason`` to any record whose UID matches
    a :class:`RemovalEvent`, then rewrites the file.  Records that already
    carry a ``removal_reason`` are left unchanged.

    Args:
        filepath: Path to program_generation_log.jsonl
        removal_events: List of ``RemovalEvent`` objects from dedup / prune.
    """
    if not removal_events or not os.path.isfile(filepath):
        return

    # Build lookup: (iteration, birth_island, batch_index) -> removal dict
    reason_lookup = {}
    for evt in removal_events:
        key = (int(evt.uid[0]), int(evt.uid[1]), int(evt.uid[2]))
        details = dict(evt.details or {})
        details.setdefault("iteration", evt.iteration)
        reason_lookup[key] = {
            "category": evt.category,
            "event_type": evt.event_type,
            "island_id": evt.island_id,
            "rule": evt.rule,
            "details": details,
        }

    # Non-atomic read/overwrite. If log corruption is observed, switch to
    # temp file + os.replace() for atomic writes.
    with open(filepath, 'r') as f:
        lines = f.readlines()
    with open(filepath, 'w') as f:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec['iteration_number'], rec['birth_island'], rec['batch_index'])
            if key in reason_lookup and 'removal_reason' not in rec:
                rec['removal_reason'] = reason_lookup[key]
            f.write(json.dumps(rec, default=str) + '\n')


def _update_generation_log_test_losses_and_mark_winner(filepath, islands):
    """Update JSONL records in-place with test_loss values and mark the winner.

    Args:
        filepath: Path to the JSONL generation log
        islands: List of island dataframes with test_loss values
    """
    if not os.path.isfile(filepath):
        return
    # Build lookup: (iteration, island, batch) -> test_loss
    test_loss_lookup = {}
    for island_idx, island_df in enumerate(islands):
        for _, row in island_df.iterrows():
            key = (int(row['iteration_number']), int(row['birth_island']), int(row['batch_index']))
            tl = row.get('test_loss')
            if tl is not None and not (isinstance(tl, float) and np.isinf(tl)):
                test_loss_lookup[key] = float(tl)

    # from test_loss_lookup, find the best test loss (i.e. smallest) and corresponding key 
    best_test_loss = float('inf')
    best_key = None
    for key, tl in test_loss_lookup.items():
        if tl < best_test_loss:
            best_test_loss = tl
            best_key = key
    logging.info(f"Best test loss found: {best_test_loss:.6g} for program {best_key}.")

    # compare best_key to winner_id if provided
    winner_id = best_key    

    # Read, update, rewrite
    with open(filepath, 'r') as f:
        lines = f.readlines()
    with open(filepath, 'w') as f:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec['iteration_number'], rec['birth_island'], rec['batch_index'])
            if key in test_loss_lookup:
                rec['test_loss'] = test_loss_lookup[key]
            # Mark the winner
            rec['is_winner'] = (winner_id is not None and key == winner_id)
            f.write(json.dumps(rec, default=str) + '\n')


async def hypothesis_engine(
        n_iterations=9, time_limit=60, k_max=2, n_islands=8, batch_size=6, 
        critical_population_size=12, min_wise_population_size=0, n_migrants=2, 
        fit_params=True, use_param_estimator=True, 
        param_penalty_weight=0.01, FAILED_PROGRAM_COST=np.inf, exploit_point=0.5,
        param_estimator_timeout_s: float | None = 5.0,
        objective_timeout_s: float | None = None,
        use_chat_mode=False,  # If True, use persistent chat sessions per island (expensive)
        chat_token_limit=50000,  # Max tokens per chat before auto-summarize and reset. 0 = unlimited
        param_estimator_refinement_rounds=0,
        exploration_topology = [1, 2, 3, 4, 5, 6, 7, 0], exploitation_topology = [1, 2, 3, 4, 5, 6, 7, 0],
        model_llm = None, param_est_llm = None, jax_translator_llm = None,
        max_iter = 1_000, learning_rate = 3e-3,
        penalty_denominator = 1,
        numpy_programs = None, param_estimators = None,
        X = None, X_eval = None,
        plot_model_fits = None, loss_fn = None,
        prompt_manager = None, trial_batch_size = None, swear_words = None,
        open_family_tree = False,
        use_simple_objective: bool = False,
        log_prompts: bool = False,
        log_jax_translations: bool = False,
        random_seed = 42, # consider setting up a seed_manager to make behaviours more robustly reproducible.
        ):
    """
    Run the full island-based hypothesis search loop.

    This is the orchestration entrypoint that:
    1) translates seed programs to JAX,
    2) scores/initializes island populations,
    3) iteratively generates new model and parameter-estimator candidates,
    4) evaluates train/test losses,
    5) migrates/prunes island populations,
    6) saves databases and visual diagnostics.

    Args:
        n_iterations (int): Maximum outer-loop iterations.
        time_limit (int | float): Wall-clock budget in minutes.
        k_max (int): Parent count sampled for generation prompts.
        n_islands (int): Number of independent island populations.
        batch_size (int): Number of proposals generated per island per iteration.
        critical_population_size (int): Target max size before pruning.
        min_wise_population_size (int): Minimum retained "wise" programs.
        n_migrants (int): Programs migrated per iteration.
        fit_params (bool): Whether gradient-based parameter fitting is enabled.
        use_param_estimator (bool): Whether to initialize parameters using estimator.
        param_penalty_weight (float): Complexity penalty applied in objective.
        FAILED_PROGRAM_COST (float): Failure sentinel cost used in scoring.
        exploit_point (float): Explore/exploit phase boundary as fraction of run.
        param_estimator_timeout_s (float | None): Per-sample timeout for
            ``param_estimator`` calls inside objective initialization.
        objective_timeout_s (float | None): Hard timeout (seconds) for each full
            objective call (initialization + optimization + evaluation).
        use_chat_mode (bool): Use persistent per-island chat sessions if True.
        chat_token_limit (int): Chat token cap before summarization/reset.
        param_estimator_refinement_rounds (int): Refinement rounds for new estimators.
        exploration_topology (list[int]): Migration destination map during explore.
        exploitation_topology (list[int]): Migration destination map during exploit.
        model_llm (str | list[str]): LLM(s) for model generation. Lists are traversed by iteration.
        param_est_llm (str | list[str]): LLM(s) for parameter estimator generation.
        jax_translator_llm (str | list[str]): LLM(s) for JAX translation.
        max_iter (int): Max optimization steps inside ``objective``.
        learning_rate (float): Optimizer learning rate inside ``objective``.
        numpy_programs (list[callable]): Seed NumPy model functions.
        param_estimators (list[callable]): Seed parameter estimators.
        X: Split data container with shape ``(2, 2)`` where each cell is a
            data dict. ``X[0,*]`` is train-sample split, ``X[1,*]`` is test-sample split.
        X_eval (np.ndarray): Evaluation grid used for diagnostics/comparison.
        plot_model_fits (callable | None): Optional plotting callback.
        loss_fn (callable | None): Objective loss function override.
        prompt_manager: PromptManager for all prompt construction.
        trial_batch_size (int | None): Trial batching size used by ``objective``.
        swear_words (list[str] | None): Blacklist for generated estimator code.
        open_family_tree (bool): Open the combined family tree HTML at the end of the run.
        use_simple_objective (bool): Use the minimal objective implementation everywhere.
        log_prompts (bool): If True, include prompt/response text in logs and generation records.
        log_jax_translations (bool): If True, include JAX translator prompt/response/code in logs.
        random_seed (int): Run seed for deterministic split-dependent operations.

    Returns:
        str: Path to the run output directory containing logs, plots, and program DBs.
    """
    has_spec_plotter = plot_model_fits is not None

    def _normalize_llm_sequence(value, label: str):
        if value is None:
            raise ValueError(f"{label} must be set (string or list).")
        if isinstance(value, (list, tuple)):
            seq = list(value)
        else:
            seq = [value]
        if not seq or any(v is None for v in seq):
            raise ValueError(f"{label} must contain at least one non-null model name.")
        return seq

    model_llm_seq = _normalize_llm_sequence(model_llm, "model_llm")
    param_est_llm_seq = _normalize_llm_sequence(param_est_llm, "param_est_llm")
    jax_llm_seq = _normalize_llm_sequence(jax_translator_llm, "jax_translator_llm")

    # load api keys
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    # Initialize IslandChatManager if using chat mode
    island_chat_manager = None
    large_model_name = None
    small_model_name = None
    if use_chat_mode:
        if len(set(model_llm_seq)) > 2:
            raise ValueError("Chat mode supports at most two model_llm entries.")
        if len(model_llm_seq) == 1:
            small_model_name = model_llm_seq[0]
            large_model_name = model_llm_seq[0]
        else:
            large_model_name = model_llm_seq[0]
            small_model_name = model_llm_seq[1]
        # Create IslandChatManager with mode-aware system instructions
        island_chat_manager = llm_helper.IslandChatManager(
            client=client,
            get_system_instruction=prompt_manager.get_system_instruction,
            small_model_name=small_model_name,
            large_model_name=large_model_name,
            explore_temperature=1.5,  # Higher temperature for creative exploration
            exploit_temperature=0.7,  # Lower temperature for focused exploitation
            thinking_budget_fraction=1.0,
            chat_token_limit=chat_token_limit,
            batch_size=batch_size
        )
        logging.info(f"Initialized IslandChatManager with models {small_model_name} / {large_model_name}")
        print(f"Chat mode enabled: using persistent chat sessions per (island, batch) pair")
        print(f"  - Total chats: {n_islands * batch_size} ({n_islands} islands × {batch_size} batches)")
        print(f"  - Explore: T=1.5, Exploit: T=0.7")
        print(f"  - Small model: {small_model_name}, Large model: {large_model_name}")
        print(f"  - Token limit per chat: {chat_token_limit} (0 = unlimited)")
    else:
        logging.info("Chat mode disabled: using independent LLM queries")
        print("Chat mode disabled: using independent LLM queries")

    X = np.asarray(X, dtype=object)
    if X.shape != (2, 2):
        raise ValueError(f"X must have shape (2, 2), got {X.shape}.")

    n_training_samples = utils.data_n_samples(X[0, 0])
    n_training_trials = utils.data_n_trials(X[0, 0])
    n_test_samples = utils.data_n_samples(X[1, 1])
    n_test_trials = utils.data_n_trials(X[1, 1])
    X_eval_train = _align_eval_grid(X_eval, n_samples=n_training_samples)
    X_eval_test = _align_eval_grid(X_eval, n_samples=n_test_samples)

    print(f"Using {n_training_trials} training trials and {n_test_trials} test trials.")
    print(f"Using {n_training_samples} samples for training and {n_test_samples} samples for testing.")

    logging.info("Translating NumPy seeds to JAX via LLM.")
    model_name = prompt_manager.get_model_name()
    seed_code_strings = [
        utils.format_function_source(program, f'{model_name}_v{i+1}', 'import numpy as np')
        for i, program in enumerate(numpy_programs)
    ]
    seed_jax_llm = jax_llm_seq[0]
    # Translate seed models sequentially to reduce burst quota/rate-limit failures.
    jax_results = []
    for code_string in seed_code_strings:
        jax_results.append(
            await translate_to_jax(code_string, client, prompt_manager, seed_jax_llm)
        )

    jax_programs = []
    jax_code_strings = []
    for i, (jax_code_string, jax_func, _jax_prompt, _jax_response) in enumerate(jax_results):
        if not callable(jax_func):
            raise RuntimeError(
                "Failed to translate seed model "
                f"{i + 1} to JAX using {seed_jax_llm}. "
                "This is commonly caused by LLM API rate limits (429 RESOURCE_EXHAUSTED). "
                "Please retry after cooldown or lower request pressure."
            )
        _run_translation_check_on_eval(
            np_func=numpy_programs[i],
            jax_func=jax_func,
            param_estimator=param_estimators[i],
            data_train_trials=X[0, 0],
            x_eval=X_eval_train,
        )
        jax_programs.append(jax_func)
        jax_code_strings.append(jax_code_string)

    # create a dataframe to store the programs in each island
    islands = []
    for _ in range(n_islands):
        islands.append(pd.DataFrame(columns=['program_code_string', 'program', 'parameter_estimator_code_string', 'parameter_estimator',
                                             'iteration_number', 'birth_island', 'batch_index', 'train_loss', 'test_loss', 'params',
                                             'initial_loss', 'initial_params', 'llm_name', 'parent1_id', 'parent2_id', 'evaluation_matrix']))
    initial_programs = pd.DataFrame([])

    # wherever you run “python script.py” from…
    base_dir = os.path.join(os.getcwd(), 'program_databases')
    print("Base directory:", base_dir)
    os.makedirs(base_dir, exist_ok=True)
    date_stamp = pd.Timestamp.now().strftime("%m-%d")
    time_stamp = pd.Timestamp.now().strftime("%H-%M-%S")
    full_dir = os.path.join(base_dir, date_stamp, time_stamp)
    os.makedirs(full_dir, exist_ok=True)
    print("Created folder:", full_dir)
    # create a directory for image diagnostics
    image_feedback_dir = os.path.join(full_dir, 'image_feedback')
    os.makedirs(image_feedback_dir, exist_ok=True)
    image_prompts_dir = os.path.join(image_feedback_dir, 'prompts')
    image_param_est_vs_gd_dir = os.path.join(image_feedback_dir, 'param_est_vs_gd')
    image_param_est_refine_dir = os.path.join(image_feedback_dir, 'param_est_refinement')
    os.makedirs(image_prompts_dir, exist_ok=True)
    os.makedirs(image_param_est_vs_gd_dir, exist_ok=True)
    os.makedirs(image_param_est_refine_dir, exist_ok=True)
    image_family_tree_fits_dir = os.path.join(image_feedback_dir, 'family_tree_fits')
    os.makedirs(image_family_tree_fits_dir, exist_ok=True)
    print("Created image feedback folder:", image_feedback_dir)
    print("Created image prompts folder:", image_prompts_dir)
    print("Created param-est vs gd folder:", image_param_est_vs_gd_dir)
    print("Created param-est refinement folder:", image_param_est_refine_dir)
    print("Created family tree fits folder:", image_family_tree_fits_dir)

    # Initialize generation log for family tree data capture
    generation_log_path = os.path.join(full_dir, 'program_generation_log.jsonl')

    # Initialize best loss tracking for live monitoring
    best_loss_log = []  # List of dicts: {iteration, timestamp, best_train_loss, best_island, ...}
    best_loss_path = os.path.join(full_dir, 'best_loss_log.csv')
    # store and compute loss of 2 initial programs
    t_start = time.time()
    seed_losses = np.zeros(2)
    seed_initial_losses = []
    seed_train_params = []
    seed_model_code_strings = []
    seed_param_est_code_strings = []
    model_name = prompt_manager.get_model_name()
    for i in range(2):
        # get the program, parameter estimator, and jax program
        program_num = numpy_programs[i]
        param_est = param_estimators[i]
        program_jax = jax_programs[i]
        # score the initial program
        seed_opt_start = time.time()
        loss_init, params_init, loss, params = _call_objective(
            use_simple_objective,
            model=program_jax,
            param_estimator=param_est,
            data=[X[0,0], X[0,1]],
            loss_fn=loss_fn,
            fit_params=fit_params,
            param_penalty_weight=param_penalty_weight,
            learning_rate=learning_rate,
            use_param_estimator=use_param_estimator,
            max_iter=max_iter,
            trial_batch_size=trial_batch_size,
            timeout_s=param_estimator_timeout_s,
            # Keep seed scoring unconstrained by objective timeout so good seeds
            # are not discarded due strict per-candidate runtime caps.
            objective_timeout_s=None,
        )
        seed_opt_time = time.time() - seed_opt_start
        print(f"Initial program {i + 1} loss before parameter fitting: {loss_init:.2f} and loss after fitting: {loss:.2f}")

        seed_losses[i] = loss
        seed_initial_losses.append(loss_init)
        seed_train_params.append(params)
        program_code_string = utils.format_function_source(
            program_num, f'{model_name}_v{i+1}', 'import numpy as np'
        )
        parameter_estimator_code_string = utils.format_function_source(
            param_est, f'parameter_estimator_v{i+1}', 'import numpy as np'
        )
        seed_model_code_strings.append(program_code_string)
        seed_param_est_code_strings.append(parameter_estimator_code_string)
        y_eval = utils.compute_evaluation_matrix(
            program_jax,
            params,
            eval_points=X_eval_train,
        )

        new_program_df = pd.DataFrame({'program_code_string': program_code_string,
                                    'program': program_jax,
                                    'parameter_estimator_code_string': parameter_estimator_code_string,
                                    'parameter_estimator': param_est,
                                    'iteration_number': -1,
                                    'birth_island': -1,  # Birth island is set to a special value for initial programs
                                    'batch_index': i,
                                    'train_loss': loss,
                                    'test_loss': None,  # all test losses will be computed at the end
                                    'optimization_time_s': seed_opt_time,
                                    'llm_name': None,
                                    'params': [params],
                                    'initial_loss': loss_init,
                                    'initial_params': [params_init],
                                    'parent1_id': None,
                                    'parent2_id': None,
                                    'evaluation_matrix': [y_eval]})
        initial_programs = pd.concat([initial_programs, new_program_df], ignore_index=True)
        print(f"Initial program {i + 1} loss: {loss:.2f}")

    # Drop invalid seed programs immediately (e.g., timed out/failed objective).
    initial_programs, _ = _drop_nonfinite_train_loss_rows(
        initial_programs,
        context="Seed initialization",
    )
    if len(initial_programs) == 0:
        raise ValueError(
            "All seed programs have non-finite train_loss and were removed. "
            "Cannot start evolutionary loop."
        )

    # Write seed programs to generation log
    n_samples=utils.data_n_samples(X[0, 0])
    for seed_idx, row in initial_programs.iterrows():
        seed_n_params = utils.params_numel_per_sample(row['params'], n_samples=n_samples)
        _append_generation_record(generation_log_path, {
            "iteration_number": -1,
            "birth_island": -1,
            "batch_index": int(seed_idx),
            "parent1_id": None,
            "parent2_id": None,
            "train_loss": float(row['train_loss']),
            "initial_loss": float(row['initial_loss']),
            "n_params": seed_n_params,
            "complexity_penalty": float(param_penalty_weight * seed_n_params),
            "model_code_numpy": row['program_code_string'],
            "param_est_code": row['parameter_estimator_code_string'],
            "model_prompt": None,
            "model_llm_response": None,
            "param_est_prompt": None,
            "param_est_llm_response": None,
            "llm_name": None,
            "temperature": None,
            "mode": "seed",
            "is_seed": True,
        })

    # seed each island with the initial programs
    for i in range(n_islands):
        islands[i] = pd.concat([islands[i], initial_programs], ignore_index=True)

    # Reset logging configuration
    log_file = os.path.join(full_dir, 'hypothesis_engine.log')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')
    
    # Log chat configuration only when prompt logging is explicitly enabled.
    if island_chat_manager is not None and log_prompts:
        island_chat_manager.log_configuration()
    
    seed_train_fit_losses = [None] * len(jax_programs)
    seed_test_fit_losses = [None] * len(jax_programs)
    seed_train_fit_paths = [None] * len(jax_programs)
    seed_test_fit_paths = [None] * len(jax_programs)
    if has_spec_plotter:
        seed_programs_list = _programs_df_to_programs_list(
            initial_programs,
            loss_func=loss_fn,
            data=X[0, 1],
            complexity_penalty=param_penalty_weight,
        )
        plot_model_fits(
            data=X[0, 0],
            programs_list=seed_programs_list,
            X_eval=X_eval_train,
            save_path=os.path.join(image_prompts_dir, 'initial_programs.png'),
            labels=['seed_1', 'seed_2'],
        )

        # Seed train/test fit plots for diagnostics.
        # Use the train-fitted params for test plots to avoid extra optimization
        # and keep cross-validated evaluation consistent.
        try:
            seed_test_params = list(seed_train_params)
            for idx, program_jax in enumerate(jax_programs):
                seed_label = f"seed_{idx+1}"
                seed_train_df = pd.DataFrame({
                    "program": [program_jax],
                    "params": [seed_train_params[idx]],
                })
                seed_train_programs_list = _programs_df_to_programs_list(
                    seed_train_df,
                    loss_func=loss_fn,
                    data=X[0, 0],
                    complexity_penalty=param_penalty_weight,
                )
                if seed_train_programs_list:
                    seed_train_fit_losses[idx] = float(
                        np.mean(np.asarray(seed_train_programs_list[0].get("losses")))
                    )
                seed_train_path = os.path.join(
                    image_family_tree_fits_dir, f'{seed_label}_train_fit.png'
                )
                seed_train_fit_paths[idx] = seed_train_path
                plot_model_fits(
                    data=X[0, 0],
                    programs_list=seed_train_programs_list,
                    X_eval=X_eval_train,
                    save_path=seed_train_path,
                    labels=[seed_label],
                    title_prefix=f"Train fits ({seed_label})",
                )

                seed_test_df = pd.DataFrame({
                    "program": [program_jax],
                    "params": [seed_test_params[idx]],
                })
                seed_test_programs_list = _programs_df_to_programs_list(
                    seed_test_df,
                    loss_func=loss_fn,
                    data=X[0, 1],
                    complexity_penalty=param_penalty_weight,
                )
                if seed_test_programs_list:
                    seed_test_fit_losses[idx] = float(
                        np.mean(np.asarray(seed_test_programs_list[0].get("losses")))
                    )
                seed_test_path = os.path.join(
                    image_family_tree_fits_dir, f'{seed_label}_test_fit.png'
                )
                seed_test_fit_paths[idx] = seed_test_path
                plot_model_fits(
                    data=X[0, 1],
                    programs_list=seed_test_programs_list,
                    X_eval=X_eval_test,
                    save_path=seed_test_path,
                    labels=[seed_label],
                    title_prefix=f"Test fits ({seed_label})",
                )
        except Exception as e:
            logging.info(f"Seed fit plotting failed: {e}")

    # Patch seed records with fit diagnostics instead of appending duplicates.
    seed_log_updates = {}
    for idx in range(len(jax_programs)):
        seed_n_params = None
        seed_complexity_penalty = None
        if idx < len(seed_train_params):
            seed_n_params = utils.params_numel_per_sample(
                seed_train_params[idx],
                n_samples=n_samples,
            )
            seed_complexity_penalty = float(param_penalty_weight * seed_n_params)
        seed_log_updates[(-1, -1, idx)] = {
            "iteration_number": -1,
            "birth_island": -1,
            "batch_index": idx,
            "train_loss": float(seed_losses[idx]),
            "initial_loss": float(seed_initial_losses[idx]) if idx < len(seed_initial_losses) else None,
            "n_params": seed_n_params,
            "complexity_penalty": seed_complexity_penalty,
            "optimization_time_s": float(initial_programs.iloc[idx].get("optimization_time_s"))
            if idx < len(initial_programs) else None,
            "train_fit_loss": seed_train_fit_losses[idx],
            "test_fit_loss": seed_test_fit_losses[idx],
            "model_prompt": None,
            "model_llm_response": None,
            "model_code_numpy": seed_model_code_strings[idx] if idx < len(seed_model_code_strings) else None,
            "model_code_jax": jax_code_strings[idx] if (log_jax_translations and idx < len(jax_code_strings)) else None,
            "param_est_prompt": None,
            "param_est_llm_response": None,
            "param_est_code": seed_param_est_code_strings[idx] if idx < len(seed_param_est_code_strings) else None,
            "param_est_refinement_prompts": [],
            "param_est_refinement_responses": [],
            "llm_name": None,
            "temperature": None,
            "mode": "seed",
            "use_large_model": None,
            "image_prompt_path": None,
            "train_fit_image_path": seed_train_fit_paths[idx],
            "test_fit_image_path": seed_test_fit_paths[idx],
            "is_seed": True,
        }
    _update_generation_log_records(generation_log_path, seed_log_updates)

    # -----------------------------
    # HYPOTHESIS ENGINE
    # -----------------------------
    for i in tqdm(range(n_iterations), desc="Hypothesis Engine Iterations"):
        # check if time limit is reached
        if time.time() - t_start > time_limit * 60:
            logging.info(f"Time limit of {time_limit} minutes reached. Stopping iterations.")
            break
        
        # Reset per-iteration token tracking (if using chat mode)
        if island_chat_manager is not None:
            island_chat_manager.start_iteration()
        
        logging.info(f"Iteration {i}")
        llm_name = model_llm_seq[i % len(model_llm_seq)]
        logging.info(f"Using model LLM: {llm_name}")
        use_large_model = use_chat_mode and (llm_name == large_model_name)
        mode = 'explore' if i < n_iterations * exploit_point else 'exploit'
        temperature = 1 + np.exp(-i / n_iterations)
        model_image_dirs = np.empty((n_islands, batch_size), dtype=object)
        # param_est_image_dirs = np.empty((n_islands, batch_size), dtype=object)
        for island_idx in range(n_islands):
            for j in range(batch_size):
                if has_spec_plotter:
                    model_image_dirs[island_idx, j] = os.path.join(image_prompts_dir, f'iter_{i}_island_{island_idx}_batch_{j}.png')
                    # param_est_image_dirs[island_idx, j] = os.path.join(image_feedback_dir, f'iter_{i}_island_{island_idx}_batch_{j}_param_est.png')
                else:
                    model_image_dirs[island_idx, j] = None
                    # param_est_image_dirs[island_idx, j] = None
        model_generation_tasks = [generate_new_model(islands[island_idx], 
                                                    llm_name=llm_name, 
                                                    client=client, 
                                                    mode=mode, 
                                                    k_max=k_max, 
                                                    temp=temperature,
                                                    data=X[0, 0],
                                                    x_eval=X_eval_train,
                                                    prompt_manager=prompt_manager,
                                                    img_dir=model_image_dirs[island_idx, j],
                                                    plot_model_fits=plot_model_fits,
                                                    island_chat_manager=island_chat_manager,
                                                    island_id=island_idx,
                                                    batch_id=j,
                                                    loss_fn=loss_fn,
                                                    loss_data=X[0, 1],
                                                    complexity_penalty=param_penalty_weight,
                                                    use_large_model=use_large_model) 
                                         for island_idx in range(n_islands) for j in range(batch_size)]
        logging.info(f"Generating {n_islands * batch_size} new programs... LLM Model: {llm_name}, mode: {mode}, temperature: {temperature:.2f}")
        print(f"Generating {n_islands * batch_size} new programs... LLM Model: {llm_name}, mode: {mode}, temperature: {temperature:.2f}")
        raw_model_results = await asyncio.gather(*model_generation_tasks)
        model_results = [
            ModelGenerationResult(
                numpy_code=model_code_string,
                prompt=prompt,
                llm_response=llm_output,
            )
            for model_code_string, prompt, llm_output, _parent_ids in raw_model_results
        ]
        parent_ids = [result[3] for result in raw_model_results]

        model_code_strings = [result.numpy_code for result in model_results]

        for candidate_idx in range(n_islands * batch_size):
            island_idx = candidate_idx // batch_size
            batch_idx = candidate_idx % batch_size
            model_result = model_results[candidate_idx]
            model_code_string = model_result.numpy_code
            model_prompt = model_result.prompt
            model_llm_response = model_result.llm_response
            parent1_id, parent2_id = parent_ids[candidate_idx]
            model_generated = model_code_string is not None
            _append_generation_record(generation_log_path, {
                "iteration_number": i,
                "birth_island": island_idx,
                "batch_index": batch_idx,
                "parent1_id": list(parent1_id) if parent1_id is not None else None,
                "parent2_id": list(parent2_id) if parent2_id is not None else None,
                "model_prompt": model_prompt,
                "model_llm_response": model_llm_response,
                "model_code_numpy": model_code_string,
                "llm_name": llm_name,
                "temperature": float(temperature),
                "mode": mode,
                "use_large_model": use_large_model,
                "image_prompt_path": model_image_dirs[island_idx, batch_idx],
                "status": "numpy_generated" if model_generated else "model_generation_failed",
                "failure_stage": None if model_generated else "model_generation",
                "failure_message": None if model_generated else "No NumPy model code generated.",
                # these will be updated later after evaluation and parameter estimation steps
                "train_loss": None,
                "initial_loss": None,
                "n_params": None,
                "complexity_penalty": None,
                "model_code_jax": None,
                "param_est_prompt": None,
                "param_est_llm_response": None,
                "param_est_code": None,
                "param_est_refinement_prompts": [],
                "param_est_refinement_responses": [],
                "train_fit_image_path": None,
                "test_fit_image_path": None,
            })

        # convert to jax
        jax_llm_name = jax_llm_seq[i % len(jax_llm_seq)]
        model_function_translation_tasks = [translate_to_jax(code_string, client, prompt_manager, jax_llm_name) for code_string in model_code_strings]
        jax_results = await asyncio.gather(*model_function_translation_tasks)
        translation_updates = {}
        for candidate_idx, (jax_code_string, jax_func, jax_prompt, jax_response) in enumerate(jax_results):
            model_result = model_results[candidate_idx]
            model_result.jax_code = jax_code_string
            model_result.jax_callable = jax_func
            model_result.jax_prompt = jax_prompt
            model_result.jax_raw_response = jax_response
            if model_code_strings[candidate_idx] is None:
                continue
            island_idx = candidate_idx // batch_size
            batch_idx = candidate_idx % batch_size
            key = (i, island_idx, batch_idx)
            update = {"model_code_jax": jax_code_string}
            if jax_code_string is None:
                update.update({
                    "status": "jax_translation_failed",
                    "failure_stage": "jax_translation",
                    "failure_message": "No JAX code block generated.",
                })
            elif jax_func is None:
                update.update({
                    "status": "jax_translation_failed",
                    "failure_stage": "jax_translation",
                    "failure_message": "Failed to parse translated JAX code into a callable.",
                })
            else:
                update.update({
                    "status": "jax_translated",
                    "failure_stage": None,
                    "failure_message": None,
                })
            translation_updates[key] = update
        _update_generation_log_records(generation_log_path, translation_updates)
        
        # build parameter‑estimator tasks
        if param_estimator_refinement_rounds > 0 and not has_spec_plotter:
            raise ValueError(
                "Parameter estimator refinement requires image diagnostics. "
                "Define plot_model_fits in the project spec or disable refinement."
            )

        # build parameter‑estimator tasks
        if param_estimator_refinement_rounds > 0 and not has_spec_plotter:
            raise ValueError(
                "Parameter estimator refinement requires image diagnostics. "
                "Define plot_model_fits in the project spec or disable refinement."
            )
        param_estimation_tasks = [
            generate_new_parameter_estimator(
                current_island=islands[island_idx],
                model_code_string=model_code_strings[island_idx * batch_size + j],
                model_fn=model_results[island_idx * batch_size + j].jax_callable,
                llm_name=param_est_llm_seq[i % len(param_est_llm_seq)],
                client=client,
                data=[X[0,0], X[0,1]],
                loss_fn=loss_fn,
                prompt_manager=prompt_manager,
                mode=mode,
                k_max=2,
                temp=temperature,
                param_estimator_max_lines=100,
                refine_rounds=param_estimator_refinement_rounds,
                param_penalty_weight=param_penalty_weight,
                random_seed=random_seed,
                swear_words=swear_words,
                island_chat_manager=island_chat_manager,
                island_id=island_idx,
                batch_id=j,
                iteration=i,
                use_simple_objective=use_simple_objective,
                plot_model_fits=plot_model_fits,
                x_eval=X_eval_train,
                image_refinement_dir=image_param_est_refine_dir,
                param_estimator_timeout_s=param_estimator_timeout_s,
                objective_timeout_s=objective_timeout_s,
            )
            for island_idx in range(n_islands)
            for j in range(batch_size)
        ]

        logging.info(
            f"Generating {n_islands * batch_size} parameter estimators "
            f"(LLM={param_est_llm_seq[i % len(param_est_llm_seq)]}, mode={mode}, T={temperature:.2f})"
        )
        logging.info(
            f"Generating {n_islands * batch_size} new parameter estimators... "
            f"Model: {param_est_llm_seq[i % len(param_est_llm_seq)]}, mode: {mode}, temperature: {temperature:.2f}"
        )
        raw_param_est_results = await asyncio.gather(*param_estimation_tasks)
        param_est_results = [
            ParamEstimatorGenerationResult(param_est_code_string, param_est_new, pe_metadata)
            for param_est_code_string, param_est_new, pe_metadata in raw_param_est_results
        ]

        param_est_updates = {}
        for candidate_idx, param_est_result in enumerate(param_est_results):
            if model_code_strings[candidate_idx] is None:
                continue
            param_est_code_string = param_est_result.code
            param_est_new = param_est_result.callable_obj
            pe_metadata = param_est_result.metadata
            island_idx = candidate_idx // batch_size
            batch_idx = candidate_idx % batch_size
            key = (i, island_idx, batch_idx)
            update = {
                "param_est_prompt": pe_metadata.get("initial_prompt"),
                "param_est_llm_response": pe_metadata.get("initial_response"),
                "param_est_code": param_est_code_string,
                "param_est_refinement_prompts": pe_metadata.get("refinement_prompts", []),
                "param_est_refinement_responses": pe_metadata.get("refinement_responses", []),
            }
            if model_results[candidate_idx].jax_callable is not None:
                if param_est_new is None:
                    update.update({
                        "status": "param_estimator_failed",
                        "failure_stage": "param_estimator",
                        "failure_message": "Failed to generate executable parameter estimator.",
                    })
                else:
                    update.update({
                        "status": "ready_for_evaluation",
                        "failure_stage": None,
                        "failure_message": None,
                    })
            param_est_updates[key] = update
        _update_generation_log_records(generation_log_path, param_est_updates)
        # combine results
        island_results = [[
            CandidateGenerationResult(
                model=model_results[island_idx * batch_size + j],
                param_estimator=param_est_results[island_idx * batch_size + j],
            )
            for j in range(batch_size)
        ] for island_idx in range(n_islands)]

        # now loop through the results and compute losses
        success_rate = 0.0
        evaluation_log_updates = {}
        for island_idx, j in np.ndindex(n_islands, batch_size):
            _clear_jax_runtime_cache()
            candidate_result = island_results[island_idx][j]
            model_result = candidate_result.model
            param_est_result = candidate_result.param_estimator
            model_code_string = model_result.numpy_code
            prompt = model_result.prompt
            model_llm_response = model_result.llm_response
            model_code_string_jax = model_result.jax_code
            model_new = model_result.jax_callable
            jax_prompt = model_result.jax_prompt
            jax_raw_response = model_result.jax_raw_response
            param_est_code_string = param_est_result.code
            param_est_new = param_est_result.callable_obj
            pe_metadata = param_est_result.metadata
            parent1_id, parent2_id = parent_ids[island_idx * batch_size + j]
            candidate_key = (i, island_idx, j)

            print(
                f"=== iter={i} island={island_idx} batch={j} === "
                f"(mode={mode})",
                flush=True,
            )
            log_lines = [
                f"=== iter={i} island={island_idx} batch={j} ===",
                f"mode={mode}",
                f"parent_ids={parent1_id},{parent2_id}",
            ]
            score_line_idx = len(log_lines)
            log_lines.append("score=<pending>")
            if log_prompts:
                log_lines.append("[Model prompt]")
                log_lines.append(prompt or "<none>")
                log_lines.append("[Model response]")
                log_lines.append(model_llm_response or "<none>")
                if isinstance(pe_metadata, dict):
                    log_lines.append("[Param estimator prompt]")
                    log_lines.append(pe_metadata.get("initial_prompt") or "<none>")
                    log_lines.append("[Param estimator response]")
                    log_lines.append(pe_metadata.get("initial_response") or "<none>")
                    if pe_metadata.get("refinement_prompts"):
                        for ridx, ref_prompt in enumerate(pe_metadata.get("refinement_prompts", []), start=1):
                            ref_resp = pe_metadata.get("refinement_responses", [None] * ridx)
                            ref_code = pe_metadata.get("refinement_codes", [None] * ridx)
                            log_lines.append(f"[Refinement {ridx} prompt]")
                            log_lines.append(ref_prompt or "<none>")
                            log_lines.append(f"[Refinement {ridx} response]")
                            log_lines.append(ref_resp[ridx - 1] if ridx - 1 < len(ref_resp) else "<none>")
                            log_lines.append(f"[Refinement {ridx} code]")
                            log_lines.append(ref_code[ridx - 1] if ridx - 1 < len(ref_code) else "<none>")

            if log_jax_translations:
                log_lines.append("[JAX translator prompt]")
                log_lines.append(jax_prompt or "<none>")
                log_lines.append("[JAX translator response]")
                log_lines.append(jax_raw_response or "<none>")
                log_lines.append("[Parsed JAX code]")
                log_lines.append(model_code_string_jax or "<none>")

            log_lines.append("[Parsed model code]")
            log_lines.append(model_code_string or "<none>")
            log_lines.append("[Parsed parameter estimator code]")
            log_lines.append(param_est_code_string or "<none>")
            if isinstance(pe_metadata, dict) and pe_metadata.get("status"):
                log_lines.append(f"[Param estimator status] {pe_metadata['status']}")
            status_notes = []
            if model_code_string is None:
                status_notes.append("model code missing")
            if param_est_code_string is None:
                status_notes.append("parameter estimator code missing")
            if model_new is None:
                status_notes.append("model parse failed")
            if param_est_new is None:
                status_notes.append("parameter estimator parse failed")
            if model_code_string_jax is None:
                status_notes.append("JAX translation missing")

            def _flush_log(extra_note: str | None = None):
                if extra_note:
                    status_notes.append(extra_note)
                if status_notes:
                    log_lines.append("[Status] " + " | ".join(status_notes))
                    print("Status:", " | ".join(status_notes), flush=True)
                logging.info("\n".join(log_lines))

            def _set_score(value):
                try:
                    scalar = float(value)
                    if np.isfinite(scalar):
                        log_lines[score_line_idx] = f"score={scalar:.6f}"
                    else:
                        log_lines[score_line_idx] = "score=inf"
                except Exception:
                    log_lines[score_line_idx] = "score=<unknown>"

            if model_new is None or param_est_new is None:
                # Provide extraction debug details when parsing fails.
                if model_code_string is None or param_est_code_string is None:
                    debug_blocks = utils.extract_code_blocks(model_llm_response)
                    debug_text = "\n\n".join(debug_blocks).strip() if debug_blocks else (model_llm_response or "").strip()
                    debug_imports = []
                    for line in debug_text.splitlines():
                        stripped = line.strip()
                        if stripped.startswith("import ") or stripped.startswith("from "):
                            debug_imports.append(stripped)
                    def _debug_extract(code: str, name_prefixes: list[str]) -> str | None:
                        for prefix in name_prefixes:
                            safe_prefix = re.escape(prefix)
                            pattern = rf"^\s*def\s+{safe_prefix}\d*\s*\(.*?(?=^\s*def\s+|\Z)"
                            match = re.search(pattern, code, flags=re.MULTILINE | re.DOTALL)
                            if match:
                                return match.group(0).strip()
                        return None

                    debug_model = _debug_extract(
                        debug_text,
                        [f"{prompt_manager.get_model_name()}_v", prompt_manager.get_model_name(), "model_v", "model"],
                    )
                    debug_param = _debug_extract(debug_text, ["parameter_estimator_v", "parameter_estimator"])
                    if log_prompts:
                        log_lines.append("[Extracted code text]")
                        log_lines.append(debug_text or "<none>")
                    log_lines.append("[Extracted imports]")
                    log_lines.append("\n".join(dict.fromkeys(debug_imports)) or "<none>")
                    log_lines.append("[Extracted model block]")
                    log_lines.append((debug_model or "<none>") if log_prompts else ("present" if debug_model else "missing"))
                    log_lines.append("[Extracted parameter_estimator block]")
                    log_lines.append((debug_param or "<none>") if log_prompts else ("present" if debug_param else "missing"))

                # update log for family tree generation
                if model_code_string is None:
                    evaluation_log_updates[candidate_key] = {
                        "status": "model_generation_failed",
                        "failure_stage": "model_generation",
                        "failure_message": "No NumPy model code generated.",
                    }
                elif model_new is None:
                    evaluation_log_updates[candidate_key] = {
                        "status": "jax_translation_failed",
                        "failure_stage": "jax_translation",
                        "failure_message": "Failed to translate NumPy model to executable JAX code.",
                    }
                else:
                    evaluation_log_updates[candidate_key] = {
                        "status": "param_estimator_failed",
                        "failure_stage": "param_estimator",
                        "failure_message": "Failed to generate executable parameter estimator.",
                    }

                _set_score(np.inf)
                _flush_log()
                continue

            model_name = prompt_manager.get_model_name()
            model_np = utils.str_to_func(model_code_string, model_name)
            if model_np is None:
                evaluation_log_updates[candidate_key] = {
                    "status": "numpy_parse_failed",
                    "failure_stage": "numpy_parse",
                    "failure_message": "Failed to parse generated NumPy model into a callable.",
                }
                _set_score(np.inf)
                _flush_log("failed to parse NumPy model")
                continue
            try:
                _run_translation_check_on_eval(
                    np_func=model_np,
                    jax_func=model_new,
                    param_estimator=param_est_new,
                    data_train_trials=X[0, 0],
                    x_eval=X_eval_train,
                )
            except Exception as e:
                evaluation_log_updates[candidate_key] = {
                    "status": "translation_check_failed",
                    "failure_stage": "translation_check",
                    "failure_message": str(e),
                }
                _set_score(np.inf)
                _flush_log(f"JAX translation check failed: {e}")
                continue
            
            opt_start = time.time()
            initial_loss, initial_params, loss, optimized_params = _call_objective(
                use_simple_objective,
                model=model_new,
                param_estimator=param_est_new,
                data=[X[0,0], X[0,1]],
                loss_fn=loss_fn,
                param_penalty_weight=param_penalty_weight,
                fit_params=fit_params,
                use_param_estimator=use_param_estimator,
                max_iter=max_iter,
                trial_batch_size=trial_batch_size,
                timeout_s=param_estimator_timeout_s,
                objective_timeout_s=objective_timeout_s,
            )
            optimization_time_s = time.time() - opt_start
            if not np.isfinite(float(loss)):
                evaluation_log_updates[candidate_key] = {
                    "status": "objective_failed",
                    "failure_stage": "objective",
                    "failure_message": "objective returned FAILED_PROGRAM_COST.",
                }

                print("Status: objective failed (non-finite loss).", flush=True)
                _set_score(np.inf)
                _flush_log("objective failed (non-finite loss)")
                logging.info('-' * 50)
                continue

            y_eval = utils.compute_evaluation_matrix(
                model_new,
                optimized_params,
                eval_points=X_eval_train,
            )
            _set_score(loss)
            _flush_log()
            logging.info(f"Loss: {loss:.2f}\n")

            train_fit_path = None
            test_fit_path = None
            train_fit_losses = []
            test_fit_losses = []
            # plot the fits of the neuron model and parameter estimator if using image feedback
            if has_spec_plotter:
                initial_params_plot = initial_params
                optimized_params_plot = optimized_params
                flat_init, _ = ravel_pytree(initial_params_plot)
                flat_opt, _ = ravel_pytree(optimized_params_plot)
                param_delta = np.asarray(flat_opt) - np.asarray(flat_init)
                mean_abs_delta = float(np.mean(np.abs(param_delta)))
                max_abs_delta = float(np.max(np.abs(param_delta)))
                if np.allclose(np.asarray(flat_init), np.asarray(flat_opt), equal_nan=True):
                    logging.info(
                        f"param_est_vs_gd: initial and optimized params are numerically identical "
                        f"(iter={i}, island={island_idx}, batch={j})."
                    )
                else:
                    logging.info(
                        f"param_est_vs_gd: param deltas (iter={i}, island={island_idx}, batch={j}) "
                        f"mean_abs={mean_abs_delta:.6g}, max_abs={max_abs_delta:.6g}"
                    )
                plot_model_fits(
                    data=X[0, 0],
                    programs_list=[
                        {
                            "model": model_new,
                            "params": initial_params_plot,
                            "losses": np.full(utils.data_n_samples(X[0, 0]), float(initial_loss)),
                        },
                        {
                            "model": model_new,
                            "params": optimized_params_plot,
                            "losses": np.full(utils.data_n_samples(X[0, 0]), float(loss)),
                        },
                    ],
                    X_eval=X_eval_train,
                    save_path=os.path.join(image_param_est_vs_gd_dir, f'iter_{i}_island_{island_idx}_batch_{j}_param_est_vs_gd.png'),
                    labels=['PE', 'GD'],
                )
                # Per-program train/test fit images for family tree sidebar
                train_fit_path = os.path.join(image_family_tree_fits_dir, f'iter_{i}_island_{island_idx}_batch_{j}_train_fit.png')
                train_programs_df = pd.DataFrame({
                    "program": [model_new, model_new],
                    "params": [initial_params_plot, optimized_params_plot],
                })
                train_programs_list = _programs_df_to_programs_list(
                    train_programs_df,
                    loss_func=loss_fn,
                    data=X[0, 0],
                    complexity_penalty=param_penalty_weight,
                )
                train_fit_losses = []
                for entry in train_programs_list:
                    if "losses" in entry:
                        train_fit_losses.append(float(np.mean(np.asarray(entry["losses"]))))
                    else:
                        train_fit_losses.append(None)
                plot_model_fits(
                    data=X[0, 0],
                    programs_list=train_programs_list,
                    X_eval=X_eval_train,
                    save_path=train_fit_path,
                    labels=['PE', 'GD'],
                    title_prefix="Train fits",
                )
                test_fit_path = os.path.join(image_family_tree_fits_dir, f'iter_{i}_island_{island_idx}_batch_{j}_test_fit.png')
                test_programs_df = pd.DataFrame({
                    "program": [model_new, model_new],
                    "params": [initial_params_plot, optimized_params_plot],
                })
                test_programs_list = _programs_df_to_programs_list(
                    test_programs_df,
                    loss_func=loss_fn,
                    data=X[0, 1],
                    complexity_penalty=param_penalty_weight,
                )
                test_fit_losses = []
                for entry in test_programs_list:
                    if "losses" in entry:
                        test_fit_losses.append(float(np.mean(np.asarray(entry["losses"]))))
                    else:
                        test_fit_losses.append(None)
                plot_model_fits(
                    data=X[0, 1],
                    programs_list=test_programs_list,
                    X_eval=X_eval_train,
                    save_path=test_fit_path,
                    labels=['PE', 'GD'],
                    title_prefix="Test fits",
                )
            # Default if plots are disabled or failed.
            if not has_spec_plotter:
                train_fit_losses = []
                test_fit_losses = []

            train_fit_loss_pe = train_fit_losses[0] if len(train_fit_losses) > 0 else None
            train_fit_loss_gd = train_fit_losses[1] if len(train_fit_losses) > 1 else train_fit_loss_pe
            train_fit_loss = train_fit_loss_gd if train_fit_loss_gd is not None else train_fit_loss_pe
            test_fit_loss_pe = test_fit_losses[0] if len(test_fit_losses) > 0 else None
            test_fit_loss_gd = test_fit_losses[1] if len(test_fit_losses) > 1 else test_fit_loss_pe
            test_fit_loss = test_fit_loss_gd if test_fit_loss_gd is not None else test_fit_loss_pe

            param_summary = utils.params_tree_summary(
                optimized_params,
                n_samples=utils.data_n_samples(X[0, 0]),
                max_lines=16,
            )
            if param_summary:
                logging.info(f"Optimized parameter structure (sample view):\n{param_summary}\n")
            t_added = time.time() - t_start
            new_program_df = pd.DataFrame({'program_code_string': model_code_string,
                                        'program': model_new,
                                        'parameter_estimator_code_string': param_est_code_string,
                                        'parameter_estimator': param_est_new,
                                        'iteration_number': i,
                                        'birth_island': island_idx,
                                        'batch_index': j,
                                        'train_loss': loss,
                                        'test_loss': None,  # will be filled later
                                        'optimization_time_s': optimization_time_s,
                                        'llm_name': llm_name,
                                        'params': [optimized_params],
                                        'initial_loss': initial_loss,
                                        'initial_params': [initial_params],
                                        'parent1_id': [parent1_id],
                                        'parent2_id': [parent2_id],
                                        'evaluation_matrix': [y_eval]
                                        })
            
            islands[island_idx] = pd.concat([islands[island_idx], new_program_df], ignore_index=True)

            n_params = int(optimized_params.shape[1])
            complexity_penalty = float(param_penalty_weight * n_params)
            evaluation_log_updates[candidate_key] = {
                "train_loss": float(loss),
                "initial_loss": float(initial_loss),
                "optimization_time_s": float(optimization_time_s),
                "model_prompt": prompt if log_prompts else None,
                "model_llm_response": model_llm_response if log_prompts else None,
                "model_code_numpy": model_code_string,
                "model_code_jax": model_code_string_jax if log_jax_translations else None,
                "param_est_prompt": pe_metadata.get("initial_prompt") if log_prompts else None,
                "param_est_llm_response": pe_metadata.get("initial_response") if log_prompts else None,
                "param_est_code": param_est_code_string,
                "param_est_refinement_prompts": pe_metadata.get("refinement_prompts", []) if log_prompts else [],
                "param_est_refinement_responses": pe_metadata.get("refinement_responses", []) if log_prompts else [],
                "llm_name": llm_name,
                "temperature": float(temperature),
                "mode": mode,
                "n_params": n_params,
                "complexity_penalty": complexity_penalty,
                "use_large_model": use_large_model,
                "image_prompt_path": model_image_dirs[island_idx, j],
                "train_fit_image_path": train_fit_path,
                "test_fit_image_path": test_fit_path,
                "train_fit_loss": train_fit_loss,
                "test_fit_loss": test_fit_loss,
                "train_fit_loss_pe": train_fit_loss_pe,
                "train_fit_loss_gd": train_fit_loss_gd,
                "test_fit_loss_pe": test_fit_loss_pe,
                "test_fit_loss_gd": test_fit_loss_gd,
                "status": "accepted",
                "failure_stage": None,
                "failure_message": None,
            }

            success_rate += 1 / (n_islands * batch_size)
            print(f"iteration {i}, island {island_idx}, batch {j}, loss: {loss:.2f}", flush=True)
            print('-' * 50, flush=True)
            logging.info("-" * 50)
        print("Success rate:", success_rate, flush=True)
        _update_generation_log_records(generation_log_path, evaluation_log_updates)

        # Remove invalid-loss programs immediately so they never participate
        # in sorting, deduplication, pruning, or migration.
        islands = _drop_nonfinite_train_loss_from_islands(
            islands,
            context=f"Iteration {i} pre-migration cleanup",
        )

        # sort each island by loss
        for island_idx in range(n_islands):
            islands[island_idx] = islands[island_idx].sort_values(by='train_loss').reset_index(drop=True)
        logging.info(f"Iteration {i} complete. The proportion of programs that successfully ran and received a loss is {success_rate:.2f}.")
        logging.info('-' * 50)
        # migrate and prune programs (better here for temperature to be in [0, 1] range)
        try:
            islands, dedup_events = genetic_helpers.perform_island_deduplication(
                islands,
                overlap_threshold=int(0.75 * critical_population_size),
                iteration=i,
            )
            islands, prune_events = genetic_helpers.perform_population_pruning(
                islands,
                critical_population_size=critical_population_size - n_migrants,
                min_wise_population_size=min_wise_population_size,
                iteration=i,
            )
            _apply_removal_reasons_to_log(generation_log_path, dedup_events + prune_events)
            islands = genetic_helpers.perform_probabilistic_migration(
                islands,
                n_migrants=n_migrants,
                destination_islands=exploration_topology if mode == 'explore' else exploitation_topology,
                temperature=(temperature - 1.0) ** 4,
                iteration=i,
            )
        except Exception as migration_error:
            logging.exception("Migration/pruning failed at iteration %s: %s", i, migration_error)
            print(
                f"Warning: migration/pruning failed at iteration {i}; "
                "continuing without migration for this iteration.",
                flush=True,
            )
        islands = _drop_nonfinite_train_loss_from_islands(
            islands,
            context=f"Iteration {i} post-migration cleanup",
        )

                                                             
        # save diagnostics
        iteration_dir = os.path.join(full_dir, 'iteration_updates', f'iteration_{i}')
        os.makedirs(iteration_dir, exist_ok=True)
        for island_idx in range(n_islands):
            pg_info = islands[island_idx][['iteration_number', 'birth_island', 'batch_index', 'train_loss']].to_string(index=False, header=False)
            print(f"Iter {i}, Island {island_idx} programs:\n{pg_info}\n")
            logging.info(f"Iter {i}, Island {island_idx} programs:\n{pg_info}\n")
        
            # Save plots of top programs
            if has_spec_plotter:
                top_df = islands[island_idx].sort_values(by='train_loss').head(3).reset_index(drop=True)
                top_df = top_df.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
                top_programs_list = _programs_df_to_programs_list(
                    top_df,
                    loss_func=loss_fn,
                    data=X[0, 1],
                    complexity_penalty=param_penalty_weight,
                )
                plot_model_fits(
                    data=X[0, 0],
                    programs_list=top_programs_list,
                    X_eval=X_eval_train,
                    save_path=os.path.join(iteration_dir, f'island_{island_idx}_top_programs.png'),
                )
        
        if has_spec_plotter:
            all_programs = pd.concat([islands[idx] for idx in range(n_islands)], ignore_index=True)
            top_programs = all_programs.sort_values(by='train_loss').head(3).reset_index(drop=True)
            top_programs = top_programs.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
            top_programs_list = _programs_df_to_programs_list(
                top_programs,
                loss_func=loss_fn,
                data=X[0, 1],
                complexity_penalty=param_penalty_weight,
            )
            plot_model_fits(
                data=X[0, 0],
                programs_list=top_programs_list,
                X_eval=X_eval_train,
                save_path=os.path.join(iteration_dir, 'top_programs_overall.png'),
            )
        
        # Log token usage summary for this iteration (if using chat mode)
        if island_chat_manager is not None:
            island_chat_manager.log_iteration_summary(i)
        
        # Log best loss across all islands for live monitoring
        all_programs = pd.concat([islands[idx] for idx in range(n_islands)], ignore_index=True)
        best_program = all_programs.loc[all_programs['train_loss'].idxmin()]
        best_loss_log.append({
            'iteration': i,
            'timestamp': pd.Timestamp.now().isoformat(),
            'best_train_loss': best_program['train_loss'],
            'best_island': best_program['birth_island'],
            'n_programs_total': len(all_programs),
            'elapsed_time': time.time() - t_start
        })
        # Write to CSV after each iteration for live monitoring
        pd.DataFrame(best_loss_log).to_csv(best_loss_path, index=False)

    # -----------------------------
    # now carry out the loss calculation on the test samples
    logging.info("Calculating loss on test set...")
    for island_idx in range(n_islands):
        logging.info(f"Island {island_idx} programs: {(len(islands[island_idx]))} programs to evaluate.")
        for j in range(len(islands[island_idx])):
            _clear_jax_runtime_cache()
            program = islands[island_idx].iloc[j]
            model = program['program']
            param_estimator = program['parameter_estimator']
            try:
                # compute the test loss
                _, _, test_loss, optimized_params = _call_objective(
                    use_simple_objective,
                    model=model,
                    param_estimator=param_estimator,
                    data=[X[1,0], X[1,1]],
                    loss_fn=loss_fn,
                    fit_params=fit_params,
                    max_iter=max_iter,
                    param_penalty_weight=param_penalty_weight,
                    use_param_estimator=use_param_estimator,
                    trial_batch_size=trial_batch_size,
                    timeout_s=param_estimator_timeout_s,
                    objective_timeout_s=objective_timeout_s,
                )
                islands[island_idx].at[j, 'test_loss'] = test_loss
                islands[island_idx].at[j, 'params'] = optimized_params
                islands[island_idx].at[j, 'mean_loss'] = np.mean(test_loss)
                print(f"Test loss: {test_loss:.2f}")
            except Exception as test_eval_error:
                logging.exception(
                    "Test evaluation failed (island=%s, idx=%s): %s",
                    island_idx,
                    j,
                    test_eval_error,
                )
                islands[island_idx].at[j, 'test_loss'] = np.inf
                islands[island_idx].at[j, 'mean_loss'] = np.inf
                print("Test loss: inf", flush=True)
            _clear_jax_runtime_cache()

    try:
        # group all islands together and save
        combined_dir = os.path.join(base_dir, date_stamp, time_stamp, 'combined')
        os.makedirs(combined_dir, exist_ok=True)
        combined_programs_dataframe = pd.concat(islands, ignore_index=True)
        combined_programs_dataframe, _ = genetic_helpers.remove_duplicates(combined_programs_dataframe, mode='complicated', loss_tol=0.025, cosine_tol=0.99, loss_type='test_loss', iteration=-1)
        # combined_programs_dataframe = combined_programs_dataframe.sort_values(by='test_loss').reset_index(drop=True)
        # sort by mean loss
        combined_programs_dataframe = combined_programs_dataframe.sort_values(by='mean_loss').reset_index(drop=True)

        # Update generation log with test losses and winner flag for family tree visualization
        _update_generation_log_test_losses_and_mark_winner(generation_log_path, islands)
        # save the combined programs dataframe, reordering columns to have order:
        # iteration_number, birth_island, batch_index, train_loss, test_loss, program_code_string, parameter_estimator_code_string, program, parameter_estimator, params, parent1_id, parent2_id
        combined_programs_dataframe = combined_programs_dataframe[['iteration_number', 'birth_island', 'batch_index',
                                                                    'train_loss', 'test_loss',
                                                                    'program_code_string', 'parameter_estimator_code_string',
                                                                    'program', 'parameter_estimator', 'params',
                                                                    'parent1_id', 'parent2_id', 'llm_name']]
        combined_programs_dataframe.to_csv(os.path.join(combined_dir, 'programs_db.csv'), index=False)

        # save island-specific results
        for island_id, island_df in enumerate(islands):
            island_dir = os.path.join(base_dir, date_stamp, time_stamp, f'island_{island_id}' if island_id < n_islands else 'meta_island')
            os.makedirs(island_dir, exist_ok=True)
            island_df.to_csv(os.path.join(island_dir, 'programs_db.csv'), index=False)
    except Exception as postprocess_error:
        logging.exception("Post-processing failed: %s", postprocess_error)
        print(
            "Warning: post-processing failed; returning partial run outputs.",
            flush=True,
        )
        return full_dir

    # ---------------------------
    # save losses plot    
    if has_spec_plotter:
        try:
            plot_train_vs_test_loss_shared(
                programs_df=combined_programs_dataframe,
                island_labels=[f'Island {i}' for i in range(n_islands)] + ['garden_of_eden'],
                save_path=os.path.join(combined_dir, 'train_vs_test_loss.png'),
            )
        except Exception as plot_error:
            logging.exception("Train-vs-test plot failed: %s", plot_error)
    
    # ---------------------------
    df_list = [combined_programs_dataframe] + islands
    combined_dir = [os.path.join(base_dir, date_stamp, time_stamp, "combined")] 
    island_dirs = [os.path.join(base_dir, date_stamp, time_stamp, f'island_{i}') for i in range(n_islands)]
    df_dirs = combined_dir + island_dirs

    if has_spec_plotter:
        try:
            for i, df in enumerate(df_list):
                df = df.head(3)
                df = df.sort_values(by='test_loss', ascending=False).reset_index(drop=True)
                programs_list = _programs_df_to_programs_list(
                    df,
                    loss_func=loss_fn,
                    data=X[1, 1],
                    complexity_penalty=param_penalty_weight,
                )
                plot_model_fits(
                    data=X[1, 1],
                    programs_list=programs_list,
                    X_eval=X_eval_test,
                    save_path=os.path.join(df_dirs[i], 'top_model_fits.png'),
                )
                # Plot top models separately using the same plot_model_fits pathway.
                for j in range(min(3, len(df))):
                    model_df = df.iloc[[j]].copy().reset_index(drop=True)
                    plot_model_fits(
                        data=X[1, 1],
                        programs_list=_programs_df_to_programs_list(
                            model_df,
                            loss_func=loss_fn,
                            data=X[1, 1],
                            complexity_penalty=param_penalty_weight,
                        ),
                        X_eval=X_eval_test,
                        save_path=os.path.join(df_dirs[i], f'top_model_fit_{min(3, len(df)) - j}.png'),
                        labels=['model'],
                    )
        except Exception as top_plot_error:
            logging.exception("Top-model plotting failed: %s", top_plot_error)
    
    # Generate family tree visualizations
    create_dynamic_progress_update(generation_log_path, full_dir)
    create_family_tree(generation_log_path, full_dir, n_islands)
    if open_family_tree:
        family_tree_path = os.path.join(full_dir, "genealogy.html")
        try:
            if os.path.isfile(family_tree_path):
                webbrowser.open(Path(family_tree_path).resolve().as_uri())
            else:
                logging.info("Family tree HTML not found at %s; skipping auto-open.", family_tree_path)
        except Exception as e:
            logging.info("Failed to open family tree HTML: %s", e)

    # Log final token usage summary (if using chat mode)
    if island_chat_manager is not None:
        island_chat_manager.log_final_summary()

    return full_dir
