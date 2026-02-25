import inspect
import json
import re
import os
import logging
import webbrowser
import asyncio
import numpy as np
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import timeout_decorator
import optax
import pandas as pd
from pathlib import Path
from . import utils, llm_helper
from . import utils, llm_helper
from . import genetic_helpers_v2 as genetic_helpers  # Using v2 with compatibility API
from .data_structures import ensure_inputs, ensure_outputs
from .evolution_diagnostics import plot_train_vs_test_loss as plot_train_vs_test_loss_shared
from .family_tree import create_family_tree
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(relativeCreated)dms | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def compute_initial_params(param_estimator, model, x, y):
    """
    Estimate per-sample initial model parameters with safe fallbacks.

    This helper calls the provided parameter estimator once per sample with a
    timeout guard. If estimation fails (timeout or exception), it falls back to
    default values extracted from the model signature.

    Args:
        param_estimator (callable): Parameter initializer with signature
            ``param_estimator(X_i, y_i) -> params_i`` for a single sample.
        model (callable): Model function used only for fallback default parameter
            inference when estimation fails.
        x (np.ndarray): Input tensor of shape ``(n_samples, n_features, n_trials)``.
        y (np.ndarray): Output tensor of shape
            ``(n_samples, n_targets, n_trials)`` or ``(n_samples, n_trials)``.

    Returns:
        pytree | None: Batched parameter pytree with leading sample axis for each
            leaf. Returns ``None`` when both estimator-based initialization and
            default-parameter fallback fail.
    """
    @timeout_decorator.timeout(5, use_signals=True)
    def _safe_estimate(pe, xi, yi):
        return pe(xi, yi)

    def _estimator_response_arg(yi):
        """
        Keep canonical 2D target format internally, but adapt scalar target for
        legacy estimators that expect y shape (n_trials,).
        """
        yi_arr = np.asarray(yi)
        if yi_arr.ndim == 2 and yi_arr.shape[0] == 1:
            return yi_arr[0]
        return yi_arr

    # Compute defaults once for per-sample fallback.
    defaults = compute_default_params(model)
    fallback = defaults
    if defaults is None:
        logging.info("Default parameters unavailable; will fall back to first successful estimate.")

    params_list = []
    n_samples = y.shape[0]
    for i in range(n_samples):
        try:
            # any call taking >5s will raise timeout_decorator.TimeoutError
            # xi has shape (n_features, n_trials)
            # yi has shape (n_trials,) for scalar or (n_targets, n_trials) for vectorized
            params_i = _safe_estimate(param_estimator, x[i], _estimator_response_arg(y[i]))
        except timeout_decorator.TimeoutError:
            logging.warning(f"param_estimator timed out for sample {i}, using defaults")
            params_i = fallback
        except Exception as e:
            logging.info(f"Error during parameter estimation for sample {i}: {e}")
            params_i = fallback
        if params_i is None:
            logging.info("Error: Unable to compute parameters (no defaults available).")
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
    x_data: jnp.ndarray,
    initial_params,
    n_samples: int,
    expected_n_targets: int = 1,
    n_validation_samples: int = 10,
) -> tuple[bool, str]:
    """
    Smoke-test model execution with JAX tracing and output-shape checks.

    A random subset of samples is evaluated to ensure the model:
    1) executes without runtime errors,
    2) is JIT/trace compatible,
    3) returns outputs with expected dimensions.

    Args:
        model (callable): Candidate model function.
        x_data (jnp.ndarray): Input tensor ``(n_samples, n_features, n_trials)``.
        initial_params (pytree): Batched parameter pytree (leading sample axis).
        n_samples (int): Number of samples in ``x_data``.
        expected_n_targets (int): Number of expected output targets.
        n_validation_samples (int): Maximum number of random samples to test.

    Returns:
        tuple[bool, str]: ``(is_valid, error_message)`` where ``error_message``
            captures the first failure reason.
    """
    try:
        model_jit = jax.jit(model)
        test_n_trials = x_data.shape[2]
        
        for sample_idx in np.random.choice(n_samples, size=min(n_validation_samples, n_samples), replace=False):
            # Validate with concrete values: x_data[sample_idx] is (n_features, n_trials)
            params_i = utils.slice_params(initial_params, sample_idx)
            output = model_jit(x_data[sample_idx], params_i)
            
            is_valid, error_msg = validate_model_output(output, test_n_trials, expected_n_targets)
            if not is_valid:
                return False, error_msg
            
            # Validate with abstract tracer values
            jax.eval_shape(model_jit, x_data[sample_idx], params_i)
        
        return True, ""
    except Exception as e:
        return False, f"Model failed to run or is incompatible with JAX tracing: {e}"


def objective(model, param_estimator, x, y,
              loss_fn=None, param_penalty_weight=0.1, fit_params=True,
              FAILED_PROGRAM_COST=jnp.inf, max_iter=1_000, learning_rate=3e-3,
              use_param_estimator=True, trial_batch_size=None,
              timeout_s: float | None = None) -> tuple[float, jnp.ndarray, float, jnp.ndarray]:
    """
    Calculate model loss using the unified Outputs representation.
    
    Supports both scalar and multi-target outputs through canonical shape
    (n_samples, n_targets, n_trials).
    
    The loss is computed as a weighted sum of per-target MSE:
        loss = sum_t(weight[t] * mean((pred[t] - true[t])^2))
    
    Args:
        model (function): The model which predicts neural activity from inputs
                          and free parameters (for a single sample).
                          Signature: model(X, params) -> activity
                          where X has shape (n_features, n_trials) for a single sample.
                          Output shape: (n_targets, n_trials) for vectorized.
                          For n_targets=1, output can be (n_trials,) and will be auto-expanded.
        param_estimator (function): Function to estimate initial parameters for the model.
                          Signature: param_estimator(X, response) -> params
                          where X has shape (n_features, n_trials) for a single sample,
                          and response has shape (n_trials,) for scalar or (n_targets, n_trials) for vectorized.
        x: Trial-split inputs for one sample split. Expected length-2 container:
           - x[0]: train-trial inputs
           - x[1]: test-trial inputs
           Each element can be an Inputs object or array with shape
           (n_samples, n_features, n_trials_split).
        y: Trial-split outputs for one sample split. Expected length-2 container:
           - y[0]: train-trial outputs
           - y[1]: test-trial outputs
           Each element can be an Outputs object or array with shape
           (n_samples, n_targets, n_trials_split).
        loss_fn (function): Per-sample loss function.
                          Signature: loss_fn(y_pred, y_true) -> scalar/array.
                          Required; no default.
        param_penalty_weight (float): Weight for the penalty on the number of parameters. Default is 0.1.
        fit_params (bool): Whether to fit the parameters of the model. Default is True.
        FAILED_PROGRAM_COST (float): Cost assigned to failed models. Default is np.inf.
        max_iter (int): Maximum number of iterations for optimization. Default is 1_000.
        use_param_estimator (bool): Whether to use the parameter estimator to compute initial parameters. Default is True.
        trial_batch_size (int | None): Ignored. Trial batching is disabled.

    Returns:
        tuple[
            - float: The cross-validated loss of the model with initial parameters,
            - pytree: The initial parameters (batched pytree).
            - float: The average loss on test set after optimization.
                     Returns FAILED_PROGRAM_COST if the model fails.
            - pytree: The optimized parameters (batched pytree).
    """
    t_start = time.time()
    initial_params = None
    if loss_fn is None:
        raise ValueError("objective requires a loss_fn; none was provided.")

    def _check_timeout():
        if timeout_s is None or timeout_s <= 0:
            return
        if time.time() - t_start > timeout_s:
            raise ObjectiveTimeout(f"objective timed out after {timeout_s}s")

    if not (isinstance(x, (list, tuple, np.ndarray)) and len(x) == 2):
        raise ValueError("objective expects x as length-2 container: [x_train_trials, x_test_trials].")
    if not (isinstance(y, (list, tuple, np.ndarray)) and len(y) == 2):
        raise ValueError("objective expects y as length-2 container: [y_train_trials, y_test_trials].")

    x_train = ensure_inputs(x[0]).to_tensor()
    x_test = ensure_inputs(x[1]).to_tensor()
    y_train_outputs = ensure_outputs(y[0])
    y_test_outputs = ensure_outputs(y[1])
    y_train = y_train_outputs.data
    y_test = y_test_outputs.data

    n_samples, n_features, _ = x_train.shape
    n_targets = y_train_outputs.n_targets
    
    try:
        _check_timeout()
        # Compute initial parameters
        # param_estimator receives y as (n_targets, n_trials) for each sample
        if use_param_estimator:
            initial_params = compute_initial_params(param_estimator, model, np.asarray(x_train), np.asarray(y_train))
        else:
            initial_params = compute_default_params(model)
    except ObjectiveTimeout:
        logging.info("Objective timed out during parameter initialization.")
        empty_params = {}
        return FAILED_PROGRAM_COST, empty_params, FAILED_PROGRAM_COST, empty_params

    if initial_params is None:
        logging.info("Error: initial_params unavailable.")
        empty_params = {}
        return FAILED_PROGRAM_COST, empty_params, FAILED_PROGRAM_COST, empty_params

    initial_params = utils.broadcast_params(initial_params, n_samples)

    # Fail immediately if fit_params is True and non-numeric params
    n_params_raw = utils.params_numel_per_sample(initial_params, n_samples=n_samples)
    n_features_in = int(n_features)
    n_features_out = int(n_targets)
    penalty_denom = max(1, n_features_in * n_features_out)
    n_params = n_params_raw / penalty_denom
    if not utils.params_all_finite(initial_params):
        logging.info("Error: Parameters contain non-numeric or non-finite values.")
        return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params
    
    # Validate model execution and output shape
    is_valid, error_msg = validate_model_execution(
        model, x_train, initial_params, n_samples,
        expected_n_targets=n_targets,
        n_validation_samples=10
    )
    if not is_valid:
        logging.info(f"Model validation failed: {error_msg}")
        return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params

    flat_init, unflatten = ravel_pytree(initial_params)

    # Per-sample loss function
    def loss_single_sample(params, x_i, y_i):
        y_pred = model(x_i, params)
        if y_i.ndim == 1:
            y_i = y_i[None, :]
        if y_pred.ndim == 1:
            y_pred = y_pred[None, :]
        sample_loss = jnp.asarray(loss_fn(y_pred, y_i))
        if sample_loss.ndim == 0:
            return sample_loss
        return jnp.mean(sample_loss)

    # Vectorize over samples
    # params: pytree (batched), x: (n_samples, n_features, n_trials_x), y: (n_samples, n_targets, n_trials_y)
    # Output: (n_samples,)
    loss_total = jax.vmap(loss_single_sample, in_axes=(0, 0, 0), out_axes=0)

    # Mini-batched loss and gradient computation to avoid GPU OOM
    n_train_trials_x = x_train.shape[2]
    n_train_trials_y = y_train.shape[2]
    trials_matched = (n_train_trials_x == n_train_trials_y)
    effective_trial_batch_size = n_train_trials_x if trial_batch_size is None else int(trial_batch_size)
    # Scalar single-feature full-batch fast path only when using default loss
    use_scalar_single_feature_fullbatch = (
        trial_batch_size is None and n_targets == 1 and n_features == 1
        and trials_matched
    )

    @jax.jit
    def loss_single_batch(params_tree, x_batch, y_batch):
        """Compute sum of losses for one batch (JIT-compiled)."""
        batch_losses = loss_total(params_tree, x_batch, y_batch)  # (n_samples,)
        return jnp.sum(batch_losses)

    # Combined loss and gradient computation - more efficient than separate calls
    loss_and_grad_single_batch = jax.jit(jax.value_and_grad(loss_single_batch))

    # JIT-compiled eval (nansum to ignore NaN losses from bad model outputs)
    @jax.jit
    def eval_single_batch(params_tree, x_batch, y_batch):
        """Compute nansum of losses for one batch (JIT-compiled, no grad)."""
        batch_losses = loss_total(params_tree, x_batch, y_batch)
        return jnp.nansum(batch_losses)

    if trials_matched:
        # Standard path: batch over trial dimension with same indices for x and y
        n_train_trials = n_train_trials_x  # same as n_train_trials_y

        def loss_and_grad_batched(flat_params):
            """Compute loss and gradient by accumulating over trial batches."""
            params_tree = unflatten(flat_params)
            total_loss = 0.0
            total_grad = jnp.zeros_like(flat_params)

            for start_idx in range(0, n_train_trials, effective_trial_batch_size):
                _check_timeout()
                end_idx = min(start_idx + effective_trial_batch_size, n_train_trials)
                batch_weight = (end_idx - start_idx) / n_train_trials
                x_batch = x_train[:, :, start_idx:end_idx]
                y_batch = y_train[:, :, start_idx:end_idx]

                batch_loss, batch_grad_tree = loss_and_grad_single_batch(params_tree, x_batch, y_batch)
                batch_grad_flat, _ = ravel_pytree(batch_grad_tree)

                total_loss += batch_loss * batch_weight
                total_grad += batch_grad_flat * batch_weight

            return total_loss / n_samples, total_grad / n_samples
    else:
        # Mismatched trials: no trial mini-batching. Pass full x_train and y_train.
        # The loss_fn handles the shape relationship internally.
        def loss_and_grad_batched(flat_params):
            """Compute loss and gradient over full data (no trial batching)."""
            params_tree = unflatten(flat_params)
            loss, grad_tree = loss_and_grad_single_batch(params_tree, x_train, y_train)
            grad_flat, _ = ravel_pytree(grad_tree)
            return loss / n_samples, grad_flat / n_samples
    
    if fit_params:
        # Adam optimizer with learning rate schedule
        # Ensure learning_rate is a Python float (not JAX array) for optax
        learning_rate = float(learning_rate)
        opt = optax.adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8)
        opt_state = opt.init(flat_init)
        
        if use_scalar_single_feature_fullbatch:
            # Match legacy scalar full-batch path for speed/consistency when no batching is requested.
            loss_param = lambda params: jnp.mean(loss_total(unflatten(params), x_train, y_train))
            loss_param_and_grad = jax.value_and_grad(loss_param)

            @jax.jit
            def train_step(params, opt_state):
                loss, grad = loss_param_and_grad(params)
                updates, opt_state = opt.update(grad, opt_state, params)
                params = optax.apply_updates(params, updates)
                return params, opt_state, loss

            print_every = 50
            params = flat_init
            initial_loss = loss_param(params)
            best_loss, best_params = initial_loss.copy(), params.copy()
            for step in range(1, max_iter + 1):
                _check_timeout()
                params, opt_state, loss_val = train_step(params, opt_state)
                _check_timeout()
                if jnp.isnan(loss_val) or jnp.isinf(loss_val) or jnp.any(jnp.isnan(params)) or jnp.any(jnp.isinf(params)):
                    logging.info(f"Loss is NaN or Inf at step {step}. Stopping optimization.")
                    print(f"Final loss: {loss_val:.4f} at step {step}")
                    break
                if loss_val < best_loss:
                    best_loss = loss_val.copy()
                    best_params = params.copy()
                if step % print_every == 0:
                    print(f"step {step:4d}  loss {loss_val:.4f}")
            params = unflatten(best_params)
            print(f"params optimized. Loss: {best_loss:.4f}")
        else:
            def train_step(params, opt_state):
                loss, grad = loss_and_grad_batched(params)
                updates, new_opt_state = opt.update(grad, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return new_params, new_opt_state, loss

            print_every = 50
            params = flat_init
            initial_loss, _ = loss_and_grad_batched(params)

            CATASTROPHIC_LOSS_THRESHOLD = 1e6
            if initial_loss > CATASTROPHIC_LOSS_THRESHOLD:
                print(f"Initial loss {initial_loss:.2e} exceeds threshold. Skipping optimization.")
                logging.info(f"Skipping optimization: initial loss {initial_loss:.2e} > {CATASTROPHIC_LOSS_THRESHOLD:.0e}")
                return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params

            best_loss, best_params = initial_loss.copy(), params.copy()
            for step in range(1, max_iter + 1):
                _check_timeout()
                params, opt_state, loss_val = train_step(params, opt_state)
                _check_timeout()
                if jnp.isnan(loss_val) or jnp.isinf(loss_val) or jnp.any(jnp.isnan(params)) or jnp.any(jnp.isinf(params)):
                    logging.info(f"Loss is NaN or Inf at step {step}. Stopping optimization.")
                    print(f"Final loss: {loss_val:.4f} at step {step}")
                    break
                if loss_val > CATASTROPHIC_LOSS_THRESHOLD:
                    logging.info(f"Loss exploded to {loss_val:.2e} at step {step}. Stopping optimization.")
                    print(f"Loss exploded to {loss_val:.2e}. Stopping optimization.")
                    return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params
                if loss_val < best_loss:
                    best_loss = loss_val.copy()
                    best_params = params.copy()
                if step % print_every == 0:
                    print(f"step {step:4d}  loss {loss_val:.4f}")
            params = unflatten(best_params)
            print(f"params optimized. Loss: {best_loss:.4f}")
    else:
        params = initial_params
    
    # Compute final loss on test set
    if trials_matched:
        def eval_loss_batched(params_tree, x_eval, y_eval):
            """Compute loss by iterating over trial batches (matched trials)."""
            n_eval_trials = x_eval.shape[2]
            weighted_sum = 0.0
            for start_idx in range(0, n_eval_trials, effective_trial_batch_size):
                _check_timeout()
                end_idx = min(start_idx + effective_trial_batch_size, n_eval_trials)
                batch_size = end_idx - start_idx
                x_batch = x_eval[:, :, start_idx:end_idx]
                y_batch = y_eval[:, :, start_idx:end_idx]
                weighted_sum += eval_single_batch(params_tree, x_batch, y_batch) * (batch_size / n_eval_trials)
            return weighted_sum / n_samples
    else:
        def eval_loss_batched(params_tree, x_eval, y_eval):
            """Compute loss over full data (mismatched trials, no batching)."""
            return eval_single_batch(params_tree, x_eval, y_eval) / n_samples

    try:
        _check_timeout()
        if use_scalar_single_feature_fullbatch:
            initial_loss = jnp.nanmean(loss_total(initial_params, x_test, y_test)) + param_penalty_weight * n_params
        else:
            initial_loss = eval_loss_batched(initial_params, x_test, y_test) + param_penalty_weight * n_params
    except ObjectiveTimeout:
        logging.info("Objective timed out during initial loss evaluation.")
        return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params
    n_nans = jnp.sum(jnp.isnan(initial_loss))
    if n_nans > 0:
        print(f"Warning: initial loss contains {n_nans} NaNs.")
    initial_loss = jnp.nan_to_num(initial_loss, nan=FAILED_PROGRAM_COST, posinf=FAILED_PROGRAM_COST, neginf=FAILED_PROGRAM_COST)
    
    try:
        _check_timeout()
        if use_scalar_single_feature_fullbatch:
            final_loss = jnp.nanmean(loss_total(params, x_test, y_test)) + param_penalty_weight * n_params
        else:
            final_loss = eval_loss_batched(params, x_test, y_test) + param_penalty_weight * n_params
    except ObjectiveTimeout:
        logging.info("Objective timed out during final loss evaluation.")
        return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params
    n_nans = jnp.sum(jnp.isnan(final_loss))
    if n_nans > 0:
        print(f"Warning: final loss contains {n_nans} NaNs.")
    final_loss = jnp.nan_to_num(final_loss, nan=FAILED_PROGRAM_COST, posinf=FAILED_PROGRAM_COST, neginf=FAILED_PROGRAM_COST)
    
    t_end = time.time()
    print(f"Time taken for optimization: {t_end - t_start:.4f} seconds")
    return float(initial_loss), initial_params, float(final_loss), params


def objective_with_timeout(
    model,
    param_estimator,
    x,
    y,
    loss_fn=None,
    param_penalty_weight=0.1,
    fit_params=True,
    FAILED_PROGRAM_COST=jnp.inf,
    max_iter=1_000,
    learning_rate=3e-3,
    use_param_estimator=True,
    trial_batch_size=None,
    timeout_s: float | None = 5.0,
) -> tuple[float, jnp.ndarray, float, jnp.ndarray]:
    """Wrapper around objective that enforces a wall-clock timeout."""
    if timeout_s is None or timeout_s <= 0:
        return objective(
            model=model,
            param_estimator=param_estimator,
            x=x,
            y=y,
            loss_fn=loss_fn,
            param_penalty_weight=param_penalty_weight,
            fit_params=fit_params,
            FAILED_PROGRAM_COST=FAILED_PROGRAM_COST,
            max_iter=max_iter,
            learning_rate=learning_rate,
            use_param_estimator=use_param_estimator,
            trial_batch_size=trial_batch_size,
        )

    @timeout_decorator.timeout(timeout_s, use_signals=True)
    def _run():
        return objective(
            model=model,
            param_estimator=param_estimator,
            x=x,
            y=y,
            loss_fn=loss_fn,
            param_penalty_weight=param_penalty_weight,
            fit_params=fit_params,
            FAILED_PROGRAM_COST=FAILED_PROGRAM_COST,
            max_iter=max_iter,
            learning_rate=learning_rate,
            use_param_estimator=use_param_estimator,
            trial_batch_size=trial_batch_size,
            timeout_s=timeout_s,
        )

    try:
        return _run()
    except timeout_decorator.TimeoutError:
        logging.info(f"Objective timed out after {timeout_s}s; skipping.")
        return float(FAILED_PROGRAM_COST), {}, float(FAILED_PROGRAM_COST), {}
    except ObjectiveTimeout:
        logging.info(f"Objective timed out after {timeout_s}s; skipping.")
        return float(FAILED_PROGRAM_COST), {}, float(FAILED_PROGRAM_COST), {}

def _programs_df_to_programs_list(programs_df: pd.DataFrame, 
                                    loss_func: callable,
                                    x: jnp.ndarray, y: jnp.ndarray,
                                    complexity_penalty: float = 0.0) -> list[dict]:
    """
    Convert a programs dataframe to the canonical programs_list plotting payload.
    Compute per sample losses for each program using the provided loss function, 
    and include them in the programs_list dicts under the key 'losses'. 
    This allows the plotting function to visualize the performance of each program.

    Args:
    programs_df (pd.DataFrame): DataFrame containing program information with columns 'model' and 'params'.
    loss_func (callable): Loss function ``loss_func(y_pred, y_true)``.
    x (jnp.ndarray): (n_samples, n_features, n_trials) input data used for computing losses.
    y (jnp.ndarray): (n_samples, n_targets, n_trials) true output data used for computing losses.
    complexity_penalty (float): Additive complexity penalty multiplier.
        Each sample loss is increased by ``complexity_penalty * n_free_params``.
    """
    programs_list = []
    if programs_df is None or len(programs_df) == 0:
        return programs_list
    x_arr = jnp.asarray(ensure_inputs(x).to_tensor())
    y_arr = jnp.asarray(ensure_outputs(y).to_tensor())
    n_samples = x_arr.shape[0]

    if loss_func is None:
        raise ValueError("_programs_df_to_programs_list requires a loss_func; none was provided.")

    for _, row in programs_df.iterrows():
        model = row.get('program', row.get('model'))
        params = row.get('params')
        if model is None or params is None:
            continue
        params_tree = utils.broadcast_params(params, n_samples)
        n_free_params_raw = utils.params_numel_per_sample(params_tree, n_samples=n_samples)
        penalty_denom = max(1, int(x_arr.shape[1]) * int(y_arr.shape[1]))
        n_free_params = n_free_params_raw / penalty_denom
        y_pred = utils.vmap_over_samples(model)(x_arr, params_tree)
        if y_pred.ndim == 2 and y_arr.ndim == 3 and y_arr.shape[1] == 1:
            y_pred = y_pred[:, None, :]
        raw_losses = jnp.asarray(loss_func(y_pred, y_arr))

        if raw_losses.ndim == 0:
            losses = jnp.full((n_samples,), raw_losses)
        else:
            # Reduce all non-sample dimensions so each sample has one loss value.
            if raw_losses.shape[0] == n_samples:
                losses = jnp.mean(raw_losses.reshape(n_samples, -1), axis=1)
            else:
                flat = raw_losses.reshape(-1)
                if flat.size == n_samples:
                    losses = flat
                elif flat.size == 1:
                    losses = jnp.full((n_samples,), flat[0])
                else:
                    losses = jnp.full((n_samples,), jnp.mean(flat))
        losses = losses + float(complexity_penalty) * n_free_params
        programs_list.append({
            'model': model,
            'params': params_tree,
            'losses': losses
        })

    return programs_list


def _align_eval_grid(X_eval, n_samples: int) -> np.ndarray:
    """
    Align evaluation grid first dimension to n_samples by tiling if needed.
    """
    eval_arr = np.asarray(X_eval)
    if eval_arr.ndim == 2:
        eval_arr = eval_arr[np.newaxis, :, :]
    if eval_arr.ndim != 3:
        raise ValueError(f"X_eval must be 3D (n_samples, n_features, n_eval_trials), got {eval_arr.shape}.")
    if eval_arr.shape[0] == n_samples:
        return eval_arr
    if eval_arr.shape[0] == 1:
        return np.broadcast_to(eval_arr, (n_samples, eval_arr.shape[1], eval_arr.shape[2]))
    idx = np.arange(n_samples, dtype=np.int64) % eval_arr.shape[0]
    return eval_arr[idx]


async def generate_new_model(current_island, llm_name, client, 
                            x, y, x_eval, prompt_manager,
                            mode='explore', k_max=2, temp=1, 
                            thinking_budget=1, img_dir=None,
                            plot_model_fits=None,
                            island_chat_manager=None, island_id: int = None,
                            batch_id: int = 0,
                            loss_fn=None,
                            loss_x=None,
                            loss_y=None,
                            complexity_penalty: float = 0.0,
                            use_large_model: bool = True):
    """
    Propose a new model program by querying the LLM from island context.

    The function samples parent programs from an island, builds a prompt
    (optionally with an image diagnostic), calls the LLM, and extracts a Python
    function code block for the next model candidate.

    Args:
        current_island (pd.DataFrame): Program population for one island.
        llm_name (str): Model name for legacy stateless LLM calls.
        client: LLM client handle used by helper wrappers.
        x: Data object forwarded to plotting for image-conditioned prompting.
        y: Data object forwarded to plotting for image-conditioned prompting.
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
        loss_x: Optional input tensor to use specifically for diagnostics loss.
        loss_y: Optional output tensor to use specifically for diagnostics loss.
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
            x_for_loss = x if loss_x is None else loss_x
            y_for_loss = y if loss_y is None else loss_y
            programs_list = _programs_df_to_programs_list(
                random_programs,
                loss_func=loss_fn,
                x=x_for_loss,
                y=y_for_loss,
                complexity_penalty=complexity_penalty,
            )
            plot_model_fits(
                X=x,
                Y=y,
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
        logging.info(
            f"Model generation prompt (island={island_id}, batch={batch_id}):\n{program_prompt}\n"
        )
        llm_output = await island_chat_manager.ask_island(
            island_id, program_prompt,
            batch_id=batch_id,
            mode=mode, 
            use_large_model=use_large_model,
            png_img=img_bytes
        )
    else:
        # Legacy: independent query
        logging.info(
            f"Model generation prompt (island={island_id}, batch={batch_id}):\n{program_prompt}\n"
        )
        llm_output = await llm_helper.call_llm_async(program_prompt, model_name=llm_name, client=client, temperature=temp, 
                                                thinking_budget=thinking_budget, img_bytes=img_bytes)
    logging.info(
        f"Model generation output (island={island_id}, batch={batch_id}):\n{llm_output}\n"
    )
    
    code_string = utils.extract_code_block(llm_output)
    if code_string is None:
        logging.info(
            f"Model generation returned no code block (island={island_id}, batch={batch_id})."
        )
        return None, None, None, (parent1_id, parent2_id)
    code_string = code_string.replace(f'def {model_name}_v{k+1}(', f'def {model_name}(')
    logging.info(
        f"Generated model candidate (island={island_id}, batch={batch_id}):\n"
        f"Prompt:\n{program_prompt}\n\n"
        f"Model (NumPy):\n{code_string}\n"
    )

    return code_string, program_prompt, llm_output, (parent1_id, parent2_id)


async def generate_new_parameter_estimator(current_island, 
                                           model_code_string: str,
                                           model_fn,
                                           llm_name, client, 
                                           x, y,
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
                                           loss_fn=None,
                                           plot_model_fits=None,
                                           x_eval=None,
                                           image_refinement_dir=None,):                                           
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
        x: Objective input split passed to ``objective`` during scoring.
            Expects length-2 container: ``[x_train_trials, x_test_trials]``.
        y: Objective target split passed to ``objective`` during scoring.
            Expects length-2 container: ``[y_train_trials, y_test_trials]``.
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

    Returns:
        tuple[str | None, callable | None, dict]: Best estimator code string, parsed
            callable, and metadata dict with prompt/response info.
            Returns ``(None, None, pe_metadata)`` when generation/validation fails.
    """
    pe_metadata = {"initial_prompt": None, "initial_response": None,
                   "refinement_prompts": [], "refinement_responses": []}
    if model_code_string is None:
        logging.info("No model code string provided, skipping parameter estimator generation.")
        return None, None, pe_metadata
    if not (isinstance(x, (list, tuple, np.ndarray)) and len(x) == 2):
        logging.info("Parameter estimator generation expects x split as [train_trials, test_trials].")
        return None, None, pe_metadata
    if not (isinstance(y, (list, tuple, np.ndarray)) and len(y) == 2):
        logging.info("Parameter estimator generation expects y split as [train_trials, test_trials].")
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
    
    # Use chat-based or legacy LLM call
    logging.info(
        f"Parameter estimator prompt (island={island_id}, batch={batch_id}):\n{prompt}\n"
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
        logging.info("No code block found in the LLM output for parameter estimator, skipping.")
        return None, None, pe_metadata
    contains_swear_word = bool(swear_words) and any(word in code_string for word in swear_words)
    if contains_swear_word:
        # find the word that is in the code_string
        swear_word = next((word for word in swear_words if word in code_string), None)
        logging.info(f"Parameter estimator code contains swear word: {swear_word}, skipping.")
        logging.info(f"Code string was:\n{code_string}")
        return None, None, pe_metadata
    code_string = re.sub(r"def\s+parameter_estimator_v\d+\s*\(", "def parameter_estimator(", code_string)
    code_string = re.sub(r"def\s+parameter_estimator_prev\s*\(", "def parameter_estimator(", code_string)
    func = utils.str_to_func(code_string, 'parameter_estimator')

    if func is None:
        logging.info("Failed to parse parameter estimator code, skipping.")
        return None, None, pe_metadata

    logging.info(
        f"Generated initial parameter estimator candidate (island={island_id}, batch={batch_id}):\n"
        f"Prompt:\n{prompt}\n\n"
        f"Parameter Estimator (initial):\n{code_string}\n"
    )

    if refine_rounds <= 0 or model_fn is None:
        return code_string, func, pe_metadata

    iter_label = "?" if iteration is None else str(iteration)
    logging.info(
        f"Param-est refinement start (iter={iter_label}, island={island_id}, "
        f"batch={batch_id}, rounds={refine_rounds})."
    )
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
    current_loss, current_params, _, _ = objective_with_timeout(
        model=model_fn,
        param_estimator=current_func,
        x=x,
        y=y,
        loss_fn=loss_fn,
        fit_params=False,  # Don't fit parameters during refinement evaluation
        param_penalty_weight=param_penalty_weight,
    )

    if current_loss < best_loss:
        best_loss = current_loss
        best_code = current_code
        best_func = current_func

    for r in range(refine_rounds):
        logging.info(
            f"Param-est refinement round {r+1}/{refine_rounds} "
            f"(iter={iter_label}, island={island_id}, batch={batch_id})."
        )
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
                X=x[0],
                Y=y[0],
                programs_list=[
                    {
                        "model": model_fn,
                        "params": current_params,
                        "losses": np.full(x[0].shape[0], float(current_loss)),
                    }
                ],
                X_eval=x_eval,
                save_path=img_path,
                labels=['PE'],
            )
            with open(img_path, "rb") as f:
                img_bytes = f.read()
        except Exception as e:
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
        logging.info(
            f"Param-est refinement prompt (iter={iter_label}, island={island_id}, "
            f"batch={batch_id}, round={r+1}):\n{refine_prompt}\n"
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
            logging.info("No code block found in refinement output; keeping current estimator.")
            continue
        if swear_words and any(word in new_code for word in swear_words):
            logging.info("Refinement code contains banned words; skipping.")
            continue

        new_code = re.sub(r"def\s+parameter_estimator_v\d+\s*\(", "def parameter_estimator(", new_code)
        new_code = re.sub(r"def\s+parameter_estimator_prev\s*\(", "def parameter_estimator(", new_code)
        logging.info(
            f"Param-est refinement code (iter={iter_label}, island={island_id}, "
            f"batch={batch_id}, round={r+1}):\n{new_code}\n"
        )
        new_func = utils.str_to_func(new_code, 'parameter_estimator')
        if new_func is None:
            logging.info("Failed to parse refined parameter estimator; keeping current.")
            continue

        new_loss, _, _, _ = objective_with_timeout(
            model=model_fn,
            param_estimator=new_func,
            x=x,
            y=y,
            loss_fn=loss_fn,
            fit_params=False,  # Don't fit parameters during refinement evaluation
            param_penalty_weight=param_penalty_weight,
        )

        logging.info(
            f"Param-est refinement eval (iter={iter_label}, island={island_id}, "
            f"batch={batch_id}, round={r+1}): loss={new_loss:.6g}."
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
            logging.info(
                f"Refinement did not improve loss ({new_loss:.4f} >= {current_loss:.4f}); keeping current."
            )

    return best_code, best_func, pe_metadata


async def translate_to_jax(code_string: str, client, prompt_manager, llm_name='gemini-2.0-flash-lite') -> tuple[str, callable]:
    """
    Translate a model code string to a JAX-compatible implementation via LLM.

    Args:
        code_string (str): Source code containing the NumPy model definition.
        client: The LLM client.
        prompt_manager: PromptManager used to build translation prompts.
        llm_name (str): LLM model name for translation.

    Returns:
        tuple[str | None, callable | None]: ``(jax_code_string, jax_callable)``.
            Returns ``(None, None)`` when translation cannot be produced/parsed.
    """
    if code_string is None:
        logging.info("No neuron model code string provided for translation.")
        return None, None
    
    prompt = prompt_manager.get_jax_translator_prompt(code_string)
    logging.info(f"JAX translation prompt:\n{prompt}\n")
    if prompt is None:
        return None, None
    
    jax_code_string = await llm_helper.call_llm_async(prompt, client=client, model_name=llm_name, temperature=0)
    logging.info(f"JAX translation output:\n{jax_code_string}\n")
    jax_code_string = utils.extract_code_block(jax_code_string)
    model_name = prompt_manager.get_model_name()
    func = utils.str_to_func(jax_code_string, model_name)
    return jax_code_string, func


def _run_translation_check_on_eval(
    np_func,
    jax_func,
    param_estimator,
    x_train_trials,
    y_train_trials,
    x_eval,
    max_samples: int = 3,
    max_eval_trials: int = 32,
):
    """
    Validate NumPy/JAX numerical agreement on a subset of evaluation points.

    Parameters are first estimated from observed train-trial data. The resulting
    parameter vectors are then used to compare NumPy vs JAX predictions on
    ``x_eval`` for a small subset of samples and eval trials.

    Args:
        np_func (callable): Original NumPy model.
        jax_func (callable): Translated JAX model.
        param_estimator (callable): Parameter estimator used to derive test params.
        x_train_trials: Train-trial input data for parameter estimation.
        y_train_trials: Train-trial output data for parameter estimation.
        x_eval (array-like): Evaluation grid ``(n_samples, n_features, n_eval_trials)``.
        max_samples (int): Maximum number of samples to check.
        max_eval_trials (int): Maximum eval trials per sample to compare.

    Returns:
        None: Raises on mismatch; otherwise completes silently.
    """
    x_obs = np.asarray(ensure_inputs(x_train_trials).to_tensor())
    y_obs = np.asarray(ensure_outputs(y_train_trials).to_tensor())
    n_samples = min(x_obs.shape[0], y_obs.shape[0])
    if n_samples <= 0:
        raise ValueError("No samples available for translation check.")

    # Use a small slice of real observed trials instead of synthetic eval grids.
    n_trials = x_obs.shape[2]
    n_eval_trials = min(5, int(max_eval_trials), n_trials)
    rng = np.random.default_rng(0)
    if n_eval_trials <= 0:
        raise ValueError("No trials available for translation check.")
    if n_eval_trials == n_trials:
        trial_idx = np.arange(n_trials)
    else:
        trial_idx = rng.choice(n_trials, size=n_eval_trials, replace=False)
    eval_points = x_obs[:, :, trial_idx]

    n_check = min(max_samples, n_samples)
    sample_idx = np.linspace(0, n_samples - 1, num=n_check, dtype=int)
    params_subset = compute_initial_params(
        param_estimator,
        np_func,
        x_obs[sample_idx],
        y_obs[sample_idx],
    )
    if params_subset is None:
        raise ValueError("Failed to compute parameters for translation check.")

    utils.check_jax_translation(
        np_func=np_func,
        jax_func=jax_func,
        eval_points=eval_points[sample_idx],
        params=params_subset,
        max_eval_trials=max_eval_trials,
    )


def _append_generation_record(filepath, record):
    """Append a single generation record as a JSON line."""
    with open(filepath, 'a') as f:
        f.write(json.dumps(record, default=str) + '\n')


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
        use_chat_mode=False,  # If True, use persistent chat sessions per island (expensive)
        chat_token_limit=50000,  # Max tokens per chat before auto-summarize and reset. 0 = unlimited
        param_estimator_refinement_rounds=0,
        exploration_topology = [1, 2, 3, 4, 5, 6, 7, 0], exploitation_topology = [1, 2, 3, 4, 5, 6, 7, 0],
        tiny_lm_name = 'gemini-2.0-flash-lite', little_lm_name = 'gemini-2.0-flash', large_lm_name = 'gemini-2.5-flash',
        use_large_every = 3, max_iter = 1_000, learning_rate = 3e-3,
        numpy_programs = None, param_estimators = None,
        X = None, Y = None, X_eval = None,
        plot_model_fits = None, loss_fn = None, 
        prompt_manager = None, trial_batch_size = None, swear_words = None,
        open_family_tree = False,
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
        use_chat_mode (bool): Use persistent per-island chat sessions if True.
        chat_token_limit (int): Chat token cap before summarization/reset.
        param_estimator_refinement_rounds (int): Refinement rounds for new estimators.
        exploration_topology (list[int]): Migration destination map during explore.
        exploitation_topology (list[int]): Migration destination map during exploit.
        tiny_lm_name (str): Model name for translation/small utility calls.
        little_lm_name (str): Default generation model.
        large_lm_name (str): Larger generation model.
        use_large_every (int): Use large model every N iterations.
        max_iter (int): Max optimization steps inside ``objective``.
        learning_rate (float): Optimizer learning rate inside ``objective``.
        numpy_programs (list[callable]): Seed NumPy model functions.
        param_estimators (list[callable]): Seed parameter estimators.
        X: Split input container with shape ``(2, 2)`` where
            ``X[0,*]`` is train-sample split and ``X[1,*]`` is test-sample split.
        Y: Split output container with the same structure as ``X``.
        X_eval (np.ndarray): Evaluation grid used for diagnostics/comparison.
        plot_model_fits (callable | None): Optional plotting callback.
        loss_fn (callable | None): Objective loss function override.
        prompt_manager: PromptManager for all prompt construction.
        trial_batch_size (int | None): Trial batching size used by ``objective``.
        swear_words (list[str] | None): Blacklist for generated estimator code.
        open_family_tree (bool): Open the combined family tree HTML at the end of the run.
        random_seed (int): Run seed for deterministic split-dependent operations.

    Returns:
        str: Path to the run output directory containing logs, plots, and program DBs.
    """
    has_spec_plotter = plot_model_fits is not None

    # load api keys
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    # Initialize IslandChatManager if using chat mode
    island_chat_manager = None
    if use_chat_mode:
        # Create IslandChatManager with mode-aware system instructions
        island_chat_manager = llm_helper.IslandChatManager(
            client=client,
            get_system_instruction=prompt_manager.get_system_instruction,
            small_model_name=little_lm_name,
            large_model_name=large_lm_name,
            explore_temperature=1.5,  # Higher temperature for creative exploration
            exploit_temperature=0.7,  # Lower temperature for focused exploitation
            thinking_budget_fraction=1.0,
            chat_token_limit=chat_token_limit,
            batch_size=batch_size
        )
        logging.info(f"Initialized IslandChatManager with models {little_lm_name} / {large_lm_name}")
        print(f"Chat mode enabled: using persistent chat sessions per (island, batch) pair")
        print(f"  - Total chats: {n_islands * batch_size} ({n_islands} islands × {batch_size} batches)")
        print(f"  - Explore: T=1.5, Exploit: T=0.7")
        print(f"  - Small model: {little_lm_name}, Large model: {large_lm_name}")
        print(f"  - Token limit per chat: {chat_token_limit} (0 = unlimited)")
    else:
        logging.info("Chat mode disabled: using independent LLM queries")
        print("Chat mode disabled: using independent LLM queries")

    X = np.asarray(X, dtype=object)
    Y = np.asarray(Y, dtype=object)
    if X.shape != (2, 2):
        raise ValueError(f"X must have shape (2, 2), got {X.shape}.")
    if Y.shape != (2, 2):
        raise ValueError(f"Y must have shape (2, 2), got {Y.shape}.")

    n_training_samples, _, n_training_trials = X[0, 0].shape
    n_test_samples, _, n_test_trials = X[1, 1].shape
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
    translation_tasks = [
        translate_to_jax(code_string, client, prompt_manager, tiny_lm_name)
        for code_string in seed_code_strings
    ]
    jax_results = await asyncio.gather(*translation_tasks)

    jax_programs = []
    for i, (_, jax_func) in enumerate(jax_results):
        _run_translation_check_on_eval(
            np_func=numpy_programs[i],
            jax_func=jax_func,
            param_estimator=param_estimators[i],
            x_train_trials=X[0, 0],
            y_train_trials=Y[0, 0],
            x_eval=X_eval_train,
        )
        jax_programs.append(jax_func)

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
    model_name = prompt_manager.get_model_name()
    for i in range(2):
        # get the program, parameter estimator, and jax program
        program_num = numpy_programs[i]
        param_est = param_estimators[i]
        program_jax = jax_programs[i]
        # score the initial program
        loss_init, params_init, loss, params = objective_with_timeout(
            program_jax,
            param_est,
            x=X[0],
            y=Y[0],
            loss_fn=loss_fn,
            fit_params=fit_params,
            param_penalty_weight=param_penalty_weight,
            learning_rate=learning_rate,
            use_param_estimator=use_param_estimator,
            max_iter=max_iter,
            trial_batch_size=trial_batch_size,
        )
        print(f"Initial program {i + 1} loss before parameter fitting: {loss_init:.2f} and loss after fitting: {loss:.2f}")

        seed_losses[i] = loss
        program_code_string = utils.format_function_source(
            program_num, f'{model_name}_v{i+1}', 'import numpy as np'
        )
        parameter_estimator_code_string = utils.format_function_source(
            param_est, f'parameter_estimator_v{i+1}', 'import numpy as np'
        )
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
                                    'llm_name': None,
                                    'params': [params],
                                    'initial_loss': loss_init,
                                    'initial_params': [params_init],
                                    'parent1_id': None,
                                    'parent2_id': None,
                                    'evaluation_matrix': [y_eval]})
        initial_programs = pd.concat([initial_programs, new_program_df], ignore_index=True)
        print(f"Initial program {i + 1} loss: {loss:.2f}")

    # seed each island with the initial programs
    for i in range(n_islands):
        islands[i] = pd.concat([islands[i], initial_programs], ignore_index=True)

    # Reset logging configuration
    log_file = os.path.join(full_dir, 'hypothesis_engine.log')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')
    
    # Log the IslandChatManager configuration (including system instruction)
    if island_chat_manager is not None:
        island_chat_manager.log_configuration()
    
    if has_spec_plotter:
        seed_programs_list = _programs_df_to_programs_list(
            initial_programs,
            loss_func=loss_fn,
            x=X[0, 1],
            y=Y[0, 1],
            complexity_penalty=param_penalty_weight,
        )
        plot_model_fits(
            X=X[0, 0],
            Y=Y[0, 0],
            programs_list=seed_programs_list,
            X_eval=X_eval_train,
            save_path=os.path.join(image_prompts_dir, 'initial_programs.png'),
            labels=['seed_1', 'seed_2'],
        )

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
        if use_large_every > 0 and i % use_large_every == 0:
            llm_name = large_lm_name
            logging.info(f"Using large LLM: {llm_name}")
        else:
            llm_name = little_lm_name
            logging.info(f"Using little LLM: {llm_name}")
        use_large_model = (llm_name == large_lm_name)  # Track whether using large model
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
        # generate new programs
        model_generation_tasks = [generate_new_model(islands[island_idx], 
                                                    llm_name=llm_name, 
                                                    client=client, 
                                                    mode=mode, 
                                                    k_max=k_max, 
                                                    temp=temperature,
                                                    x=X[0, 0],
                                                    y=Y[0, 0],
                                                    x_eval=X_eval_train,
                                                    prompt_manager=prompt_manager,
                                                    img_dir=model_image_dirs[island_idx, j],
                                                    plot_model_fits=plot_model_fits,
                                                    island_chat_manager=island_chat_manager,
                                                    island_id=island_idx,
                                                    batch_id=j,
                                                    loss_fn=loss_fn,
                                                    loss_x=X[0, 1],
                                                    loss_y=Y[0, 1],
                                                    complexity_penalty=param_penalty_weight,
                                                    use_large_model=use_large_model) 
                                         for island_idx in range(n_islands) for j in range(batch_size)]
        logging.info(f"Generating {n_islands * batch_size} new programs... LLM Model: {llm_name}, mode: {mode}, temperature: {temperature:.2f}")
        print(f"Generating {n_islands * batch_size} new programs... LLM Model: {llm_name}, mode: {mode}, temperature: {temperature:.2f}")
        model_results = await asyncio.gather(*model_generation_tasks)
        model_code_strings = [result[0] for result in model_results]
        model_prompts = [result[1] for result in model_results]
        model_llm_responses = [result[2] for result in model_results]
        parent_ids = [result[3] for result in model_results]
        
        # convert to jax
        model_function_translation_tasks = [translate_to_jax(code_string, client, prompt_manager, tiny_lm_name) for code_string in model_code_strings]
        jax_results = await asyncio.gather(*model_function_translation_tasks)
        model_results = [(model_code_strings[j], model_prompts[j], model_llm_responses[j], jax_results[j][0], jax_results[j][1]) for j in range(n_islands * batch_size)]
        
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
                model_fn=model_results[island_idx * batch_size + j][4],
                llm_name=little_lm_name,
                client=client,
                x=X[0],
                y=Y[0],
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
                plot_model_fits=plot_model_fits,
                x_eval=X_eval_train,
                image_refinement_dir=image_param_est_refine_dir,
            )
            for island_idx in range(n_islands)
            for j in range(batch_size)
        ]

        logging.info(
            f"Generating {n_islands * batch_size} parameter estimators "
            f"(LLM={little_lm_name}, mode={mode}, T={temperature:.2f})"
        )
        logging.info(f"Generating {n_islands * batch_size} new parameter estimators... Model: {little_lm_name}, mode: {mode}, temperature: {temperature:.2f}")
        param_est_results = await asyncio.gather(*param_estimation_tasks)
        # combine results
        island_results = [[model_results[island_idx * batch_size + j] + param_est_results[island_idx * batch_size + j] for j in range(batch_size)] for island_idx in range(n_islands)]

        # now loop through the results and compute losses
        success_rate = 0.0
        for island_idx, j in np.ndindex(n_islands, batch_size):
            logging.info(f"id={i},{island_idx},{j}")
            model_code_string, prompt, model_llm_response, model_code_string_jax, model_new, param_est_code_string, param_est_new, pe_metadata = island_results[island_idx][j]
            parent1_id, parent2_id = parent_ids[island_idx * batch_size + j]

            logging.info(f"Prompt: \n{prompt}\n")
            logging.info(f"Model: \n{model_code_string}\n")
            logging.info(f"Model (JAX): \n{model_code_string_jax}\n")
            logging.info(f"Parameter Estimator: \n{param_est_code_string}\n")

            if model_new is None or param_est_new is None:
                logging.info(f"Skipping island {island_idx}, batch {j} due to LLM generation failure.")
                logging.info('-' * 50)
                continue

            model_name = prompt_manager.get_model_name()
            model_np = utils.str_to_func(model_code_string, model_name)
            if model_np is None:
                logging.info(
                    f"Skipping island {island_idx}, batch {j}: failed to parse NumPy model."
                )
                logging.info('-' * 50)
                continue
            try:
                _run_translation_check_on_eval(
                    np_func=model_np,
                    jax_func=model_new,
                    param_estimator=param_est_new,
                    x_train_trials=X[0, 0],
                    y_train_trials=Y[0, 0],
                    x_eval=X_eval_train,
                )
            except Exception as e:
                logging.info(
                    f"Skipping island {island_idx}, batch {j}: JAX translation check failed: {e}"
                )
                logging.info('-' * 50)
                continue
            
            initial_loss, initial_params, loss, optimized_params = objective_with_timeout(
                model_new,
                param_est_new,
                x=X[0],
                y=Y[0],
                loss_fn=loss_fn,
                param_penalty_weight=param_penalty_weight,
                fit_params=fit_params,
                use_param_estimator=use_param_estimator,
                max_iter=max_iter,
                trial_batch_size=trial_batch_size,
            )
            if loss == FAILED_PROGRAM_COST:
                logging.info('-' * 50)
                continue

            y_eval = utils.compute_evaluation_matrix(
                model_new,
                optimized_params,
                eval_points=X_eval_train,
            )
            logging.info(f"Loss: {loss:.2f}\n")

            train_fit_path = None
            test_fit_path = None
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
                    X=X[0, 0],
                    Y=Y[0, 0],
                    programs_list=[
                        {
                            "model": model_new,
                            "params": initial_params_plot,
                            "losses": np.full(X[0, 0].shape[0], float(initial_loss)),
                        },
                        {
                            "model": model_new,
                            "params": optimized_params_plot,
                            "losses": np.full(X[0, 0].shape[0], float(loss)),
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
                    x=X[0, 0],
                    y=Y[0, 0],
                    complexity_penalty=param_penalty_weight,
                )
                plot_model_fits(
                    X=X[0, 0],
                    Y=Y[0, 0],
                    programs_list=train_programs_list,
                    X_eval=X_eval_train,
                    save_path=train_fit_path,
                    labels=['PE', 'GD'],
                )
                test_fit_path = os.path.join(image_family_tree_fits_dir, f'iter_{i}_island_{island_idx}_batch_{j}_test_fit.png')
                try:
                    test_initial_loss, test_initial_params, test_final_loss, test_params = objective_with_timeout(
                        model_new,
                        param_est_new,
                        x=X[1],
                        y=Y[1],
                        loss_fn=loss_fn,
                        param_penalty_weight=param_penalty_weight,
                        fit_params=fit_params,
                        use_param_estimator=use_param_estimator,
                        max_iter=max_iter,
                        trial_batch_size=trial_batch_size,
                    )
                    if test_final_loss == FAILED_PROGRAM_COST:
                        raise ValueError("test objective failed")
                except Exception as e:
                    logging.info(
                        "Skipping test fit plot; failed to compute test parameters "
                        f"(iter={i}, island={island_idx}, batch={j}): {e}"
                    )
                else:
                    test_programs_df = pd.DataFrame({
                        "program": [model_new, model_new],
                        "params": [test_initial_params, test_params],
                    })
                    test_programs_list = _programs_df_to_programs_list(
                        test_programs_df,
                        loss_func=loss_fn,
                        x=X[1, 1],
                        y=Y[1, 1],
                        complexity_penalty=param_penalty_weight,
                    )
                    plot_model_fits(
                        X=X[1, 1],
                        Y=Y[1, 1],
                        programs_list=test_programs_list,
                        X_eval=X_eval_test,
                        save_path=test_fit_path,
                        labels=['PE', 'GD'],
                    )

            param_summary = utils.params_tree_summary(
                optimized_params,
                n_samples=X[0, 0].shape[0],
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
                                        'llm_name': llm_name,
                                        'params': [optimized_params],
                                        'initial_loss': initial_loss,
                                        'initial_params': [initial_params],
                                        'parent1_id': [parent1_id],
                                        'parent2_id': [parent2_id],
                                        'evaluation_matrix': [y_eval]
                                        })
            
            islands[island_idx] = pd.concat([islands[island_idx], new_program_df], ignore_index=True)

            # Write generation record to JSON log for family tree
            _append_generation_record(generation_log_path, {
                "iteration_number": i,
                "birth_island": island_idx,
                "batch_index": j,
                "parent1_id": list(parent1_id),
                "parent2_id": list(parent2_id),
                "train_loss": float(loss),
                "initial_loss": float(initial_loss),
                "model_prompt": prompt,
                "model_llm_response": model_llm_response,
                "model_code_numpy": model_code_string,
                "model_code_jax": model_code_string_jax,
                "param_est_prompt": pe_metadata.get("initial_prompt"),
                "param_est_llm_response": pe_metadata.get("initial_response"),
                "param_est_code": param_est_code_string,
                "param_est_refinement_prompts": pe_metadata.get("refinement_prompts", []),
                "param_est_refinement_responses": pe_metadata.get("refinement_responses", []),
                "llm_name": llm_name,
                "temperature": float(temperature),
                "mode": mode,
                "use_large_model": use_large_model,
                "image_prompt_path": model_image_dirs[island_idx, j],
                "train_fit_image_path": train_fit_path,
                "test_fit_image_path": test_fit_path,
            })

            success_rate += 1 / (n_islands * batch_size)
            print(f"iteration {i}, island {island_idx}, batch {j}, loss: {loss:.2f}", flush=True)
            print('-' * 50, flush=True)
            logging.info("-" * 50)
        print("Success rate:", success_rate, flush=True)

        # sort each island by loss
        for island_idx in range(n_islands):
            islands[island_idx] = islands[island_idx].sort_values(by='train_loss').reset_index(drop=True)
        logging.info(f"Iteration {i} complete. The proportion of programs that successfully ran and received a loss is {success_rate:.2f}.")
        logging.info('-' * 50)
        # migrate and prune programs (better here for temperature to be in [0, 1] range)
        islands = genetic_helpers.perform_island_deduplication(islands, overlap_threshold=int(0.75 * critical_population_size))
        islands = genetic_helpers.perform_population_pruning(islands, critical_population_size=critical_population_size - n_migrants,
                                                min_wise_population_size=min_wise_population_size,)
        islands = genetic_helpers.perform_probabilistic_migration(islands, 
                                                                  n_migrants=n_migrants,
                                                                  destination_islands=exploration_topology if mode == 'explore' else exploitation_topology, 
                                                                  temperature=(temperature - 1.0)**4)

                                                             
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
                    x=X[0, 1],
                    y=Y[0, 1],
                    complexity_penalty=param_penalty_weight,
                )
                plot_model_fits(
                    X=X[0, 0],
                    Y=Y[0, 0],
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
                x=X[0, 1],
                y=Y[0, 1],
                complexity_penalty=param_penalty_weight,
            )
            plot_model_fits(
                X=X[0, 0],
                Y=Y[0, 0],
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
            program = islands[island_idx].iloc[j]
            model = program['program']
            param_estimator = program['parameter_estimator']
            # compute the test loss
            _, _, test_loss, optimized_params = objective_with_timeout(
                model,
                param_estimator,
                x=X[1],
                y=Y[1],
                loss_fn=loss_fn,
                fit_params=fit_params,
                max_iter=max_iter,
                param_penalty_weight=param_penalty_weight,
                use_param_estimator=use_param_estimator,
                trial_batch_size=trial_batch_size,
            )
            islands[island_idx].at[j, 'test_loss'] = test_loss
            islands[island_idx].at[j, 'params'] = optimized_params
            islands[island_idx].at[j, 'mean_loss'] = np.mean(test_loss)
            print(f"Test loss: {test_loss:.2f}")

    # group all islands together and save
    combined_dir = os.path.join(base_dir, date_stamp, time_stamp, 'combined')
    os.makedirs(combined_dir, exist_ok=True)
    combined_programs_dataframe = pd.concat(islands, ignore_index=True)
    combined_programs_dataframe = genetic_helpers.remove_duplicates(combined_programs_dataframe, mode='complicated', loss_tol=0.025, cosine_tol=0.99, loss_type='test_loss')
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

    # ---------------------------
    # save losses plot    
    if has_spec_plotter:
        plot_train_vs_test_loss_shared(
            programs_df=combined_programs_dataframe,
            island_labels=[f'Island {i}' for i in range(n_islands)] + ['garden_of_eden'],
            save_path=os.path.join(combined_dir, 'train_vs_test_loss.png'),
        )
    
    # ---------------------------
    df_list = [combined_programs_dataframe] + islands
    combined_dir = [os.path.join(base_dir, date_stamp, time_stamp, "combined")] 
    island_dirs = [os.path.join(base_dir, date_stamp, time_stamp, f'island_{i}') for i in range(n_islands)]
    df_dirs = combined_dir + island_dirs

    if has_spec_plotter:
        for i, df in enumerate(df_list):
            df = df.head(3)
            df = df.sort_values(by='test_loss', ascending=False).reset_index(drop=True)
            programs_list = _programs_df_to_programs_list(
                df,
                loss_func=loss_fn,
                x=X[1, 1],
                y=Y[1, 1],
                complexity_penalty=param_penalty_weight,
            )
            plot_model_fits(
                X=X[1, 1],
                Y=Y[1, 1],
                programs_list=programs_list,
                X_eval=X_eval_test,
                save_path=os.path.join(df_dirs[i], 'top_model_fits.png'),
            )
            # Plot top models separately using the same plot_model_fits pathway.
            for j in range(min(3, len(df))):
                model_df = df.iloc[[j]].copy().reset_index(drop=True)
                plot_model_fits(
                    X=X[1, 1],
                    Y=Y[1, 1],
                    programs_list=_programs_df_to_programs_list(
                        model_df,
                        loss_func=loss_fn,
                        x=X[1, 1],
                        y=Y[1, 1],
                        complexity_penalty=param_penalty_weight,
                    ),
                    X_eval=X_eval_test,
                    save_path=os.path.join(df_dirs[i], f'top_model_fit_{min(3, len(df)) - j}.png'),
                    labels=['model'],
                )
    
    # Generate family tree visualizations
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
