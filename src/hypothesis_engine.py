import inspect
import re
import os
import logging
import asyncio
import numpy as np
import jax, jax.numpy as jnp
import timeout_decorator
import optax
import pandas as pd
from pathlib import Path
from . import utils, llm_helper, loss_functions
from . import genetic_helpers_v2 as genetic_helpers  # Using v2 with compatibility API
from .data_structures import ensure_inputs, ensure_outputs
from .evolution_diagnostics import plot_train_vs_test_loss as plot_train_vs_test_loss_shared
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(relativeCreated)dms | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def compute_initial_params(param_estimator, model, x, y) -> jnp.ndarray:
    """
    Compute initial parameters for the model using the provided parameter estimator. 
    The parameter estimator is written in numpy, but the model is written in JAX. 
    So the data x and y will be numpy arrays, but the output will be a JAX array.
    
    Args:
        param_estimator (function): Function to estimate initial parameters for the model.
                                    Signature: param_estimator(X, response) -> params
                                    where X has shape (n_features, n_trials) for a single sample,
                                    and response has shape (n_trials,) for scalar or (n_targets, n_trials) for vectorized.
        model (function): The model which predicts neural activity from stimuli and free parameters.
                                 Signature: model(X, *params) -> activity
                                 where X has shape (n_features, n_trials) for a single sample.
        x (np.ndarray): Input data, shape (n_samples, n_features, n_trials).
        y (np.ndarray): Response data, shape (n_samples, n_trials) for scalar
                        or (n_samples, n_targets, n_trials) for vectorized.
    Returns:
        jnp.ndarray: The estimated parameters for each sample, shape (n_samples, n_params).
                     If the parameter estimation fails, returns an array of default parameters based on the model's signature.
                     If this also fails, returns None.
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

    try:
        # any call taking >5s will raise timeout_decorator.TimeoutError
        # xi has shape (n_features, n_trials)
        # yi has shape (n_trials,) for scalar or (n_targets, n_trials) for vectorized
        return jnp.array([
            _safe_estimate(param_estimator, x[i], _estimator_response_arg(y[i]))
            for i in range(y.shape[0])
        ])
    except timeout_decorator.TimeoutError:
        logging.warning("param_estimator timed out, falling back to defaults")
    except Exception as e:
        logging.info(f"Error during parameter estimation: {e}")

    # If parameter estimation fails, compute default parameters based on the model's signature
    params = compute_default_params(model)
    if params is not None:
        # default params is a 2D array with shape (1, n_params), so we need to repeat it for each sample
        n_samples = y.shape[0]
        return jnp.repeat(params, n_samples, axis=0)
    else:
        logging.info("Error: Unable to compute default parameters for the neuron model.")
        return None


def compute_default_params(model) -> jnp.ndarray:
    """
    Compute default parameters for the model based on its signature.
    Args:
        model (function): The model which predicts neural activity from stimuli and free parameters.
                                 Signature: model(X, *params) -> activity
                                 where X has shape (n_features, n_trials) for a single sample.
    Returns:
        jnp.ndarray: The default parameters for the model, shape (1, n_params).
                     If the parameter estimation fails, returns None.
    """
    try:
        sig = inspect.signature(model)
        # First parameter is the input (X or theta), skip it
        all_param_names = list(sig.parameters.keys())
        # The first param is the input (could be named 'X', 'theta', or anything)
        # All subsequent params are the model parameters to fit
        param_names = all_param_names[1:] if all_param_names else []
        defaults = [sig.parameters[n].default if sig.parameters[n].default is not inspect._empty else 0.0 for n in param_names]
        default_arr = jnp.array(defaults, dtype=np.float32)
        return default_arr.reshape(1, -1)  # reshape to (1, n_params)
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
    Validate that model output has the expected shape.
    
    For scalar outputs (n_targets=1):
        - Expected shape: (n_trials,) - 1D array
        - If allow_1d_for_single_target=True, also accepts 2D (1, n_trials)
    
    For vectorized outputs (n_targets>1):
        - Expected shape: (n_targets, n_trials) - 2D array
    
    Args:
        output: The model output to validate
        expected_n_trials: Expected number of trials (last dimension)
        expected_n_targets: Expected number of targets. Default 1 (scalar output).
        allow_1d_for_single_target: If True and n_targets=1, accept both 1D and 2D output.
    
    Returns:
        tuple[bool, str]: (is_valid, error_message)
            - is_valid: True if output shape is correct
            - error_message: Empty string if valid, otherwise describes the issue
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
    initial_params: jnp.ndarray,
    n_samples: int,
    expected_n_targets: int = 1,
    n_validation_samples: int = 10,
) -> tuple[bool, str]:
    """
    Validate that a model can execute correctly and produces expected output shapes.
    
    Tests the model on a random subset of samples to verify:
    1. Model runs without exceptions
    2. Model is compatible with JAX JIT and tracing
    3. Output shape matches expected (n_trials,) for scalar or (n_targets, n_trials) for vectorized
    
    Args:
        model: The model function to validate
        x_data: Input data of shape (n_samples, n_features, n_trials)
        initial_params: Initial parameters of shape (n_samples, n_params)
        n_samples: Number of samples
        expected_n_targets: Expected number of output targets (1 for scalar, >1 for vectorized)
        n_validation_samples: Number of random samples to test (default 10)
    
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    try:
        model_jit = jax.jit(model)
        test_n_trials = x_data.shape[2]
        
        for sample_idx in np.random.choice(n_samples, size=min(n_validation_samples, n_samples), replace=False):
            # Validate with concrete values: x_data[sample_idx] is (n_features, n_trials)
            output = model_jit(x_data[sample_idx], *initial_params[sample_idx])
            
            is_valid, error_msg = validate_model_output(output, test_n_trials, expected_n_targets)
            if not is_valid:
                return False, error_msg
            
            # Validate with abstract tracer values
            jax.eval_shape(model_jit, x_data[sample_idx], *initial_params[sample_idx])
        
        return True, ""
    except Exception as e:
        return False, f"Model failed to run or is incompatible with JAX tracing: {e}"


def _default_model_loss_fn(model, x_i, y_i, params):
    """
    Default per-sample loss: mean quadratic loss over all targets and trials.
    """
    pred = model(x_i, *params)
    if y_i.ndim == 1:
        y_i = y_i[None, :]
    if pred.ndim == 1:
        pred = pred[None, :]
    return jnp.mean(loss_functions.quadratic_loss(pred, y_i))


def objective(model, param_estimator, x, y,
              loss_fn=None, param_penalty_weight=0.1, fit_params=True,
              FAILED_PROGRAM_COST=jnp.inf, max_iter=1_000, learning_rate=3e-3,
              use_param_estimator=True, trial_batch_size=None) -> tuple[float, jnp.ndarray, float, jnp.ndarray]:
    """
    Calculate model loss using the unified Outputs representation.
    
    Supports both scalar and multi-target outputs through canonical shape
    (n_samples, n_targets, n_trials).
    
    The loss is computed as a weighted sum of per-target MSE:
        loss = sum_t(weight[t] * mean((pred[t] - true[t])^2))
    
    Args:
        model (function): The model which predicts neural activity from inputs
                          and free parameters (for a single sample).
                          Signature: model(X, *params) -> activity
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
                          Signature: loss_fn(model, x_i, y_i, params) -> scalar.
                          If None, defaults to quadratic loss.
        param_penalty_weight (float): Weight for the penalty on the number of parameters. Default is 0.1.
        fit_params (bool): Whether to fit the parameters of the model. Default is True.
        FAILED_PROGRAM_COST (float): Cost assigned to failed models. Default is np.inf.
        max_iter (int): Maximum number of iterations for optimization. Default is 1_000.
        use_param_estimator (bool): Whether to use the parameter estimator to compute initial parameters. Default is True.
        trial_batch_size (int | None): Ignored. Trial batching is disabled.

    Returns:
        tuple[
            - float: The cross-validated loss of the model with initial parameters,
            - jnp.ndarray: The initial parameters (n_samples, n_params).
            - float: The average loss on test set after optimization.
                     Returns FAILED_PROGRAM_COST if the model fails.
            - jnp.ndarray: The optimized parameters (n_samples, n_params).
    """
    t_start = time.time()
    if loss_fn is None:
        loss_fn = _default_model_loss_fn
    
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
    
    # Compute initial parameters
    # param_estimator receives y as (n_targets, n_trials) for each sample
    if use_param_estimator:
        initial_params = compute_initial_params(param_estimator, model, np.asarray(x_train), np.asarray(y_train))
    else:
        initial_params = compute_default_params(model)
        if initial_params is not None:
            initial_params = jnp.repeat(initial_params, n_samples, axis=0)
    
    # Fail immediately if initial_params is None or not a JAX array
    if initial_params is None or not isinstance(initial_params, jnp.ndarray):
        logging.info("Error: initial_params should be a JAX array.")
        return FAILED_PROGRAM_COST, jnp.zeros((n_samples, 0)), FAILED_PROGRAM_COST, jnp.zeros((n_samples, 0))
    if initial_params.ndim != 2 or initial_params.shape[0] != n_samples:
        logging.info(f"Error: initial_params should be a 2D array with shape ({n_samples}, n_params).")
        return FAILED_PROGRAM_COST, jnp.zeros((n_samples, 0)), FAILED_PROGRAM_COST, jnp.zeros((n_samples, 0))
    
    # Fail immediately if fit_params is True and non-numeric params
    n_params = initial_params.shape[1]
    all_numeric = (initial_params.dtype.kind in 'biufc' and 
                   jnp.all(jnp.isfinite(initial_params)))
    if fit_params and not all_numeric:
        logging.info("Error: Cannot fit non-numeric parameters.")
        return FAILED_PROGRAM_COST, jnp.zeros((n_samples, n_params)), FAILED_PROGRAM_COST, jnp.zeros((n_samples, n_params))
    
    # Validate model execution and output shape
    is_valid, error_msg = validate_model_execution(
        model, x_train, initial_params, n_samples,
        expected_n_targets=n_targets,
        n_validation_samples=10
    )
    if not is_valid:
        logging.info(f"Model validation failed: {error_msg}")
        return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params

    # Per-sample loss function
    def loss_single_sample(params, x_i, y_i):
        return loss_fn(model, x_i, y_i, params)

    # Vectorize over samples
    # params: (n_samples, n_params), x: (n_samples, n_features, n_trials_x), y: (n_samples, n_targets, n_trials_y)
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
    def loss_single_batch(params_2d, x_batch, y_batch):
        """Compute sum of losses for one batch (JIT-compiled)."""
        batch_losses = loss_total(params_2d, x_batch, y_batch)  # (n_samples,)
        return jnp.sum(batch_losses)

    # Combined loss and gradient computation - more efficient than separate calls
    loss_and_grad_single_batch = jax.jit(jax.value_and_grad(loss_single_batch))

    # JIT-compiled eval (nansum to ignore NaN losses from bad model outputs)
    @jax.jit
    def eval_single_batch(params_2d, x_batch, y_batch):
        """Compute nansum of losses for one batch (JIT-compiled, no grad)."""
        batch_losses = loss_total(params_2d, x_batch, y_batch)
        return jnp.nansum(batch_losses)

    if trials_matched:
        # Standard path: batch over trial dimension with same indices for x and y
        n_train_trials = n_train_trials_x  # same as n_train_trials_y

        def loss_and_grad_batched(params):
            """Compute loss and gradient by accumulating over trial batches."""
            params_2d = params.reshape(-1, n_params)
            total_loss = 0.0
            total_grad = jnp.zeros_like(params)

            for start_idx in range(0, n_train_trials, effective_trial_batch_size):
                end_idx = min(start_idx + effective_trial_batch_size, n_train_trials)
                batch_weight = (end_idx - start_idx) / n_train_trials
                x_batch = x_train[:, :, start_idx:end_idx]
                y_batch = y_train[:, :, start_idx:end_idx]

                batch_loss, batch_grad = loss_and_grad_single_batch(params_2d, x_batch, y_batch)

                total_loss += batch_loss * batch_weight
                total_grad += batch_grad.reshape(-1) * batch_weight

            return total_loss / n_samples, total_grad / n_samples
    else:
        # Mismatched trials: no trial mini-batching. Pass full x_train and y_train.
        # The loss_fn handles the shape relationship internally.
        def loss_and_grad_batched(params):
            """Compute loss and gradient over full data (no trial batching)."""
            params_2d = params.reshape(-1, n_params)
            loss, grad = loss_and_grad_single_batch(params_2d, x_train, y_train)
            return loss / n_samples, grad.reshape(-1) / n_samples
    
    if fit_params:
        # Adam optimizer with learning rate schedule
        # Ensure learning_rate is a Python float (not JAX array) for optax
        learning_rate = float(learning_rate)
        opt = optax.adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8)
        opt_state = opt.init(initial_params.reshape(-1))
        
        if use_scalar_single_feature_fullbatch:
            # Match legacy scalar full-batch path for speed/consistency when no batching is requested.
            loss_param = lambda params: jnp.mean(loss_total(params.reshape(-1, n_params), x_train, y_train))
            loss_param_and_grad = jax.value_and_grad(loss_param)

            @jax.jit
            def train_step(params, opt_state):
                loss, grad = loss_param_and_grad(params)
                updates, opt_state = opt.update(grad, opt_state, params)
                params = optax.apply_updates(params, updates)
                return params, opt_state, loss

            print_every = 50
            params = initial_params.reshape(-1)
            initial_loss = loss_param(params)
            best_loss, best_params = initial_loss.copy(), params.copy()
            for step in range(1, max_iter + 1):
                params, opt_state, loss_val = train_step(params, opt_state)
                if jnp.isnan(loss_val) or jnp.isinf(loss_val) or jnp.any(jnp.isnan(params)) or jnp.any(jnp.isinf(params)):
                    logging.info(f"Loss is NaN or Inf at step {step}. Stopping optimization.")
                    print(f"Final loss: {loss_val:.4f} at step {step}")
                    break
                if loss_val < best_loss:
                    best_loss = loss_val.copy()
                    best_params = params.copy()
                if step % print_every == 0:
                    print(f"step {step:4d}  loss {loss_val:.4f}")
            params = best_params.reshape(n_samples, n_params)
            print(f"params optimized. Loss: {best_loss:.4f}")
        else:
            def train_step(params, opt_state):
                loss, grad = loss_and_grad_batched(params)
                updates, new_opt_state = opt.update(grad, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return new_params, new_opt_state, loss

            print_every = 50
            params = initial_params.reshape(-1)
            initial_loss, _ = loss_and_grad_batched(params)

            CATASTROPHIC_LOSS_THRESHOLD = 1e6
            if initial_loss > CATASTROPHIC_LOSS_THRESHOLD:
                print(f"Initial loss {initial_loss:.2e} exceeds threshold. Skipping optimization.")
                logging.info(f"Skipping optimization: initial loss {initial_loss:.2e} > {CATASTROPHIC_LOSS_THRESHOLD:.0e}")
                return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params

            best_loss, best_params = initial_loss.copy(), params.copy()
            for step in range(1, max_iter + 1):
                params, opt_state, loss_val = train_step(params, opt_state)
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
            params = best_params.reshape(n_samples, n_params)
            print(f"params optimized. Loss: {best_loss:.4f}")
    else:
        params = compute_initial_params(param_estimator, model, np.asarray(x_train), np.asarray(y_train))
        if params is None or not isinstance(params, jnp.ndarray):
            logging.info("Error: params should be a JAX array.")
            return FAILED_PROGRAM_COST, jnp.zeros((n_samples, n_params)), FAILED_PROGRAM_COST, jnp.zeros((n_samples, n_params))
    
    # Compute final loss on test set
    if trials_matched:
        def eval_loss_batched(params_2d, x_eval, y_eval):
            """Compute loss by iterating over trial batches (matched trials)."""
            n_eval_trials = x_eval.shape[2]
            weighted_sum = 0.0
            for start_idx in range(0, n_eval_trials, effective_trial_batch_size):
                end_idx = min(start_idx + effective_trial_batch_size, n_eval_trials)
                batch_size = end_idx - start_idx
                x_batch = x_eval[:, :, start_idx:end_idx]
                y_batch = y_eval[:, :, start_idx:end_idx]
                weighted_sum += eval_single_batch(params_2d, x_batch, y_batch) * (batch_size / n_eval_trials)
            return weighted_sum / n_samples
    else:
        def eval_loss_batched(params_2d, x_eval, y_eval):
            """Compute loss over full data (mismatched trials, no batching)."""
            return eval_single_batch(params_2d, x_eval, y_eval) / n_samples

    if use_scalar_single_feature_fullbatch:
        initial_loss = jnp.nanmean(loss_total(initial_params, x_test, y_test)) + param_penalty_weight * n_params
    else:
        initial_loss = eval_loss_batched(initial_params, x_test, y_test) + param_penalty_weight * n_params
    n_nans = jnp.sum(jnp.isnan(initial_loss))
    if n_nans > 0:
        print(f"Warning: initial loss contains {n_nans} NaNs.")
    initial_loss = jnp.nan_to_num(initial_loss, nan=FAILED_PROGRAM_COST, posinf=FAILED_PROGRAM_COST, neginf=FAILED_PROGRAM_COST)
    
    if use_scalar_single_feature_fullbatch:
        final_loss = jnp.nanmean(loss_total(params, x_test, y_test)) + param_penalty_weight * n_params
    else:
        final_loss = eval_loss_batched(params, x_test, y_test) + param_penalty_weight * n_params
    n_nans = jnp.sum(jnp.isnan(final_loss))
    if n_nans > 0:
        print(f"Warning: final loss contains {n_nans} NaNs.")
    final_loss = jnp.nan_to_num(final_loss, nan=FAILED_PROGRAM_COST, posinf=FAILED_PROGRAM_COST, neginf=FAILED_PROGRAM_COST)
    
    t_end = time.time()
    print(f"Time taken for optimization: {t_end - t_start:.4f} seconds")
    return float(initial_loss), initial_params, float(final_loss), params


async def generate_new_model(current_island, llm_name, client, 
                                    spike_matrix, stimuli, prompt_manager,
                                    loss_fn=None,
                                    mode='explore', k_max=2, temp=1, 
                                    thinking_budget=1, img_dir=None,
                                    plot_model_fits=None,
                                    island_chat_manager=None, island_id: int = None,
                                    batch_id: int = 0,
                                    use_large_model: bool = True):
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
            sup_title = "".join([f"{model_name}_v{i+1}: Loss = {random_programs['train_loss'][i]:.2f} \n" for i in range(min(3, len(random_programs)))])
            plot_model_fits(
                X=stimuli,
                Y=spike_matrix,
                programs_df=random_programs,
                params_col="params",
                loss_col="train_loss",
                save_path=img_dir,
                labels=[f"{model_name}_v_{i+1}" for i in range(len(random_programs))],
                colours=['tab:green', 'tab:red'],
                title=sup_title,
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
        logging.info(
            f"Model generation returned no code block (island={island_id}, batch={batch_id})."
        )
        return None, None, (parent1_id, parent2_id)
    code_string = code_string.replace(f'def {model_name}_v{k+1}(', f'def {model_name}(')
    logging.info(
        f"Generated model candidate (island={island_id}, batch={batch_id}):\n"
        f"Prompt:\n{program_prompt}\n\n"
        f"Model (NumPy):\n{code_string}\n"
    )
    
    return code_string, program_prompt, (parent1_id, parent2_id)


async def generate_new_parameter_estimator(current_island, 
                                           model_code_string: str,
                                           model_fn,
                                           llm_name, client, 
                                           spike_matrix, stimuli, prompt_manager,
                                           mode='explore', k_max=1, temp=1,
                                           param_estimator_max_lines=100, img_dir=None,
                                           swear_words=None,
                                           refine_rounds: int = 0,
                                           param_penalty_weight: float = 0.1,
                                           objective_X=None,
                                           objective_Y=None,
                                           plot_model_fits=None,
                                           island_chat_manager=None, island_id: int = None,
                                           batch_id: int = 0,
                                           use_large_model: bool = False,
                                           loss_fn=None,):                                           
    if model_code_string is None:
        logging.info("No model code string provided, skipping parameter estimator generation.")
        return None, None
    if objective_X is None or objective_Y is None:
        logging.info("No objective split data provided, skipping parameter estimator generation.")
        return None, None
    k = min(k_max, len(current_island))
    random_programs = current_island.sample(k, replace=False).reset_index(drop=True)
    # sort from worst to best (loss descending)
    random_programs = random_programs.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
    use_image = (
        img_dir is not None
        and plot_model_fits is not None
    )
    use_chat_mode = island_chat_manager is not None and island_id is not None
    
    # Use appropriate prompt function based on mode
    if use_chat_mode:
        prompt = prompt_manager.get_parameter_estimator_prompt(random_programs,
                                                        model_code_string=model_code_string,
                                                        max_lines=param_estimator_max_lines,
                                                        use_image=use_image)
    else:
        prompt = prompt_manager.get_parameter_estimator_prompt_legacy(random_programs,
                                                        model_code_string=model_code_string,
                                                        max_lines=param_estimator_max_lines,
                                                        use_image=use_image)
    
    random_programs_crude = random_programs.copy()
    random_programs_crude['params'] = random_programs['initial_params']
    # now try generating an image from the random programs
    if use_image:
        try:
            sup_title = "".join([f"model_v{i+1}: Loss = {random_programs['train_loss'][i]:.2f} \n" for i in range(min(3, len(random_programs)))])
            plot_model_fits(
                X=stimuli,
                Y=spike_matrix,
                programs_df=random_programs_crude,
                params_col="params",
                loss_col="train_loss",
                save_path=img_dir,
                labels=[f"v_{i+1}" for i in range(len(random_programs_crude))],
                colours=['tab:green', 'tab:red'],
                title=sup_title,
            )
            img_path = Path(img_dir)
            with img_path.open("rb") as f:
                img_bytes = f.read()
        except Exception as e:
            logging.info(f"Error generating image for parameter estimator prompt: {e}")
            img_bytes = None
            # if we can't generate an image, we will just use the text prompt without image
            use_image = False
    else:
        img_bytes = None
    
    # Use chat-based or legacy LLM call
    if island_chat_manager is not None and island_id is not None:
        llm_output = await island_chat_manager.ask_island(
            island_id, prompt,
            batch_id=batch_id,
            mode=mode,
            use_large_model=use_large_model,
            png_img=img_bytes
        )
    else:
        # Legacy: independent query
        llm_output = await llm_helper.call_llm_async(prompt, model_name=llm_name, client=client, temperature=temp,
                                                thinking_budget=0.25, img_bytes=img_bytes)
    # extract the code block from the LLM output
    code_string = utils.extract_code_block(llm_output)
    if code_string is None:
        logging.info("No code block found in the LLM output for parameter estimator, skipping.")
        return None, None
    contains_swear_word = any(word in code_string for word in swear_words)
    if contains_swear_word:
        # find the word that is in the code_string
        swear_word = next((word for word in swear_words if word in code_string), None)
        logging.info(f"Parameter estimator code contains swear word: {swear_word}, skipping.")
        logging.info(f"Code string was:\n{code_string}")
        return None, None
    code_string = code_string.replace(f'def parameter_estimator_v{k+1}(', 'def parameter_estimator(')
    func = utils.str_to_func(code_string, 'parameter_estimator')

    if func is None:
        logging.info("Failed to parse parameter estimator code, skipping.")
        return None, None

    logging.info(
        f"Generated initial parameter estimator candidate (island={island_id}, batch={batch_id}):\n"
        f"Prompt:\n{prompt}\n\n"
        f"Parameter Estimator (initial):\n{code_string}\n"
    )

    if refine_rounds <= 0 or model_fn is None:
        return code_string, func

    best_code = code_string
    best_func = func
    best_loss = float(jnp.inf)

    current_code = code_string
    current_func = func
    current_loss, current_params, _, _ = objective(
        model=model_fn,
        param_estimator=current_func,
        x=objective_X,
        y=objective_Y,
        loss_fn=loss_fn,
        fit_params=False,  # Don't fit parameters during refinement evaluation
        param_penalty_weight=param_penalty_weight,
    )

    if current_loss < best_loss:
        best_loss = current_loss
        best_code = current_code
        best_func = current_func

    for r in range(refine_rounds):
        img_bytes = None
        refine_img_path = None
        if plot_model_fits is not None and img_dir is not None and current_params is not None:
            try:
                base_path = Path(img_dir)
                refine_img_path = base_path.with_name(f"{base_path.stem}_refine_{r+1}{base_path.suffix}")
                current_losses = np.full(spike_matrix.shape[0], float(current_loss))
                programs_list = [{
                    "model": model_fn,
                    "params": np.asarray(current_params),
                    "losses": current_losses,
                }]
                plot_model_fits(
                    X=stimuli,
                    Y=spike_matrix,
                    programs_list=programs_list,
                    save_path=str(refine_img_path),
                    labels=[f"refine_{r+1}"],
                    colours=['tab:green'],
                    title=f"Param estimator refinement {r+1}/{refine_rounds} (no-GD loss={current_loss:.4f})",
                )
                with refine_img_path.open("rb") as f:
                    img_bytes = f.read()
            except Exception as e:
                logging.info(f"Error generating refinement image: {e}")
                img_bytes = None

        # Build refinement prompt using current estimator as the only parent
        refinement_df = pd.DataFrame({
            'train_loss': [current_loss],
            'program_code_string': [model_code_string],
            'parameter_estimator_code_string': [current_code],
        })

        refine_header = (
            f"Refinement round {r+1}/{refine_rounds}.\n"
            f"Current no-GD loss: {current_loss:.4f}.\n"
            "Improve the parameter estimator without using gradient descent or external optimizers.\n"
            "Return only the updated parameter_estimator code.\n"
        )

        if use_chat_mode:
            refine_prompt = prompt_manager.get_parameter_estimator_prompt(
                refinement_df,
                model_code_string=model_code_string,
                max_lines=param_estimator_max_lines,
                use_image=img_bytes is not None,
            )
        else:
            refine_prompt = prompt_manager.get_parameter_estimator_prompt_legacy(
                refinement_df,
                model_code_string=model_code_string,
                max_lines=param_estimator_max_lines,
                use_image=img_bytes is not None,
            )
        refine_prompt = refine_header + "\n" + refine_prompt

        # Call LLM for refinement
        if island_chat_manager is not None and island_id is not None:
            llm_output = await island_chat_manager.ask_island(
                island_id, refine_prompt,
                batch_id=batch_id,
                mode=mode,
                use_large_model=use_large_model,
                png_img=img_bytes
            )
        else:
            llm_output = await llm_helper.call_llm_async(
                refine_prompt,
                model_name=llm_name,
                client=client,
                temperature=temp,
                thinking_budget=0.25,
                img_bytes=img_bytes
            )

        new_code = utils.extract_code_block(llm_output)
        if new_code is None:
            logging.info("No code block found in refinement output; keeping current estimator.")
            continue
        if any(word in new_code for word in swear_words):
            logging.info("Refinement code contains banned words; skipping.")
            continue

        new_code = re.sub(r"def\s+parameter_estimator_v\d+\s*\(", "def parameter_estimator(", new_code)
        new_func = utils.str_to_func(new_code, 'parameter_estimator')
        if new_func is None:
            logging.info("Failed to parse refined parameter estimator; keeping current.")
            continue

        new_loss, new_params, _, _ = objective(
            model=model_fn,
            param_estimator=new_func,
            x=objective_X,
            y=objective_Y,
            loss_fn=loss_fn,
            fit_params=False,  # Don't fit parameters during refinement evaluation
            param_penalty_weight=param_penalty_weight,
        )

        current_code = new_code
        current_func = new_func
        current_loss = new_loss
        current_params = new_params

        if new_loss < best_loss:
            best_loss = new_loss
            best_code = new_code
            best_func = new_func

    return best_code, best_func


async def translate_to_jax(code_string: str, client, prompt_manager, llm_name='gemini-2.0-flash-lite') -> tuple[str, callable]:
    """
    Translates a neuron model code string to JAX format.
    Args:
        code_string (str): The neuron model code string to translate.
        client: The LLM client.
        prompt_manager: PromptManager instance for generating prompts.
        llm_name (str): The LLM model name to use.
    Returns:
        callable: The translated JAX function.
    """
    if code_string is None:
        logging.info("No neuron model code string provided for translation.")
        return None, None
    
    prompt = prompt_manager.get_jax_translator_prompt(code_string)
    # print(f"Translating neuron model to JAX with prompt:\n{prompt}")
    if prompt is None:
        return None, None
    
    jax_code_string = await llm_helper.call_llm_async(prompt, client=client, model_name=llm_name, temperature=0)
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
    Validate NumPy/JAX agreement on a small subset of evaluation points.
    Parameters are estimated from observed train-trial data for the same subset.
    """
    x_obs = np.asarray(ensure_inputs(x_train_trials).to_tensor())
    y_obs = np.asarray(ensure_outputs(y_train_trials).to_tensor())
    eval_points = np.asarray(x_eval)
    if eval_points.ndim != 3:
        raise ValueError(
            f"X_eval must have shape (n_samples, n_features, n_eval_trials), got {eval_points.shape}."
        )

    n_samples = min(x_obs.shape[0], y_obs.shape[0], eval_points.shape[0])
    if n_samples <= 0:
        raise ValueError("No samples available for translation check.")

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
        params=np.asarray(params_subset),
        max_eval_trials=max_eval_trials,
    )


async def hypothesis_engine(
        n_iterations=9, time_limit=60, k_max=2, n_islands=8, batch_size=6, 
        critical_population_size=12, min_wise_population_size=0, 
        n_migrants=2, fit_params=True, use_param_estimator=True, exploit_point=0.5,
        param_penalty_weight=0.01, FAILED_PROGRAM_COST=np.inf,
        use_chat_mode=False,  # If True, use persistent chat sessions per island (expensive)
        chat_token_limit=50000,  # Max tokens per chat before auto-summarize and reset. 0 = unlimited
        param_estimator_refinement_rounds=0,
        exploration_topology = [1, 2, 3, 4, 5, 6, 7, 0], exploitation_topology = [1, 2, 3, 4, 5, 6, 7, 0],
        tiny_lm_name = 'gemini-2.0-flash-lite', little_lm_name = 'gemini-2.0-flash', large_lm_name = 'gemini-2.5-flash',
        use_large_every = 3, max_iter = 1_000, learning_rate = 3e-3,
        use_large_model_for_param_estimators=False,
        numpy_programs = None, param_estimators = None,
        X = None, Y = None, X_eval = None,
        plot_model_fits = None, loss_fn = None, 
        prompt_manager = None, trial_batch_size = None, swear_words = None,
        random_seed = 42, # consider setting up a seed_manager to make behaviours more robustly reproducible.
        ):
    """ 
    Main function to run the hypothesis engine.
    
    Args:
        X: Trial-split data container with shape (2, 2):
           X[0, 0]=train samples/train trials, X[0, 1]=train samples/test trials,
           X[1, 0]=test samples/train trials,  X[1, 1]=test samples/test trials.
           Entries are Inputs (or Inputs-compatible tensors).
        Y: Same structure as X but Outputs entries.
        X_eval: Precomputed evaluation grid for diagnostics.
        plot_model_fits: Optional plot callback callable.
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
            x_eval=X_eval,
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
    print("Created image feedback folder:", image_feedback_dir)

    # census[i] = [generation, island, batch_index, llm_name, loss, time, parent1_id, parent2_id, evaluation_matrix, n_free_params]
    census = []
    
    # Initialize best loss tracking for live monitoring
    best_loss_log = []  # List of dicts: {iteration, timestamp, best_train_loss, best_island, ...}
    best_loss_path = os.path.join(full_dir, 'best_loss_log.csv') if log_best_loss else None
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
        loss_init, params_init, loss, params = objective(
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
        # format strings
        program_code_string = utils.format_function_source(
            program_num, f'{model_name}_v{i+1}', 'import numpy as np'
        )
        parameter_estimator_code_string = utils.format_function_source(
            param_est, f'parameter_estimator_v{i+1}', 'import numpy as np'
        )
        y_eval = utils.compute_evaluation_matrix(
            program_jax,
            params,
            eval_points=X_eval,
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
        census.append([-1, -1, i, None, loss, time.time() - t_start, None, None, y_eval, params.shape[1]])

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
        plot_model_fits(
            X=X[0, 0],
            Y=Y[0, 0],
            programs_df=initial_programs,
            params_col="params",
            loss_col="train_loss",
            save_path=os.path.join(image_feedback_dir, 'initial_programs.png'),
            labels=['seed_1', 'seed_2'],
            colours=['tab:green', 'tab:red'],
            title="Seed Programs",
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
                    model_image_dirs[island_idx, j] = os.path.join(image_feedback_dir, f'iter_{i}_island_{island_idx}_batch_{j}.png')
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
                                                                   spike_matrix=Y[0,0], 
                                                                   stimuli=X[0,0],
                                                                   prompt_manager=prompt_manager,
                                                                   loss_fn=loss_fn,
                                                                   img_dir=model_image_dirs[island_idx, j],
                                                                   plot_model_fits=plot_model_fits,
                                                                   island_chat_manager=island_chat_manager,
                                                                   island_id=island_idx,
                                                                   batch_id=j,
                                                                   use_large_model=use_large_model) 
                                         for island_idx in range(n_islands) for j in range(batch_size)]
        logging.info(f"Generating {n_islands * batch_size} new programs... LLM Model: {llm_name}, mode: {mode}, temperature: {temperature:.2f}")
        print(f"Generating {n_islands * batch_size} new programs... LLM Model: {llm_name}, mode: {mode}, temperature: {temperature:.2f}")
        model_results = await asyncio.gather(*model_generation_tasks)
        model_code_strings = [result[0] for result in model_results]
        model_prompts = [result[1] for result in model_results]
        parent_ids = [result[2] for result in model_results]
        
        # convert to jax
        model_function_translation_tasks = [translate_to_jax(code_string, client, prompt_manager, tiny_lm_name) for code_string in model_code_strings]
        jax_results = await asyncio.gather(*model_function_translation_tasks)
        model_results = [(model_code_strings[j], model_prompts[j], jax_results[j][0], jax_results[j][1]) for j in range(n_islands * batch_size)]
        
        # build parameter‑estimator tasks
        param_estimation_tasks = [
            generate_new_parameter_estimator(
                current_island=islands[island_idx],
                model_code_string=model_code_strings[island_idx * batch_size + j],
                model_fn=model_results[island_idx * batch_size + j][3],
                llm_name=little_lm_name,
                client=client,
                spike_matrix=Y[0,0],
                stimuli=X[0,0],
                prompt_manager=prompt_manager,
                mode=mode,
                k_max=2,
                temp=temperature,
                param_estimator_max_lines=100,
                img_dir=os.path.join(image_feedback_dir, f'iter_{i}_island_{island_idx}_batch_{j}_param_estimator.png') if use_large_model_for_param_estimators else None,
                refine_rounds=param_estimator_refinement_rounds,
                param_penalty_weight=param_penalty_weight,
                objective_X=X[0],
                objective_Y=Y[0],
                random_seed=random_seed,
                swear_words=swear_words,
                plot_model_fits=plot_model_fits,
                island_chat_manager=island_chat_manager,
                island_id=island_idx,
                batch_id=j,
                use_large_model=use_large_model_for_param_estimators,
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
            model_code_string, prompt, model_code_string_jax, model_new, param_est_code_string, param_est_new = island_results[island_idx][j]
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
                    x_eval=X_eval,
                )
            except Exception as e:
                logging.info(
                    f"Skipping island {island_idx}, batch {j}: JAX translation check failed: {e}"
                )
                logging.info('-' * 50)
                continue
            
            initial_loss, initial_params, loss, optimized_params = objective(
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
                eval_points=X_eval,
            )
            logging.info(f"Loss: {loss:.2f}\n")


            # plot the fits of the neuron model and parameter estimator if using image feedback
            if has_spec_plotter:
                initial_params_plot = np.asarray(initial_params).copy()
                optimized_params_plot = np.asarray(optimized_params).copy()
                param_delta = optimized_params_plot - initial_params_plot
                mean_abs_delta = float(np.mean(np.abs(param_delta)))
                max_abs_delta = float(np.max(np.abs(param_delta)))
                if np.allclose(initial_params_plot, optimized_params_plot, equal_nan=True):
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
                    save_path=os.path.join(image_feedback_dir, f'iter_{i}_island_{island_idx}_batch_{j}_param_est_vs_gd.png'),
                    colours=['tab:green', 'tab:red'],
                    labels=['Param Estimator', 'Gradient Descent'],
                    title=(
                        "param_est_vs_gd\n"
                        f"Initial Loss: {initial_loss:.4f}, Final Loss: {loss:.4f}\n"
                        f"|Δparams| mean={mean_abs_delta:.3e}, max={max_abs_delta:.3e}"
                    ),
                )
            
            param_names = [n for n in inspect.signature(model_new).parameters if n != "theta"]
            if optimized_params.shape[1] == len(param_names):
                df = pd.DataFrame(np.array(optimized_params)[:10], columns=param_names)
                logging.info(f"Optimized Parameters for 10 samples:\n{df}\n")
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
            census.append([i, island_idx, j, llm_name, loss, t_added, parent1_id, parent2_id, y_eval, optimized_params.shape[1]])
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
                sup_title = f"Iteration {i}, Island {island_idx}, Top {len(top_df)} Programs\n"
                sup_title += "\n".join([f"model {j+1}: iter {top_df['iteration_number'][j]}, birth island {top_df['birth_island'][j]}, batch {top_df['batch_index'][j]}, loss: {top_df['train_loss'][j]:.2f}" for j in range(len(top_df))])
                plot_model_fits(
                    X=X[0, 0],
                    Y=Y[0, 0],
                    programs_df=top_df,
                    params_col="params",
                    loss_col="train_loss",
                    save_path=os.path.join(iteration_dir, f'island_{island_idx}_top_programs.png'),
                    title=sup_title,
                )
        
        if has_spec_plotter:
            all_programs = pd.concat([islands[idx] for idx in range(n_islands)], ignore_index=True)
            top_programs = all_programs.sort_values(by='train_loss').head(3).reset_index(drop=True)
            top_programs = top_programs.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
            sup_title = f"Iteration {i}, Top 3 Programs Overall\n"
            sup_title += "\n".join([f"model {j+1}: iter {top_programs['iteration_number'][j]}, birth island {top_programs['birth_island'][j]}, batch {top_programs['batch_index'][j]}, loss: {top_programs['train_loss'][j]:.2f}" for j in range(len(top_programs))])
            plot_model_fits(
                X=X[0, 0],
                Y=Y[0, 0],
                programs_df=top_programs,
                params_col="params",
                loss_col="train_loss",
                save_path=os.path.join(iteration_dir, 'top_programs_overall.png'),
                title=sup_title,
            )
        
        # save census
        census_path = os.path.join(iteration_dir, 'census.npy')
        census_np = np.array(census, dtype=object)
        np.save(census_path, census_np)
        
        # Log token usage summary for this iteration (if using chat mode)
        if island_chat_manager is not None:
            island_chat_manager.log_iteration_summary(i)
        
        # Log best loss across all islands for live monitoring
        if log_best_loss:
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
            _, _, test_loss, optimized_params = objective(
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
    # save the combined programs dataframe, reordering columns to have order:
    # iteration_number, birth_island, batch_index, train_loss, test_loss, program_code_string, parameter_estimator_code_string, program, parameter_estimator, params, parent1_id, parent2_id
    combined_programs_dataframe = combined_programs_dataframe[['iteration_number', 'birth_island', 'batch_index',
                                                                'train_loss', 'test_loss',
                                                                'program_code_string', 'parameter_estimator_code_string',
                                                                'program', 'parameter_estimator', 'params',
                                                                'parent1_id', 'parent2_id', 'llm_name']]
    combined_programs_dataframe.to_csv(os.path.join(combined_dir, 'programs_db.csv'), index=False)

    # save census npy array
    census_path = os.path.join(combined_dir, 'census.npy')
    census_np = np.array(census, dtype=object)
    np.save(census_path, census_np)

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
    config_str = f"n_islands={n_islands}, batch_size={batch_size}, n_iterations={n_iterations},\n"
    config_str += f"llm_names={little_lm_name, large_lm_name}, fit_params={fit_params}, \n"
    config_str += f"critical_population_size={critical_population_size}.\n"

    if has_spec_plotter:
        for i, df in enumerate(df_list):
            df_sup = config_str
            df = df.head(3)
            df = df.sort_values(by='test_loss', ascending=False).reset_index(drop=True)
            df_sup += "".join([f"model {len(df) - i}: iter {df['iteration_number'][i]}, birth_island {df['birth_island'][i]}, batch {df['batch_index'][i]}, total loss {0.5 * (df['test_loss'][i] + df['train_loss'][i]):.2f}\n" for i in range(min(3, len(df)))])
            plot_model_fits(
                X=X[1, 1],
                Y=Y[1, 1],
                programs_df=df,
                params_col="params",
                loss_col="test_loss",
                save_path=os.path.join(df_dirs[i], 'top_model_fits.png'),
                title=df_sup,
            )
            # Plot top models separately using the same plot_model_fits pathway.
            for j in range(min(3, len(df))):
                birth_island = df['birth_island'][j]
                iteration_number = df['iteration_number'][j]
                batch_index = df['batch_index'][j]
                model_df = df.iloc[[j]].copy().reset_index(drop=True)
                model_title = (
                    f"Island {birth_island}, Iteration {iteration_number}, "
                    f"Batch {batch_index}, loss: {df['test_loss'][j]:.2f}"
                )
                plot_model_fits(
                    X=X[1, 1],
                    Y=Y[1, 1],
                    programs_df=model_df,
                    params_col="params",
                    loss_col="test_loss",
                    save_path=os.path.join(df_dirs[i], f'top_model_fit_{min(3, len(df)) - j}.png'),
                    labels=['model'],
                    colours=['tab:green'],
                    title=model_title,
                )
    
    # Log final token usage summary (if using chat mode)
    if island_chat_manager is not None:
        island_chat_manager.log_final_summary()

    return full_dir
