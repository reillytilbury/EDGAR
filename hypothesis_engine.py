import inspect
import ast
import re
import os
import logging
import asyncio
import importlib
import numpy as np
import jax, jax.numpy as jnp
import timeout_decorator
import jaxopt, optax
import pandas as pd
from pathlib import Path
import utils, diagnostic, genetic_helpers, loss_functions
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
print(jax.default_backend())    # should print "gpu"
print(jax.devices())

def _load_project_seed_models(project: str):
    return importlib.import_module(f"projects.{project}.seed_models")

def _load_project_image_diagnostics(project: str):
    return importlib.import_module(f"projects.{project}.image_diagnostics")

def _load_project_data(project: str, data_path: str | None = None, data_kwargs: dict | None = None):
    data_module = importlib.import_module(f"projects.{project}.data_loading")
    if not hasattr(data_module, "load_data"):
        raise ValueError(f"Project '{project}' must define data_loading.load_data.")
    kwargs = {} if data_kwargs is None else dict(data_kwargs)
    if "data_path" not in kwargs:
        if data_path is not None:
            kwargs["data_path"] = data_path
        else:
            project_config = utils._load_prompt_config(project)
            config_path = project_config.get("DATA_PATH", None)
            if isinstance(config_path, str) and config_path.strip():
                kwargs["data_path"] = config_path
            elif isinstance(config_path, (list, tuple)) and len(config_path) > 0:
                kwargs["data_path"] = config_path
    return data_module.load_data(**kwargs)

def _get_module_import_lines(module) -> str:
    try:
        src = inspect.getsource(module)
    except Exception:
        return ""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ""
    lines = src.splitlines()
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", node.lineno)
            imports.extend(lines[start:end])
    return "\n".join(imports) + ("\n" if imports else "")

def _ensure_xy(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if y.ndim < 2:
        raise ValueError(f"Y must have shape (n_cells, n_trials, ...). Got shape {y.shape}.")
    n_trials = y.shape[1]
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f"X must be 2D (n_stim_dim, n_trials). Got shape {x.shape}.")
    if x.shape[1] != n_trials:
        if x.shape[0] == n_trials:
            x = x.T
        else:
            raise ValueError(
                f"X and Y must share n_trials. X shape {x.shape}, Y shape {y.shape}."
            )
    return jnp.asarray(x), jnp.asarray(y)

def _plot_stimuli_for_cells(x, n_cells: int):
    if x.ndim == 2 and x.shape[0] == n_cells:
        return x
    if x.ndim == 2 and x.shape[0] == 1:
        return jnp.broadcast_to(x[0], (n_cells, x.shape[1]))
    if x.ndim == 2 and x.shape[0] != n_cells:
        return x
    return None


def _effective_param_count(n_params: int, x_model, y_data, x_shared: bool) -> float:
    """
    Estimate effective parameter count by normalizing scalar params by n_input * n_output.
    This makes large matrices count as ~1 parameter.
    """
    if x_model is None:
        n_input = 1
    else:
        if x_shared:
            if x_model.ndim <= 1:
                n_input = 1
            else:
                n_input = int(x_model.shape[1])
        else:
            if x_model.ndim <= 2:
                n_input = 1
            else:
                n_input = int(x_model.shape[2])
    n_output = int(y_data.shape[2]) if y_data.ndim >= 3 else 1
    denom = max(1, n_input * n_output)
    return float(n_params) / float(denom)

def _stimuli_trials_first(stimuli):
    if stimuli.ndim == 1:
        return stimuli
    if stimuli.ndim != 2:
        raise ValueError(f"Stimuli must be 1D or 2D, got {stimuli.ndim}D.")
    trials_first = stimuli.T
    if trials_first.shape[1] == 1:
        return trials_first[:, 0]
    return trials_first

def _is_shared_stimuli(x, n_cells: int):
    if x.ndim == 1:
        return True
    return x.ndim == 2 and x.shape[0] != n_cells

def _extract_first_function_name(code_string: str) -> str | None:
    try:
        import ast
        tree = ast.parse(code_string)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                return node.name
    except Exception:
        return None
    return None

def _translation_eval_stimuli(x, n_cells: int, max_trials: int = 256):
    x = np.asarray(x)
    if _is_shared_stimuli(x, n_cells):
        stim = _stimuli_trials_first(x)
    else:
        stim = x[0]
    stim = np.asarray(stim)
    if stim.ndim == 1:
        return stim[:max_trials]
    return stim[:max_trials, ...]

def compute_initial_params(param_estimator, neuron_model, x, y) -> jnp.ndarray:
    """
    Compute initial parameters for the neuron model using the provided parameter estimator. Confusingly, the parameter estimator will be written in numpy,
    but the neuron model will be written in JAX. So the data x and y will be numpy arrays, but the output will be a JAX array.
    Args:
        param_estimator (function): Function to estimate initial parameters for the neuron model.
                                    Signature: param_estimator(stimuli, response) -> params
        neuron_model (function): The model which predicts neural activity from stimuli and free parameters.
                                 Signature: neuron_model(stimuli, *params) -> activity
        x (np.ndarray): Stimuli data. Either:
            - (n_stim_dim, n_trials) shared across cells, or
            - (n_cells, n_trials) per-cell stimuli (legacy).
        y (np.ndarray): Response data, shape (n_cells, n_trials).
    Returns:
        jnp.ndarray: The estimated parameters for each cell, shape (n_cells, n_params).
                     If the parameter estimation fails, returns an array of default parameters based on the neuron model's signature.
                     If this also fails, returns None.
    """
    @timeout_decorator.timeout(5, use_signals=True)
    def _safe_estimate(pe, xi, yi):
        return pe(xi, yi)
    try:
        # any call taking >5s will raise timeout_decorator.TimeoutError
        n_cells = y.shape[0]
        if _is_shared_stimuli(x, n_cells):
            x_shared = _stimuli_trials_first(np.asarray(x))
            return jnp.array([_safe_estimate(param_estimator, x_shared, y[i]) for i in range(n_cells)])
        return jnp.array([_safe_estimate(param_estimator, x[i], y[i]) for i in range(n_cells)])
    except timeout_decorator.TimeoutError:
        logging.warning("param_estimator timed out, falling back to defaults")
    except Exception as e:
        logging.info(f"Error during parameter estimation: {e}")

    # If parameter estimation fails, compute default parameters based on the neuron model's signature
    params = compute_default_params(neuron_model)
    if params is not None:
        # default params is a 2D array with shape (1, n_params), so we need to repeat it for each cell
        n_cells = y.shape[0]
        return jnp.repeat(params, n_cells, axis=0)
    else:
        logging.info("Error: Unable to compute default parameters for the neuron model.")
        return None

def compute_default_params(neuron_model) -> jnp.ndarray:
    """
    Compute default parameters for the neuron model based on its signature.
    Args:
        neuron_model (function): The model which predicts neural activity from stimuli and free parameters.
                                 Signature: neuron_model(stimuli, *params) -> activity
    Returns:
        jnp.ndarray: The default parameters for the neuron model, shape (1, n_params).
                     If the parameter estimation fails, returns None.
    """
    try:
        sig = inspect.signature(neuron_model)
        param_names = list(sig.parameters.keys())[1:]
        defaults = [sig.parameters[n].default if sig.parameters[n].default is not inspect._empty else 0.0 for n in param_names]
        default_arr = jnp.array(defaults, dtype=np.float32)
        return default_arr.reshape(1, -1)  # reshape to (1, n_params)
    except Exception as e:
        logging.info(f"Error while generating default parameters: {e}")
        return None    

def objective(neuron_model, param_estimator, loss_func, x, y, 
              param_penalty_weight=0.1, fit_params=True, random_seed=0,
              FAILED_PROGRAM_COST=jnp.inf, tol=1e-2, max_iter=1_000) -> tuple[float, jnp.ndarray, float, jnp.ndarray]:
    """
    Calculate the loss of the model. 
    
    The loss is calculated as the mean over cells and trials of the loss function provided.
    Args:
        neuron_model (function): The model which predicts neural activity from stimuli
                                and free parameters (for a single cell).
                                Signature: neuron_model(stimuli, *params) -> activity
        param_estimator (function): Function to estimate initial parameters for the neuron model.
                                Signature: param_estimator(stimuli, response) -> params
        loss_func (function): The loss function to use for calculating the loss.
        x (jnp.ndarray): Stimuli data. Either:
            - (n_stim_dim, n_trials) shared across cells, or
            - (n_cells, n_trials) per-cell stimuli (legacy).
        y (jnp.ndarray): Response data, shape (n_cells, n_trials, ...).
        param_penalty_weight (float): Weight for the penalty on the number of parameters. Default is 0.1.
        fit_params (bool): Whether to fit the parameters of the model. Default is True.
        random_seed (int or None): Random seed for reproducibility. Default is 0. If None, will not split the data into training and test sets.
        FAILED_PROGRAM_COST (float): Cost assigned to failed models. Default is np.inf.
        tol (float): Tolerance for optimization convergence. Default is 1e-2.
        max_iter (int): Maximum number of iterations for optimization. Default is 1_000.
    Returns:
        tuple[
            - float: The cross-validated loss of the model with data fit by the parameter estimator,
            - jnp.ndarray: The parameters fit by the parameter estimator.
            - float: The average loss (MSE on test set) across all cells. 
                     Returns FAILED_PROGRAM_COST if the model fails for ANY cell.
            - jnp.ndarray: The parameters for each cell (n_cells, n_params).
    """
    t_start = time.time()
    if y.ndim < 2:
        logging.info(f"Error: y must be at least 2D (n_cells, n_trials, ...). Got shape {y.shape}.")
        return FAILED_PROGRAM_COST, jnp.zeros((y.shape[0] if y.ndim > 0 else 0, 0)), FAILED_PROGRAM_COST, jnp.zeros((y.shape[0] if y.ndim > 0 else 0, 0))
    population_mode = y.ndim >= 3 and y.shape[0] == 1
    n_cells, n_trials = y.shape[0], y.shape[1]
    if x.ndim not in (1, 2):
        logging.info(f"Error: x must be 1D or 2D, got shape {x.shape}.")
        return FAILED_PROGRAM_COST, jnp.zeros((n_cells, 0)), FAILED_PROGRAM_COST, jnp.zeros((n_cells, 0))
    x_shared = _is_shared_stimuli(x, n_cells)
    # train/test split over trials
    key = jax.random.PRNGKey(random_seed)
    training_size = n_trials // 2
    shuffled_indices = jax.random.permutation(key, jnp.arange(n_trials))
    training_trials_idx = shuffled_indices[:training_size]
    test_trials_idx = shuffled_indices[training_size:]
    if x.ndim == 1:
        x_train = x[training_trials_idx]
        x_test = x[test_trials_idx]
    else:
        x_train = x[:, training_trials_idx]
        x_test = x[:, test_trials_idx]
    y_train = y[:, training_trials_idx]
    y_test = y[:, test_trials_idx]
    x_train_model = _stimuli_trials_first(x_train) if x_shared else x_train
    x_test_model = _stimuli_trials_first(x_test) if x_shared else x_test
    x_full_model = _stimuli_trials_first(x) if x_shared else x

    n_model_params = len(inspect.signature(neuron_model).parameters) - 1

    def _population_dims(x_model, y_data, x_shared_local: bool):
        if x_shared_local:
            if x_model.ndim <= 1:
                n_source = 1
            else:
                n_source = int(x_model.shape[1])
        else:
            if x_model.ndim <= 2:
                n_source = 1
            else:
                n_source = int(x_model.shape[2])
        if y_data.ndim >= 3:
            n_target = int(y_data.shape[2])
        elif y_data.ndim == 2:
            n_target = int(y_data.shape[1])
        else:
            n_target = 1
        return n_source, n_target

    # Perform initial param calc. x and y must be numpy arrays of shape (n_cells, n_trials, ...)
    if population_mode:
        x_est = x_train_model if x_shared else x_train_model[0]
        try:
            raw_params = param_estimator(np.asarray(x_est), np.asarray(y_train[0]))
            n_source, n_target = _population_dims(x_train_model, y_train, x_shared)
            packed = utils.pack_population_params(raw_params, n_source, n_target, n_model_params)
            initial_params = jnp.asarray(packed)[None, :]
        except Exception as e:
            logging.info(f"Error during population parameter estimation: {e}")
            n_source, n_target = _population_dims(x_train_model, y_train, x_shared)
            try:
                total_size = int(sum(np.prod(s) for s in utils._population_param_shapes(n_source, n_target, n_model_params)))
            except Exception:
                total_size = 0
            initial_params = jnp.zeros((1, total_size))
    else:
        initial_params = compute_initial_params(param_estimator, neuron_model, np.asarray(x_train), np.asarray(y_train))
    
    # Fail immediately if initial_params is None or not a JAX array
    if initial_params is None or not isinstance(initial_params, jnp.ndarray):
        logging.info("Error: initial_params should be a JAX array.")
        return FAILED_PROGRAM_COST, jnp.zeros((n_cells, 0)), FAILED_PROGRAM_COST, jnp.zeros((n_cells, 0))
    if initial_params.ndim != 2 or initial_params.shape[0] != n_cells:
        logging.info(f"Error: initial_params should be a 2D array with shape ({n_cells}, n_params).")
        return FAILED_PROGRAM_COST, jnp.zeros((n_cells, 0)), FAILED_PROGRAM_COST, jnp.zeros((n_cells, 0))

    # Fail immediately if fit_params is True and non-numeric params
    n_params = initial_params.shape[1]
    all_numeric = (initial_params.dtype.kind in 'biufc' and 
                  jnp.all(jnp.isfinite(initial_params)))
    if fit_params and not all_numeric:
        logging.info("Error: Cannot fit non-numeric parameters.")
        return FAILED_PROGRAM_COST, jnp.zeros((n_cells, n_params)), FAILED_PROGRAM_COST, jnp.zeros((n_cells, n_params))

    # Fail immediately if neuron_model doesn't run
    try:
        # Check compatibility with JAX's tracing mechanism
        neuron_model_jit = jax.jit(neuron_model)
        def _call_model_jit(stim, params, y_data):
            if population_mode:
                n_source, n_target = _population_dims(stim, y_data, x_shared)
                parts = utils.unpack_population_params(params, n_source, n_target, n_model_params)
                return neuron_model_jit(stim, *parts)
            return utils.call_model_with_params(neuron_model_jit, stim, params, n_model_params)
        for cell_idx in np.random.choice(n_cells, size=min(10, n_cells), replace=False):
            # Validate with concrete values
            stim = x_full_model if x_shared else x_full_model[cell_idx]
            output = _call_model_jit(stim, initial_params[cell_idx], y_train[cell_idx] if y_train.ndim > 1 else y_train)
            if output.ndim < 1 or output.shape[0] != n_trials:
                logging.info(f"Error: model output shape {output.shape} does not match trials {n_trials}.")
                return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params
            # Validate with abstract tracer values
            if population_mode:
                n_source, n_target = _population_dims(stim, y_train[cell_idx] if y_train.ndim > 1 else y_train, x_shared)
                param_parts = utils.unpack_population_params(initial_params[cell_idx], n_source, n_target, n_model_params)
                jax.eval_shape(neuron_model_jit, stim, *param_parts)
            else:
                param_parts = utils._split_params_for_model(stim, initial_params[cell_idx], n_model_params)
                jax.eval_shape(neuron_model_jit, stim, *param_parts)
    except Exception as e:
        logging.info(f"Model failed to run or is incompatible with JAX tracing: {e}")
        return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params

    scale_eps = 1e-6
    y_scale = jnp.std(y_train, axis=1, keepdims=True)
    y_scale = jnp.where(y_scale < scale_eps, 1.0, y_scale)
    def _call_model(stim, params, y_data):
        if population_mode:
            n_source, n_target = _population_dims(stim, y_data, x_shared)
            parts = utils.unpack_population_params(params, n_source, n_target, n_model_params)
            return neuron_model(stim, *parts)
        return utils.call_model_with_params(neuron_model, stim, params, n_model_params)

    def loss_single_cell(params, x_data, y_data, scale):
        pred = _call_model(x_data, params, y_data)
        pred = pred / scale
        y_scaled = y_data / scale
        return jnp.mean(loss_func(pred, y_scaled))
    # vectorize the loss function for all cells. The inputs will have shapes:
    # - params: (n_cells, n_params)
    # - x_data: (n_cells, n_trials) or shared (n_trials, n_stim_dim) / (n_trials,)
    # - y_data: (n_cells, n_trials)
    # The output will have shape (n_cells,)
    x_in_axes = None if x_shared else 0
    loss_total = jax.vmap(loss_single_cell, in_axes=(0, x_in_axes, 0, 0), out_axes=0)

    if fit_params:
        # define the loss function wrt params. This will have input shape n_cells * n_params (note that params is flattened) and output shape (1,)
        loss_param = lambda params: jnp.mean(loss_total(params.reshape(-1, n_params), x_train_model, y_train, y_scale))
        loss_param_and_grad = jax.value_and_grad(loss_param)

        # solver = jaxopt.ScipyMinimize(
        #     fun=loss_param_and_grad,
        #     value_and_grad=True,
        #     method='L-BFGS-B',
        #     maxiter=max_iter,
        #     tol=tol,
        #     jit=True)
        # try:
        #     result = solver.run(initial_params.reshape(-1))
        #     params = jnp.asarray(result.params).reshape(n_cells, n_params)
        #     print(f"Optimization success: {result.state.success}, iterations: {result.state.iter_num}")
        # except Exception as e:
        #     params = initial_params
        #     logging.info(f"Error during optimization: {e}")

        # 1.  build adam
        learning_rate = 3e-3
        beta1, beta2  = 0.9, 0.999
        opt = optax.adam(learning_rate, b1=beta1, b2=beta2, eps=1e-8)
        opt_state = opt.init(initial_params.reshape(-1))
        
        # 2. jit single step
        @jax.jit
        def train_step(params, opt_state):
            loss, grad = loss_param_and_grad(params)
            updates, opt_state = opt.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss
        
        # 3.  iterate
        print_every = 50
        params = initial_params.reshape(-1)  # Flatten params for the optimizer
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
        params = best_params.reshape(n_cells, n_params)
        print(f"params optimized. Loss: {best_loss:.4f}")
    else:
        if population_mode:
            params = initial_params
        else:
            params = compute_initial_params(param_estimator, neuron_model, np.asarray(x_train), np.asarray(y_train))
            if params is None or not isinstance(params, jnp.ndarray):
                logging.info("Error: params should be a JAX array.")
                return FAILED_PROGRAM_COST, jnp.zeros((n_cells, n_params))

    # compute the final loss on the test set for the initial and optimized parameters
    param_complexity = _effective_param_count(n_params, x_train_model, y_train, x_shared)
    initial_loss = jnp.nanmean(loss_total(initial_params, x_test_model, y_test, y_scale)) + param_penalty_weight * param_complexity
    # print number of nans in initial_loss
    n_nans = jnp.sum(jnp.isnan(initial_loss))
    if n_nans > 0:
        print(f"Warning: initial loss contains {n_nans} NaNs. This may indicate a problem with the model or data.")
    initial_loss = jnp.nan_to_num(initial_loss, nan=FAILED_PROGRAM_COST, posinf=FAILED_PROGRAM_COST, neginf=FAILED_PROGRAM_COST)
    final_loss = jnp.nanmean(loss_total(params, x_test_model, y_test, y_scale)) + param_penalty_weight * param_complexity
    # print number of nans in final_loss
    n_nans = jnp.sum(jnp.isnan(final_loss))
    if n_nans > 0:
        print(f"Warning: final loss contains {n_nans} NaNs. This may indicate a problem with the model or data.")
    final_loss = jnp.nan_to_num(final_loss, nan=FAILED_PROGRAM_COST, posinf=FAILED_PROGRAM_COST, neginf=FAILED_PROGRAM_COST)
    # Round final losses to 2 decimal places
    # final_loss = jnp.round(final_loss, 2)
    t_end = time.time()
    print(f"Time taken for optimization: {t_end - t_start:.4f} seconds")
    return float(initial_loss), initial_params, float(final_loss), params


def evaluate_param_estimator_loss(neuron_model, param_estimator, loss_func, x, y, param_penalty_weight=0.1):
    """
    Evaluate parameter estimator on provided data without gradient descent.
    Returns (loss, params). Loss is mean over cells/trials with same scaling as objective.
    """
    try:
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        if y.ndim < 2:
            return float(jnp.inf), None
        population_mode = y.ndim >= 3 and y.shape[0] == 1
        n_cells, n_trials = y.shape[0], y.shape[1]
        if x.ndim not in (1, 2):
            return float(jnp.inf), None
        x_shared = _is_shared_stimuli(x, n_cells)
        x_model = _stimuli_trials_first(x) if x_shared else x

        n_model_params = len(inspect.signature(neuron_model).parameters) - 1
        def _population_dims(x_model_local, y_data_local, x_shared_local: bool):
            if x_shared_local:
                if x_model_local.ndim <= 1:
                    n_source = 1
                else:
                    n_source = int(x_model_local.shape[1])
            else:
                if x_model_local.ndim <= 2:
                    n_source = 1
                else:
                    n_source = int(x_model_local.shape[2])
            if y_data_local.ndim >= 3:
                n_target = int(y_data_local.shape[2])
            elif y_data_local.ndim == 2:
                n_target = int(y_data_local.shape[1])
            else:
                n_target = 1
            return n_source, n_target

        if population_mode:
            x_est = x_model if x_shared else x_model[0]
            try:
                raw_params = param_estimator(np.asarray(x_est), np.asarray(y[0]))
                n_source, n_target = _population_dims(x_model, y, x_shared)
                packed = utils.pack_population_params(raw_params, n_source, n_target, n_model_params)
                params = jnp.asarray(packed)[None, :]
            except Exception:
                n_source, n_target = _population_dims(x_model, y, x_shared)
                total_size = int(sum(np.prod(s) for s in utils._population_param_shapes(n_source, n_target, n_model_params)))
                params = jnp.zeros((1, total_size))
        else:
            params = compute_initial_params(param_estimator, neuron_model, np.asarray(x), np.asarray(y))
        if params is None or not isinstance(params, jnp.ndarray):
            return float(jnp.inf), None
        if params.ndim != 2 or params.shape[0] != n_cells:
            return float(jnp.inf), None
        n_params = params.shape[1]
        all_numeric = (params.dtype.kind in 'biufc' and jnp.all(jnp.isfinite(params)))
        if not all_numeric:
            return float(jnp.inf), params

        scale_eps = 1e-6
        y_scale = jnp.std(y, axis=1, keepdims=True)
        y_scale = jnp.where(y_scale < scale_eps, 1.0, y_scale)

        def loss_single_cell(p, x_data, y_data, scale):
            if population_mode:
                n_source, n_target = _population_dims(x_data, y_data, x_shared)
                parts = utils.unpack_population_params(p, n_source, n_target, n_model_params)
                pred = neuron_model(x_data, *parts)
            else:
                pred = utils.call_model_with_params(neuron_model, x_data, p, n_model_params)
            pred = pred / scale
            y_scaled = y_data / scale
            return jnp.mean(loss_func(pred, y_scaled))

        x_in_axes = None if x_shared else 0
        loss_total = jax.vmap(loss_single_cell, in_axes=(0, x_in_axes, 0, 0), out_axes=0)
        param_complexity = _effective_param_count(n_params, x_model, y, x_shared)
        loss_val = jnp.nanmean(loss_total(params, x_model, y, y_scale)) + param_penalty_weight * param_complexity
        loss_val = jnp.nan_to_num(loss_val, nan=jnp.inf, posinf=jnp.inf, neginf=jnp.inf)
        return float(loss_val), params
    except Exception as e:
        logging.info(f"Error evaluating parameter estimator loss: {e}")
        return float(jnp.inf), None


async def generate_new_neuron_model(current_island, llm_name, client, 
                                    spike_matrix, stimuli,
                                    mode='explore', k_max=2, temp=1, 
                                    thinking_budget=1, img_dir=None,
                                    project: str = "grid_cells",
                                    image_fn=None,
                                    model_fn_name: str | None = None,
                                    model_label: str | None = None):
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
    if model_fn_name is None:
        model_fn_name = utils._get_model_function_name(project)
    if model_label is None:
        config = utils._load_prompt_config(project)
        model_label = config.get("MODEL_LABEL", "Neuron Model")
    use_image = img_dir is not None and image_fn is not None
    program_prompt = utils.create_program_prompt(
        random_programs, mode=mode, llm_type=llm_name[0], use_image=use_image, project=project
    )

    if use_image:
        try:
            sup_title = "".join([f"{model_fn_name}_v{i+1}: Loss = {random_programs['train_loss'][i]:.2f} \n" for i in range(min(3, len(random_programs)))])
            image_fn(programs_df=random_programs,
                                    loss_function=loss_functions.quadratic_loss,
                                    x=stimuli, y=spike_matrix,
                                    cell_selection=np.random.choice(spike_matrix.shape[0], size=4, replace=False),
                                    save_path=img_dir,
                                    labels=['v_1', 'v_2'],
                                    colours=['tab:green', 'tab:red'],
                                    dpi=384*3/20,
                                    title=sup_title,
                                    legend_fontsize=20,
                                    line_alpha=0.9,
                                    line_width=4,)
            
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
    llm_output = await utils.call_llm_async(program_prompt, model_name=llm_name, client=client, temperature=temp, 
                                            thinking_budget=thinking_budget, img_bytes=img_bytes)
    code_string = utils.extract_code_block(llm_output)
    if code_string is None:
        return None, None, (parent1_id, parent2_id)
    code_string = code_string.replace(f'def {model_fn_name}_v{k+1}(', f'def {model_fn_name}(')
    
    return code_string, program_prompt, (parent1_id, parent2_id)

async def generate_new_parameter_estimator(current_island, 
                                           neuron_model_code_string: str,
                                           neuron_model_fn,
                                           llm_name, client, 
                                           spike_matrix, stimuli,
                                           k_max=1, temp=1,
                                           param_estimator_max_lines=100, img_dir=None,
                                           swear_words=['lstsq', 'scipy.optimize', 'optimize.minimize', 'curve_fit', 'sklearn'],
                                           project: str = "grid_cells",
                                           image_fn=None,
                                           thinking_budget: float = 0.25,
                                           refine_rounds: int = 3,
                                           param_penalty_weight: float = 0.1,
                                           log_prefix: str = ""):
    prefix = f"{log_prefix} " if log_prefix else ""
    if neuron_model_code_string is None:
        logging.info(f"{prefix}No neuron model code string provided, skipping parameter estimator generation.")
        return None, None, None
    k = min(k_max, len(current_island))
    random_programs = current_island.sample(k, replace=False).reset_index(drop=True)
    # sort from worst to best (loss descending)
    random_programs = random_programs.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
    model_fn_name = utils._get_model_function_name(project)
    use_image = img_dir is not None and image_fn is not None
    prompt = utils.create_parameter_estimator_prompt(
        random_programs,
        neuron_model_code_string=neuron_model_code_string,
        llm_type=llm_name[0], max_lines=param_estimator_max_lines,
        use_image=use_image,
        project=project,
    )
    
    random_programs_crude = random_programs.copy()
    random_programs_crude['params'] = random_programs['initial_params']
    # now try generating an image from the random programs
    if use_image:
        try:
            sup_title = "".join([f"{model_fn_name}_v{i+1}: Loss = {random_programs['train_loss'][i]:.2f} \n" for i in range(min(3, len(random_programs)))])
            image_fn(programs_df=random_programs_crude,
                                    loss_function=loss_functions.quadratic_loss,
                                    x=stimuli, y=spike_matrix,
                                    cell_selection=np.random.choice(spike_matrix.shape[0], size=4, replace=False),
                                    save_path=img_dir,
                                    labels=['v_1', 'v_2'],
                                    colours=['tab:green', 'tab:red'],
                                    dpi=384*2/20,
                                    title=sup_title,
                                    legend_fontsize=20,
                                    line_alpha=0.9,
                                    line_width=4,)
            img_path = Path(img_dir)
            with img_path.open("rb") as f:
                img_bytes = f.read()
        except Exception as e:
            logging.info(f"{prefix}Error generating image for parameter estimator prompt: {e}")
            img_bytes = None
            # if we can't generate an image, we will just use the text prompt without image
            use_image = False
    else:
        img_bytes = None
    
    llm_output = await utils.call_llm_async(prompt, model_name=llm_name, client=client, temperature=temp,
                                            thinking_budget=thinking_budget, img_bytes=img_bytes)
    # extract the code block from the LLM output
    code_string = utils.extract_code_block(llm_output)
    if code_string is None:
        logging.info(f"{prefix}No code block found in the LLM output for parameter estimator, skipping.")
        return None, None, prompt
    contains_swear_word = any(word in code_string for word in swear_words)
    if contains_swear_word:
        # find the word that is in the code_string
        swear_word = next((word for word in swear_words if word in code_string), None)
        logging.info(f"{prefix}Parameter estimator code contains swear word: {swear_word}, skipping.")
        return None, None, prompt
    code_string = re.sub(r"def\s+parameter_estimator_v\d+\s*\(", "def parameter_estimator(", code_string)
    func = utils.str_to_func(code_string, 'parameter_estimator')

    # optional refinement loop
    if refine_rounds > 0 and neuron_model_fn is not None:
        best_code = code_string
        best_func = func
        best_loss = float(jnp.inf)
        current_code = code_string
        current_func = func
        n_cells = spike_matrix.shape[0]
        n_cells_img = min(4, n_cells)
        for r in range(1, refine_rounds + 1):
            current_loss, current_params = evaluate_param_estimator_loss(
                neuron_model_fn, current_func, loss_functions.quadratic_loss,
                x=stimuli, y=spike_matrix, param_penalty_weight=param_penalty_weight
            )
            logging.info(f"{prefix}Param estimator refinement round {r}/{refine_rounds}: train loss {current_loss:.4f}")
            if current_loss < best_loss:
                best_loss = current_loss
                best_code = current_code
                best_func = current_func

            img_bytes = None
            if image_fn is not None and img_dir is not None and current_params is not None:
                try:
                    img_path = Path(img_dir)
                    refine_path = img_path.with_name(f"{img_path.stem}_refine_{r}{img_path.suffix}")
                    image_fn(
                        programs_df=pd.DataFrame({'program': [neuron_model_fn], 'params': [current_params]}),
                        loss_function=loss_functions.quadratic_loss,
                        x=stimuli, y=spike_matrix,
                        cell_selection=np.random.choice(n_cells, size=n_cells_img, replace=False),
                        save_path=str(refine_path),
                        labels=['Param Estimator'],
                        dpi=384*2/20,
                        title=f"Param Estimator Refinement (round {r})\nLoss: {current_loss:.2f}",
                    )
                    with refine_path.open("rb") as f:
                        img_bytes = f.read()
                except Exception as e:
                    logging.info(f"{prefix}Error generating refinement image: {e}")
                    img_bytes = None

            refine_prompt = utils.create_parameter_estimator_refinement_prompt(
                neuron_model_code_string=neuron_model_code_string,
                param_estimator_code_string=current_code,
                current_loss=current_loss,
                round_idx=r,
                n_rounds=refine_rounds,
                max_lines=param_estimator_max_lines,
                llm_type=llm_name[0],
                use_image=img_bytes is not None,
                project=project,
            )
            llm_output = await utils.call_llm_async(refine_prompt, model_name=llm_name, client=client, temperature=temp,
                                                   thinking_budget=thinking_budget, img_bytes=img_bytes)
            new_code = utils.extract_code_block(llm_output)
            if new_code is None:
                continue
            contains_swear_word = any(word in new_code for word in swear_words)
            if contains_swear_word:
                continue
            new_code = re.sub(r"def\s+parameter_estimator_v\d+\s*\(", "def parameter_estimator(", new_code)
            new_func = utils.str_to_func(new_code, 'parameter_estimator')
            logging.info(f"{prefix}Parameter Estimator (refinement round {r}/{refine_rounds}):\n{new_code}\n")
            current_code = new_code
            current_func = new_func

        return best_code, best_func, prompt

    return code_string, func, prompt

async def generate_new_parameter_estimator_from_image_feedback(image_prompt: str,
                                                               image_dir: str,
                                                               model_name='gemini-2.0-flash',
                                                               swear_words=['lstsq', 'scipy.optimize', 'optimize.minimize', 'curve_fit', 'sklearn'],
                                                               max_lines=100,
                                                               temp=1,
                                                               client=None) -> tuple[str, callable]:
    """ Generates a new parameter estimator from an image feedback prompt.
    Args:
        image_prompt (str): The prompt string for the AI to generate a new parameter estimator.
        image_dir (str): Directory where the image is stored.
        swear_words (list): List of words that should not be present in the generated code.
        max_lines (int): Maximum number of lines for the generated code.
        client: The genai client to use for LLM calls.
    Returns:
        tuple[str, callable]: The generated parameter estimator code string and the function object.
    """
    if image_prompt is None or image_dir is None:
        logging.info("No image prompt or image directory provided for parameter estimator generation.")
        return None, None
    # load image as bytes
    image_path = Path(image_dir)
    if not image_path.exists():
        logging.info(f"Image path {image_path} does not exist, skipping parameter estimator generation from image feedback.")
        return None, None
    with image_path.open("rb") as f:
        img_bytes = f.read()
    # call the LLM with the image prompt and image bytes
    llm_output = await utils.call_llm_async(image_prompt, model_name=model_name, client=client, temperature=temp, img_bytes=img_bytes)
    code_string = utils.extract_code_block(llm_output) # extract the code block from the LLM output
    if code_string is None:
        logging.info("No code block found in the LLM output for parameter estimator from image feedback, skipping.")
        return None, None
    # check for swear words
    contains_swear_word = any(word in code_string for word in swear_words)
    if contains_swear_word:
        swear_word = next((word for word in swear_words if word in code_string), None)
        logging.info(f"Parameter estimator code contains swear word: {swear_word}, skipping.")
        return None, None
    # extract the function from the code string
    func = utils.str_to_func(code_string, 'parameter_estimator')
    return code_string, func

async def translate_to_jax(code_string: str, client, llm_name='gemini-2.0-flash-lite',
                           thinking_budget: float = 0.0, function_name: str | None = None,
                           project: str | None = None,
                           eval_stimuli=None, max_attempts: int = 3,
                           max_abs_err: float = 1e-5, max_rel_err: float = 1e-3) -> tuple[str, callable]:
    """
    Translates a neuron model code string to JAX format.
    Args:
        code_string (str): The neuron model code string to translate.
    Returns:
        callable: The translated JAX function.
    """
    if code_string is None:
        logging.info("No neuron model code string provided for translation.")
        return None, None
    
    prompt = utils.create_jax_translater_prompt(code_string, project=project or "orientation")
    if prompt is None:
        return None, None

    population_mode = False
    if project is not None:
        try:
            population_mode = bool(utils._load_prompt_config(project).get("POPULATION_MODE", False))
        except Exception:
            population_mode = False

    fallback_name = utils._get_model_function_name(project) if project else 'neuron_model'
    orig_name = function_name or _extract_first_function_name(code_string) or fallback_name
    orig_func = utils.str_to_func(code_string, orig_name)

    def _make_params_row(stim, n_model_params: int, vector_first: bool = False):
        if n_model_params <= 0:
            return np.asarray([], dtype=float)
        if not vector_first:
            return np.zeros(n_model_params, dtype=float)
        n_feat = int(stim.shape[-1]) if getattr(stim, "ndim", 0) > 1 else 1
        if n_model_params == 1:
            return np.zeros(n_feat, dtype=float)
        return np.zeros(n_feat + (n_model_params - 1), dtype=float)

    def _make_structured_params(stim, n_model_params: int):
        n_source = int(stim.shape[1]) if getattr(stim, "ndim", 0) > 1 else 1
        n_target = 3
        if n_model_params == 1:
            return (np.zeros((n_source, n_target), dtype=float),)
        if n_model_params == 2:
            return (np.zeros((n_source, n_target), dtype=float), np.zeros(n_target, dtype=float))
        if n_model_params == 4:
            zeros = np.zeros(n_target, dtype=float)
            return (np.zeros((n_source, n_target), dtype=float), zeros, zeros, zeros)
        return tuple(np.zeros(n_target, dtype=float) for _ in range(n_model_params))

    def _try_validate(jax_func, stim, params_row, n_model_params: int):
        try:
            if population_mode:
                parts = _make_structured_params(stim, n_model_params)
                y_np = np.asarray(orig_func(stim, *parts))
                y_jax = np.asarray(jax_func(jnp.asarray(stim), *[jnp.asarray(p) for p in parts]))
            else:
                y_np = np.asarray(utils.call_model_with_params(orig_func, stim, params_row, n_model_params))
                y_jax = np.asarray(utils.call_model_with_params(jax_func, jnp.asarray(stim), jnp.asarray(params_row), n_model_params))
        except Exception as e:
            logging.info(f"Translation validation failed to execute: {e}")
            return False
        if y_np.shape != y_jax.shape:
            logging.info(f"Translation validation shape mismatch: {y_np.shape} vs {y_jax.shape}")
            return False
        abs_err = float(np.mean(np.abs(y_np - y_jax)))
        rel_err = float(abs_err / (np.mean(np.abs(y_np)) + 1e-8))
        if abs_err > max_abs_err and rel_err > max_rel_err:
            logging.info(f"Translation validation failed (abs_err={abs_err:.3e}, rel_err={rel_err:.3e})")
            return False
        return True

    def _validate_translation(jax_func):
        if eval_stimuli is None:
            return True
        if orig_func is None or jax_func is None:
            logging.info("Translation validation skipped: missing original or JAX function.")
            return False
        stim = np.asarray(eval_stimuli)
        n_model_params = len(inspect.signature(orig_func).parameters) - 1
        params_row = _make_params_row(stim, n_model_params, vector_first=False)
        if _try_validate(jax_func, stim, params_row, n_model_params):
            return True
        if not population_mode and getattr(stim, "ndim", 0) > 1:
            params_row = _make_params_row(stim, n_model_params, vector_first=True)
            if _try_validate(jax_func, stim, params_row, n_model_params):
                return True
        return False

    last_code = None
    for attempt in range(1, max_attempts + 1):
        jax_code_string = await utils.call_llm_async(
            prompt,
            client=client,
            model_name=llm_name,
            temperature=0,
            thinking_budget=thinking_budget,
        )
        jax_code_string = utils.extract_code_block(jax_code_string)
        if jax_code_string is None:
            last_code = None
            continue
        last_code = jax_code_string
        needle = function_name or fallback_name
        func = utils.str_to_func(jax_code_string, needle)
        if func is None:
            first_fn = _extract_first_function_name(jax_code_string)
            if first_fn and first_fn != needle:
                logging.info(f"Function '{needle}' not found in JAX translation; trying '{first_fn}'.")
                func = utils.str_to_func(jax_code_string, first_fn)
        if _validate_translation(func):
            return jax_code_string, func
        logging.info(f"Rejecting JAX translation attempt {attempt}/{max_attempts}; retrying.")

    return last_code, None

def compute_evaluation_matrix(program: callable, params: jnp.ndarray, n_evaluation_points: int = 100,
                              eval_stimuli=None) -> jnp.ndarray:
    """
    Computes the evaluation matrix for a given program and parameters.
    Args:
        program (callable): The neuron model function.
        params (jnp.ndarray): The parameters for the neuron model. (n_cells, n_params)
        n_evaluation_points (int): Number of points to evaluate the model at.
        eval_stimuli (jnp.ndarray or None): Optional stimuli array to evaluate on.
                                            If None, defaults to a 1D angle grid.
    Returns:
        jnp.ndarray: The evaluation matrix of shape (n_cells, n_evaluation_points).
    """
    if eval_stimuli is None:
        eval_stimuli = jnp.linspace(0, 2 * jnp.pi, n_evaluation_points)
    params = jnp.asarray(params)
    stim = jnp.asarray(eval_stimuli)
    n_model_params = len(inspect.signature(program).parameters) - 1
    if params.ndim == 2 and params.shape[0] == 1 and stim.ndim >= 2 and n_model_params > 0:
        n_source = int(stim.shape[1]) if stim.ndim > 1 else 1
        params_len = int(params.shape[1])
        if n_model_params == 1:
            n_target = max(1, params_len // n_source)
        elif n_model_params == 2:
            n_target = max(1, params_len // (n_source + 1))
        elif n_model_params == 4:
            n_target = max(1, params_len // (n_source + 3))
        else:
            n_target = max(1, params_len // max(1, n_model_params))
        parts = utils.unpack_population_params(params[0], n_source, n_target, n_model_params)
        y_eval = program(stim, *parts)
        return jnp.asarray(y_eval)[None, ...]
    program_vmap = utils.vmap_over_cells(program)
    y_eval = program_vmap(eval_stimuli, params)
    return y_eval

async def main(project: str = "place_cells",
                data_kwargs: dict | None = None,
                model_thinking_budget: float = 0.0,
                param_est_thinking_budget: float = 0.0,
                translation_thinking_budget: float = 0.0,
                param_est_refine_rounds: int = 3,
                n_iterations=9, time_limit=60, k_max=2, n_islands=8, batch_size=6, 
                critical_population_size=12, min_wise_population_size=0, 
                n_migrants=2, fit_params=True, tol=1e-6, exploit_point=0.5,
                param_penalty_weight=0.01, FAILED_PROGRAM_COST=np.inf,
                exploration_topology = [1, 2, 3, 4, 5, 6, 7, 0],
                exploitation_topology = [1, 2, 3, 4, 5, 6, 7, 0],
                tiny_lm_name = 'gemini-2.0-flash-lite',
                little_lm_name = 'gemini-2.0-flash',
                large_lm_name = 'gemini-2.5-flash',
                use_large_every = 3):
    """ 
    Main function to run the hypothesis engine.
    """
    # load api keys
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    X, Y = _load_project_data(project, data_kwargs=data_kwargs)
    X, Y = _ensure_xy(X, Y)
    project_config = utils._load_prompt_config(project)
    population_mode = bool(project_config.get("POPULATION_MODE", False))
    if population_mode:
        Y = jnp.asarray(Y)
        if Y.ndim == 2:
            Y = jnp.transpose(Y)[None, ...]
        elif Y.ndim == 3 and Y.shape[0] != 1:
            raise ValueError(f"Population mode expects Y with leading dimension 1; got shape {tuple(Y.shape)}.")
    n_cells, n_trials = Y.shape[0], Y.shape[1]
    if n_cells == 0 or n_trials == 0:
        logging.info("No cells or trials after data loading/filtering; aborting run.")
        print("No cells or trials after data loading/filtering; aborting run.")
        return
    # normalize response per cell
    norms = jnp.linalg.norm(Y, axis=tuple(range(1, Y.ndim)), keepdims=True)
    norms = jnp.where(norms == 0, 1.0, norms)
    Y = 100 * Y / norms
    if population_mode:
        training_cells = jnp.array([0])
        test_cells = jnp.array([0])
        response_train, response_test = Y, Y
        x_train, x_test = X, X
        n_targets = int(response_train.shape[2]) if response_train.ndim > 2 else 0
        print(f"Population mode: using {n_targets} target cells (no cell split).")
    else:
        key = jax.random.PRNGKey(42)
        training_size = n_cells // 2
        shuffled_indices = jax.random.permutation(key, jnp.arange(n_cells))
        training_cells, test_cells = shuffled_indices[:training_size], shuffled_indices[training_size:]
        response_train, response_test = Y[training_cells, :], Y[test_cells, :]
        x_train, x_test = X, X
        print(f"Using {len(training_cells)} cells for training and {len(test_cells)} cells for testing.")
    translation_eval_stimuli = _translation_eval_stimuli(x_train, response_train.shape[0])

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

    seed_module = _load_project_seed_models(project)
    image_diagnostics_module = _load_project_image_diagnostics(project)
    model_fn_name = project_config.get("MODEL_FUNCTION_NAME", "neuron_model")
    if not isinstance(model_fn_name, str) or not model_fn_name.strip():
        model_fn_name = "neuron_model"
    model_label = project_config.get("MODEL_LABEL", "Neuron Model")
    if not isinstance(model_label, str) or not model_label.strip():
        model_label = "Neuron Model"
    model_diag_name = project_config.get("MODEL_IMAGE_DIAGNOSTIC_FN", "")
    param_diag_name = project_config.get("PARAM_EST_IMAGE_DIAGNOSTIC_FN", "")
    if isinstance(model_diag_name, str) and model_diag_name.strip():
        model_image_diagnostic_fn = getattr(image_diagnostics_module, model_diag_name, None)
    else:
        model_image_diagnostic_fn = None
    if isinstance(param_diag_name, str) and param_diag_name.strip():
        param_est_image_diagnostic_fn = getattr(image_diagnostics_module, param_diag_name, None)
    else:
        param_est_image_diagnostic_fn = None

    # determine evaluation stimuli (supports multi-dimensional stimuli)
    eval_stimuli = None
    if X.shape[0] > 1:
        n_eval = min(100, X.shape[1])
        eval_stimuli = _stimuli_trials_first(X[:, :n_eval])
    def _diag_entity_count(resp):
        if population_mode and resp.ndim >= 3:
            return int(resp.shape[2])
        return int(resp.shape[0])

    diag_count_train = _diag_entity_count(response_train)
    diag_count_test = _diag_entity_count(response_test)
    n_diag_cells = min(9, diag_count_test, diag_count_train)
    stimuli_plot_train = _plot_stimuli_for_cells(x_train, response_train.shape[0])
    stimuli_plot_test = _plot_stimuli_for_cells(x_test, response_test.shape[0])
    use_model_images = model_image_diagnostic_fn is not None
    use_param_est_images = param_est_image_diagnostic_fn is not None
    if stimuli_plot_train is None and use_model_images:
        logging.info("Disabling model image feedback: stimuli not compatible with diagnostics.")
        use_model_images = False
    if stimuli_plot_train is None and use_param_est_images:
        logging.info("Disabling parameter-estimator image feedback: stimuli not compatible with diagnostics.")
        use_param_est_images = False
    
    # store and compute loss of 2 initial programs
    t_start = time.time()
    seed_models_config = project_config.get("SEED_MODELS", [])
    if not isinstance(seed_models_config, list) or len(seed_models_config) < 2:
        raise ValueError(f"Project '{project}' must define at least two SEED_MODELS entries in config.json.")
    seed_models = []
    for entry in seed_models_config:
        if not isinstance(entry, (list, tuple)) or len(entry) not in (2, 3):
            raise ValueError("Each SEED_MODELS entry must be [numpy_fn, param_est_fn] or [numpy_fn, jax_fn, param_est_fn].")
        if len(entry) == 2:
            numpy_name, est_name = entry
            jax_name = None
        else:
            numpy_name, jax_name, est_name = entry
        if not all(isinstance(name, str) and name.strip() for name in (numpy_name, est_name)):
            raise ValueError("SEED_MODELS entries must be non-empty function names.")
        if jax_name is not None and (not isinstance(jax_name, str) or not jax_name.strip()):
            raise ValueError("SEED_MODELS JAX function name must be non-empty when provided.")
        try:
            numpy_fn = getattr(seed_module, numpy_name)
        except AttributeError:
            raise AttributeError(f"Seed model function '{numpy_name}' not found in projects.{project}.seed_models.")
        jax_fn = None
        if jax_name is not None:
            try:
                jax_fn = getattr(seed_module, jax_name)
            except AttributeError:
                raise AttributeError(f"Seed model JAX function '{jax_name}' not found in projects.{project}.seed_models.")
        try:
            est_fn = getattr(seed_module, est_name)
        except AttributeError:
            raise AttributeError(f"Seed parameter estimator '{est_name}' not found in projects.{project}.seed_models.")
        seed_models.append((numpy_fn, jax_fn, est_fn))
    if len(seed_models) < 2:
        raise ValueError(f"Project '{project}' must define at least two seed models.")
    numpy_programs = [seed_models[0][0], seed_models[1][0]]
    jax_programs = [seed_models[0][1], seed_models[1][1]]
    param_estimators = [seed_models[0][2], seed_models[1][2]]
    seed_losses = np.zeros(2)
    seed_imports = _get_module_import_lines(seed_module)
    for i in range(2):
        # get the program, parameter estimator, and jax program
        program_num = numpy_programs[i]
        param_est = param_estimators[i]
        # format strings
        import_string = seed_imports
        if "import numpy as np" not in import_string:
            import_string += "import numpy as np\n"
        import_string_jax = seed_imports
        if "import jax.numpy as jnp" not in import_string_jax:
            import_string_jax += "import jax.numpy as jnp\n"
        program_name = program_num.__name__
        param_est_name = param_est.__name__
        program_source = inspect.getsource(program_num)
        program_code_string = program_source.replace(f'def {program_name}(', f'def {model_fn_name}_v{i+1}(')
        program_code_string = import_string + program_code_string
        parameter_estimator_code_string = inspect.getsource(param_est).replace(f'def {param_est_name}(', f'def parameter_estimator_v{i+1}(')
        parameter_estimator_code_string = import_string + parameter_estimator_code_string
        program_jax = jax_programs[i]
        program_jax_code_string = None
        if program_jax is None:
            program_code_for_translation = import_string + program_source
            program_jax_code_string, program_jax = await translate_to_jax(
                program_code_for_translation,
                client,
                tiny_lm_name,
                thinking_budget=translation_thinking_budget,
                function_name=program_name,
                project=project,
                eval_stimuli=translation_eval_stimuli,
            )
            if program_jax is None:
                raise ValueError("Failed to translate seed model to JAX.")
        else:
            program_jax_name = program_jax.__name__
            program_jax_code_string = inspect.getsource(program_jax).replace(f'def {program_jax_name}(', f'def {model_fn_name}_v{i+1}(')
            program_jax_code_string = import_string_jax + program_jax_code_string

        # score the initial program
        loss_init, params_init, loss, params = objective(program_jax, param_est, 
                                        loss_func=loss_functions.quadratic_loss, 
                                        x=x_train, y=response_train, 
                                        fit_params=fit_params, param_penalty_weight=param_penalty_weight, tol=tol)
        seed_losses[i] = loss
        y_eval = compute_evaluation_matrix(program_jax, params, n_evaluation_points=100, eval_stimuli=eval_stimuli)

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
    if use_model_images and stimuli_plot_train is not None:
        model_image_diagnostic_fn(programs_df=initial_programs,
                                   loss_function=loss_functions.quadratic_loss,
                                   x=stimuli_plot_train, y=response_train,
                                   cell_selection=np.random.choice(diag_count_train, size=n_diag_cells, replace=False),
                                   save_path=os.path.join(image_feedback_dir, 'initial_programs.png'),
                                   labels=['seed_1', 'seed_2'],
                                   colours=['tab:green', 'tab:red'],
                                   dpi=100.0,
                                   title="Seed Programs",
                                   legend_fontsize=20,
                                   line_alpha=0.9,
                                   line_width=4,)

    # -----------------------------
    # HYPOTHESIS ENGINE
    # -----------------------------
    pbar = tqdm(range(n_iterations), desc="Hypothesis Engine Iterations")
    for i in pbar:
        # check if time limit is reached
        if time.time() - t_start > time_limit * 60:
            logging.info(f"Time limit of {time_limit} minutes reached. Stopping iterations.")
            break
        logging.info(f"Iteration {i}")
        if use_large_every > 0 and i % use_large_every == 0:
            llm_name = large_lm_name
            logging.info(f"Using large LLM: {llm_name}")
        else:
            llm_name = little_lm_name
            logging.info(f"Using little LLM: {llm_name}")
        mode = 'explore' if i < n_iterations * exploit_point else 'exploit'
        temperature = 1 + np.exp(-i / n_iterations)
        pbar.set_description(f"Iteration {i+1}/{n_iterations} ({mode})")
        pbar.set_postfix({"refine": param_est_refine_rounds})

        model_image_dirs = np.empty((n_islands, batch_size), dtype=object)
        param_est_image_dirs = np.empty((n_islands, batch_size), dtype=object)
        for island_idx in range(n_islands):
            for j in range(batch_size):
                if use_model_images:
                    model_image_dirs[island_idx, j] = os.path.join(image_feedback_dir, f'iter_{i}_island_{island_idx}_batch_{j}.png')
                else:
                    model_image_dirs[island_idx, j] = None
                if use_param_est_images:
                    param_est_image_dirs[island_idx, j] = os.path.join(image_feedback_dir, f'iter_{i}_island_{island_idx}_batch_{j}_param_est.png')
                else:
                    param_est_image_dirs[island_idx, j] = None
        # generate new programs
        neuron_model_generation_tasks = [generate_new_neuron_model(islands[island_idx], 
                                                                   llm_name=llm_name, 
                                                                   client=client, 
                                                                   mode=mode, 
                                                                   k_max=k_max, 
                                                                   temp=temperature,
                                                                   spike_matrix=response_train, 
                                                                   stimuli=stimuli_plot_train,
                                                                   img_dir=model_image_dirs[island_idx, j],
                                                                   project=project,
                                                                   image_fn=model_image_diagnostic_fn,
                                                                   model_fn_name=model_fn_name,
                                                                   model_label=model_label,
                                                                   thinking_budget=model_thinking_budget) 
                                         for island_idx in range(n_islands) for j in range(batch_size)]
        logging.info(f"Generating {n_islands * batch_size} new programs... Model: {llm_name}, mode: {mode}, temperature: {temperature:.2f}")
        print(f"Generating {n_islands * batch_size} new programs... Model: {llm_name}, mode: {mode}, temperature: {temperature:.2f}")
        logging.info(f"Stage 1: {model_label} generation (prompts + code)")
        neuron_model_results = await asyncio.gather(*neuron_model_generation_tasks)
        neuron_model_code_strings = [result[0] for result in neuron_model_results]
        neuron_model_prompts = [result[1] for result in neuron_model_results]
        parent_ids = [result[2] for result in neuron_model_results]
        n_candidates = n_islands * batch_size
        for idx in range(n_candidates):
            island_idx = idx // batch_size
            j = idx % batch_size
            log_prefix = f"id={i},{island_idx},{j}"
            prompt = neuron_model_prompts[idx] if neuron_model_prompts[idx] is not None else "<failed>"
            code_string = neuron_model_code_strings[idx] if neuron_model_code_strings[idx] is not None else "<failed>"
            logging.info(f"{log_prefix} {model_label} Prompt (initial):\n{prompt}\n")
            logging.info(f"{log_prefix} {model_label} (initial):\n{code_string}\n")
        
        # convert to jax
        neuron_model_function_translation_tasks = []
        for code_string in neuron_model_code_strings:
            fn_name = _extract_first_function_name(code_string) if code_string is not None else None
            neuron_model_function_translation_tasks.append(
                translate_to_jax(
                    code_string,
                    client,
                    tiny_lm_name,
                    thinking_budget=translation_thinking_budget,
                    function_name=fn_name,
                    project=project,
                    eval_stimuli=translation_eval_stimuli,
                )
            )
        jax_results = await asyncio.gather(*neuron_model_function_translation_tasks)
        neuron_model_results = [(neuron_model_code_strings[j], neuron_model_prompts[j], jax_results[j][0], jax_results[j][1]) for j in range(n_islands * batch_size)]
        
        # build parameter‑estimator tasks (initial only, no refinement here)
        param_estimation_tasks = [
            generate_new_parameter_estimator(
                current_island=islands[island_idx],
                neuron_model_code_string=neuron_model_code_strings[island_idx * batch_size + j],
                neuron_model_fn=neuron_model_results[island_idx * batch_size + j][3],
                llm_name=little_lm_name,  # same model used for programs
                client=client,
                spike_matrix=response_train, # training data
                stimuli=stimuli_plot_train,
                k_max=2,
                temp=temperature,
                param_estimator_max_lines=100,
                img_dir=param_est_image_dirs[island_idx, j],
                project=project,
                image_fn=param_est_image_diagnostic_fn,
                thinking_budget=param_est_thinking_budget,
                refine_rounds=0,
                param_penalty_weight=param_penalty_weight,
                log_prefix=f"id={i},{island_idx},{j}",
            )
            for island_idx in range(n_islands)
            for j in range(batch_size)
        ]

        logging.info(
            f"Generating {n_candidates} parameter estimators (initial) "
            f"(LLM={little_lm_name}, mode={mode}, T={temperature:.2f})"
        )
        logging.info("Stage 2: Parameter estimator generation (initial, prompts + code)")
        param_est_results = await asyncio.gather(*param_estimation_tasks)
        param_est_code_strings = [result[0] for result in param_est_results]
        param_est_funcs = [result[1] for result in param_est_results]
        param_est_prompts = [result[2] for result in param_est_results]
        for idx in range(n_candidates):
            island_idx = idx // batch_size
            j = idx % batch_size
            log_prefix = f"id={i},{island_idx},{j}"
            prompt = param_est_prompts[idx] if param_est_prompts[idx] is not None else "<failed>"
            code_string = param_est_code_strings[idx] if param_est_code_strings[idx] is not None else "<failed>"
            logging.info(f"{log_prefix} Parameter Estimator Prompt (initial):\n{prompt}\n")
            logging.info(f"{log_prefix} Parameter Estimator (initial):\n{code_string}\n")

        # track current/best parameter estimators per candidate
        param_est_states = []
        for code_string, func in zip(param_est_code_strings, param_est_funcs):
            param_est_states.append({
                "current_code": code_string,
                "current_func": func,
                "current_round": 0,
                "best_code": code_string,
                "best_func": func,
                "best_round": 0,
                "best_loss": float(jnp.inf),
            })

        # refinement stages (from previous estimators, not parents)
        param_est_swear_words = ['lstsq', 'scipy.optimize', 'optimize.minimize', 'curve_fit', 'sklearn']
        if param_est_refine_rounds > 0:
            logging.info(
                f"Refining {n_candidates} parameter estimators for {param_est_refine_rounds} rounds "
                f"(LLM={little_lm_name}, T={temperature:.2f})"
            )
            n_cells = diag_count_train
            n_cells_img = min(4, n_cells)
            for r in range(1, param_est_refine_rounds + 1):
                logging.info(f"Parameter estimator refinement round {r}/{param_est_refine_rounds}")
                refine_tasks = []
                refine_meta = []
                for idx in range(n_candidates):
                    island_idx = idx // batch_size
                    j = idx % batch_size
                    log_prefix = f"id={i},{island_idx},{j}"
                    state = param_est_states[idx]
                    neuron_model_new = neuron_model_results[idx][3]
                    if neuron_model_new is None or state["current_func"] is None or state["current_code"] is None:
                        continue

                    current_loss, current_params = evaluate_param_estimator_loss(
                        neuron_model_new,
                        state["current_func"],
                        loss_functions.quadratic_loss,
                        x=x_train,
                        y=response_train,
                        param_penalty_weight=param_penalty_weight,
                    )
                    logging.info(f"{log_prefix} Param Estimator (round {state['current_round']}) train loss {current_loss:.4f}")
                    if current_loss < state["best_loss"]:
                        state["best_loss"] = current_loss
                        state["best_code"] = state["current_code"]
                        state["best_func"] = state["current_func"]
                        state["best_round"] = state["current_round"]

                    img_bytes = None
                    if use_param_est_images and param_est_image_diagnostic_fn is not None and current_params is not None:
                        img_path = param_est_image_dirs[island_idx, j]
                        if img_path is not None:
                            try:
                                base_path = Path(img_path)
                                refine_path = base_path.with_name(f"{base_path.stem}_refine_{r}{base_path.suffix}")
                                param_est_image_diagnostic_fn(
                                    programs_df=pd.DataFrame({'program': [neuron_model_new], 'params': [current_params]}),
                                    loss_function=loss_functions.quadratic_loss,
                                    x=stimuli_plot_train,
                                    y=response_train,
                                    cell_selection=np.random.choice(n_cells, size=n_cells_img, replace=False),
                                    save_path=str(refine_path),
                                    labels=[f'Refine {r}'],
                                    dpi=384*2/20,
                                    title=f"Param Estimator Refinement (round {r})\nLoss: {current_loss:.2f}",
                                )
                                with refine_path.open("rb") as f:
                                    img_bytes = f.read()
                            except Exception as e:
                                logging.info(f"{log_prefix} Error generating refinement image: {e}")
                                img_bytes = None

                    refine_prompt = utils.create_parameter_estimator_refinement_prompt(
                        neuron_model_code_string=neuron_model_results[idx][0],
                        param_estimator_code_string=state["current_code"],
                        current_loss=current_loss,
                        round_idx=r,
                        n_rounds=param_est_refine_rounds,
                        max_lines=100,
                        llm_type=little_lm_name[0],
                        use_image=img_bytes is not None,
                        project=project,
                    )
                    logging.info(
                        f"{log_prefix} Parameter Estimator Prompt (refinement round {r}/{param_est_refine_rounds}):\n{refine_prompt}\n"
                    )
                    refine_tasks.append(
                        utils.call_llm_async(
                            refine_prompt,
                            model_name=little_lm_name,
                            client=client,
                            temperature=temperature,
                            thinking_budget=param_est_thinking_budget,
                            img_bytes=img_bytes,
                        )
                    )
                    refine_meta.append((idx, log_prefix))

                if refine_tasks:
                    refine_outputs = await asyncio.gather(*refine_tasks)
                    for llm_output, meta in zip(refine_outputs, refine_meta):
                        idx, log_prefix = meta
                        new_code = utils.extract_code_block(llm_output)
                        if new_code is None:
                            logging.info(f"{log_prefix} No code block found in LLM output for refinement; keeping previous.")
                            continue
                        if any(word in new_code for word in param_est_swear_words):
                            swear_word = next((word for word in param_est_swear_words if word in new_code), None)
                            logging.info(f"{log_prefix} Parameter estimator refinement contains swear word: {swear_word}; keeping previous.")
                            continue
                        new_code = re.sub(r"def\s+parameter_estimator_v\d+\s*\(", "def parameter_estimator(", new_code)
                        new_func = utils.str_to_func(new_code, 'parameter_estimator')
                        state = param_est_states[idx]
                        state["current_code"] = new_code
                        state["current_func"] = new_func
                        state["current_round"] = r
                        logging.info(
                            f"{log_prefix} Parameter Estimator (refinement round {r}/{param_est_refine_rounds}):\n{new_code}\n"
                        )

            # final evaluation to include last refinement
            for idx in range(n_candidates):
                island_idx = idx // batch_size
                j = idx % batch_size
                log_prefix = f"id={i},{island_idx},{j}"
                state = param_est_states[idx]
                neuron_model_new = neuron_model_results[idx][3]
                if neuron_model_new is None or state["current_func"] is None:
                    continue
                final_loss, _ = evaluate_param_estimator_loss(
                    neuron_model_new,
                    state["current_func"],
                    loss_functions.quadratic_loss,
                    x=x_train,
                    y=response_train,
                    param_penalty_weight=param_penalty_weight,
                )
                logging.info(f"{log_prefix} Param Estimator (round {state['current_round']}) train loss {final_loss:.4f} [post-refinement]")
                if final_loss < state["best_loss"]:
                    state["best_loss"] = final_loss
                    state["best_code"] = state["current_code"]
                    state["best_func"] = state["current_func"]
                    state["best_round"] = state["current_round"]

        # fallback to current estimator if no best selected
        for state in param_est_states:
            if state["best_func"] is None and state["current_func"] is not None:
                state["best_code"] = state["current_code"]
                state["best_func"] = state["current_func"]
                state["best_round"] = state["current_round"]

        param_est_best_results = [
            (state["best_code"], state["best_func"], state["best_round"], state["best_loss"])
            for state in param_est_states
        ]

        # combine results
        island_results = [
            [
                neuron_model_results[island_idx * batch_size + j]
                + param_est_best_results[island_idx * batch_size + j]
                for j in range(batch_size)
            ]
            for island_idx in range(n_islands)
        ]

        # now loop through the results and compute losses
        success_rate = 0.0
        for island_idx, j in np.ndindex(n_islands, batch_size):
            logging.info(f"id={i},{island_idx},{j}")
            neuron_model_code_string, prompt, neuron_model_code_string_jax, neuron_model_new, param_est_code_string, param_est_new, param_est_best_round, param_est_best_loss = island_results[island_idx][j]
            parent1_id, parent2_id = parent_ids[island_idx * batch_size + j]
            if neuron_model_new is None or param_est_new is None:
                logging.info(f"Skipping island {island_idx}, batch {j} due to LLM generation failure.")
                logging.info('-' * 50)
                continue
            
            initial_loss, initial_params, loss, optimized_params = objective(neuron_model_new, param_est_new, 
                                                                                loss_func=loss_functions.quadratic_loss,
                                                                                x=x_train, y=response_train,
                                                                                param_penalty_weight=param_penalty_weight,
                                                                                fit_params=fit_params, tol=tol)
            if loss == FAILED_PROGRAM_COST:
                logging.info('-' * 50)
                continue

            y_eval = compute_evaluation_matrix(neuron_model_new, optimized_params, n_evaluation_points=100, eval_stimuli=eval_stimuli)
            logging.info(f"Loss: {loss:.2f}\n")
            logging.info(f"{model_label}: \n{neuron_model_code_string}\n")
            logging.info(f"{model_label} (JAX): \n{neuron_model_code_string_jax}\n")
            best_loss_str = f"{param_est_best_loss:.4f}" if np.isfinite(param_est_best_loss) else "inf"
            logging.info(
                f"Parameter Estimator (best, round {param_est_best_round}, no-GD loss {best_loss_str}): \n{param_est_code_string}\n"
            )


            # plot the fits of the neuron model and parameter estimator if using image feedback
            if use_param_est_images and stimuli_plot_train is not None:
                param_est_image_diagnostic_fn(
                    programs_df=pd.DataFrame({'program': [neuron_model_new, neuron_model_new], 'params': [initial_params, optimized_params]}),
                    loss_function=loss_functions.quadratic_loss,
                    x=stimuli_plot_train,
                    y=response_train,
                    cell_selection=np.random.choice(diag_count_train, size=4, replace=False),
                    colours=['tab:green', 'tab:red'],
                    labels=['Param Estimator', 'Gradient Descent'],
                    line_alpha=1.0,
                    line_width=5.0,
                    point_alpha=0.2,
                    point_size=120,
                    legend_fontsize=20,
                    title=f"Updated Parameter Estimator and Gradient Descent Fit \n"
                        f"Initial Loss: {initial_loss:.2f}, Final Loss: {loss:.2f}",
                    save_path=os.path.join(image_feedback_dir, f'iter_{i}_island_{island_idx}_batch_{j}_updated_param_est.png')
                )
            
            param_names = list(inspect.signature(neuron_model_new).parameters.keys())[1:]
            if optimized_params.shape[1] == len(param_names):
                df = pd.DataFrame(np.array(optimized_params)[:10], columns=param_names)
                logging.info(f"Optimized Parameters for 10 cells:\n{df}\n")
            t_added = time.time() - t_start
            new_program_df = pd.DataFrame({'program_code_string': neuron_model_code_string,
                                        'program': neuron_model_new,
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
            print(f"iteration {i}, island {island_idx}, batch {j}, loss: {loss:.2f}")
            print('-' * 50)
            logging.info("-" * 50)
        print("Success rate:", success_rate)

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
            top_df = islands[island_idx].sort_values(by='train_loss').head(3).reset_index(drop=True)
            top_df = top_df.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
            sup_title = f"Iteration {i}, Island {island_idx}, Top {len(top_df)} Programs\n"
            sup_title += "\n".join([f"model {j+1}: iter {top_df['iteration_number'][j]}, birth island {top_df['birth_island'][j]}, batch {top_df['batch_index'][j]}, loss: {top_df['train_loss'][j]:.2f}" for j in range(len(top_df))])
            if use_model_images and stimuli_plot_train is not None:
                model_image_diagnostic_fn(
                    programs_df=top_df,
                    loss_function=loss_functions.quadratic_loss,
                    x=stimuli_plot_train,
                    y=response_train,
                    cell_selection=np.random.choice(diag_count_train, size=n_diag_cells, replace=False),
                    title=sup_title,
                    save_path=os.path.join(iteration_dir, f'island_{island_idx}_top_programs.png'),
                    dpi=300.0)
        
        all_programs = pd.concat([islands[idx] for idx in range(n_islands)], ignore_index=True)
        top_programs = all_programs.sort_values(by='train_loss').head(3).reset_index(drop=True)
        top_programs = top_programs.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
        sup_title = f"Iteration {i}, Top 3 Programs Overall\n"
        sup_title += "\n".join([f"model {j+1}: iter {top_programs['iteration_number'][j]}, birth island {top_programs['birth_island'][j]}, batch {top_programs['batch_index'][j]}, loss: {top_programs['train_loss'][j]:.2f}" for j in range(len(top_programs))])
        if use_model_images and stimuli_plot_train is not None:
            model_image_diagnostic_fn(
                programs_df=top_programs,
                loss_function=loss_functions.quadratic_loss,
                x=stimuli_plot_train,
                y=response_train,
                cell_selection=np.random.choice(diag_count_train, size=n_diag_cells, replace=False),
                title=sup_title,
                save_path=os.path.join(iteration_dir, 'top_programs_overall.png'),
                dpi=300.0)
        
        # save census
        census_path = os.path.join(iteration_dir, 'census.npy')
        census_np = np.array(census, dtype=object)
        np.save(census_path, census_np)

    pbar.close()

    # -----------------------------
    # now carry out the loss calculation on the test cells
    logging.info("Calculating loss on test set...")
    for island_idx in range(n_islands):
        logging.info(f"Island {island_idx} programs:")
        for j in range(len(islands[island_idx])):
            program = islands[island_idx].iloc[j]
            neuron_model = program['program']
            param_estimator = program['parameter_estimator']
            # compute the test loss
            _, _, test_loss, optimized_params = objective(neuron_model, param_estimator,
                                                          loss_func=loss_functions.quadratic_loss,
                                                          x=x_test, y=response_test, fit_params=fit_params,
                                                          max_iter=2_000, 
                                                          param_penalty_weight=param_penalty_weight, tol=tol)
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
    diagnostic.plot_train_vs_test_loss(programs_df=combined_programs_dataframe,
                                       island_labels=[f'Island {i}' for i in range(n_islands)] + ['garden_of_eden'],
                                       save_path=os.path.join(combined_dir, 'train_vs_test_loss.png'))
    
    # ---------------------------
    df_list = [combined_programs_dataframe] + islands
    combined_dir = [os.path.join(base_dir, date_stamp, time_stamp, "combined")] 
    island_dirs = [os.path.join(base_dir, date_stamp, time_stamp, f'island_{i}') for i in range(n_islands)]
    df_dirs = combined_dir + island_dirs
    config_str = f"n_islands={n_islands}, batch_size={batch_size}, n_iterations={n_iterations},\n"
    config_str += f"llm_names={little_lm_name, large_lm_name}, fit_params={fit_params}, \n"
    config_str += f"critical_population_size={critical_population_size}.\n"

    for i, df in enumerate(df_list):
        df_sup = config_str
        df = df.head(3)
        df = df.sort_values(by='test_loss', ascending=True).reset_index(drop=True)
        df_sup += "".join([f"model {i + 1}: iter {df['iteration_number'][i]}, birth_island {df['birth_island'][i]}, batch {df['batch_index'][i]}, test loss {df['test_loss'][i]:.2f}\n" for i in range(min(3, len(df)))])
        if use_model_images and stimuli_plot_test is not None:
            model_image_diagnostic_fn(
                programs_df=df,
                loss_function=loss_functions.quadratic_loss,
                x=stimuli_plot_test,
                y=response_test,
                cell_selection=np.random.choice(diag_count_test, size=n_diag_cells, replace=False),
                title=df_sup,
                save_path=os.path.join(df_dirs[i], 'top_model_fits.png')
            )
            # plot top 3 models separately
        for j in range(min(3, len(df))):
            birth_island = df['birth_island'][j]
            iteration_number = df['iteration_number'][j]
            batch_index = df['batch_index'][j]
            cell_selection = np.random.choice(diag_count_test, size=n_diag_cells, replace=False)
            if use_model_images and stimuli_plot_test is not None:
                x_for_plot = stimuli_plot_test
                if not _is_shared_stimuli(stimuli_plot_test, response_test.shape[0]):
                    x_for_plot = stimuli_plot_test[cell_selection]
                single_df = df.iloc[[j]].copy()
                model_image_diagnostic_fn(
                    programs_df=single_df,
                    loss_function=loss_functions.quadratic_loss,
                    x=x_for_plot,
                    y=response_test,
                    cell_selection=cell_selection,
                    title=f"Island {birth_island}, Iteration {iteration_number}, Batch {batch_index}, test loss {df['test_loss'][j]:.2f}",
                    save_path=os.path.join(df_dirs[i], f'top_model_fit_{min(3, len(df)) - j}.png')
                )

if __name__ == "__main__":
    print("running place cells smoke test")
    asyncio.run(main(
        project="place_cells",
        model_thinking_budget=0.0,
        param_est_thinking_budget=0.0,
        translation_thinking_budget=0.0,
        param_est_refine_rounds=0,
        n_iterations=1,
        n_islands=2,
        batch_size=2,
        k_max=2,
        use_large_every=3,
        tiny_lm_name="gemini-2.0-flash-lite",
        little_lm_name="gemini-2.0-flash",
        large_lm_name="gemini-2.0-flash",
    ))
