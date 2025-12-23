import inspect
import os
import logging
import asyncio
import csv
import json
from typing import Sequence, Optional
import numpy as np
import jax, jax.numpy as jnp
import timeout_decorator
import optax
from pathlib import Path
import utils, genetic_helpers, loss_functions
from entities import Program, Island, ProgramSnapshot
from tqdm import tqdm
import google.genai
from dotenv import load_dotenv
import time
print(jax.default_backend())    # should print "gpu"
print(jax.devices())

CENSUS_CSV_COLUMNS = [
    "program_index",
    "generation",
    "birth_island",
    "batch_index",
    "train_loss",
    "test_loss",
    "llm_name",
    "parent1_id",
    "parent2_id",
    "timestamp",
    "param_count",
    "function_code_string",
    "parameter_estimator_code_string",
    "evaluation_matrix",
    "is_seed",
    "notes",
]

def _serialize_value(value):
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return json.dumps(np.asarray(value).tolist())
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    return value

def save_records_csv(path: str, records: Sequence[dict], columns: Optional[Sequence[str]] = None) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    records = list(records)
    if not records:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    if columns is None:
        columns = list(records[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for rec in records:
            row = {col: _serialize_value(rec.get(col)) for col in columns}
            writer.writerow(row)

def load_program_snapshots(path: str) -> list[ProgramSnapshot]:
    if not os.path.exists(path):
        return []
    snapshots: list[ProgramSnapshot] = []

    def _parse_optional_json(value: Optional[str]):
        if value in (None, "", "None"):
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parent1_raw = _parse_optional_json(row.get("parent1_id"))
            parent2_raw = _parse_optional_json(row.get("parent2_id"))

            def _as_tuple(val):
                if val is None:
                    return None
                if isinstance(val, list):
                    try:
                        return tuple(int(x) for x in val)
                    except Exception:
                        return None
                return None

            snapshot = ProgramSnapshot(
                program_index=int(row["program_index"]),
                generation=int(row["generation"]),
                birth_island=int(row["birth_island"]),
                batch_index=int(row["batch_index"]),
                train_loss=float(row["train_loss"]),
                test_loss=float(row["test_loss"]) if row.get("test_loss") not in (None, "", "None") else None,
                llm_name=row.get("llm_name"),
                parent1_id=_as_tuple(parent1_raw),
                parent2_id=_as_tuple(parent2_raw),
                timestamp=float(row["timestamp"]) if row.get("timestamp") else 0.0,
                param_count=int(row["param_count"]) if row.get("param_count") else 0,
                function_code_string=row.get("function_code_string", ""),
                parameter_estimator_code_string=row.get("parameter_estimator_code_string", ""),
                evaluation_matrix=_parse_optional_json(row.get("evaluation_matrix")),
                is_seed=row.get("is_seed", "False") in ("True", "true", "1"),
                notes=row.get("notes"),
            )
            snapshots.append(snapshot)
    return snapshots


def _default_sine_numpy(theta, A=1.0, B=0.2, phase=0.0):
    theta = np.asarray(theta)
    return B + A * np.sin(theta - phase)


def _default_cosine_numpy(theta, A=1.0, B=0.2, phase=0.0):
    theta = np.asarray(theta)
    return B + A * np.cos(theta - phase)


def _default_parameter_estimator(theta, spikes):
    theta = np.asarray(theta)
    spikes = np.asarray(spikes)
    amplitude = float(np.max(spikes) - np.min(spikes))
    bias = float(np.median(spikes))
    phase = float(theta[np.argmax(spikes)])
    return np.array([amplitude, bias, phase])

def compute_initial_params(param_estimator, func, X, Y) -> jnp.ndarray:
    """
    Compute initial parameters for func using the provided parameter estimator. Confusingly, the parameter estimator will be written in numpy,
    but the func will be written in JAX. So the data x and y will be numpy arrays, but the output will be a JAX array.
    Args:
        param_estimator (function): Function to estimate initial parameters for the func.
                                    Signature: param_estimator(stimuli, response) -> params
        func (function): The model which predicts neural activity from stimuli and free parameters.
                                 Signature: func(stimuli, *params) -> activity
        X (np.ndarray): Stimuli data, shape (n_units, n_points).
        Y (np.ndarray): Response data, shape (n_units, n_points).
    Returns:
        jnp.ndarray: The estimated parameters for each unit, shape (n_units, n_params).
                     If the parameter estimation fails, returns an array of default parameters based on the func's signature.
                     If this also fails, returns None.
    """
    if param_estimator is not None:
        @timeout_decorator.timeout(5, use_signals=True)
        def _safe_estimate(pe, xi, yi):
            return pe(xi, yi)
        try:
            # any call taking >5s will raise timeout_decorator.TimeoutError
            return jnp.array([_safe_estimate(param_estimator, X[i], Y[i])for i in range(Y.shape[0])])
        except timeout_decorator.TimeoutError:
            logging.warning("param_estimator timed out, falling back to defaults")
        except Exception as e:
            logging.info(f"Error during parameter estimation: {e}")

    # If parameter estimation fails or not provided, compute default parameters based on the func's signature
    params = return_default_params(func)
    if params is not None:
        # default params is a 2D array with shape (1, n_params), so we need to repeat it for each unit
        n_units = Y.shape[0]
        return jnp.repeat(params, n_units, axis=0)
    else:
        logging.info("Error: Unable to compute default parameters for the func.")
        return None

def return_default_params(func) -> jnp.ndarray:
    """
    Compute default parameters for the func based on its signature.
    Args:
        func (function): The model which predicts neural activity from stimuli and free parameters.
                                 Signature: func(stimuli, *params) -> activity
    Returns:
        jnp.ndarray: The default parameters for the func, shape (1, n_params).
                     If the parameter estimation fails, returns None.
    """
    try:
        sig = inspect.signature(func)
        param_names = [n for n in sig.parameters if n != "theta"]
        defaults = [sig.parameters[n].default if sig.parameters[n].default is not inspect._empty else 0.0 for n in param_names]
        default_arr = jnp.array(defaults, dtype=np.float32)
        return default_arr.reshape(1, -1)  # reshape to (1, n_params)
    except Exception as e:
        logging.info(f"Error while generating default parameters: {e}")
        return None    

def objective(func, param_estimator, loss_func, X_train, Y_train, X_test, Y_test, 
              param_penalty_weight=0.1, fit_params=True,
              FAILED_PROGRAM_COST=jnp.inf, max_iter=1_000,
              beta1=0.9, beta2=0.999, learning_rate=3e-3, eps=1e-8) -> tuple[float, jnp.ndarray, float, jnp.ndarray]:
    """
    Calculate the loss of the model. 
    
    The loss is calculated as the mean over units and points of the loss function provided.
    Args:
        func (function): The model which predicts neural activity from stimuli
                                and free parameters (for a single unit).
                                Signature: func(stimuli, *params) -> activity
        param_estimator (function or None): Optional estimator for initial parameters.
                                Signature: param_estimator(stimuli, response) -> params
        loss_func (function): The loss function to use for calculating the loss.
        X_train (jnp.ndarray): Stimuli data for training, shape (n_units, n_train_points).
        Y_train (jnp.ndarray): Response data for training, shape (n_units, n_train_points).
        X_test (jnp.ndarray): Stimuli data for evaluation, shape (n_units, n_test_points).
        Y_test (jnp.ndarray): Response data for evaluation, shape (n_units, n_test_points).
        param_penalty_weight (float): Weight for the penalty on the number of parameters. Default is 0.1.
        fit_params (bool): Whether to fit the parameters of the model. Default is True.
        FAILED_PROGRAM_COST (float): Cost assigned to failed models. Default is np.inf.
        max_iter (int): Maximum number of iterations for optimization. Default is 1_000.
        beta1 (float): Beta1 parameter for the Adam optimizer. Default is 0.9.
        beta2 (float): Beta2 parameter for the Adam optimizer. Default is 0.999.
        learning_rate (float): Learning rate for the Adam optimizer. Default is 3e-3.
        eps (float): Epsilon parameter for the Adam optimizer. Default is 1e-8.

    Returns:
        tuple[
            - float: The cross-validated loss of the model with data fit by the parameter estimator,
            - jnp.ndarray: The parameters fit by the parameter estimator.
            - float: The average loss (MSE on test set) across all units. 
                     Returns FAILED_PROGRAM_COST if the model fails for ANY unit.
            - jnp.ndarray: The parameters for each unit (n_units, n_params).
    """
    t_start = time.time()
    n_units, n_points = Y_train.shape

    # Perform initial param calc. X and Y must be numpy arrays of shape (n_units, n_points)
    initial_params = compute_initial_params(param_estimator, func, np.asarray(X_train), np.asarray(Y_train))
    
    # Fail immediately if initial_params is None or not a JAX array
    if initial_params is None or not isinstance(initial_params, jnp.ndarray):
        logging.info("Error: initial_params should be a JAX array.")
        return FAILED_PROGRAM_COST, jnp.zeros((n_units, 0)), FAILED_PROGRAM_COST, jnp.zeros((n_units, 0))
    if initial_params.ndim != 2 or initial_params.shape[0] != n_units:
        logging.info(f"Error: initial_params should be a 2D array with shape ({n_units}, n_params).")
        return FAILED_PROGRAM_COST, jnp.zeros((n_units, 0)), FAILED_PROGRAM_COST, jnp.zeros((n_units, 0))

    # Fail immediately if fit_params is True and non-numeric params
    n_params = initial_params.shape[1]
    all_numeric = (initial_params.dtype.kind in 'biufc' and 
                  jnp.all(jnp.isfinite(initial_params)))
    if fit_params and not all_numeric:
        logging.info("Error: Cannot fit non-numeric parameters.")
        return FAILED_PROGRAM_COST, jnp.zeros((n_units, n_params)), FAILED_PROGRAM_COST, jnp.zeros((n_units, n_params))

    # Fail immediately if neuron_model doesn't run
    try:
        # Check compatibility with JAX's tracing mechanism
        func_jit = jax.jit(func)
        for unit_idx in np.random.choice(n_units, size=min(10, n_units), replace=False):
            # Validate with concrete values
            output = func_jit(X_train[unit_idx], *initial_params[unit_idx])
            if output.ndim != 1 or output.shape[0] != X_train.shape[1]:
                logging.info(f"Error: func output shape {output.shape[0]} does not match input shape {X_train.shape[1]}.")
                return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params
            # Validate with abstract tracer values
            jax.eval_shape(func_jit, X_train[unit_idx], *initial_params[unit_idx])
    except Exception as e:
        logging.info(f"Func failed to run or is incompatible with JAX tracing: {e}")
        return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params

    loss_single_unit = lambda params, X_data, Y_data: jnp.mean(loss_func(func(X_data, *params), Y_data), axis=-1)
    # vectorize the loss function for all units. The inputs will have shapes:
    # - params: (n_units, n_params)
    # - x_data: (n_units, n_points)
    # - y_data: (n_units, n_points)
    # The output will have shape (n_units,)
    loss_total = jax.vmap(loss_single_unit, in_axes=(0, 0, 0), out_axes=0)

    if fit_params:
        # define the loss function wrt params. This will have input shape n_units * n_params (note that params is flattened) and output shape (1,)
        loss_param = lambda params: jnp.mean(loss_total(params.reshape(-1, n_params), X_train, Y_train))
        loss_param_and_grad = jax.value_and_grad(loss_param)

        # 1.  build adam
        opt = optax.adam(learning_rate, b1=beta1, b2=beta2, eps=eps)
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
        params = best_params.reshape(n_units, n_params)
        print(f"params optimized. Loss: {best_loss:.4f}")
    else:
        params = compute_initial_params(param_estimator, func, np.asarray(X_train), np.asarray(Y_train))
        if params is None or not isinstance(params, jnp.ndarray):
            logging.info("Error: params should be a JAX array.")
            return FAILED_PROGRAM_COST, jnp.zeros((n_units, n_params))

    # compute the final loss on the test set for the initial and optimized parameters
    initial_loss = jnp.nanmean(loss_total(initial_params, X_test, Y_test)) + param_penalty_weight * n_params
    # print number of nans in initial_loss
    n_nans = jnp.sum(jnp.isnan(initial_loss))
    if n_nans > 0:
        print(f"Warning: initial loss contains {n_nans} NaNs. This may indicate a problem with the model or data.")
    initial_loss = jnp.nan_to_num(initial_loss, nan=FAILED_PROGRAM_COST, posinf=FAILED_PROGRAM_COST, neginf=FAILED_PROGRAM_COST)
    final_loss = jnp.nanmean(loss_total(params, X_test, Y_test)) + param_penalty_weight * n_params
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

async def create_new_function(current_island: Island, llm_name, client, Y, X,
                              mode='explore', k_max=2, temp=1, 
                              thinking_budget=1, img_dir=None,
                              function_name='neuron_model',
                              diagnostic_image_fn=None,
                              diagnostic_metadata=None):
    k = min(k_max, len(current_island))
    random_programs = current_island.sample(k)
    random_programs.sort(key=lambda p: p.train_loss, reverse=True)
    parent1_id = random_programs[0].identifier() if random_programs else None
    parent2_id = random_programs[1].identifier() if len(random_programs) > 1 else None
    use_image = diagnostic_image_fn is not None
    program_prompt = utils.create_program_prompt(random_programs, mode=mode,
                                                 use_image=use_image, function_name=function_name)

    img_bytes = None
    if use_image and random_programs and diagnostic_image_fn:
        metadata = diagnostic_metadata or {}
        if img_dir:
            metadata.setdefault("save_path", img_dir)
        try:
            img_bytes = diagnostic_image_fn(programs=random_programs, X=X, Y=Y, metadata=metadata)
            if img_dir and img_bytes:
                Path(img_dir).write_bytes(img_bytes)
        except Exception as e:
            logging.info(f"Diagnostic image generation failed: {e}")
            img_bytes = None

    llm_output = await utils.call_llm_async(program_prompt, llm_name=llm_name, client=client, temperature=temp, 
                                            thinking_budget=thinking_budget, img_bytes=img_bytes)
    code_string = utils.extract_code_block(llm_output)
    if code_string is None:
        return None, None, (parent1_id, parent2_id)
    code_string = code_string.replace(f'def {function_name}_v{k+1}(', f'def {function_name}(')
    
    return code_string, program_prompt, (parent1_id, parent2_id)

async def create_new_parameter_estimator(current_island: Island, func_code_string: str,
                                           llm_name, client,
                                           k_max=1, temp=1, thinking_budget=0.25,
                                           param_estimator_max_lines=100,
                                           swear_words=['lstsq', 'scipy.optimize', 'optimize.minimize', 'curve_fit', 'sklearn'],
                                           function_name='neuron_model'):
    if func_code_string is None:
        logging.info("No function code string provided, skipping parameter estimator generation.")
        return None, None
    k = min(k_max, len(current_island))
    random_programs = current_island.sample(k)
    random_programs.sort(key=lambda p: p.train_loss, reverse=True)

    prompt = utils.create_parameter_estimator_prompt(random_programs,
                                                     func_code_string=func_code_string,
                                                     max_lines=param_estimator_max_lines,
                                                     function_name=function_name)
    llm_output = await utils.call_llm_async(prompt, model_name=llm_name, client=client, temperature=temp,
                                            thinking_budget=thinking_budget)
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
        return None, None
    code_string = code_string.replace(f'def parameter_estimator_v{k+1}(', 'def parameter_estimator(')
    func = utils.str_to_func(code_string, 'parameter_estimator')
    return code_string, func

async def translate_to_jax(code_string: str, client, llm_name='gemini-2.0-flash',
                           function_name: str = 'neuron_model') -> tuple[str, callable]:
    """
    Translates a neuron model code string to JAX format.
    Args:
        code_string (str): The neuron model code string to translate.
    Returns:
        callable: The translated JAX function.
    """
    if code_string is None:
        logging.info(f"No {function_name} code string provided for translation.")
        return None, None
    
    prompt = utils.create_jax_translater_prompt(code_string, function_name=function_name)
    # print(f"Translating neuron model to JAX with prompt:\n{prompt}")
    if prompt is None:
        return None, None
    
    jax_code_string = await utils.call_llm_async(prompt, client=client, model_name=llm_name, temperature=0)
    jax_code_string = utils.extract_code_block(jax_code_string)
    func = utils.str_to_func(jax_code_string, function_name)
    return jax_code_string, func

def sample_function(func: callable, params: jnp.ndarray, 
                    sample_points: jnp.ndarray = jnp.linspace(0, 2 * jnp.pi, 100)) -> jnp.ndarray:
    """
    evaluates the func at the given evaluation points.
    Args:
        func (callable): The neuron model function.
        params (jnp.ndarray): The parameters for the neuron model. (n_units, n_params)
        sample_points (jnp.ndarray): Points to evaluate the model at.
    Returns:
        jnp.ndarray: The evaluation matrix of shape (n_units, n_evaluation_points).
    """
    func_vmap = utils.vmap_over_units(func)
    y_eval = func_vmap(sample_points, params)
    return y_eval

async def _run_engine(X, Y,n_generations=9, time_limit=60, k_max=2, n_islands=8, batch_size=6, 
                critical_population_size=12, min_wise_population_size=0, 
                n_migrants=2, fit_params=True, exploit_point=0.5,
                param_penalty_weight=0.01, FAILED_PROGRAM_COST=np.inf,
                exploration_topology = None,
                exploitation_topology = None,
                seed_functions_numpy = None,
                seed_parameter_estimators = None,
                func_name = 'neuron_model',
                tiny_lm_name = 'gemini-2.0-flash',
                little_lm_name = 'gemini-2.0-flash',
                large_lm_name = 'gemini-2.5-flash',
                use_large_every = None,
                diagnostic_image_fn = None):
    """ 
    Main function to run the hypothesis engine.
    """
    # load api keys
    load_dotenv()
    client = google.genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    # load and preprocess data
    Y_loop, Y_eval, X_loop, X_eval = utils.split_arrays(X, Y, axis=0)
    X_loop_train_points, Y_loop_train_points, X_loop_test_points, Y_loop_test_points = utils.split_arrays(X_loop, Y_loop, axis=1)
    X_eval_train_points, Y_eval_train_points, X_eval_test_points, Y_eval_test_points = utils.split_arrays(X_eval, Y_eval, axis=1)

    if seed_functions_numpy is None:
        seed_functions_numpy = [_default_sine_numpy, _default_cosine_numpy]
    else:
        seed_functions_numpy = list(seed_functions_numpy)
    if not seed_functions_numpy:
        raise ValueError("At least one seed function must be provided.")
    if seed_parameter_estimators is None:
        seed_parameter_estimators = [None] * len(seed_functions_numpy)
    else:
        seed_parameter_estimators = list(seed_parameter_estimators)
        if len(seed_parameter_estimators) < len(seed_functions_numpy):
            seed_parameter_estimators.extend([None] * (len(seed_functions_numpy) - len(seed_parameter_estimators)))

    use_image_feedback = diagnostic_image_fn is not None

    islands = [Island(idx) for idx in range(n_islands)]
    initial_programs: list[Program] = []
    program_snapshots: list[ProgramSnapshot] = []
    best_train_history: list[float] = []

    def record_program(program: Program, timestamp: float, is_seed: bool = False, notes: Optional[str] = None) -> ProgramSnapshot:
        snapshot = ProgramSnapshot.from_program(
            program=program,
            program_index=len(program_snapshots),
            timestamp=timestamp,
            is_seed=is_seed,
            notes=notes,
        )
        program.record_index = snapshot.program_index
        program_snapshots.append(snapshot)
        running_best = min(best_train_history[-1], snapshot.train_loss) if best_train_history else snapshot.train_loss
        best_train_history.append(running_best)
        return snapshot

    # create output directories
    base_dir, date_stamp, time_stamp, full_dir, image_feedback_dir = utils.create_output_directories(use_image=use_image_feedback)
    # store and compute loss of initial programs
    t_start = time.time()
    n_seeds = min(2, len(seed_functions_numpy), len(seed_parameter_estimators))
    for i in range(n_seeds):
        func_numpy = seed_functions_numpy[i]
        param_est = seed_parameter_estimators[i]
        func_numpy_name = func_numpy.__name__
        param_est_name = param_est.__name__ if param_est else None

        import_string = "import numpy as np \n"
        numpy_source = inspect.getsource(func_numpy).replace(f'def {func_numpy_name}(', f'def {func_name}(')
        function_code_string = import_string + numpy_source
        jax_code_string, func_jax = await translate_to_jax(function_code_string, client, tiny_lm_name, function_name=func_name)
        if func_jax is None:
            logging.info(f"Skipping seed {i} because translation failed.")
            continue
        if not utils.validate_jax_translation(func_numpy, func_jax):
            logging.info(f"Skipping seed {i} because translation validation failed.")
            continue

        loss_init, params_init, loss, params = objective(func_jax, param_est, 
                                        loss_func=loss_functions.quadratic_loss, 
                                        X_train=X_loop_train_points, Y_train=Y_loop_train_points,
                                        X_test=X_loop_test_points, Y_test=Y_loop_test_points,
                                        fit_params=fit_params, param_penalty_weight=param_penalty_weight)

        if param_est is not None:
            parameter_estimator_code_string = inspect.getsource(param_est).replace(f'def {param_est_name}(', f'def parameter_estimator_v{i+1}(')
            parameter_estimator_code_string = import_string + parameter_estimator_code_string
        else:
            parameter_estimator_code_string = ""
        display_function_code_string = function_code_string.replace(f'def {func_name}(', f'def {func_name}_v{i+1}(')
        Y_SAMPLE = sample_function(func_jax, params, sample_points=jnp.linspace(0, 2 * jnp.pi, 100))

        new_program = Program(
            function_code_string=display_function_code_string,
            function=func_jax,
            parameter_estimator_code_string=parameter_estimator_code_string,
            parameter_estimator=param_est,
            generation=-1,
            birth_island=-1,
            batch_index=i,
            train_loss=float(loss),
            test_loss=None,
            llm_name=None,
            params=params,
            initial_loss=float(loss_init),
            initial_params=params_init,
            parent1_id=None,
            parent2_id=None,
            evaluation_matrix=Y_SAMPLE,
        )
        initial_programs.append(new_program)
        print(f"Initial program {i + 1} loss: {loss:.2f}")
        record_program(new_program, time.time() - t_start, is_seed=True, notes="seed program")

    # seed each island with the initial programs
    for island in islands:
        island.extend(initial_programs)

    # Reset logging configuration
    log_file = os.path.join(full_dir, 'engine.log')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')
    # -----------------------------
    # HYPOTHESIS ENGINE
    # -----------------------------
    for i in tqdm(range(n_generations), desc="Hypothesis Engine generations"):
        # check if time limit is reached
        if time.time() - t_start > time_limit * 60:
            logging.info(f"Time limit of {time_limit} minutes reached. Stopping generations.")
            break
        logging.info(f"generation {i}")
        assert use_large_every is None or use_large_every > 0, "use_large_every must be > 0 or None"
        if use_large_every is not None and i % use_large_every == 0:
            llm_name = large_lm_name
            logging.info(f"Using large LLM: {llm_name}")
        else:
            llm_name = little_lm_name
            logging.info(f"Using little LLM: {llm_name}")
        mode = 'explore' if i < n_generations * exploit_point else 'exploit'
        temperature = 1 + np.exp(-i / n_generations)
        if use_image_feedback:
            model_image_dirs = np.empty((n_islands, batch_size), dtype=object)
            for island_idx in range(n_islands):
                for j in range(batch_size):
                    model_image_dirs[island_idx, j] = os.path.join(image_feedback_dir, f'iter_{i}_island_{island_idx}_batch_{j}.png')
        else:
            model_image_dirs = None
        # generate new programs
        function_creation_tasks = []
        for island_idx in range(n_islands):
            for j in range(batch_size):
                img_path = model_image_dirs[island_idx, j] if model_image_dirs is not None else None
                diag_meta = {
                    "generation": i,
                    "island": island_idx,
                    "batch": j,
                    "save_path": img_path,
                }
                function_creation_tasks.append(
                    create_new_function(islands[island_idx], llm_name=llm_name, 
                                        client=client, mode=mode, 
                                        k_max=k_max, temp=temperature,
                                        Y=Y_loop, X=X_loop,
                                        img_dir=img_path,
                                        function_name=func_name,
                                        diagnostic_image_fn=diagnostic_image_fn,
                                        diagnostic_metadata=diag_meta)
                )
        logging.info(f"Creating {n_islands * batch_size} new programs... Model: {llm_name}, mode: {mode}, temperature: {temperature:.2f}")
        print(f"Creating {n_islands * batch_size} new programs... Model: {llm_name}, mode: {mode}, temperature: {temperature:.2f}")
        llm_function_results = await asyncio.gather(*function_creation_tasks)
        llm_function_code_strings = [result[0] for result in llm_function_results]
        llm_function_prompts = [result[1] for result in llm_function_results]
        parent_ids = [result[2] for result in llm_function_results]
        
        # convert to jax
        llm_function_translation_tasks = [translate_to_jax(code_string, client, tiny_lm_name, function_name=func_name) for code_string in llm_function_code_strings]
        jax_results = await asyncio.gather(*llm_function_translation_tasks)
        validated_results = []
        for idx in range(n_islands * batch_size):
            code_str = llm_function_code_strings[idx]
            prompt_str = llm_function_prompts[idx]
            jax_code_str, jax_func = jax_results[idx]
            numpy_func = utils.str_to_func(code_str, func_name) if code_str else None
            if numpy_func and jax_func and utils.validate_jax_translation(numpy_func, jax_func):
                validated_results.append((code_str, prompt_str, jax_code_str, jax_func))
            else:
                validated_results.append((None, None, None, None))
                llm_function_code_strings[idx] = None
                logging.info(f"Skipping program {idx} due to failed translation validation.")
        llm_function_results = validated_results
        
        # build parameter‑estimator tasks
        param_estimation_tasks = [
            create_new_parameter_estimator(
                current_island=islands[island_idx],
                func_code_string=llm_function_code_strings[island_idx * batch_size + j],
                llm_name=little_lm_name,
                client=client,
                k_max=2,
                temp=temperature,
                param_estimator_max_lines=100,
                function_name=func_name
            )
            for island_idx in range(n_islands)
            for j in range(batch_size)
        ]

        logging.info(
            f"Creating {n_islands * batch_size} parameter estimators "
            f"(LLM={llm_name}, mode={mode}, T={temperature:.2f})"
        )
        logging.info(f"Creating {n_islands * batch_size} new parameter estimators... Model: {little_lm_name}, mode: {mode}, temperature: {temperature:.2f}")
        param_est_results = await asyncio.gather(*param_estimation_tasks)
        # combine results
        island_results = [[llm_function_results[island_idx * batch_size + j] + param_est_results[island_idx * batch_size + j] for j in range(batch_size)] for island_idx in range(n_islands)]

        # now loop through the results and compute losses
        success_rate = 0.0
        for island_idx, j in np.ndindex(n_islands, batch_size):
            logging.info(f"id={i},{island_idx},{j}")
            func_code_string, prompt, func_code_string_jax, func_new, param_est_code_string, param_est_new = island_results[island_idx][j]
            param_est_code_string = param_est_code_string or ""
            parent1_id, parent2_id = parent_ids[island_idx * batch_size + j]
            if func_new is None:
                logging.info(f"Skipping island {island_idx}, batch {j} due to LLM generation failure.")
                logging.info('-' * 50)
                continue
            
            initial_loss, initial_params, loss, optimized_params = objective(func_new, param_est_new, 
                                                                                loss_func=loss_functions.quadratic_loss,
                                                                                X_train=X_loop_train_points, Y_train=Y_loop_train_points,
                                                                                X_test=X_loop_test_points, Y_test=Y_loop_test_points,
                                                                                param_penalty_weight=param_penalty_weight,
                                                                                fit_params=fit_params)
            if loss == FAILED_PROGRAM_COST:
                logging.info('-' * 50)
                continue

            Y_SAMPLE = sample_function(func_new, optimized_params, sample_points=jnp.linspace(0, 2 * jnp.pi, 100))
            logging.info(f"Prompt: \n{prompt}\n")
            logging.info(f"Loss: {loss:.2f}\n")
            logging.info(f"Function: \n{func_code_string}\n")
            logging.info(f"Function (JAX): \n{func_code_string_jax}\n")
            logging.info(f"Parameter Estimator: \n{param_est_code_string}\n")
            
            param_names = [n for n in inspect.signature(func_new).parameters if n != "theta"]
            if optimized_params.shape[1] == len(param_names):
                sample_params = np.array(optimized_params)[:10]
                logging.info(f"Optimized Parameters for 10 units:\n{sample_params}\n")
            t_added = time.time() - t_start
            new_program = Program(
                function_code_string=func_code_string,
                function=func_new,
                parameter_estimator_code_string=param_est_code_string,
                parameter_estimator=param_est_new,
                generation=i,
                birth_island=island_idx,
                batch_index=j,
                train_loss=float(loss),
                test_loss=None,
                llm_name=llm_name,
                params=optimized_params,
                initial_loss=float(initial_loss),
                initial_params=initial_params,
                parent1_id=parent1_id,
                parent2_id=parent2_id,
                evaluation_matrix=Y_SAMPLE,
            )
            islands[island_idx].add(new_program)
            record_program(new_program, t_added)
            success_rate += 1 / (n_islands * batch_size)
            print(f"generation {i}, island {island_idx}, batch {j}, loss: {loss:.2f}")
            print('-' * 50)
            logging.info("-" * 50)
        print("Success rate:", success_rate)

        # sort each island by loss
        for island in islands:
            island.sort_by('train_loss')
        logging.info(f"generation {i} complete. The proportion of programs that successfully ran and received a loss is {success_rate:.2f}.")
        logging.info('-' * 50)
        # migrate and prune programs (better here for temperature to be in [0, 1] range)
        if exploration_topology is None:
            exploration_topology = [[(j + 1) % n_islands for j in range(n_islands)]]
        if exploitation_topology is None:
            exploitation_topology = exploration_topology
        islands = genetic_helpers.perform_island_deduplication(islands, overlap_threshold=int(0.75 * critical_population_size))
        islands = genetic_helpers.perform_population_pruning(islands, critical_population_size=critical_population_size - n_migrants,
                                                min_wise_population_size=min_wise_population_size,)
        islands = genetic_helpers.perform_probabilistic_migration(islands, 
                                                                  n_migrants=n_migrants,
                                                                  destination_islands=exploration_topology if mode == 'explore' else exploitation_topology, 
                                                                  temperature=(temperature - 1.0)**4)

                                                             
        generation_dir = os.path.join(full_dir, 'generation_updates', f'generation_{i}')
        os.makedirs(generation_dir, exist_ok=True)
        for island_idx in range(n_islands):
            island_programs = list(islands[island_idx].programs)
            if not island_programs:
                continue
            pg_info = "\n".join(
                f"{prog.generation:>3} {prog.birth_island:>3} {prog.batch_index:>3} {prog.train_loss:6.2f}"
                for prog in island_programs
            )
            print(f"Gen {i}, Island {island_idx} programs:\n{pg_info}\n")
            logging.info(f"Gen {i}, Island {island_idx} programs:\n{pg_info}\n")
        census_path = os.path.join(generation_dir, 'census.csv')
        census_records = [snapshot.to_dict() for snapshot in program_snapshots]
        save_records_csv(census_path, census_records, columns=CENSUS_CSV_COLUMNS)

    # -----------------------------
    # now carry out the loss calculation on the test units
    logging.info("Calculating loss on test set...")
    for island_idx in range(n_islands):
        logging.info(f"Island {island_idx} programs:")
        for program in islands[island_idx]:
            neuron_model = program.function
            param_estimator = program.parameter_estimator
            _, _, test_loss, optimized_params = objective(neuron_model, param_estimator,
                                                          loss_func=loss_functions.quadratic_loss,
                                                          X_train=X_eval_train_points, Y_train=Y_eval_train_points,
                                                          X_test=X_eval_test_points, Y_test=Y_eval_test_points,
                                                          fit_params=fit_params,
                                                          max_iter=2_000, 
                                                          param_penalty_weight=param_penalty_weight)
            program.test_loss = test_loss
            program.params = optimized_params
            program.mean_loss = np.mean(test_loss)
            if program.record_index is not None and 0 <= program.record_index < len(program_snapshots):
                program_snapshots[program.record_index].test_loss = float(test_loss)
            print(f"Test loss: {test_loss:.2f}")

    # compute best test history after evaluating all programs
    best_test_history: list[float] = []
    running_best_test = float("inf")
    for snapshot in program_snapshots:
        if snapshot.test_loss is not None:
            running_best_test = min(running_best_test, snapshot.test_loss)
        best_test_history.append(running_best_test)

    # group all islands together and save
    combined_dir = os.path.join(base_dir, date_stamp, time_stamp, 'combined')
    os.makedirs(combined_dir, exist_ok=True)
    combined_island = Island(-1, [prog for island in islands for prog in island.programs])
    combined_island = genetic_helpers.remove_duplicates(
        combined_island, mode='complicated', loss_tol=0.025, cosine_tol=0.99, loss_type='test_loss'
    )
    combined_programs = sorted(
        combined_island.programs,
        key=lambda p: p.mean_loss if p.mean_loss is not None else float("inf")
    )
    census_records = [snapshot.to_dict() for snapshot in program_snapshots]
    save_records_csv(os.path.join(combined_dir, 'census.csv'), census_records, CENSUS_CSV_COLUMNS)

    return {
        "islands": islands,
        "combined_programs": combined_programs,
        "snapshots": program_snapshots,
        "best_train_history": best_train_history,
        "best_test_history": best_test_history,
        "output_dir": full_dir,
    }


class Edgar:
    """High-level interface for running the hypothesis engine."""

    def __init__(self, **config):
        defaults = dict(
            n_generations=9,
            time_limit=60,
            k_max=2,
            n_islands=8,
            batch_size=6,
            critical_population_size=12,
            min_wise_population_size=0,
            n_migrants=2,
            fit_params=True,
            exploit_point=0.5,
            param_penalty_weight=0.01,
            FAILED_PROGRAM_COST=np.inf,
            exploration_topology=[1, 2, 3, 4, 5, 6, 7, 0],
            exploitation_topology=[1, 2, 3, 4, 5, 6, 7, 0],
            seed_functions_numpy=None,
            seed_parameter_estimators=None,
            func_name='neuron_model',
            tiny_lm_name='gemini-2.0-flash',
            little_lm_name='gemini-2.0-flash',
            large_lm_name='gemini-2.5-flash',
            use_large_every=3,
            diagnostic_image_fn=None,
        )
        defaults.update(config)
        self.config = defaults
        self.islands_: Optional[list[Island]] = None
        self.combined_programs_: Optional[list[Program]] = None
        self.snapshots_: Optional[list[ProgramSnapshot]] = None
        self.census_: Optional[list[ProgramSnapshot]] = None
        self.best_train_history_: Optional[list[float]] = None
        self.best_test_history_: Optional[list[float]] = None
        self.output_dir_: Optional[str] = None

    async def run_async(self, X, Y):
        result = await _run_engine(X, Y, **self.config)
        self.islands_ = result["islands"]
        self.combined_programs_ = result["combined_programs"]
        self.snapshots_ = result["snapshots"]
        self.best_train_history_ = result["best_train_history"]
        self.best_test_history_ = result["best_test_history"]
        self.output_dir_ = result["output_dir"]
        # backwards compatibility alias
        self.census_ = self.snapshots_
        self.output_dir_ = result["output_dir"]
        return result

    def run(self, X, Y):
        return asyncio.run(self.run_async(X, Y))

    @classmethod
    def from_config(cls, path: str) -> "Edgar":
        config = utils.load_edgar_config(path)
        return cls(**config)

    def get_lineage_edges(self) -> list[tuple[int, int]]:
        if not self.snapshots_:
            return []
        id_to_index = {
            (snap.generation, snap.birth_island, snap.batch_index): snap.program_index
            for snap in self.snapshots_
        }
        edges = []
        for snap in self.snapshots_:
            for parent in (snap.parent1_id, snap.parent2_id):
                if parent is not None and parent in id_to_index:
                    edges.append((id_to_index[parent], snap.program_index))
        return edges

    def plot_progress(self, metric: str = "train", ax=None):
        """Plot running best loss over the course of the run."""
        import matplotlib.pyplot as plt

        if metric not in {"train", "test"}:
            raise ValueError("metric must be 'train' or 'test'")
        history = self.best_train_history_ if metric == "train" else self.best_test_history_
        if not history:
            raise ValueError("No run history is available. Call `run` first.")
        xs = np.arange(len(history))
        ys = np.array(history, dtype=float)
        ys[~np.isfinite(ys)] = np.nan
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xs, ys, color="red", linewidth=2.5, label=f"Best {metric} loss")
        ax.set_xlabel("Program index")
        ax.set_ylabel(f"{metric.capitalize()} loss")
        ax.set_title(f"Running best {metric} loss")
        ax.legend()
        return ax

    def plot_lineage(self, ax=None):
        """Visualize parent-child relationships and losses."""
        import matplotlib.pyplot as plt

        if not self.snapshots_:
            raise ValueError("No snapshots available to plot.")
        nodes_x = [snap.generation for snap in self.snapshots_]
        nodes_y = [snap.train_loss for snap in self.snapshots_]
        colors = []
        colour_map = {}
        palette = ["#ffb703", "#219ebc", "#8ecae6", "#023047", "#fb8500"]
        for snap in self.snapshots_:
            key = snap.llm_name or "seed"
            if key not in colour_map:
                colour_map[key] = palette[len(colour_map) % len(palette)]
            colors.append(colour_map[key])
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))
        edges = self.get_lineage_edges()
        for parent_idx, child_idx in edges:
            parent = self.snapshots_[parent_idx]
            child = self.snapshots_[child_idx]
            ax.plot(
                [parent.generation, child.generation],
                [parent.train_loss, child.train_loss],
                color="gray",
                linewidth=0.75,
                alpha=0.4,
            )
        ax.scatter(nodes_x, nodes_y, c=colors, s=60, edgecolor="k", alpha=0.8)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Train loss")
        ax.set_title("Program lineage graph")
        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w", label=name, markerfacecolor=colour_map[name], markersize=8, markeredgecolor="k")
            for name in colour_map
        ]
        ax.legend(handles=legend_handles, title="LLM source", bbox_to_anchor=(1.05, 1), loc="upper left")
        return ax
