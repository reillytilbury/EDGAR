import inspect
import os
import logging
import asyncio
import numpy as np
import jax, jax.numpy as jnp
import timeout_decorator
import jaxopt, optax
import pandas as pd
from pathlib import Path
import utils, diagnostic, seed_programs, genetic_helpers, loss_functions
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

def compute_initial_params(param_estimator, neuron_model, x, y) -> jnp.ndarray:
    """
    Compute initial parameters for the neuron model using the provided parameter estimator. Confusingly, the parameter estimator will be written in numpy,
    but the neuron model will be written in JAX. So the data x and y will be numpy arrays, but the output will be a JAX array.
    Args:
        param_estimator (function): Function to estimate initial parameters for the neuron model.
                                    Signature: param_estimator(stimuli, response) -> params
        neuron_model (function): The model which predicts neural activity from stimuli and free parameters.
                                 Signature: neuron_model(stimuli, *params) -> activity
        x (np.ndarray): Stimuli data, shape (n_cells, n_trials).
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
        return jnp.array([_safe_estimate(param_estimator, x[i], y[i])for i in range(y.shape[0])])
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
        param_names = [n for n in sig.parameters if n != "theta"]
        defaults = [sig.parameters[n].default if sig.parameters[n].default is not inspect._empty else 0.0 for n in param_names]
        default_arr = jnp.array(defaults, dtype=np.float32)
        return default_arr.reshape(1, -1)  # reshape to (1, n_params)
    except Exception as e:
        logging.info(f"Error while generating default parameters: {e}")
        return None    

def objective(neuron_model, param_estimator, loss_func, x, y, 
              param_penalty_weight=0.1, fit_params=True, random_seed=0,
              FAILED_PROGRAM_COST=jnp.inf, tol=1e-2, max_iter=1_000,
              use_param_estimator=True) -> tuple[float, jnp.ndarray, float, jnp.ndarray]:
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
        x (jnp.ndarray): Stimuli data, shape (n_cells, n_trials).
        y (jnp.ndarray): Response data, shape (n_cells, n_trials).
        param_penalty_weight (float): Weight for the penalty on the number of parameters. Default is 0.1.
        fit_params (bool): Whether to fit the parameters of the model. Default is True.
        random_seed (int or None): Random seed for reproducibility. Default is 0. If None, will not split the data into training and test sets.
        FAILED_PROGRAM_COST (float): Cost assigned to failed models. Default is np.inf.
        tol (float): Tolerance for optimization convergence. Default is 1e-2.
        max_iter (int): Maximum number of iterations for optimization. Default is 1_000.
        use_param_estimator (bool): Whether to use the parameter estimator to compute initial parameters. Default is True.

    Returns:
        tuple[
            - float: The cross-validated loss of the model with data fit by the parameter estimator,
            - jnp.ndarray: The parameters fit by the parameter estimator.
            - float: The average loss (MSE on test set) across all cells. 
                     Returns FAILED_PROGRAM_COST if the model fails for ANY cell.
            - jnp.ndarray: The parameters for each cell (n_cells, n_params).
    """
    t_start = time.time()
    n_cells, n_trials = y.shape
    # train/test split over trials
    key = jax.random.PRNGKey(random_seed)
    training_size = n_trials // 2
    shuffled_indices = jax.random.permutation(key, jnp.arange(n_trials))
    training_trials_idx = shuffled_indices[:training_size]
    test_trials_idx = shuffled_indices[training_size:]
    x_train = x[:, training_trials_idx]
    y_train = y[:, training_trials_idx]
    x_test = x[:, test_trials_idx]
    y_test = y[:, test_trials_idx]

    # Perform initial param calc. x and y must be numpy arrays of shape (n_cells, n_trials)
    if use_param_estimator:
        initial_params = compute_initial_params(param_estimator, neuron_model, np.asarray(x_train), np.asarray(y_train))
    else:
        initial_params = compute_default_params(neuron_model)
        # if initial_params not none, reshape from (1, n_params) to (n_cells, n_params)
        if initial_params is not None:
            n_params = initial_params.shape[1]
            initial_params = jnp.repeat(initial_params, n_cells, axis=0)
    
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
        for cell_idx in np.random.choice(n_cells, size=min(10, n_cells), replace=False):
            # Validate with concrete values
            output = neuron_model_jit(x[cell_idx], *initial_params[cell_idx])
            if output.ndim != 1 or output.shape[0] != x.shape[1]:
                logging.info(f"Error: neuron_model output shape {output.shape[0]} does not match input shape {x.shape[1]}.")
                return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params
            # Validate with abstract tracer values
            jax.eval_shape(neuron_model_jit, x[cell_idx], *initial_params[cell_idx])
    except Exception as e:
        logging.info(f"Neuron model failed to run or is incompatible with JAX tracing: {e}")
        return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params

    loss_single_cell = lambda params, x_data, y_data: jnp.mean(loss_func(neuron_model(x_data, *params), y_data), axis=-1)
    # vectorize the loss function for all cells. The inputs will have shapes:
    # - params: (n_cells, n_params)
    # - x_data: (n_cells, n_trials)
    # - y_data: (n_cells, n_trials)
    # The output will have shape (n_cells,)
    loss_total = jax.vmap(loss_single_cell, in_axes=(0, 0, 0), out_axes=0)

    if fit_params:
        # define the loss function wrt params. This will have input shape n_cells * n_params (note that params is flattened) and output shape (1,)
        loss_param = lambda params: jnp.mean(loss_total(params.reshape(-1, n_params), x_train, y_train))
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
        params = compute_initial_params(param_estimator, neuron_model, np.asarray(x_train), np.asarray(y_train))
        if params is None or not isinstance(params, jnp.ndarray):
            logging.info("Error: params should be a JAX array.")
            return FAILED_PROGRAM_COST, jnp.zeros((n_cells, n_params))

    # compute the final loss on the test set for the initial and optimized parameters
    initial_loss = jnp.nanmean(loss_total(initial_params, x_test, y_test)) + param_penalty_weight * n_params
    # print number of nans in initial_loss
    n_nans = jnp.sum(jnp.isnan(initial_loss))
    if n_nans > 0:
        print(f"Warning: initial loss contains {n_nans} NaNs. This may indicate a problem with the model or data.")
    initial_loss = jnp.nan_to_num(initial_loss, nan=FAILED_PROGRAM_COST, posinf=FAILED_PROGRAM_COST, neginf=FAILED_PROGRAM_COST)
    final_loss = jnp.nanmean(loss_total(params, x_test, y_test)) + param_penalty_weight * n_params
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


async def generate_new_neuron_model(current_island, llm_name, client, 
                                    spike_matrix, stimuli,
                                    mode='explore', k_max=2, temp=1, 
                                    thinking_budget=1, img_dir=None):
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
    use_image = img_dir is not None
    program_prompt = utils.create_program_prompt(random_programs, mode=mode, llm_type=llm_name[0], use_image=use_image)

    if use_image:
        try:
            sup_title = "".join([f"neuron_model_v{i+1}: Loss = {random_programs['train_loss'][i]:.2f} \n" for i in range(min(3, len(random_programs)))])
            diagnostic.plot_model_fits(programs_df=random_programs,
                                    loss_function=loss_functions.quadratic_loss,
                                    x=stimuli, y=spike_matrix,
                                    cell_selection=np.random.choice(spike_matrix.shape[0], size=9, replace=False),
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
    code_string = code_string.replace(f'def neuron_model_v{k+1}(', 'def neuron_model(')
    
    return code_string, program_prompt, (parent1_id, parent2_id)

async def generate_new_parameter_estimator(current_island, 
                                           neuron_model_code_string: str,
                                           llm_name, client, 
                                           spike_matrix, stimuli,
                                           k_max=1, temp=1,
                                           param_estimator_max_lines=100, img_dir=None,
                                           swear_words=['lstsq', 'scipy.optimize', 'optimize.minimize', 'curve_fit', 'sklearn']):
    if neuron_model_code_string is None:
        logging.info("No neuron model code string provided, skipping parameter estimator generation.")
        return None, None
    k = min(k_max, len(current_island))
    random_programs = current_island.sample(k, replace=False).reset_index(drop=True)
    # sort from worst to best (loss descending)
    random_programs = random_programs.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
    use_image = img_dir is not None
    prompt = utils.create_parameter_estimator_prompt(random_programs,
                                                    neuron_model_code_string=neuron_model_code_string,
                                                    llm_type=llm_name[0], max_lines=param_estimator_max_lines,
                                                    use_image=use_image)
    
    random_programs_crude = random_programs.copy()
    random_programs_crude['params'] = random_programs['initial_params']
    # now try generating an image from the random programs
    if use_image:
        try:
            sup_title = "".join([f"neuron_model_v{i+1}: Loss = {random_programs['train_loss'][i]:.2f} \n" for i in range(min(3, len(random_programs)))])
            diagnostic.plot_model_fits(programs_df=random_programs_crude,
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
            logging.info(f"Error generating image for parameter estimator prompt: {e}")
            img_bytes = None
            # if we can't generate an image, we will just use the text prompt without image
            use_image = False
    else:
        img_bytes = None
    
    llm_output = await utils.call_llm_async(prompt, model_name=llm_name, client=client, temperature=temp,
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
        return None, None
    code_string = code_string.replace(f'def parameter_estimator_v{k+1}(', 'def parameter_estimator(')
    func = utils.str_to_func(code_string, 'parameter_estimator')
    return code_string, func

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

async def translate_to_jax(code_string: str, client, llm_name='gemini-1.5-flash-8b') -> tuple[str, callable]:
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
    
    prompt = utils.create_jax_translater_prompt(code_string)
    # print(f"Translating neuron model to JAX with prompt:\n{prompt}")
    if prompt is None:
        return None, None
    
    jax_code_string = await utils.call_llm_async(prompt, client=client, model_name=llm_name, temperature=0)
    jax_code_string = utils.extract_code_block(jax_code_string)
    func = utils.str_to_func(jax_code_string, 'neuron_model')
    return jax_code_string, func

def compute_evaluation_matrix(program: callable, params: jnp.ndarray, n_evaluation_points: int = 100) -> jnp.ndarray:
    """
    Computes the evaluation matrix for a given program and parameters.
    Args:
        program (callable): The neuron model function.
        params (jnp.ndarray): The parameters for the neuron model. (n_cells, n_params)
        n_evaluation_points (int): Number of points to evaluate the model at.
    Returns:
        jnp.ndarray: The evaluation matrix of shape (n_cells, n_evaluation_points).
    """
    angles = jnp.linspace(0, 2 * jnp.pi, n_evaluation_points)
    program_vmap = utils.vmap_over_cells(program)
    y_eval = program_vmap(angles, params)
    return y_eval

async def main(n_iterations=9, time_limit=60, k_max=2, n_islands=8, batch_size=6, 
                critical_population_size=12, min_wise_population_size=0, 
                n_migrants=2, fit_params=True, tol=1e-6, exploit_point=0.5,
                param_penalty_weight=0.01, FAILED_PROGRAM_COST=np.inf,
                use_image_feedback=True, use_param_estimator=True,
                exploration_topology = [1, 2, 3, 4, 5, 6, 7, 0],
                exploitation_topology = [1, 2, 3, 4, 5, 6, 7, 0],
                tiny_lm_name = 'gemini-1.5-flash-8b',
                little_lm_name = 'gemini-2.0-flash',
                large_lm_name = 'gemini-2.5-flash',
                use_large_every = 3,
                conc_thresh = 0.55, activity_thresh = 0.4,
                data_path = '/home/reilly/Downloads/8279387/gratings_drifting_GT1_2019_04_12_1.npy'):
    """ 
    Main function to run the hypothesis engine.
    """
    # load api keys
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    # load and preprocess data
    neural_data = np.load(data_path, allow_pickle=True)
    neural_data = neural_data.item()
    response = utils.extract_stimulus_related_response(neural_data, n_pcs=0)
    angles = neural_data['istim']
    n_trials = response.shape[1]
    n_trials_small = int(n_trials * activity_thresh)

    # filter 
    active = (response > 0).astype(np.float32)
    firing_probs = np.mean(active, axis=1)
    conc = np.abs(np.sum(np.exp(2j * angles)[np.newaxis, :] * response, axis=1) / np.sum(response, axis=1))
    good_cells = np.where((firing_probs > activity_thresh) & (conc > conc_thresh))[0]
    n_good_cells = len(good_cells)

    # update angles and response to be (n_cells_small, n_trials_small) and (n_cells_small, n_trials_small)
    response_cropped, angles_cropped = np.zeros((len(good_cells), n_trials_small)), np.zeros((len(good_cells), n_trials_small))
    for i, cell in enumerate(good_cells):
        active_trials = response[cell] > 0
        active_trials_idx = np.where(active_trials)[0][:n_trials_small]
        response_cropped[i] = response[cell, active_trials_idx]
        angles_cropped[i] = angles[active_trials_idx]
        
    # update response and angles to be the cropped versions and convert to JAX arrays, normalize and split into train/test
    response, angles = jnp.asarray(response_cropped), jnp.asarray(angles_cropped)
    response = 100 * response / jnp.linalg.norm(response, axis=1, keepdims=True)  # normalize response
    key = jax.random.PRNGKey(42)
    training_size = n_good_cells // 2
    shuffled_indices = jax.random.permutation(key, jnp.arange(n_good_cells))
    training_cells, test_cells = shuffled_indices[:training_size], shuffled_indices[training_size:]
    response_train, response_test = response[training_cells, :], response[test_cells, :]
    angles_train, angles_test = angles[training_cells, :], angles[test_cells, :]
    print(f"Selected {len(good_cells)} cells with activity > {activity_thresh} and concentration > {conc_thresh}.")
    print(f"Using {len(training_cells)} cells for training and {len(test_cells)} cells for testing.")

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
    
    # store and compute loss of 2 initial programs
    t_start = time.time()
    numpy_programs = [seed_programs.neuron_model_1, seed_programs.neuron_model_2]
    jax_programs = [seed_programs.neuron_model_1_jax, seed_programs.neuron_model_2_jax]
    param_estimators = [seed_programs.parameter_estimator_1, seed_programs.parameter_estimator_2]
    seed_losses = np.zeros(2)
    for i in range(2):
        # get the program, parameter estimator, and jax program
        program_num = numpy_programs[i]
        param_est = param_estimators[i]
        program_jax = jax_programs[i]
        # score the initial program
        loss_init, params_init, loss, params = objective(program_jax, param_est, 
                                        loss_func=loss_functions.quadratic_loss, 
                                        x=angles_train, y=response_train, 
                                        fit_params=fit_params, param_penalty_weight=param_penalty_weight, tol=tol,
                                        use_param_estimator=use_param_estimator)
        seed_losses[i] = loss
        # format strings
        import_string = "import numpy as np \n"
        import_string_jax = "import jax.numpy as jnp \n"
        program_name = program_num.__name__
        param_est_name = param_est.__name__
        program_jax_name = program_jax.__name__
        program_code_string = inspect.getsource(program_num).replace(f'def {program_name}(', f'def neuron_model_v{i+1}(')
        program_code_string = import_string + program_code_string
        parameter_estimator_code_string = inspect.getsource(param_est).replace(f'def {param_est_name}(', f'def parameter_estimator_v{i+1}(')
        parameter_estimator_code_string = import_string + parameter_estimator_code_string
        program_jax_code_string = inspect.getsource(program_jax).replace(f'def {program_jax_name}(', f'def neuron_model_v{i+1}(')
        program_jax_code_string = import_string_jax + program_jax_code_string
        y_eval = compute_evaluation_matrix(program_jax, params, n_evaluation_points=100)

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
    diagnostic.plot_model_fits(programs_df=initial_programs,
                               loss_function=loss_functions.quadratic_loss,
                               x=angles_train, y=response_train,
                               cell_selection=np.random.choice(len(angles_train), size=9, replace=False),
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
    for i in tqdm(range(n_iterations), desc="Hypothesis Engine Iterations"):
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
        model_image_dirs = np.empty((n_islands, batch_size), dtype=object)
        # param_est_image_dirs = np.empty((n_islands, batch_size), dtype=object)
        for island_idx in range(n_islands):
            for j in range(batch_size):
                if use_image_feedback:
                    model_image_dirs[island_idx, j] = os.path.join(image_feedback_dir, f'iter_{i}_island_{island_idx}_batch_{j}.png')
                    # param_est_image_dirs[island_idx, j] = os.path.join(image_feedback_dir, f'iter_{i}_island_{island_idx}_batch_{j}_param_est.png')
                else:
                    model_image_dirs[island_idx, j] = None
                    # param_est_image_dirs[island_idx, j] = None
        # generate new programs
        neuron_model_generation_tasks = [generate_new_neuron_model(islands[island_idx], 
                                                                   llm_name=llm_name, 
                                                                   client=client, 
                                                                   mode=mode, 
                                                                   k_max=k_max, 
                                                                   temp=temperature,
                                                                   spike_matrix=response_train, 
                                                                   stimuli=angles_train,
                                                                   img_dir=model_image_dirs[island_idx, j]) 
                                         for island_idx in range(n_islands) for j in range(batch_size)]
        logging.info(f"Generating {n_islands * batch_size} new programs... Model: {llm_name}, mode: {mode}, temperature: {temperature:.2f}")
        print(f"Generating {n_islands * batch_size} new programs... Model: {llm_name}, mode: {mode}, temperature: {temperature:.2f}")
        neuron_model_results = await asyncio.gather(*neuron_model_generation_tasks)
        neuron_model_code_strings = [result[0] for result in neuron_model_results]
        neuron_model_prompts = [result[1] for result in neuron_model_results]
        parent_ids = [result[2] for result in neuron_model_results]
        
        # convert to jax
        neuron_model_function_translation_tasks = [translate_to_jax(code_string, client, tiny_lm_name) for code_string in neuron_model_code_strings]
        jax_results = await asyncio.gather(*neuron_model_function_translation_tasks)
        neuron_model_results = [(neuron_model_code_strings[j], neuron_model_prompts[j], jax_results[j][0], jax_results[j][1]) for j in range(n_islands * batch_size)]
        
        # build parameter‑estimator tasks
        param_estimation_tasks = [
            generate_new_parameter_estimator(
                current_island=islands[island_idx],
                neuron_model_code_string=neuron_model_code_strings[island_idx * batch_size + j],
                llm_name=little_lm_name,  # same model used for programs
                client=client,
                spike_matrix=response_train, # training data
                stimuli=angles_train,
                k_max=2,
                temp=temperature,
                param_estimator_max_lines=100,
                img_dir=None # no image feedback for parameter estimator generation
            )
            for island_idx in range(n_islands)
            for j in range(batch_size)
        ]

        logging.info(
            f"Generating {n_islands * batch_size} parameter estimators "
            f"(LLM={llm_name}, mode={mode}, T={temperature:.2f})"
        )
        logging.info(f"Generating {n_islands * batch_size} new parameter estimators... Model: {little_lm_name}, mode: {mode}, temperature: {temperature:.2f}")
        param_est_results = await asyncio.gather(*param_estimation_tasks)
        # combine results
        island_results = [[neuron_model_results[island_idx * batch_size + j] + param_est_results[island_idx * batch_size + j] for j in range(batch_size)] for island_idx in range(n_islands)]

        # get images of param estimation if using image feedback
        # if use_image_feedback:
        #     for island_idx, j in np.ndindex(n_islands, batch_size):
        #         logging.info(f"Iteration {i}, Island {island_idx}, Batch Index {j}")
        #         neuron_model_code_string, prompt, neuron_model_code_string_jax, neuron_model_new, param_est_code_string, param_est_new = island_results[island_idx][j]
        #         parent1_id, parent2_id = parent_ids[island_idx * batch_size + j]
        #         if neuron_model_new is None or param_est_new is None:
        #             logging.info(f"Skipping island {island_idx}, batch {j} due to LLM generation failure.")
        #             logging.info('-' * 50)
        #             continue

        #         initial_loss, initial_params, _, _ = objective(neuron_model_new, param_est_new, loss_func=loss_functions.quadratic_loss,
        #                                                        x=angles_train, y=response_train,param_penalty_weight=param_penalty_weight,
        #                                                        fit_params=False, tol=tol, use_param_estimator=use_param_estimator)
        #         if initial_loss == FAILED_PROGRAM_COST:
        #             logging.info('-' * 50)
        #             continue
                
        #         cells = np.random.choice(n_good_cells // 2, size=4, replace=False)
        #         diagnostic.plot_model_fits(
        #             programs_df=pd.DataFrame({'program': [neuron_model_new], 'params': [initial_params]}),
        #             loss_function=loss_functions.quadratic_loss,
        #             x=angles_train,
        #             y=response_train,
        #             cell_selection=cells,
        #             colours=['tab:red'],
        #             labels=['Neuron Model'],
        #             line_alpha=1.0,
        #             line_width=5.0,
        #             point_alpha=0.2,
        #             point_size=120,
        #             legend_fontsize=20,
        #             title=f"4 Example Fits for Neuron Model and Parameter Estimator. \n"
        #                   f"Loss: {initial_loss:.2f}",
        #             save_path=os.path.join(image_feedback_dir, f'iter_{i}_island_{island_idx}_batch_{j}_param_est.png'),
        #         )
            
        #     # now create a new prompt to update the parameter estimator
        #     param_est_image_dirs = np.empty((n_islands, batch_size), dtype=object)
        #     for island_idx, j in np.ndindex(n_islands, batch_size):
        #         if use_image_feedback:
        #             param_est_image_dirs[island_idx, j] = os.path.join(image_feedback_dir, f'iter_{i}_island_{island_idx}_batch_{j}_param_est.png')
        #         else:
        #             param_est_image_dirs[island_idx, j] = None

        #     # collect the new neuron model and parameter estimator code strings
        #     image_prompts = np.empty((n_islands, batch_size), dtype=object)
        #     for island_idx, j in np.ndindex(n_islands, batch_size):
        #         neuron_model_code_string = island_results[island_idx][j][0]
        #         param_est_code_string = island_results[island_idx][j][4]
        #         if neuron_model_code_string is None or param_est_code_string is None:
        #             image_prompts[island_idx, j] = None
        #         else:
        #             image_prompts[island_idx, j] = utils.create_parameter_estimator_image_prompt(
        #                 neuron_model_code_string=neuron_model_code_string,
        #                 param_estimator_code_string=param_est_code_string)
        
        #     # generate new parameter estimators from image feedback
        #     param_est_image_tasks = [generate_new_parameter_estimator_from_image_feedback(image_prompt=image_prompts[island_idx, j],
        #                                                                                   image_dir=param_est_image_dirs[island_idx, j],
        #                                                                                   model_name=llm_name,
        #                                                                                   temp=temperature,
        #                                                                                   max_lines=100,
        #                                                                                   client=client)
        #                             for island_idx in range(n_islands) for j in range(batch_size)]
        #     logging.info(f"Generating {n_islands * batch_size} new parameter estimators from image feedback... Model: {llm_name}, temperature: {temperature:.2f}")
        #     param_est_image_results = await asyncio.gather(*param_est_image_tasks)
            
        #     # update the island results with the new parameter estimators (if they are not None)
        #     for island_idx, j in np.ndindex(n_islands, batch_size):
        #         if param_est_image_results[island_idx * batch_size + j][0] is not None:
        #             print(f"Island {island_idx}, batch {j} image feedback generation successful.")
        #             print(f"Initial Parameter Estimator:\n{island_results[island_idx][j][4]}")
        #             print(f"New Parameter Estimator:\n{param_est_image_results[island_idx * batch_size + j][0]}")
        #             result_list = list(island_results[island_idx][j])
        #             result_list[4] = param_est_image_results[island_idx * batch_size + j][0]
        #             result_list[5] = param_est_image_results[island_idx * batch_size + j][1]
        #             island_results[island_idx][j] = tuple(result_list)
        #         else:
        #             logging.info(f"Skipping island {island_idx}, batch {j} due to image feedback generation failure.")

        # now loop through the results and compute losses
        success_rate = 0.0
        for island_idx, j in np.ndindex(n_islands, batch_size):
            logging.info(f"id={i},{island_idx},{j}")
            neuron_model_code_string, prompt, neuron_model_code_string_jax, neuron_model_new, param_est_code_string, param_est_new = island_results[island_idx][j]
            parent1_id, parent2_id = parent_ids[island_idx * batch_size + j]
            if neuron_model_new is None or param_est_new is None:
                logging.info(f"Skipping island {island_idx}, batch {j} due to LLM generation failure.")
                logging.info('-' * 50)
                continue
            
            initial_loss, initial_params, loss, optimized_params = objective(neuron_model_new, param_est_new, 
                                                                                loss_func=loss_functions.quadratic_loss,
                                                                                x=angles_train, y=response_train,
                                                                                param_penalty_weight=param_penalty_weight,
                                                                                fit_params=fit_params, tol=tol, 
                                                                                use_param_estimator=use_param_estimator)
            if loss == FAILED_PROGRAM_COST:
                logging.info('-' * 50)
                continue

            y_eval = compute_evaluation_matrix(neuron_model_new, optimized_params, n_evaluation_points=100)
            logging.info(f"Prompt: \n{prompt}\n")
            logging.info(f"Loss: {loss:.2f}\n")
            logging.info(f"Neuron Model: \n{neuron_model_code_string}\n")
            logging.info(f"Neuron Model (JAX): \n{neuron_model_code_string_jax}\n")
            logging.info(f"Parameter Estimator: \n{param_est_code_string}\n")


            # plot the fits of the neuron model and parameter estimator if using image feedback
            if use_image_feedback:
                diagnostic.plot_model_fits(
                    programs_df=pd.DataFrame({'program': [neuron_model_new, neuron_model_new], 'params': [initial_params, optimized_params]}),
                    loss_function=loss_functions.quadratic_loss,
                    x=angles_train,
                    y=response_train,
                    cell_selection=np.random.choice(len(angles_train), size=4, replace=False),
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
            
            param_names = [n for n in inspect.signature(neuron_model_new).parameters if n != "theta"]
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
            diagnostic.plot_model_fits(
                programs_df=top_df,
                loss_function=loss_functions.quadratic_loss,
                x=angles_train,
                y=response_train,
                cell_selection=np.random.choice(response_train.shape[0], size=9, replace=False),
                title=sup_title,
                save_path=os.path.join(iteration_dir, f'island_{island_idx}_top_programs.png'),
                dpi=300.0)
        
        all_programs = pd.concat([islands[idx] for idx in range(n_islands)], ignore_index=True)
        top_programs = all_programs.sort_values(by='train_loss').head(3).reset_index(drop=True)
        top_programs = top_programs.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
        sup_title = f"Iteration {i}, Top 3 Programs Overall\n"
        sup_title += "\n".join([f"model {j+1}: iter {top_programs['iteration_number'][j]}, birth island {top_programs['birth_island'][j]}, batch {top_programs['batch_index'][j]}, loss: {top_programs['train_loss'][j]:.2f}" for j in range(len(top_programs))])
        diagnostic.plot_model_fits(
            programs_df=top_programs,
            loss_function=loss_functions.quadratic_loss,
            x=angles_train,
            y=response_train,
            cell_selection=np.random.choice(response_train.shape[0], size=9, replace=False),
            title=sup_title,
            save_path=os.path.join(iteration_dir, 'top_programs_overall.png'),
            dpi=300.0)
        
        # save census
        census_path = os.path.join(iteration_dir, 'census.npy')
        census_np = np.array(census, dtype=object)
        np.save(census_path, census_np)

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
                                                          x=angles_test, y=response_test, fit_params=fit_params,
                                                          max_iter=2_000, 
                                                          param_penalty_weight=param_penalty_weight, tol=tol,
                                                          use_param_estimator=use_param_estimator)
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
        df = df.sort_values(by='test_loss', ascending=False).reset_index(drop=True)
        df_sup += "".join([f"model {len(df) - i}: iter {df['iteration_number'][i]}, birth_island {df['birth_island'][i]}, batch {df['batch_index'][i]}, total loss {0.5 * (df['test_loss'][i] + df['train_loss'][i]):.2f}\n" for i in range(min(3, len(df)))])
        diagnostic.plot_model_fits(
            programs_df=df,
            loss_function=loss_functions.quadratic_loss,
            x=angles_test,
            y=response_test,
            cell_selection=np.random.choice(response_test.shape[0], size=9, replace=False),
            title=df_sup,
            save_path=os.path.join(df_dirs[i], 'top_model_fits.png')
        )
        # plot top 3 models separately
        for j in range(min(3, len(df))):
            birth_island = df['birth_island'][j]
            iteration_number = df['iteration_number'][j]
            batch_index = df['batch_index'][j]
            cell_selection = np.random.choice(response_test.shape[0], size=9, replace=False)
            diagnostic.plot_single_model_fit(
                model=df['program'][j],
                loss_function=loss_functions.quadratic_loss,
                x=angles_test[cell_selection],
                y=response_test[cell_selection],
                params=df['params'][j][cell_selection],
                title=f"Island {birth_island}, Iteration {iteration_number}, Batch {batch_index}, loss: {df['test_loss'][j]:.2f}",
                save_path=os.path.join(df_dirs[i], f'top_model_fit_{min(3, len(df)) - j}.png')
            )

if __name__ == "__main__":
    for i in range(4):
        print("running without penalty")
        asyncio.run(main(n_iterations=9, time_limit=100, use_image_feedback=False, use_large_every=-1,
                         param_penalty_weight=0.0,
                         exploration_topology=[1, 2, 3, 4, 5, 6, 7, 0], exploit_point=1.0))
    for i in range(4):
        print("running with penalty")
        asyncio.run(main(n_iterations=9, time_limit=40, 
                         use_image_feedback=False, use_large_every=-1,
                         exploration_topology=[1, 2, 3, 4, 5, 6, 7, 0], exploit_point=1.0))