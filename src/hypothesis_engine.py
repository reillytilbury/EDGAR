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
from . import utils, loss_functions, llm_helper
from . import genetic_helpers_v2 as genetic_helpers  # Using v2 with compatibility API
from .prompt_manager import PromptManager
from .data_structures import Inputs, Outputs, ensure_inputs, ensure_outputs
import experiments.orientation_tuning.seed_programs # delete this once we read seed_programs from experiment.yaml
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

print(jax.default_backend())    # should print "gpu"
print(jax.devices())


def save_data_summary(
    response: np.ndarray,
    inputs: np.ndarray,
    training_samples: jnp.ndarray,
    test_samples: jnp.ndarray,
    output_dir: str,
    random_seed: int = 0
) -> pd.DataFrame:
    """
    Save a summary of data splits and matrix sizes to a CSV file.
    
    This function documents:
    1. Sample split (training vs test cells)
    2. Trial split (training vs test inputs within objective function)
    3. Feature counts and data shapes
    4. Data types and estimated memory sizes
    
    Args:
        response: Full response matrix, shape (n_samples, n_trials)
        inputs: Full inputs matrix, shape (n_samples, n_trials) or (n_samples, n_features, n_trials)
        training_samples: Indices of samples used for training
        test_samples: Indices of samples used for testing
        output_dir: Directory to save the CSV
        random_seed: Random seed used for trial split (for verification)
    
    Returns:
        DataFrame with the summary information
    """
    n_total_samples, n_trials = response.shape
    n_train_samples = len(training_samples)
    n_test_samples = len(test_samples)
    
    # Determine inputs shape and features
    if inputs.ndim == 2:
        n_features = 1
        inputs_shape_str = f"({inputs.shape[0]}, {inputs.shape[1]})"
    else:
        n_features = inputs.shape[1]
        inputs_shape_str = f"({inputs.shape[0]}, {inputs.shape[1]}, {inputs.shape[2]})"
    
    # Calculate trial split (same logic as in objective function)
    n_trial_splits = 10
    trials_per_split = n_trials // n_trial_splits
    n_training_trials = trials_per_split * 5  # odd chunks (5 of 10)
    n_test_trials = trials_per_split * 5       # even chunks (5 of 10)
    
    # Helper to calculate size in bytes
    def calc_size(shape, dtype):
        n_elements = np.prod(shape)
        bytes_per_element = np.dtype(dtype).itemsize
        return n_elements * bytes_per_element
    
    def format_size(size_bytes):
        if size_bytes >= 1e9:
            return f"{size_bytes / 1e9:.2f} GB"
        elif size_bytes >= 1e6:
            return f"{size_bytes / 1e6:.2f} MB"
        elif size_bytes >= 1e3:
            return f"{size_bytes / 1e3:.2f} KB"
        else:
            return f"{size_bytes} B"
    
    # Build summary rows
    rows = []
    
    # === SAMPLE SPLIT SUMMARY ===
    rows.append({
        'category': 'SAMPLE_SPLIT',
        'matrix_name': 'total_samples',
        'description': 'Total number of cells/samples in dataset',
        'shape': f"({n_total_samples},)",
        'dtype': '-',
        'size_bytes': '-',
        'size_human': '-',
        'n_elements': n_total_samples
    })
    rows.append({
        'category': 'SAMPLE_SPLIT',
        'matrix_name': 'training_samples',
        'description': 'Cells used for training (held-in)',
        'shape': f"({n_train_samples},)",
        'dtype': str(training_samples.dtype),
        'size_bytes': calc_size((n_train_samples,), training_samples.dtype),
        'size_human': format_size(calc_size((n_train_samples,), training_samples.dtype)),
        'n_elements': n_train_samples
    })
    rows.append({
        'category': 'SAMPLE_SPLIT',
        'matrix_name': 'test_samples',
        'description': 'Cells used for testing (held-out)',
        'shape': f"({n_test_samples},)",
        'dtype': str(test_samples.dtype),
        'size_bytes': calc_size((n_test_samples,), test_samples.dtype),
        'size_human': format_size(calc_size((n_test_samples,), test_samples.dtype)),
        'n_elements': n_test_samples
    })
    
    # === TRIAL SPLIT SUMMARY (within objective function) ===
    rows.append({
        'category': 'TRIAL_SPLIT',
        'matrix_name': 'total_trials',
        'description': 'Total number of trials per sample',
        'shape': f"({n_trials},)",
        'dtype': '-',
        'size_bytes': '-',
        'size_human': '-',
        'n_elements': n_trials
    })
    rows.append({
        'category': 'TRIAL_SPLIT',
        'matrix_name': 'training_trials',
        'description': 'Trials used for param fitting (odd 10-chunks)',
        'shape': f"({n_training_trials},)",
        'dtype': 'int32',
        'size_bytes': calc_size((n_training_trials,), 'int32'),
        'size_human': format_size(calc_size((n_training_trials,), 'int32')),
        'n_elements': n_training_trials
    })
    rows.append({
        'category': 'TRIAL_SPLIT',
        'matrix_name': 'test_trials',
        'description': 'Trials used for loss evaluation (even 10-chunks)',
        'shape': f"({n_test_trials},)",
        'dtype': 'int32',
        'size_bytes': calc_size((n_test_trials,), 'int32'),
        'size_human': format_size(calc_size((n_test_trials,), 'int32')),
        'n_elements': n_test_trials
    })
    rows.append({
        'category': 'TRIAL_SPLIT',
        'matrix_name': 'trial_split_method',
        'description': f'Deterministic: 10 equal chunks, odd->train, even->test. Seed={random_seed}',
        'shape': '-',
        'dtype': '-',
        'size_bytes': '-',
        'size_human': '-',
        'n_elements': '-'
    })
    
    # === DATA MATRICES ===
    # Response matrices
    response_dtype = response.dtype
    rows.append({
        'category': 'DATA_MATRIX',
        'matrix_name': 'response (full)',
        'description': 'All cells, all trials',
        'shape': str(response.shape),
        'dtype': str(response_dtype),
        'size_bytes': calc_size(response.shape, response_dtype),
        'size_human': format_size(calc_size(response.shape, response_dtype)),
        'n_elements': np.prod(response.shape)
    })
    
    response_train_shape = (n_train_samples, n_trials)
    rows.append({
        'category': 'DATA_MATRIX',
        'matrix_name': 'response_train',
        'description': 'Training cells, all trials',
        'shape': str(response_train_shape),
        'dtype': str(response_dtype),
        'size_bytes': calc_size(response_train_shape, response_dtype),
        'size_human': format_size(calc_size(response_train_shape, response_dtype)),
        'n_elements': np.prod(response_train_shape)
    })
    
    response_test_shape = (n_test_samples, n_trials)
    rows.append({
        'category': 'DATA_MATRIX',
        'matrix_name': 'response_test',
        'description': 'Test cells, all trials',
        'shape': str(response_test_shape),
        'dtype': str(response_dtype),
        'size_bytes': calc_size(response_test_shape, response_dtype),
        'size_human': format_size(calc_size(response_test_shape, response_dtype)),
        'n_elements': np.prod(response_test_shape)
    })
    
    # Input matrices
    inputs_dtype = inputs.dtype
    rows.append({
        'category': 'DATA_MATRIX',
        'matrix_name': 'inputs (full)',
        'description': f'All cells, {n_features} features, all trials',
        'shape': inputs_shape_str,
        'dtype': str(inputs_dtype),
        'size_bytes': calc_size(inputs.shape, inputs_dtype),
        'size_human': format_size(calc_size(inputs.shape, inputs_dtype)),
        'n_elements': np.prod(inputs.shape)
    })
    
    if inputs.ndim == 2:
        inputs_train_shape = (n_train_samples, n_trials)
        inputs_test_shape = (n_test_samples, n_trials)
    else:
        inputs_train_shape = (n_train_samples, n_features, n_trials)
        inputs_test_shape = (n_test_samples, n_features, n_trials)
    
    rows.append({
        'category': 'DATA_MATRIX',
        'matrix_name': 'inputs_train',
        'description': f'Training cells, {n_features} features, all trials',
        'shape': str(inputs_train_shape),
        'dtype': str(inputs_dtype),
        'size_bytes': calc_size(inputs_train_shape, inputs_dtype),
        'size_human': format_size(calc_size(inputs_train_shape, inputs_dtype)),
        'n_elements': np.prod(inputs_train_shape)
    })
    
    rows.append({
        'category': 'DATA_MATRIX',
        'matrix_name': 'inputs_test',
        'description': f'Test cells, {n_features} features, all trials',
        'shape': str(inputs_test_shape),
        'dtype': str(inputs_dtype),
        'size_bytes': calc_size(inputs_test_shape, inputs_dtype),
        'size_human': format_size(calc_size(inputs_test_shape, inputs_dtype)),
        'n_elements': np.prod(inputs_test_shape)
    })
    
    # === OBJECTIVE FUNCTION SUB-MATRICES (within training samples) ===
    # These are created inside objective() for the training samples
    if inputs.ndim == 2:
        x_train_shape = (n_train_samples, 1, n_training_trials)
        x_test_shape = (n_train_samples, 1, n_test_trials)
    else:
        x_train_shape = (n_train_samples, n_features, n_training_trials)
        x_test_shape = (n_train_samples, n_features, n_test_trials)
    
    y_train_shape = (n_train_samples, n_training_trials)
    y_test_shape = (n_train_samples, n_test_trials)
    
    rows.append({
        'category': 'OBJECTIVE_SUBMATRIX',
        'matrix_name': 'x_train (in objective)',
        'description': 'Training samples, training trials (param fitting)',
        'shape': str(x_train_shape),
        'dtype': str(inputs_dtype),
        'size_bytes': calc_size(x_train_shape, inputs_dtype),
        'size_human': format_size(calc_size(x_train_shape, inputs_dtype)),
        'n_elements': np.prod(x_train_shape)
    })
    
    rows.append({
        'category': 'OBJECTIVE_SUBMATRIX',
        'matrix_name': 'y_train (in objective)',
        'description': 'Training samples, training trials (param fitting)',
        'shape': str(y_train_shape),
        'dtype': str(response_dtype),
        'size_bytes': calc_size(y_train_shape, response_dtype),
        'size_human': format_size(calc_size(y_train_shape, response_dtype)),
        'n_elements': np.prod(y_train_shape)
    })
    
    rows.append({
        'category': 'OBJECTIVE_SUBMATRIX',
        'matrix_name': 'x_test (in objective)',
        'description': 'Training samples, test trials (loss evaluation)',
        'shape': str(x_test_shape),
        'dtype': str(inputs_dtype),
        'size_bytes': calc_size(x_test_shape, inputs_dtype),
        'size_human': format_size(calc_size(x_test_shape, inputs_dtype)),
        'n_elements': np.prod(x_test_shape)
    })
    
    rows.append({
        'category': 'OBJECTIVE_SUBMATRIX',
        'matrix_name': 'y_test (in objective)',
        'description': 'Training samples, test trials (loss evaluation)',
        'shape': str(y_test_shape),
        'dtype': str(response_dtype),
        'size_bytes': calc_size(y_test_shape, response_dtype),
        'size_human': format_size(calc_size(y_test_shape, response_dtype)),
        'n_elements': np.prod(y_test_shape)
    })
    
    # === FEATURE INFO ===
    rows.append({
        'category': 'FEATURES',
        'matrix_name': 'n_features',
        'description': 'Number of input features per sample',
        'shape': '-',
        'dtype': '-',
        'size_bytes': '-',
        'size_human': '-',
        'n_elements': n_features
    })
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'data_summary.csv')
    df.to_csv(csv_path, index=False)
    
    # Also print a summary
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"Sample Split: {n_train_samples}/{n_total_samples} train, {n_test_samples}/{n_total_samples} test")
    print(f"Trial Split:  {n_training_trials}/{n_trials} train, {n_test_trials}/{n_trials} test (per sample, in objective)")
    print(f"Features:     {n_features} per sample")
    print(f"Data Types:   response={response_dtype}, inputs={inputs_dtype}")
    total_size = sum(r['size_bytes'] for r in rows if isinstance(r['size_bytes'], (int, float)))
    print(f"Total Data:   {format_size(total_size)}")
    print(f"Saved to:     {csv_path}")
    print("=" * 70 + "\n")
    
    logging.info(f"Data summary saved to {csv_path}")
    
    return df


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
    try:
        # any call taking >5s will raise timeout_decorator.TimeoutError
        # xi has shape (n_features, n_trials)
        # yi has shape (n_trials,) for scalar or (n_targets, n_trials) for vectorized
        return jnp.array([_safe_estimate(param_estimator, x[i], y[i]) for i in range(y.shape[0])])
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


def objective_legacy(model, param_estimator, loss_func, x, y, 
              param_penalty_weight=0.1, fit_params=True, random_seed=0,
              FAILED_PROGRAM_COST=jnp.inf, tol=1e-2, max_iter=1_000,
              use_param_estimator=True, trial_batch_size=5000) -> tuple[float, jnp.ndarray, float, jnp.ndarray]:
    """
    LEGACY: Calculate the loss of the model for scalar (single-target) outputs.
    
    This is the original objective function preserved for backward compatibility.
    For new code, use objective() which handles both scalar and vectorized outputs.
    
    The loss is calculated as the mean over samples and trials of the loss function provided.
    
    Args:
        model (function): The model which predicts neural activity from inputs
                                and free parameters (for a single sample).
                                Signature: model(X, *params) -> activity
                                where X has shape (n_features, n_trials) for a single sample.
        param_estimator (function): Function to estimate initial parameters for the model.
                                Signature: param_estimator(X, response) -> params
                                where X has shape (n_features, n_trials) for a single sample.
        loss_func (function): The loss function to use for calculating the loss.
        x: Input data. Can be:
           - 2D array (n_samples, n_trials) - will be auto-expanded to (n_samples, 1, n_trials)
           - 3D array (n_samples, n_features, n_trials)
           - Inputs object
        y (jnp.ndarray): Response data, shape (n_samples, n_trials). MUST be 2D for legacy.
        param_penalty_weight (float): Weight for the penalty on the number of parameters. Default is 0.1.
        fit_params (bool): Whether to fit the parameters of the model. Default is True.
        random_seed (int or None): Random seed for reproducibility. Default is 0.
        FAILED_PROGRAM_COST (float): Cost assigned to failed models. Default is np.inf.
        tol (float): Tolerance for optimization convergence. Default is 1e-2.
        max_iter (int): Maximum number of iterations for optimization. Default is 1_000.
        use_param_estimator (bool): Whether to use the parameter estimator to compute initial parameters. Default is True.
        trial_batch_size (int): Number of trials to process per mini-batch to avoid GPU OOM. Default is 5000.

    Returns:
        tuple[
            - float: The cross-validated loss of the model with data fit by the parameter estimator,
            - jnp.ndarray: The parameters fit by the parameter estimator.
            - float: The average loss (MSE on test set) across all samples. 
                     Returns FAILED_PROGRAM_COST if the model fails for ANY cell.
            - jnp.ndarray: The parameters for each sample (n_samples, n_params).
    """
    t_start = time.time()
    
    # Normalize x to Inputs format: (n_samples, n_features, n_trials)
    x_inputs = ensure_inputs(x)
    x_data = x_inputs.to_tensor()  # shape: (n_samples, n_features, n_trials)
    
    n_samples, n_features, n_trials = x_data.shape
    
    # train/test split over trials (axis 2)
    # split the trials into 10 equal length chunks and allocate all odd chunks to train and even chunks to test 
    key = jax.random.PRNGKey(random_seed)        
    n_trial_splits = 10 
    trials_per_split = n_trials // n_trial_splits
    split_indices = [jnp.arange(i * trials_per_split, (i + 1) * trials_per_split) for i in range(n_trial_splits)]
    training_trials_idx = jnp.concatenate([split_indices[i] for i in range(n_trial_splits) if i % 2 == 1])
    test_trials_idx = jnp.concatenate([split_indices[i] for i in range(n_trial_splits) if i % 2 == 0])
    # # old code 
    # key = jax.random.PRNGKey(random_seed)
    # training_size = n_trials // 2
    # shuffled_indices = jax.random.permutation(key, jnp.arange(n_trials))
    # training_trials_idx = shuffled_indices[:training_size]
    # test_trials_idx = shuffled_indices[training_size:]

    # Split inputs and response: x has shape (n_samples, n_features, n_trials)
    x_train = x_data[:, :, training_trials_idx]  # (n_samples, n_features, training_size)
    y_train = y[:, training_trials_idx]           # (n_samples, training_size)
    x_test = x_data[:, :, test_trials_idx]        # (n_samples, n_features, test_size)
    y_test = y[:, test_trials_idx]                # (n_samples, test_size)

    # Perform initial param calc. x must be numpy array of shape (n_samples, n_features, n_trials)
    if use_param_estimator:
        initial_params = compute_initial_params(param_estimator, model, np.asarray(x_train), np.asarray(y_train))
    else:
        initial_params = compute_default_params(model)
        # if initial_params not none, reshape from (1, n_params) to (n_samples, n_params)
        if initial_params is not None:
            n_params = initial_params.shape[1]
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

    # Fail immediately if model doesn't run
    # x_data[sample_idx] has shape (n_features, n_trials)
    try:
        # Check compatibility with JAX's tracing mechanism
        model_jit = jax.jit(model)
        test_n_trials = x_data.shape[2]  # full n_trials for validation
        for sample_idx in np.random.choice(n_samples, size=min(10, n_samples), replace=False):
            # Validate with concrete values: x_data[sample_idx] is (n_features, n_trials)
            output = model_jit(x_data[sample_idx], *initial_params[sample_idx])
            if output.ndim != 1 or output.shape[0] != test_n_trials:
                logging.info(f"Error: model output shape {output.shape[0]} does not match input n_trials {test_n_trials}.")
                return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params
            # Validate with abstract tracer values
            jax.eval_shape(model_jit, x_data[sample_idx], *initial_params[sample_idx])
    except Exception as e:
        logging.info(f"Model failed to run or is incompatible with JAX tracing: {e}")
        return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params

    # Loss for a single sample: x_data has shape (n_features, n_trials), y_data has shape (n_trials,)
    loss_single_cell = lambda params, x_i, y_i: jnp.mean(loss_func(model(x_i, *params), y_i), axis=-1)
    # vectorize the loss function for all samples. The inputs will have shapes:
    # - params: (n_samples, n_params)
    # - x_i: (n_samples, n_features, n_trials) -> batched over axis 0
    # - y_i: (n_samples, n_trials) -> batched over axis 0
    # The output will have shape (n_samples,)
    loss_total = jax.vmap(loss_single_cell, in_axes=(0, 0, 0), out_axes=0)

    # Mini-batched loss and gradient computation to avoid GPU OOM
    # Key: we JIT only the per-batch computation, NOT the loop over batches
    n_train_trials = x_train.shape[2]
    
    # JIT-compiled loss for a single batch
    @jax.jit
    def loss_single_batch(params_2d, x_batch, y_batch):
        """Compute sum of losses for one batch (JIT-compiled)."""
        batch_losses = loss_total(params_2d, x_batch, y_batch)  # (n_samples,)
        return jnp.sum(batch_losses)
    
    # Gradient of per-batch loss
    grad_single_batch = jax.jit(jax.grad(loss_single_batch))
    
    def loss_and_grad_batched(params):
        """Compute loss and gradient by accumulating over trial batches (not JIT-compiled)."""
        params_2d = params.reshape(-1, n_params)
        total_loss = 0.0
        total_grad = jnp.zeros_like(params)
        
        for start_idx in range(0, n_train_trials, trial_batch_size):
            end_idx = min(start_idx + trial_batch_size, n_train_trials)
            batch_weight = (end_idx - start_idx) / n_train_trials
            x_batch = x_train[:, :, start_idx:end_idx]
            y_batch = y_train[:, start_idx:end_idx]
            
            # Compute loss and gradient for this batch
            batch_loss = loss_single_batch(params_2d, x_batch, y_batch)
            batch_grad = grad_single_batch(params_2d, x_batch, y_batch)
            
            # Accumulate with proper weighting
            # We need to batch_weight because loss_single_batch calculates the mean loss per batch. 
            # An alternative would have been to let loss_single_cell calculate the sum of loss for each cell 
            # and then divide by n_trials at the end rather than batch_weight.
            total_loss += batch_loss * batch_weight
            total_grad += batch_grad.reshape(-1) * batch_weight
        
        # Normalize by n_samples
        return total_loss / n_samples, total_grad / n_samples

    if fit_params:
        # solver = jaxopt.ScipyMinimize(
        #     fun=loss_param_and_grad,
        #     value_and_grad=True,
        #     method='L-BFGS-B',
        #     maxiter=max_iter,
        #     tol=tol,
        #     jit=True)
        # try:
        #     result = solver.run(initial_params.reshape(-1))
        #     params = jnp.asarray(result.params).reshape(n_samples, n_params)
        #     print(f"Optimization success: {result.state.success}, iterations: {result.state.iter_num}")
        # except Exception as e:
        #     params = initial_params
        #     logging.info(f"Error during optimization: {e}")
        # 

        # 1.  build adam with learning rate schedule for better convergence
        #     Higher initial LR helps parameters with different scales converge
        peak_lr = 0.001
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=peak_lr * 0.1,
            peak_value=peak_lr,
            warmup_steps=50,
            decay_steps=max_iter,
            end_value=peak_lr * 0.01
        )
        opt = optax.adam(schedule, b1=0.9, b2=0.999, eps=1e-8)
        opt_state = opt.init(initial_params.reshape(-1))
        
        # 2. Define update step (NOT jit-compiled because loss_and_grad_batched has Python loop)
        def train_step(params, opt_state):
            loss, grad = loss_and_grad_batched(params)
            updates, new_opt_state = opt.update(grad, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss
        
        # 3.  iterate
        print_every = 50
        params = initial_params.reshape(-1)  # Flatten params for the optimizer
        initial_loss, _ = loss_and_grad_batched(params)
        
        # Early exit for catastrophically bad programs (loss > 1e10 suggests garbage outputs)
        CATASTROPHIC_LOSS_THRESHOLD = 1e6
        if initial_loss > CATASTROPHIC_LOSS_THRESHOLD:
            print(f"Initial loss {initial_loss:.2e} exceeds threshold {CATASTROPHIC_LOSS_THRESHOLD:.0e}. Skipping optimization.")
            logging.info(f"Skipping optimization: initial loss {initial_loss:.2e} > {CATASTROPHIC_LOSS_THRESHOLD:.0e}")
            return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params
        
        best_loss, best_params = initial_loss.copy(), params.copy()
        for step in range(1, max_iter + 1):
            params, opt_state, loss_val = train_step(params, opt_state)
            if jnp.isnan(loss_val) or jnp.isinf(loss_val) or jnp.any(jnp.isnan(params)) or jnp.any(jnp.isinf(params)):
                logging.info(f"Loss is NaN or Inf at step {step}. Stopping optimization.")
                print(f"Final loss: {loss_val:.4f} at step {step}")
                break
            # Also exit early if loss explodes during training
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
            return FAILED_PROGRAM_COST, jnp.zeros((n_samples, n_params))

    # compute the final loss on the test set for the initial and optimized parameters
    # Use mini-batched evaluation to avoid GPU OOM
    n_test_trials = x_test.shape[2]
    
    def eval_loss_batched(params_2d, x_eval, y_eval):
        """Compute loss by iterating over trial batches."""
        n_eval_trials = x_eval.shape[2]
        # Accumulate weighted sum: each batch contributes (batch_size / total_trials) weight
        weighted_sum = 0.0
        for start_idx in range(0, n_eval_trials, trial_batch_size):
            end_idx = min(start_idx + trial_batch_size, n_eval_trials)
            batch_size = end_idx - start_idx
            x_batch = x_eval[:, :, start_idx:end_idx]
            y_batch = y_eval[:, start_idx:end_idx]
            # loss_total returns (n_samples,) - mean loss per sample over trials in this batch
            batch_losses = loss_total(params_2d, x_batch, y_batch)  # (n_samples,)
            weighted_sum += jnp.nansum(batch_losses) * (batch_size / n_eval_trials)
        # Divide by n_samples to get mean over samples  
        return weighted_sum / n_samples
    
    initial_loss = eval_loss_batched(initial_params, x_test, y_test) + param_penalty_weight * n_params
    # print number of nans in initial_loss
    n_nans = jnp.sum(jnp.isnan(initial_loss))
    if n_nans > 0:
        print(f"Warning: initial loss contains {n_nans} NaNs. This may indicate a problem with the model or data.")
    initial_loss = jnp.nan_to_num(initial_loss, nan=FAILED_PROGRAM_COST, posinf=FAILED_PROGRAM_COST, neginf=FAILED_PROGRAM_COST)
    final_loss = eval_loss_batched(params, x_test, y_test) + param_penalty_weight * n_params
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


def objective_vectorized(model, param_estimator, loss_func, x, y,
                         target_weights=None,
                         param_penalty_weight=0.1, fit_params=True, random_seed=0,
                         FAILED_PROGRAM_COST=jnp.inf, tol=1e-2, max_iter=1_000,
                         use_param_estimator=True, trial_batch_size=10000) -> tuple[float, jnp.ndarray, float, jnp.ndarray]:
    """
    Calculate the loss of a model that predicts multiple targets (vectorized outputs).
    
    This function handles models where the output has shape (n_targets, n_trials) for
    each sample, allowing prediction of multiple target cells simultaneously.
    
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
        loss_func (function): Element-wise loss function. Applied to (pred, true) arrays.
                          Should return array of same shape as inputs.
        x: Input data. Can be:
           - 2D array (n_samples, n_trials) - will be auto-expanded to (n_samples, 1, n_trials)
           - 3D array (n_samples, n_features, n_trials)
           - Inputs object
        y: Output/response data. Can be:
           - 2D array (n_samples, n_trials) - auto-expanded to (n_samples, 1, n_trials)
           - 3D array (n_samples, n_targets, n_trials)
           - Outputs object
        target_weights: Optional weights for each target. Can be:
           - None: uniform weights (1/n_targets for each target, sums to 1)
           - 1D array of shape (n_targets,): custom weights (will be normalized to sum to 1)
        param_penalty_weight (float): Weight for the penalty on the number of parameters. Default is 0.1.
        fit_params (bool): Whether to fit the parameters of the model. Default is True.
        random_seed (int or None): Random seed for reproducibility. Default is 0.
        FAILED_PROGRAM_COST (float): Cost assigned to failed models. Default is np.inf.
        tol (float): Tolerance for optimization convergence. Default is 1e-2.
        max_iter (int): Maximum number of iterations for optimization. Default is 1_000.
        use_param_estimator (bool): Whether to use the parameter estimator to compute initial parameters. Default is True.
        trial_batch_size (int): Number of trials to process per mini-batch to avoid GPU OOM. Default is 5000.

    Returns:
        tuple[
            - float: The cross-validated loss of the model with initial parameters,
            - jnp.ndarray: The initial parameters (n_samples, n_params).
            - float: The average loss on test set after optimization.
                     Returns FAILED_PROGRAM_COST if the model fails.
            - jnp.ndarray: The optimized parameters (n_samples, n_params).
    """
    t_start = time.time()
    
    # Normalize inputs to Inputs format: (n_samples, n_features, n_trials)
    x_inputs = ensure_inputs(x)
    x_data = x_inputs.to_tensor()  # shape: (n_samples, n_features, n_trials)
    
    # Normalize outputs to Outputs format: (n_samples, n_targets, n_trials)
    y_outputs = ensure_outputs(y)
    y_data = y_outputs.data  # shape: (n_samples, n_targets, n_trials)
    
    n_samples, n_features, n_trials = x_data.shape
    n_targets = y_outputs.n_targets
    
    # Set up target weights (normalize to sum to 1)
    if target_weights is None:
        target_weights = jnp.ones(n_targets) / n_targets
    else:
        target_weights = jnp.asarray(target_weights)
        if target_weights.shape != (n_targets,):
            raise ValueError(f"target_weights shape {target_weights.shape} does not match n_targets={n_targets}")
        target_weights = target_weights / jnp.sum(target_weights)  # normalize
    
    # Train/test split over trials (axis 2)
    # Split the trials into 10 equal length chunks: odd chunks -> train, even chunks -> test
    key = jax.random.PRNGKey(random_seed)
    n_trial_splits = 10
    trials_per_split = n_trials // n_trial_splits
    split_indices = [jnp.arange(i * trials_per_split, (i + 1) * trials_per_split) for i in range(n_trial_splits)]
    training_trials_idx = jnp.concatenate([split_indices[i] for i in range(n_trial_splits) if i % 2 == 1])
    test_trials_idx = jnp.concatenate([split_indices[i] for i in range(n_trial_splits) if i % 2 == 0])
    
    # Split inputs and outputs
    x_train = x_data[:, :, training_trials_idx]  # (n_samples, n_features, training_size)
    y_train = y_data[:, :, training_trials_idx]  # (n_samples, n_targets, training_size)
    x_test = x_data[:, :, test_trials_idx]       # (n_samples, n_features, test_size)
    y_test = y_data[:, :, test_trials_idx]       # (n_samples, n_targets, test_size)
    
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
        model, x_data, initial_params, n_samples,
        expected_n_targets=n_targets,
        n_validation_samples=10
    )
    if not is_valid:
        logging.info(f"Model validation failed: {error_msg}")
        return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params
    
    # Helper: normalize model output to 2D (n_targets, n_trials)
    # This handles n_targets=1 case where model might return 1D
    def normalize_model_output(output, n_targets):
        """Ensure model output is 2D: (n_targets, n_trials)."""
        if output.ndim == 1 and n_targets == 1:
            return output[None, :]  # (1, n_trials)
        return output
    
    # Wrapped model that normalizes output
    def model_normalized(x_i, *params):
        output = model(x_i, *params)
        return normalize_model_output(output, n_targets)
    
    # Loss for a single sample:
    # x_i: (n_features, n_trials), y_i: (n_targets, n_trials)
    # model output: (n_targets, n_trials)
    # Returns: scalar (weighted sum of per-target MSE)
    def loss_single_sample(params, x_i, y_i):
        pred = model_normalized(x_i, *params)  # (n_targets, n_trials)
        # loss_func returns element-wise loss: (n_targets, n_trials)
        elementwise_loss = loss_func(pred, y_i)  # (n_targets, n_trials)
        # Mean over trials for each target: (n_targets,)
        per_target_mse = jnp.mean(elementwise_loss, axis=-1)
        # Weighted sum over targets: scalar
        return jnp.sum(target_weights * per_target_mse)
    
    # Vectorize over samples
    # params: (n_samples, n_params), x: (n_samples, n_features, n_trials), y: (n_samples, n_targets, n_trials)
    # Output: (n_samples,)
    loss_total = jax.vmap(loss_single_sample, in_axes=(0, 0, 0), out_axes=0)
    
    # Mini-batched loss and gradient computation to avoid GPU OOM
    n_train_trials = x_train.shape[2]
    
    @jax.jit
    def loss_single_batch(params_2d, x_batch, y_batch):
        """Compute sum of losses for one batch (JIT-compiled)."""
        batch_losses = loss_total(params_2d, x_batch, y_batch)  # (n_samples,)
        return jnp.sum(batch_losses)
    
    grad_single_batch = jax.jit(jax.grad(loss_single_batch))
    
    def loss_and_grad_batched(params):
        """Compute loss and gradient by accumulating over trial batches."""
        params_2d = params.reshape(-1, n_params)
        total_loss = 0.0
        total_grad = jnp.zeros_like(params)
        
        for start_idx in range(0, n_train_trials, trial_batch_size):
            end_idx = min(start_idx + trial_batch_size, n_train_trials)
            batch_weight = (end_idx - start_idx) / n_train_trials
            x_batch = x_train[:, :, start_idx:end_idx]
            y_batch = y_train[:, :, start_idx:end_idx]  # Note: 3D now
            
            batch_loss = loss_single_batch(params_2d, x_batch, y_batch)
            batch_grad = grad_single_batch(params_2d, x_batch, y_batch)
            
            total_loss += batch_loss * batch_weight
            total_grad += batch_grad.reshape(-1) * batch_weight
        
        return total_loss / n_samples, total_grad / n_samples
    
    if fit_params:
        # Adam optimizer with learning rate schedule
        peak_lr = 0.001
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=peak_lr * 0.1,
            peak_value=peak_lr,
            warmup_steps=50,
            decay_steps=max_iter,
            end_value=peak_lr * 0.01
        )
        opt = optax.adam(schedule, b1=0.9, b2=0.999, eps=1e-8)
        opt_state = opt.init(initial_params.reshape(-1))
        
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
    n_test_trials = x_test.shape[2]
    
    def eval_loss_batched(params_2d, x_eval, y_eval):
        """Compute loss by iterating over trial batches."""
        n_eval_trials = x_eval.shape[2]
        weighted_sum = 0.0
        for start_idx in range(0, n_eval_trials, trial_batch_size):
            end_idx = min(start_idx + trial_batch_size, n_eval_trials)
            batch_size = end_idx - start_idx
            x_batch = x_eval[:, :, start_idx:end_idx]
            y_batch = y_eval[:, :, start_idx:end_idx]  # Note: 3D now
            batch_losses = loss_total(params_2d, x_batch, y_batch)
            weighted_sum += jnp.nansum(batch_losses) * (batch_size / n_eval_trials)
        return weighted_sum / n_samples
    
    initial_loss = eval_loss_batched(initial_params, x_test, y_test) + param_penalty_weight * n_params
    n_nans = jnp.sum(jnp.isnan(initial_loss))
    if n_nans > 0:
        print(f"Warning: initial loss contains {n_nans} NaNs.")
    initial_loss = jnp.nan_to_num(initial_loss, nan=FAILED_PROGRAM_COST, posinf=FAILED_PROGRAM_COST, neginf=FAILED_PROGRAM_COST)
    
    final_loss = eval_loss_batched(params, x_test, y_test) + param_penalty_weight * n_params
    n_nans = jnp.sum(jnp.isnan(final_loss))
    if n_nans > 0:
        print(f"Warning: final loss contains {n_nans} NaNs.")
    final_loss = jnp.nan_to_num(final_loss, nan=FAILED_PROGRAM_COST, posinf=FAILED_PROGRAM_COST, neginf=FAILED_PROGRAM_COST)
    
    t_end = time.time()
    print(f"Time taken for optimization: {t_end - t_start:.4f} seconds")
    return float(initial_loss), initial_params, float(final_loss), params


def objective(model, param_estimator, loss_func, x, y,
              target_weights=None,
              param_penalty_weight=0.1, fit_params=True, random_seed=0,
              FAILED_PROGRAM_COST=jnp.inf, tol=1e-2, max_iter=1_000,
              use_param_estimator=True, trial_batch_size=5000) -> tuple[float, jnp.ndarray, float, jnp.ndarray]:
    """
    Calculate the loss of the model. Always uses vectorized Outputs representation.
    
    This is the main entry point for model evaluation. All outputs are normalized
    to Outputs objects with shape (n_samples, n_targets, n_trials), even for
    single-target (scalar) cases where n_targets=1.
    
    For n_targets=1, this currently delegates to objective_legacy for backward
    compatibility. Once objective_vectorized is fully tested, we can remove this
    special case and always use objective_vectorized.
    
    Args:
        model (function): The model which predicts neural activity from inputs
                          and free parameters (for a single sample).
                          Signature: model(X, *params) -> activity
                          where X has shape (n_features, n_trials) for a single sample.
                          Output shape: (n_trials,) for scalar, (n_targets, n_trials) for vectorized.
        param_estimator (function): Function to estimate initial parameters for the model.
                          Signature: param_estimator(X, response) -> params
                          where X has shape (n_features, n_trials) for a single sample.
        loss_func (function): The loss function to use for calculating the loss.
        x: Input data. Can be:
           - 2D array (n_samples, n_trials) - will be auto-expanded to (n_samples, 1, n_trials)
           - 3D array (n_samples, n_features, n_trials)
           - Inputs object
        y: Output/response data. Always normalized to Outputs object. Can be:
           - 2D array (n_samples, n_trials) - auto-expanded to (n_samples, 1, n_trials)
           - 3D array (n_samples, n_targets, n_trials)
           - Outputs object
        target_weights: Optional weights for each target. Only used for n_targets > 1.
           - None: uniform weights (1/n_targets for each)
           - 1D array (n_targets,): custom weights (normalized to sum to 1)
        param_penalty_weight (float): Weight for the penalty on the number of parameters. Default is 0.1.
        fit_params (bool): Whether to fit the parameters of the model. Default is True.
        random_seed (int or None): Random seed for reproducibility. Default is 0.
        FAILED_PROGRAM_COST (float): Cost assigned to failed models. Default is np.inf.
        tol (float): Tolerance for optimization convergence. Default is 1e-2.
        max_iter (int): Maximum number of iterations for optimization. Default is 1_000.
        use_param_estimator (bool): Whether to use the parameter estimator to compute initial parameters. Default is True.
        trial_batch_size (int): Number of trials to process per mini-batch to avoid GPU OOM. Default is 5000.

    Returns:
        tuple[
            - float: The cross-validated loss with initial parameters,
            - jnp.ndarray: The initial parameters.
            - float: The average loss on test set after optimization.
            - jnp.ndarray: The optimized parameters for each sample (n_samples, n_params).
    """
    # Normalize y to Outputs format: always (n_samples, n_targets, n_trials)
    y_outputs = ensure_outputs(y)
    n_targets = y_outputs.n_targets
    
    if n_targets == 1:
        # Single target: use legacy implementation for backward compatibility
        # TODO: Once objective_vectorized is fully tested with n_targets=1,
        # remove this branch and always use objective_vectorized
        y_2d = y_outputs.to_2d(0)  # Extract first (only) target as 2D
        if target_weights is not None:
            logging.info("Warning: target_weights ignored for single-target outputs.")
        return objective_legacy(
            model=model,
            param_estimator=param_estimator,
            loss_func=loss_func,
            x=x,
            y=y_2d,
            param_penalty_weight=param_penalty_weight,
            fit_params=fit_params,
            random_seed=random_seed,
            FAILED_PROGRAM_COST=FAILED_PROGRAM_COST,
            tol=tol,
            max_iter=max_iter,
            use_param_estimator=use_param_estimator,
            trial_batch_size=trial_batch_size,
        )
    else:
        # Multiple targets: use vectorized implementation
        return objective_vectorized(
            model=model,
            param_estimator=param_estimator,
            loss_func=loss_func,
            x=x,
            y=y_outputs,
            target_weights=target_weights,
            param_penalty_weight=param_penalty_weight,
            fit_params=fit_params,
            random_seed=random_seed,
            FAILED_PROGRAM_COST=FAILED_PROGRAM_COST,
            tol=tol,
            max_iter=max_iter,
            use_param_estimator=use_param_estimator,
            trial_batch_size=trial_batch_size,
        )


async def generate_new_model(current_island, llm_name, client, 
                                    spike_matrix, stimuli, prompt_manager,
                                    mode='explore', k_max=2, temp=1, 
                                    thinking_budget=1, img_dir=None, diagnostics_module=None,
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
    use_image = img_dir is not None
    use_chat_mode = island_chat_manager is not None and island_id is not None
    model_name = prompt_manager.get_model_name()
    
    # Use appropriate prompt function based on mode
    if use_chat_mode:
        program_prompt = prompt_manager.get_program_prompt(random_programs, mode=mode, use_image=use_image)
    else:
        program_prompt = prompt_manager.get_program_prompt_legacy(random_programs, mode=mode, use_image=use_image)

    if use_image and diagnostics_module is not None:
        try:
            sup_title = "".join([f"{model_name}_v{i+1}: Loss = {random_programs['train_loss'][i]:.2f} \n" for i in range(min(3, len(random_programs)))])
            diagnostics_module.plot_model_fits(programs_df=random_programs,
                                    loss_function=loss_functions.quadratic_loss,
                                    inputs=stimuli, response=spike_matrix,
                                    sample_selection=np.random.choice(spike_matrix.shape[0], size=9, replace=False),
                                    save_path=img_dir,
                                    labels=[f'{model_name}_v_1', f'{model_name}_v_2'],
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
        return None, None, (parent1_id, parent2_id)
    code_string = code_string.replace(f'def {model_name}_v{k+1}(', f'def {model_name}(')
    
    return code_string, program_prompt, (parent1_id, parent2_id)

async def generate_new_parameter_estimator(current_island, 
                                           model_code_string: str,
                                           llm_name, client, 
                                           spike_matrix, stimuli, prompt_manager,
                                           mode='explore', k_max=1, temp=1,
                                           param_estimator_max_lines=100, img_dir=None,
                                           swear_words=['lstsq', 'scipy.optimize', 'optimize.minimize', 'curve_fit', 'sklearn'], 
                                           island_chat_manager=None, island_id: int = None,
                                           batch_id: int = 0,
                                           diagnostics_module=None,
                                           use_large_model: bool = False):                                           
    if model_code_string is None:
        logging.info("No model code string provided, skipping parameter estimator generation.")
        return None, None
    k = min(k_max, len(current_island))
    random_programs = current_island.sample(k, replace=False).reset_index(drop=True)
    # sort from worst to best (loss descending)
    random_programs = random_programs.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
    use_image = img_dir is not None
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
    if use_image and diagnostics_module is not None:
        try:
            sup_title = "".join([f"model_v{i+1}: Loss = {random_programs['train_loss'][i]:.2f} \n" for i in range(min(3, len(random_programs)))])
            diagnostics_module.plot_model_fits(programs_df=random_programs_crude,
                                    loss_function=loss_functions.quadratic_loss,
                                    inputs=stimuli, response=spike_matrix,
                                    sample_selection=np.random.choice(spike_matrix.shape[0], size=4, replace=False),
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
        return None, None
    code_string = code_string.replace(f'def parameter_estimator_v{k+1}(', 'def parameter_estimator(')
    func = utils.str_to_func(code_string, 'parameter_estimator')
    return code_string, func

async def not_used_yet_generate_new_parameter_estimator_from_image_feedback(image_prompt: str,
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
    llm_output = await llm_helper.call_llm_async(image_prompt, model_name=model_name, client=client, temperature=temp, img_bytes=img_bytes)
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

async def hypothesis_engine(n_iterations=9, time_limit=60, k_max=2, n_islands=8, batch_size=6, 
                critical_population_size=12, min_wise_population_size=0, 
                n_migrants=2, fit_params=True, tol=1e-6, exploit_point=0.5,
                param_penalty_weight=0.01, FAILED_PROGRAM_COST=np.inf,
                use_image_feedback=True, use_param_estimator=True,
                use_chat_mode=False,  # If True, use persistent chat sessions per island (expensive)
                chat_token_limit=50000,  # Max tokens per chat before auto-summarize and reset. 0 = unlimited
                exploration_topology = [1, 2, 3, 4, 5, 6, 7, 0],
                exploitation_topology = [1, 2, 3, 4, 5, 6, 7, 0],
                tiny_lm_name = 'gemini-2.0-flash-lite',
                little_lm_name = 'gemini-2.0-flash',
                large_lm_name = 'gemini-2.5-flash',
                use_large_every = 3,
                training_ratio = 0.5, 
                max_iter = 1_000,
                use_large_model_for_param_estimators=False,
                numpy_programs = None,
                jax_programs = None,
                param_estimators = None,
                load_and_process_data_fn = None,
                data_config = None,
                diagnostics_module = None,
                prompts_config_path = None,
                log_best_loss = True,
                random_seed = 42):
    """ 
    Main function to run the hypothesis engine.
    
    Args:
        data_config: Dict containing all data loading parameters. This is passed
                     directly to load_and_process_data_fn, which extracts whatever
                     parameters it needs. This allows different experiments to have
                     different parameter sets without changing hypothesis_engine.
        log_best_loss: If True, logs the best loss at each iteration to a CSV file
                       for live monitoring. The file is saved to the experiment
                       output directory as 'best_loss_log.csv'.
    """
    if data_config is None:
        data_config = {}
    
    # Initialize prompt manager with experiment-specific prompts
    if prompts_config_path is None:
        # Fall back to default config path
        prompts_config_path = Path(__file__).parent.parent / "config" / "prompts.yaml"
    prompt_manager = PromptManager(config_path=prompts_config_path)
    
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

    # raise error if numpy_programs, jax_programs, or param_estimators are None or have length /= 2
    if numpy_programs is None or len(numpy_programs) != 2:
        raise ValueError("numpy_programs must be a list of 2 functions.")
    if jax_programs is None or len(jax_programs) != 2:
        raise ValueError("jax_programs must be a list of 2 functions.")
    if param_estimators is None or len(param_estimators) != 2:
        raise ValueError("param_estimators must be a list of 2 functions.")

    data_dict = load_and_process_data_fn(**data_config)
    response = data_dict['response']
    # Use 'inputs' if available (new format), fall back to 'trials' (deprecated)
    if 'inputs' in data_dict:
        inputs = data_dict['inputs'].data
    else:
        inputs = data_dict['trials'].data  # Keep for backward compat during transition

    n_good_samples, n_trials = response.shape
    if inputs.ndim == 2:
        n_features = 1
    else: 
        n_features = inputs.shape[1]
        
    key = jax.random.PRNGKey(random_seed)
    def create_train_test_split(n_samples, training_ratio):
        training_size = int(n_samples * training_ratio)
        shuffled_indices = jax.random.permutation(key, jnp.arange(n_samples))
        training_samples = shuffled_indices[:training_size]
        test_samples = shuffled_indices[training_size:]
        return training_samples, test_samples
    training_samples, test_samples = create_train_test_split(n_good_samples, training_ratio)
    response_train, response_test = response[training_samples, :], response[test_samples, :]
    inputs_train, inputs_test = inputs[training_samples, :], inputs[test_samples, :]  # has shape (n_samples, n_features, n_trials)
    print(f"Loaded {n_good_samples} samples, {n_trials} trials per sample.")
    print(f"Using {len(training_samples)} samples for training and {len(test_samples)} samples for testing.")

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

    # Save data summary CSV for inspection
    save_data_summary(
        response=response,
        inputs=inputs,
        training_samples=training_samples,
        test_samples=test_samples,
        output_dir=full_dir,
        random_seed=random_seed
    )

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
        loss_init, params_init, loss, params = objective(program_jax, param_est, 
                                        loss_func=loss_functions.quadratic_loss, 
                                        x=inputs_train, y=response_train, 
                                        fit_params=fit_params, param_penalty_weight=param_penalty_weight, tol=tol,
                                        use_param_estimator=use_param_estimator, max_iter=max_iter)
        seed_losses[i] = loss
        # format strings
        program_code_string = utils.format_function_source(
            program_num, f'{model_name}_v{i+1}', 'import numpy as np'
        )
        parameter_estimator_code_string = utils.format_function_source(
            param_est, f'parameter_estimator_v{i+1}', 'import numpy as np'
        )
        program_jax_code_string = utils.format_function_source(
            program_jax, f'{model_name}_v{i+1}', 'import jax.numpy as jnp'
        )
        if inputs_train.shape[1] == 1 : 
            # evenly spaced evaluation points for 1D inputs
            eval_points = None
        else : 
            eval_points = inputs_train
        y_eval = diagnostics_module.compute_evaluation_matrix(program_jax, params, eval_points=eval_points, n_evaluation_points=100, n_features=n_features)

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
    
    if diagnostics_module is not None:
        diagnostics_module.plot_model_fits(programs_df=initial_programs,
                               loss_function=loss_functions.quadratic_loss,
                               inputs=inputs_train, response=response_train,
                               sample_selection=np.random.choice(len(inputs_train), size=9, replace=False),
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
                if use_image_feedback:
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
                                                                   spike_matrix=response_train, 
                                                                   stimuli=inputs_train,
                                                                   prompt_manager=prompt_manager,
                                                                   img_dir=model_image_dirs[island_idx, j],
                                                                   diagnostics_module=diagnostics_module,
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
                llm_name=little_lm_name,
                client=client,
                spike_matrix=response_train,
                stimuli=inputs_train,
                prompt_manager=prompt_manager,
                mode=mode,
                k_max=2,
                temp=temperature,
                param_estimator_max_lines=100,
                img_dir=os.path.join(image_feedback_dir, f'iter_{i}_island_{island_idx}_batch_{j}_param_estimator.png') if use_large_model_for_param_estimators else None,
                island_chat_manager=island_chat_manager,
                island_id=island_idx,
                batch_id=j,
                diagnostics_module=diagnostics_module,
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
            if model_new is None or param_est_new is None:
                logging.info(f"Skipping island {island_idx}, batch {j} due to LLM generation failure.")
                logging.info('-' * 50)
                continue
            
            initial_loss, initial_params, loss, optimized_params = objective(model_new, param_est_new, 
                                                                                loss_func=loss_functions.quadratic_loss,
                                                                                x=inputs_train, y=response_train,
                                                                                param_penalty_weight=param_penalty_weight,
                                                                                fit_params=fit_params, tol=tol, 
                                                                                use_param_estimator=use_param_estimator, 
                                                                                max_iter=max_iter)
            if loss == FAILED_PROGRAM_COST:
                logging.info('-' * 50)
                continue

            if inputs_train.shape[1] == 1 : 
                # evenly spaced evaluation points for 1D inputs
                eval_points = None
            else : 
                eval_points = inputs_train
            y_eval = diagnostics_module.compute_evaluation_matrix(model_new, optimized_params, eval_points=eval_points, n_evaluation_points=100)
            logging.info(f"Prompt: \n{prompt}\n")
            logging.info(f"Loss: {loss:.2f}\n")
            logging.info(f"Model: \n{model_code_string}\n")
            logging.info(f"Model (JAX): \n{model_code_string_jax}\n")
            logging.info(f"Parameter Estimator: \n{param_est_code_string}\n")


            # plot the fits of the neuron model and parameter estimator if using image feedback
            if use_image_feedback and diagnostics_module is not None:
                diagnostics_module.plot_model_fits(
                    programs_df=pd.DataFrame({'program': [model_new, model_new], 'params': [initial_params, optimized_params]}),
                    loss_function=loss_functions.quadratic_loss,
                    inputs=inputs_train,
                    response=response_train,
                    sample_selection=np.random.choice(len(inputs_train), size=4, replace=False),
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
            if diagnostics_module is not None:
                top_df = islands[island_idx].sort_values(by='train_loss').head(3).reset_index(drop=True)
                top_df = top_df.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
                sup_title = f"Iteration {i}, Island {island_idx}, Top {len(top_df)} Programs\n"
                sup_title += "\n".join([f"model {j+1}: iter {top_df['iteration_number'][j]}, birth island {top_df['birth_island'][j]}, batch {top_df['batch_index'][j]}, loss: {top_df['train_loss'][j]:.2f}" for j in range(len(top_df))])
                diagnostics_module.plot_model_fits(
                    programs_df=top_df,
                    loss_function=loss_functions.quadratic_loss,
                    inputs=inputs_train,
                    response=response_train,
                    sample_selection=np.random.choice(response_train.shape[0], size=6, replace=False),
                    title=sup_title,
                    save_path=os.path.join(iteration_dir, f'island_{island_idx}_top_programs.png'),
                    dpi=300.0)
        
        if diagnostics_module is not None:
            all_programs = pd.concat([islands[idx] for idx in range(n_islands)], ignore_index=True)
            top_programs = all_programs.sort_values(by='train_loss').head(3).reset_index(drop=True)
            top_programs = top_programs.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
            sup_title = f"Iteration {i}, Top 3 Programs Overall\n"
            sup_title += "\n".join([f"model {j+1}: iter {top_programs['iteration_number'][j]}, birth island {top_programs['birth_island'][j]}, batch {top_programs['batch_index'][j]}, loss: {top_programs['train_loss'][j]:.2f}" for j in range(len(top_programs))])
            diagnostics_module.plot_model_fits(
                programs_df=top_programs,
                loss_function=loss_functions.quadratic_loss,
                inputs=inputs_train,
                response=response_train,
                sample_selection=np.random.choice(response_train.shape[0], size=6, replace=False),
                title=sup_title,
                save_path=os.path.join(iteration_dir, 'top_programs_overall.png'),
                dpi=300.0)
        
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
            _, _, test_loss, optimized_params = objective(model, param_estimator,
                                                          loss_func=loss_functions.quadratic_loss,
                                                          x=inputs_test, y=response_test, fit_params=fit_params,
                                                          max_iter=max_iter, 
                                                          param_penalty_weight=param_penalty_weight, tol=tol,
                                                          use_param_estimator=use_param_estimator, 
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
    if diagnostics_module is not None:
        diagnostics_module.plot_train_vs_test_loss(programs_df=combined_programs_dataframe,
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

    if diagnostics_module is not None:
        for i, df in enumerate(df_list):
            df_sup = config_str
            df = df.head(3)
            df = df.sort_values(by='test_loss', ascending=False).reset_index(drop=True)
            df_sup += "".join([f"model {len(df) - i}: iter {df['iteration_number'][i]}, birth_island {df['birth_island'][i]}, batch {df['batch_index'][i]}, total loss {0.5 * (df['test_loss'][i] + df['train_loss'][i]):.2f}\n" for i in range(min(3, len(df)))])
            diagnostics_module.plot_model_fits(
                programs_df=df,
                loss_function=loss_functions.quadratic_loss,
                inputs=inputs_test,
                response=response_test,
                sample_selection=np.random.choice(response_test.shape[0], size=6, replace=False),
                title=df_sup,
                save_path=os.path.join(df_dirs[i], 'top_model_fits.png')
            )
            # plot top 3 models separately
            for j in range(min(3, len(df))):
                birth_island = df['birth_island'][j]
                iteration_number = df['iteration_number'][j]
                batch_index = df['batch_index'][j]
                sample_selection = np.random.choice(response_test.shape[0], size=6, replace=False)
                diagnostics_module.plot_single_model_fit(
                    model=df['program'][j],
                    loss_function=loss_functions.quadratic_loss,
                    x=inputs_test[sample_selection],
                    y=response_test[sample_selection],
                    params=df['params'][j][sample_selection],
                    title=f"Island {birth_island}, Iteration {iteration_number}, Batch {batch_index}, loss: {df['test_loss'][j]:.2f}",
                    save_path=os.path.join(df_dirs[i], f'top_model_fit_{min(3, len(df)) - j}.png')
                )
    
    # Log final token usage summary (if using chat mode)
    if island_chat_manager is not None:
        island_chat_manager.log_final_summary()