import inspect
import jax
import numpy as np
import jax.numpy as jnp
from typing import Callable, Union
from .data_structures import Inputs
# Set up logging to suppress warnings from httpx, urllib3, and google.genai
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.ERROR)
# load dotenv to load environment variables from .env file


def format_function_source(func: Callable, new_name: str, import_statement: str = "") -> str:
    """
    Format a function's source code with a new name and import statement.
    
    Args:
        func: The function to extract source code from
        new_name: The new name for the function
        import_statement: Import statement to prepend to the source code
        
    Returns:
        Formatted source code string
    """
    source = inspect.getsource(func)
    original_name = func.__name__
    formatted_source = source.replace(f'def {original_name}(', f'def {new_name}(')
    
    if import_statement and not import_statement.endswith('\n'):
        import_statement += '\n'
    
    return import_statement + formatted_source


def vmap_over_samples(model_fn):
    """Return a version of `model_fn` that accepts
       (X, params_matrix) and runs one row per sample.
       
    Args:
        model_fn: A model function with signature:
                  model_fn(X, *params) -> (n_trials,)
                  where X has shape (n_features, n_trials).
    
    Returns:
        A vmapped function that accepts:
        - X: shape (n_samples, n_features, n_trials)
        - params_matrix: shape (n_samples, n_params)
        Returns shape (n_samples, n_trials).
    """
    def _wrapped(X_cell, params_row):
        # X_cell shape: (n_features, n_trials) for one cell
        # params_row shape: (k,) - one cell's parameters
        return model_fn(X_cell, *params_row)   # unpack to scalars
    return jax.vmap(_wrapped, in_axes=(0, 0))   # both X and params batched over cells


def extract_code_block(text: Union[str, None], start_marker: str = "```python\n", end_marker: str = "```") -> Union[str, None]:
    """
    Extracts a code block from a given text string, using specified start and end markers.
    If the text is None, it returns an empty string.
    If start and end markers not found returns the whole text.
    Args:
        text (str or None): The input text containing the code block.
        start_marker (str): The marker indicating the start of the code block.
        end_marker (str): The marker indicating the end of the code block.
    Returns:
        str: The extracted code block, or an empty string if the text is None.
    """
    if text is None:
        return None
    
    # find the start of the code block
    start = text.find(start_marker)
    if start == -1:
        start = 0
    else:
        # move the start index to the end of the marker
        start += len(start_marker)

    # find the closing fence after that
    end = text.find(end_marker, start)
    if end == -1:
        end = len(text) 

    # return just the code between the fences
    return text[start:end].rstrip()


def str_to_func(code_string: Union[str, None], needle: str) -> Union[Callable, None]:
    """
    Convert a string containing Python code into a callable function,

    Args:
        code_string (str or None): The string containing the Python function definition.
        needle (str): The name of the function to be extracted from the string.

    Returns:
        function: The callable function defined in the string, or None if not found.
    """
    # check if code sting is None, if so, return None
    if code_string is None:
        return None
    
    # Prepare a namespace dictionary for exec to run in. 
    execution_namespace = {}

    # Execute the code string within the specified namespace
    try:
        exec(code_string, execution_namespace)  # Pass the dictionary
    except Exception as e:
        print(f"Error executing code string: {e}\nCode:\n{code_string}") # Print code on error
        return None
    else:
        # Retrieve the function object from the namespace dictionary
        if needle in execution_namespace:
            return execution_namespace[needle]
        else:
            print(f"Function {needle} not found in executed code.")
            return None


def check_jax_translation(
    np_func,
    jax_func,
    eval_points,
    params,
    sample_indices=None,
    max_eval_trials=32,
    rtol=1e-4,
    atol=1e-4,
):
    """
    Check NumPy vs JAX model agreement on a subset of eval points and samples.

    Args:
        np_func: Original NumPy model function with signature model(X, *params).
        jax_func: Translated JAX model function with signature model(X, *params).
        eval_points: Array with shape (n_samples, n_features, n_eval_trials) or
            (n_features, n_eval_trials) for a single sample.
        params: Array with shape (n_samples, n_params) or (n_params,).
        sample_indices: Optional sample indices to validate. If None, checks up to 3 samples.
        max_eval_trials: Max number of eval-trial points per sample used for comparison.
        rtol: Relative tolerance for numeric comparison.
        atol: Absolute tolerance for numeric comparison.

    Raises:
        ValueError: If shapes are incompatible or predictions mismatch.
    """
    eval_arr = np.asarray(eval_points)
    if eval_arr.ndim == 2:
        eval_arr = eval_arr[None, :, :]
    if eval_arr.ndim != 3:
        raise ValueError(
            f"eval_points must have shape (n_samples, n_features, n_eval_trials), got {eval_arr.shape}."
        )

    params_arr = np.asarray(params)
    if params_arr.ndim == 1:
        params_arr = np.broadcast_to(params_arr[None, :], (eval_arr.shape[0], params_arr.shape[0]))
    elif params_arr.ndim == 2:
        if params_arr.shape[0] == 1 and eval_arr.shape[0] > 1:
            params_arr = np.broadcast_to(params_arr, (eval_arr.shape[0], params_arr.shape[1]))
    else:
        raise ValueError(f"params must be 1D or 2D, got {params_arr.shape}.")

    if params_arr.shape[0] != eval_arr.shape[0]:
        raise ValueError(
            f"Sample mismatch between eval_points ({eval_arr.shape[0]}) and params ({params_arr.shape[0]})."
        )

    n_samples = eval_arr.shape[0]
    if sample_indices is None:
        n_check = min(3, n_samples)
        sample_indices = np.linspace(0, n_samples - 1, num=n_check, dtype=int)
    else:
        sample_indices = np.asarray(sample_indices).reshape(-1).astype(np.int64, copy=False)

    for sample_idx in sample_indices:
        if sample_idx < 0 or sample_idx >= n_samples:
            raise ValueError(f"sample index {sample_idx} out of range for n_samples={n_samples}.")
        x_eval = eval_arr[sample_idx]
        if max_eval_trials is not None and x_eval.shape[1] > max_eval_trials:
            keep_idx = np.linspace(0, x_eval.shape[1] - 1, num=max_eval_trials, dtype=int)
            x_eval = x_eval[:, keep_idx]

        sample_params = params_arr[sample_idx]
        np_pred = np.asarray(np_func(x_eval, *sample_params))
        jax_pred = np.asarray(jax_func(jnp.asarray(x_eval), *sample_params))

        if np_pred.shape != jax_pred.shape:
            raise ValueError(
                f"Shape mismatch at sample {sample_idx}: numpy={np_pred.shape}, jax={jax_pred.shape}"
            )

        if not np.allclose(np_pred, jax_pred, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(np_pred - jax_pred))
            raise ValueError(
                f"Numeric mismatch at sample {sample_idx}: max_diff={max_diff:.6g}"
            )


def build_evaluation_points(inputs: Inputs, x_min = None, x_max = None, n_bins = 100):
    """
    Build evaluation grid from config bounds.
    Parameters:
    - inputs: Inputs object containing the shape information (n_samples, n_features, n_trials).
    - x_min: Scalar or array-like of shape (n_features,) specifying the minimum value for each feature. 
            If scalar, the same minimum is applied to all features. If none, set to min value in inputs.
    - x_max: Scalar or array-like of shape (n_features,) specifying the maximum value for each feature.
            If scalar, the same maximum is applied to all features. If none, set to max value in inputs.
    - n_bins: Number of evaluation points to generate per feature.
    Returns:
    - eval_points: A numpy array of shape (n_samples, n_features, n_bins) containing the evaluation points for each sample and feature.
    """
    n_samples, n_features, _ = inputs.shape
    x_min = np.min(inputs.data, axis=(0, 2)) if x_min is None else x_min
    x_max = np.max(inputs.data, axis=(0, 2)) if x_max is None else x_max
    # now check that x_min and x_max are either scalars or arrays of shape (n_features,)
    assert np.isscalar(x_min) or (isinstance(x_min, (list, np.ndarray)) and len(x_min) == n_features), \
        "x_min must be a scalar or array of shape (n_features,)"
    assert np.isscalar(x_max) or (isinstance(x_max, (list, np.ndarray)) and len(x_max) == n_features), \
        "x_max must be a scalar or array of shape (n_features,)"
    # set vector bounds for each feature
    x_min_vec = np.full(n_features, x_min) if np.isscalar(x_min) else np.asarray(x_min)
    x_max_vec = np.full(n_features, x_max) if np.isscalar(x_max) else np.asarray(x_max)
    per_feature = np.stack(
        [np.linspace(x_min_vec[i], x_max_vec[i], n_bins) for i in range(n_features)],
        axis=0,
    )
    return np.broadcast_to(per_feature[None, :, :], (n_samples, n_features, n_bins))


def compute_evaluation_matrix(program, params, eval_points):
    """
    Compute model evaluations used for logging/comparison.
    """
    if eval_points is None:
        raise ValueError("eval_points must be provided.")

    params_arr = jnp.asarray(params)
    eval_arr = jnp.asarray(eval_points)

    if eval_arr.ndim == 1:
        eval_arr = eval_arr[:, None]

    program_vmap = vmap_over_samples(program)
    return program_vmap(eval_arr, params_arr)
