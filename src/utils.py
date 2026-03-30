import inspect
import re
import jax
import numpy as np
import jax.numpy as jnp
from typing import Callable, Dict, Union
# Set up logging to suppress warnings from httpx, urllib3, and google.genai
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Data dict helpers
# ---------------------------------------------------------------------------
# The data structure is a plain dict[str, np.ndarray] where all values share
# the same last dimension (n_trials). These helpers provide validation,
# slicing, and conversion utilities.
# ---------------------------------------------------------------------------

def validate_data(X: Dict[str, np.ndarray]) -> None:
    """Validate that X is a dict of arrays sharing the same first and last dimensions.

    All arrays must have at least 2 dimensions (n_samples, ..., n_trials), share
    the same leading n_samples axis, and share the same trailing n_trials axis.

    Raises ValueError with a clear message if validation fails.
    """
    if not isinstance(X, dict):
        raise ValueError(f"Data must be a dict, got {type(X).__name__}.")
    if len(X) == 0:
        raise ValueError("Data dict must not be empty.")
    n_samples = None
    n_trials = None
    for key, arr in X.items():
        if not isinstance(arr, (np.ndarray, jnp.ndarray)):
            raise ValueError(
                f"Data['{key}'] must be a numpy or JAX array, got {type(arr).__name__}."
            )
        if arr.ndim < 2:
            raise ValueError(
                f"Data['{key}'] has {arr.ndim} dimension(s); all arrays must have at least "
                f"2 dimensions (n_samples, ..., n_trials)."
            )
        if n_samples is None:
            n_samples = arr.shape[0]
        elif arr.shape[0] != n_samples:
            raise ValueError(
                f"All arrays must share the same first dimension (n_samples). "
                f"First key has n_samples={n_samples}, but '{key}' has n_samples={arr.shape[0]}."
            )
        if n_trials is None:
            n_trials = arr.shape[-1]
        elif arr.shape[-1] != n_trials:
            raise ValueError(
                f"All arrays must share the same last dimension (n_trials). "
                f"First key has n_trials={n_trials}, but '{key}' has n_trials={arr.shape[-1]}."
            )


def data_n_trials(X: Dict[str, np.ndarray]) -> int:
    """Return the shared last dimension (n_trials) of the data dict."""
    first_arr = next(iter(X.values()))
    return int(first_arr.shape[-1])


def data_n_samples(X: Dict[str, np.ndarray]) -> int:
    """Return the first dimension (n_samples) of the data dict."""
    first_arr = next(iter(X.values()))
    return int(first_arr.shape[0])


def slice_data_samples(X: Dict[str, np.ndarray], indices) -> Dict[str, np.ndarray]:
    """Slice the sample axis (dim 0) of every array in the data dict."""
    return {k: v[indices] for k, v in X.items()}


def slice_data_trials(X: Dict[str, np.ndarray], indices) -> Dict[str, np.ndarray]:
    """Slice the trial axis (last dim) of every array in the data dict."""
    return {k: v[..., indices] for k, v in X.items()}


def get_data_sample(X: Dict[str, np.ndarray], idx: int) -> Dict[str, np.ndarray]:
    """Extract a single sample from the data dict, removing the sample axis."""
    return {k: v[idx] for k, v in X.items()}


def data_as_jax(X: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
    """Convert all arrays in the data dict to JAX arrays."""
    return {k: jnp.asarray(v) for k, v in X.items()}


def data_as_numpy(X: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert all arrays in the data dict to numpy arrays."""
    return {k: np.asarray(v) for k, v in X.items()}


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
    """Return a version of ``model_fn`` that maps over the sample axis.

    Args:
        model_fn: A model function with signature
            ``model_fn(data_i, params) -> output`` for a single sample,
            where ``data_i`` is a dict of arrays (no sample axis).

    Returns:
        A vmapped function that accepts:
        - data: dict of arrays with leading sample axis
        - params_tree: pytree with leading sample axis for each leaf
        Maps over dim 0 of both data and params.
    """
    def _wrapped(data_i, params_i):
        return model_fn(data_i, params_i)
    return jax.vmap(_wrapped, in_axes=(0, 0))


def tree_to_jax(params):
    """Convert all leaves in a pytree to jnp arrays."""
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), params)


def call_model(model_fn, data, params, prefer_jax: bool = True):
    """Invoke a model with data/params converted to JAX arrays when possible.

    Args:
        model_fn: Model function with signature ``model_fn(data_i, params)``.
        data: Single-sample data dict (no sample axis).
        params: Parameter pytree for a single sample.
        prefer_jax: If True, try JAX arrays first, fall back to numpy.
    """
    if prefer_jax:
        try:
            data_jax = data_as_jax(data) if isinstance(data, dict) else jnp.asarray(data)
            params_jax = tree_to_jax(params)
            return model_fn(data_jax, params_jax)
        except Exception as jax_exc:
            try:
                data_np = data_as_numpy(data) if isinstance(data, dict) else np.asarray(data)
                return model_fn(data_np, params)
            except Exception:
                raise jax_exc
    data_np = data_as_numpy(data) if isinstance(data, dict) else np.asarray(data)
    return model_fn(data_np, params)


def stack_params(params_list):
    """Stack a list of per-sample param pytrees into a batched pytree."""
    if not params_list:
        return None
    return jax.tree_util.tree_map(
        lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], axis=0),
        *params_list,
    )


def broadcast_params(params, n_samples: int):
    """Ensure params have a leading sample axis, broadcasting as needed."""
    params = tree_to_jax(params)

    def _broadcast(arr):
        arr = jnp.asarray(arr)
        if arr.ndim == 0:
            return jnp.broadcast_to(arr, (n_samples,))
        if arr.shape[0] == n_samples:
            return arr
        if arr.shape[0] == 1:
            return jnp.broadcast_to(arr, (n_samples,) + arr.shape[1:])
        arr = arr[None, ...]
        return jnp.broadcast_to(arr, (n_samples,) + arr.shape)

    return jax.tree_util.tree_map(_broadcast, params)


def slice_params(params, idx: int):
    """Slice a batched params pytree at the given sample index."""
    return jax.tree_util.tree_map(lambda x: x if jnp.ndim(x) == 0 else x[idx], params)


def params_numel_per_sample(params, n_samples: int | None = None) -> int:
    """Count scalar parameters for a single sample in a params pytree."""
    if n_samples is not None:
        params = slice_params(broadcast_params(params, n_samples), 0)
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(np.asarray(leaf).size for leaf in leaves))


def params_all_finite(params) -> bool:
    """Return True if all leaves are numeric and finite."""
    leaves = jax.tree_util.tree_leaves(params)
    for leaf in leaves:
        arr = np.asarray(leaf)
        if arr.dtype.kind not in "biufc":
            return False
        try:
            if not np.all(np.isfinite(arr)):
                return False
        except TypeError:
            return False
    return True


def params_signature(params, n_samples: int | None = None):
    """Return a structural signature of params (treedef, leaf shapes, dtypes)."""
    if n_samples is not None:
        params = slice_params(broadcast_params(params, n_samples), 0)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    shapes = [np.asarray(leaf).shape for leaf in leaves]
    dtypes = [np.asarray(leaf).dtype for leaf in leaves]
    return treedef, shapes, dtypes


def _format_tree_path(path) -> str:
    parts = []
    for entry in path:
        if isinstance(entry, jax.tree_util.DictKey):
            parts.append(str(entry.key))
        elif isinstance(entry, jax.tree_util.SequenceKey):
            parts.append(str(entry.idx))
        elif isinstance(entry, jax.tree_util.GetAttrKey):
            parts.append(entry.name)
        else:
            parts.append(str(entry))
    return ".".join(parts) if parts else "<root>"


def params_tree_summary(params, n_samples: int | None = None, max_lines: int = 24) -> str:
    """Summarize param pytree structure for logging/debugging."""
    if n_samples is not None:
        params = slice_params(broadcast_params(params, n_samples), 0)
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(params)
    lines = []
    for path, leaf in leaves_with_path:
        arr = np.asarray(leaf)
        lines.append(f"{_format_tree_path(path)}: shape={arr.shape}, dtype={arr.dtype}")
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more)"]
    return "\n".join(lines)


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


def extract_code_blocks(text: Union[str, None]) -> list[str]:
    """
    Extract all fenced code blocks (```python ... ``` or ``` ... ```).
    Returns a list of code strings without the fences.
    """
    if text is None:
        return []
    blocks = re.findall(r"```(?:python)?\\s*\\n(.*?)```", text, flags=re.DOTALL)
    if blocks:
        return [b.rstrip() for b in blocks]
    # Fallback: any fenced blocks without language tag on same line.
    blocks = re.findall(r"```\\s*\\n(.*?)```", text, flags=re.DOTALL)
    return [b.rstrip() for b in blocks]


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
    data,
    params,
    sample_indices=None,
    max_eval_trials=32,
    rtol=1e-4,
    atol=1e-4,
):
    """
    Check NumPy vs JAX model agreement on a subset of data samples.

    Args:
        np_func: Original NumPy model with signature ``model(data_i, params)``.
        jax_func: Translated JAX model with signature ``model(data_i, params)``.
        data (dict[str, np.ndarray]): Data dict with sample axis at dim 0.
        params: Pytree with leading sample axis for each leaf.
        sample_indices: Optional sample indices to validate. If None, checks up to 3 samples.
        max_eval_trials: Max number of trial points per sample used for comparison.
        rtol: Relative tolerance for numeric comparison.
        atol: Absolute tolerance for numeric comparison.

    Raises:
        ValueError: If predictions mismatch between NumPy and JAX.
    """
    n_samples = data_n_samples(data)
    params_arr = broadcast_params(params, n_samples)

    if sample_indices is None:
        n_check = min(3, n_samples)
        sample_indices = np.linspace(0, n_samples - 1, num=n_check, dtype=int)
    else:
        sample_indices = np.asarray(sample_indices).reshape(-1).astype(np.int64, copy=False)

    for sample_idx in sample_indices:
        if sample_idx < 0 or sample_idx >= n_samples:
            raise ValueError(f"sample index {sample_idx} out of range for n_samples={n_samples}.")
        data_i = get_data_sample(data, sample_idx)
        n_trials = data_n_trials(data)
        if max_eval_trials is not None and n_trials > max_eval_trials:
            keep_idx = np.linspace(0, n_trials - 1, num=max_eval_trials, dtype=int)
            data_i = slice_data_trials(data_i, keep_idx)

        sample_params = slice_params(params_arr, sample_idx)
        np_pred = np.asarray(np_func(data_i, sample_params))
        jax_pred = np.asarray(jax_func(data_as_jax(data_i), sample_params))

        if np_pred.shape != jax_pred.shape:
            raise ValueError(
                f"Shape mismatch at sample {sample_idx}: numpy={np_pred.shape}, jax={jax_pred.shape}"
            )

        if not np.allclose(np_pred, jax_pred, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(np_pred - jax_pred))
            raise ValueError(
                f"Numeric mismatch at sample {sample_idx}: max_diff={max_diff:.6g}"
            )


def build_evaluation_points(data, eval_keys=None, x_min=None, x_max=None, n_bins=100):
    """Build evaluation grid as a data dict.

    Creates a linspace grid for each specified key, broadcast to n_samples and
    any intermediate dimensions present in the original key.

    Args:
        data (dict[str, np.ndarray]): Data dict with sample axis at dim 0,
            used to infer n_samples and default min/max bounds.
        eval_keys (list[str] | None): Keys to create eval grids for. If None,
            uses all keys in the data dict.
        x_min: Scalar (applied to all keys), list (one per eval key), or None
            (derived from data per key).
        x_max: Same format as x_min.
        n_bins (int): Number of evaluation points per key.

    Returns:
        dict[str, np.ndarray]: Eval data dict where each key has shape
            ``(n_samples, *mid_dims, n_bins)``, where ``mid_dims`` are the
            intermediate dimensions of the original key between the sample and
            trial axes (empty for 2D keys, giving shape ``(n_samples, n_bins)``).
    """
    n_samples = data_n_samples(data)
    if eval_keys is None:
        eval_keys = list(data.keys())
    n_keys = len(eval_keys)

    missing = [k for k in eval_keys if k not in data]
    if missing:
        raise ValueError(
            f"eval_keys contains keys not present in data: {missing}. "
            f"Available keys: {list(data.keys())}."
        )

    # Resolve per-key bounds
    if x_min is None:
        x_min_vec = [float(np.min(data[k])) for k in eval_keys]
    elif np.isscalar(x_min):
        x_min_vec = [float(x_min)] * n_keys
    else:
        x_min_vec = [float(v) for v in x_min]
    if x_max is None:
        x_max_vec = [float(np.max(data[k])) for k in eval_keys]
    elif np.isscalar(x_max):
        x_max_vec = [float(x_max)] * n_keys
    else:
        x_max_vec = [float(v) for v in x_max]

    if len(x_min_vec) != n_keys or len(x_max_vec) != n_keys:
        raise ValueError(
            f"x_min/x_max length ({len(x_min_vec)}/{len(x_max_vec)}) "
            f"must match eval_keys length ({n_keys})."
        )

    result = {}
    for i, key in enumerate(eval_keys):
        grid = np.linspace(x_min_vec[i], x_max_vec[i], n_bins)
        # mid_dims are intermediate dims between sample and trial axes.
        # For 2D arrays (n_samples, n_trials) this is an empty tuple.
        mid_dims = data[key].shape[1:-1]
        out_shape = (n_samples,) + mid_dims + (n_bins,)
        # grid_reshape is (1, 1, ..., 1, n_bins) — one leading 1 per mid dim —
        # so that np.broadcast_to spreads the linspace across all intermediate dims.
        grid_reshape = (1,) + (1,) * len(mid_dims) + (n_bins,)
        result[key] = np.broadcast_to(grid.reshape(grid_reshape), out_shape)
    return result


def compute_evaluation_matrix(program, params, eval_points):
    """Compute model evaluations on an evaluation grid.

    Args:
        program: Model function with signature ``model(data_i, params)``.
        params: Batched parameter pytree with leading sample axis.
        eval_points (dict[str, np.ndarray]): Evaluation data dict with sample
            axis at dim 0, as returned by ``build_evaluation_points``.

    Returns:
        jnp.ndarray: Model output evaluated on the grid.
    """
    if eval_points is None:
        raise ValueError("eval_points must be provided.")

    params_arr = tree_to_jax(params)
    eval_data = data_as_jax(eval_points)
    n_samples = data_n_samples(eval_data)
    params_arr = broadcast_params(params_arr, n_samples)

    program_vmap = vmap_over_samples(program)
    return program_vmap(eval_data, params_arr)
