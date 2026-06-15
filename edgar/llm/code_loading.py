import warnings
from collections.abc import Callable


def load_function_from_source(
    source: str | None,
    entrypoint_name: str,
) -> Callable | None:
    """Executes Python source code in a fresh namespace and returns a specified callable object.

    This utility is crucial for dynamically loading LLM-generated code (e.g., models, parameter estimators)
    or user-defined project functions (e.g., `load_data_fn`, `loss_fn`, `plot_fn`) into the EDGAR system.
    The source code is executed within a completely new and isolated namespace to prevent unintended
    side effects or variable conflicts. After execution, the function attempts to retrieve the
    object named by `entrypoint_name`.

    The function returns `None` under the following conditions:
    - The provided `source` string is `None` or contains only whitespace.
    - An exception occurs during the execution of the `source` code (e.g., syntax error, runtime error).
      In such cases, a warning is issued.
    - The object identified by `entrypoint_name` is not found in the executed namespace or is not callable.

    Args:
        source: The Python source code string to execute. If `None` or empty, no execution occurs.
        entrypoint_name: The name of the callable object expected to be defined in the `source`
            and to be returned.

    Returns:
        The callable object corresponding to `entrypoint_name` if successfully loaded and callable,
        otherwise `None`.
    """
    # Reject missing or whitespace-only source strings.
    if not source or not source.strip():
        return None

    # Execute the source in a fresh module-like namespace.
    ns = {}
    try:
        exec(source, ns)
    except Exception as e:
        warnings.warn(f"[code_loading] exec failed for '{entrypoint_name}': {e}")
        return None

    # Return the requested object only if it is callable.
    func = ns.get(entrypoint_name)
    return func if callable(func) else None
