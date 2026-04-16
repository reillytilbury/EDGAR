import logging
import warnings
from collections.abc import Callable
from typing import Any


def load_function_from_source(
    source: str | None,
    entrypoint_name: str,
    namespace: dict[str, Any] | None = None,
) -> Callable | None:
    """
    Execute a generated Python module and return a required public entrypoint.

    Generated modules may include imports, constants, helper functions, and
    classes. The only contract this loader enforces is that the module defines a
    callable named by ``entrypoint_name``.
    """
    if source is None or not str(source).strip():
        return None

    execution_namespace = {} if namespace is None else dict(namespace)
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            exec(str(source), execution_namespace)
        for warn in captured:
            logging.warning(
                "Warning executing generated source for %s: %s (%s:%s)",
                entrypoint_name,
                warn.message,
                warn.filename,
                warn.lineno,
            )
    except Exception as exc:
        logging.warning(
            "Error executing generated source for %s: %s\nCode:\n%s",
            entrypoint_name,
            exc,
            source,
        )
        return None

    func = execution_namespace.get(entrypoint_name)
    if not callable(func):
        logging.warning("Generated source did not define callable %s.", entrypoint_name)
        return None

    try:
        setattr(func, "__source_code__", str(source))
        setattr(func, "__function_name__", entrypoint_name)
    except Exception:
        pass
    return func

