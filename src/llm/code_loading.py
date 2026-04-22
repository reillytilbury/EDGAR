from collections.abc import Callable

# TODO: This is a very simple implementation. The best improvement is to run the code in a new process, which would allow for better timeouts.

def load_function_from_source(
    source: str | None,
    entrypoint_name: str,
) -> Callable | None:
    """
    Execute Python source code and return a named callable.

    The source is executed in a fresh namespace. After execution, the object
    named by ``entrypoint_name`` is returned if it exists and is callable.

    Returns ``None`` if:
    - the source is empty,
    - execution fails,
    - the named object is missing or not callable.
    """
    # Reject missing or whitespace-only source strings.
    if not source or not source.strip():
        return None

    # Execute the source in a fresh module-like namespace.
    ns = {}
    try:
        exec(source, ns)
    except Exception:
        return None

    # Return the requested object only if it is callable.
    func = ns.get(entrypoint_name)
    return func if callable(func) else None