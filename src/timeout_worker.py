import numpy as np


def run_estimator_from_source(source_code: str, function_name: str, xi, yi, conn):
    """
    Execute estimator source in an isolated process and return result via pipe.
    """
    namespace = {"np": np}
    try:
        exec(source_code, namespace)
        func = namespace.get(function_name)
        if not callable(func):
            raise RuntimeError(f"Function '{function_name}' not found after exec.")
        conn.send(("ok", func(xi, yi), None))
    except Exception as e:
        conn.send(("error", repr(e), str(e)))
    finally:
        conn.close()
