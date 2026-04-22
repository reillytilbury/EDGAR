import inspect
import logging
import multiprocessing as mp
import textwrap
import time

import jax
import numpy as np
import timeout_decorator
from jax.flatten_util import ravel_pytree

from .. import utils
from ..timeout_worker import run_estimator_from_source


class ObjectiveTimeout(Exception):
    """Raised when objective exceeds wall-clock timeout."""

class ProcessTimeoutUnavailable(RuntimeError):
    """Raised when process-based timeout backend cannot be used."""


def _get_callable_source(func) -> tuple[str | None, str | None]:
    """
    Best-effort extraction of callable source for spawn-safe subprocess execution.
    """
    code = getattr(func, "__source_code__", None)
    name = getattr(func, "__function_name__", None) or getattr(func, "__name__", None)
    if isinstance(code, str) and code.strip() and isinstance(name, str) and name:
        return code, name
    try:
        source = textwrap.dedent(inspect.getsource(func)).strip()
        if not source:
            return None, None
        inferred_name = getattr(func, "__name__", None)
        if not isinstance(inferred_name, str) or not inferred_name:
            return None, None
        return source, inferred_name
    except Exception:
        return None, None

def _run_param_estimator_with_timeout(param_estimator, data_i, timeout_s: float):
    """
    Robust timeout wrapper for potentially hanging estimators.

    Uses a dedicated ``spawn`` subprocess so estimator execution can be force-
    terminated without ``fork`` conflicts in JAX multithreaded runtimes.
    """
    source_code, function_name = _get_callable_source(param_estimator)
    if not source_code or not function_name:
        raise ProcessTimeoutUnavailable(
            "Unable to extract source for estimator; process timeout unavailable."
        )

    start_methods = mp.get_all_start_methods()
    if "spawn" not in start_methods:
        raise ProcessTimeoutUnavailable("Process timeout requires 'spawn' start method.")

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=run_estimator_from_source,
        args=(source_code, function_name, utils.data_as_numpy(data_i), child_conn),
        daemon=True,
    )
    proc.start()
    child_conn.close()
    try:
        if not parent_conn.poll(timeout_s):
            raise TimeoutError(f"param_estimator timed out after {timeout_s:.2f}s")
        status, payload, detail = parent_conn.recv()
        if status == "ok":
            return payload
        raise RuntimeError(f"param_estimator failed: {payload} ({detail})")
    finally:
        parent_conn.close()
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=0.2)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=0.2)

def compute_initial_params(
    param_estimator,
    model,
    data,
    timeout_s: float | None = 5.0,
    deadline_s: float | None = None,
):
    """
    Estimate per-sample initial model parameters with safe fallbacks.

    Uses per-sample timeouts when provided and clamps each estimate to the
    remaining wall-clock time before ``deadline_s``.

    Args:
        param_estimator (callable): Parameter initializer with signature
            ``param_estimator(data_i) -> params_i`` for a single sample.
        model (callable): Model function used only for fallback default parameter
            inference when estimation fails.
        data (dict[str, np.ndarray]): Data dict where all arrays have shape
            ``(n_samples, ..., n_trials)``.
        timeout_s (float | None): Per-sample timeout in seconds. If ``None`` or
            non-positive, no explicit per-sample timeout is applied.
        deadline_s (float | None): Absolute wall-clock deadline expressed in
            ``time.time()`` seconds. If exceeded, raises ``ObjectiveTimeout``.

    Returns:
        pytree | None: Batched parameter pytree with leading sample axis for each
            leaf, or ``None`` if no fallback parameters are available.
    """
    defaults = compute_default_params(model)
    fallback = defaults
    if defaults is None:
        logging.info(
            "Default parameters unavailable; will fall back to the first successful estimate."
        )

    timeout_s_val = float(timeout_s) if timeout_s is not None and timeout_s > 0 else None
    n_samples = utils.data_n_samples(data)
    params_list = []

    def _remaining_time(sample_idx: int) -> float | None:
        if deadline_s is None:
            return None
        remaining_s = float(deadline_s) - time.time()
        if remaining_s <= 0:
            raise ObjectiveTimeout(
                f"objective timed out during parameter initialization at sample {sample_idx}"
            )
        return remaining_s

    def _effective_timeout(sample_idx: int) -> float | None:
        remaining_s = _remaining_time(sample_idx)
        if timeout_s_val is None:
            return remaining_s
        if remaining_s is None:
            return timeout_s_val
        return min(timeout_s_val, remaining_s)

    def _estimate_once(data_i, timeout_limit: float | None):
        if timeout_limit is None:
            return param_estimator(data_i)
        try:
            wrapped_estimator = timeout_decorator.timeout(
                timeout_limit, use_signals=True
            )(param_estimator)
        except Exception as exc:
            logging.info(
                "Signal timeout wrapper unavailable (%s); using process timeout backend.",
                exc,
            )
            return _run_param_estimator_with_timeout(param_estimator, data_i, timeout_limit)
        return wrapped_estimator(data_i)

    def _normalize_params(params_i):
        return jax.tree_util.tree_map(
            lambda value: (
                np.asarray(value)[0]
                if np.asarray(value).ndim >= 1 and np.asarray(value).shape[0] == 1
                else value
            ),
            params_i,
        )

    for sample_idx in range(n_samples):
        data_i = utils.data_as_numpy(utils.get_data_sample(data, sample_idx))
        timeout_limit = _effective_timeout(sample_idx)

        try:
            params_i = _estimate_once(data_i, timeout_limit)
            if params_i is None:
                raise ValueError("param_estimator returned None")
            params_i = _normalize_params(params_i)
        except ObjectiveTimeout:
            raise
        except (TimeoutError, timeout_decorator.TimeoutError):
            timeout_desc = (
                f"{timeout_limit:.2f}s" if timeout_limit is not None else "unknown timeout"
            )
            logging.info(
                "param_estimator timed out for sample %s after %s; using fallback.",
                sample_idx,
                timeout_desc,
            )
            params_i = fallback
        except ProcessTimeoutUnavailable as exc:
            logging.info(
                "param_estimator timeout backend unavailable for sample %s: %s",
                sample_idx,
                exc,
            )
            params_i = fallback
        except Exception as exc:
            logging.info(
                "param_estimator failed for sample %s: %s; using fallback.",
                sample_idx,
                exc,
            )
            params_i = fallback

        if params_i is None:
            logging.info(
                "Unable to initialize parameters for sample %s: no defaults or prior successful estimate available.",
                sample_idx,
            )
            return None

        fallback = params_i
        params_list.append(params_i)

    return utils.stack_params(params_list)

def compute_default_params(model):
    """
    Build default parameters from model metadata or signature.

    Preferred sources:
    1) ``model.default_params`` (callable or value)
    2) ``model.DEFAULT_PARAMS`` (value)
    3) Signature default for ``params`` when using ``model(X, params=...)``
    4) Legacy positional defaults for ``model(X, *params)``

    Returns:
        pytree | None: Default parameter pytree for a single sample, or ``None`` if
            defaults cannot be determined.
    """
    try:
        default_attr = getattr(model, "default_params", None)
        if default_attr is not None:
            return default_attr() if callable(default_attr) else default_attr
        default_attr = getattr(model, "DEFAULT_PARAMS", None)
        if default_attr is not None:
            return default_attr

        sig = inspect.signature(model)
        param_names = list(sig.parameters.keys())
        if len(param_names) >= 2 and param_names[1] == "params":
            params_param = sig.parameters["params"]
            if params_param.default is not inspect._empty:
                return params_param.default
            return None

        # Legacy path: positional parameters after X/theta.
        if len(param_names) <= 1:
            return None
        defaults = [
            sig.parameters[n].default if sig.parameters[n].default is not inspect._empty else 0.0
            for n in param_names[1:]
        ]
        import jax.numpy as jnp
        default_arr = jnp.array(defaults, dtype=np.float32)
        return default_arr.reshape(1, -1)
    except Exception as e:
        logging.info(f"Error while generating default parameters: {e}")
        return None
