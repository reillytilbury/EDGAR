"""Utilities for running isolated EDGAR subprocesses."""

from __future__ import annotations

import multiprocessing as mp
import os
import warnings
from queue import Empty
from typing import Any, Callable
import traceback


class SubprocessTimeoutError(TimeoutError):
    """Raised when an isolated worker does not return before its deadline."""


class SubprocessWorkerError(RuntimeError):
    """Raised when an isolated worker exits with an uncaught exception."""

    def __init__(self, message: str, traceback_text: str):
        super().__init__(message)
        self.traceback_text = traceback_text


def _worker_entry(queue, target, args):
    try:
        target(queue, *args)
    except Exception as exc:
        queue.put(
            (
                "__edgar_worker_error__",
                f"{type(exc).__name__}: {exc}",
                traceback.format_exc(),
            )
        )


def get_mp_context() -> mp.context.BaseContext:
    """Return the configured multiprocessing context.

    ``spawn`` is the safe default for JAX. Non-spawn methods remain available
    for notebooks and debugging, but cannot reliably select a JAX platform
    after the parent process has initialized JAX.
    """
    start_method = os.environ.get("EDGAR_MP_START_METHOD", "spawn")
    if start_method != "spawn":
        warnings.warn(
            f"EDGAR_MP_START_METHOD={start_method!r} may be unsafe with JAX; "
            "jax_backend selection is only guaranteed with 'spawn'.",
            stacklevel=2,
        )
    return mp.get_context(start_method)


def run_in_subprocess(
    target: Callable[..., Any], args: tuple[Any, ...], timeout: float
) -> Any:
    """Run a queue-based worker in an isolated process and return its result.

    The worker must accept a multiprocessing queue as its first argument and
    place its result on that queue. The queue is created by this function and
    prepended to ``args``.
    """
    ctx = get_mp_context()
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker_entry, args=(queue, target, args))
    proc.start()
    try:
        result = queue.get(timeout=timeout)
    except Empty as exc:
        if proc.is_alive():
            proc.kill()
        proc.join()
        raise SubprocessTimeoutError(
            f"Worker exceeded timeout of {timeout} seconds"
        ) from exc

    proc.join()
    if (
        isinstance(result, tuple)
        and len(result) == 3
        and result[0] == "__edgar_worker_error__"
    ):
        raise SubprocessWorkerError(result[1], result[2])
    return result
