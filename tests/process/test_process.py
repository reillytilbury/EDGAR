import time

import pytest

from edgar.process import (
    SubprocessTimeoutError,
    SubprocessWorkerError,
    get_mp_context,
    run_in_subprocess,
)


def _return_worker(queue, value):
    queue.put(value)


def _slow_worker(queue, delay):
    time.sleep(delay)
    queue.put("finished")


def _raising_worker(queue):
    raise ValueError("worker failed")


def test_run_in_subprocess_returns_worker_result(monkeypatch):
    monkeypatch.delenv("EDGAR_MP_START_METHOD", raising=False)

    assert run_in_subprocess(_return_worker, (42,), timeout=5) == 42


def test_run_in_subprocess_raises_on_timeout():
    with pytest.raises(SubprocessTimeoutError, match="0.05 seconds"):
        run_in_subprocess(_slow_worker, (1,), timeout=0.05)


def test_run_in_subprocess_propagates_worker_errors():
    with pytest.raises(SubprocessWorkerError, match="ValueError: worker failed") as exc:
        run_in_subprocess(_raising_worker, (), timeout=5)

    assert "ValueError: worker failed" in exc.value.traceback_text


def test_spawn_is_the_default_context(monkeypatch):
    monkeypatch.delenv("EDGAR_MP_START_METHOD", raising=False)

    assert get_mp_context().get_start_method() == "spawn"


def test_non_spawn_context_warns(monkeypatch):
    monkeypatch.setenv("EDGAR_MP_START_METHOD", "fork")

    with pytest.warns(UserWarning, match="may be unsafe with JAX"):
        context = get_mp_context()

    assert context.get_start_method() == "fork"
