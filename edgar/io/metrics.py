"""
edgar/io/metrics.py

In-process metrics accumulator for an EDGAR run. One ``RunMetrics`` instance
per run. Three consumers read from it:

- ``run.log`` — streaming start/end lines for each stage, plus tick lines
  during scoring (the slowest stage, and the only one that's serial).
- ``status.json`` — ``current_stage`` string + the last completed gen's metrics
  row, so the dashboard can show "now: score 23/48" without re-reading
  ``metrics.jsonl``.
- ``metrics.jsonl`` — one JSON row per generation, with stage timings, per-role
  LLM call stats (n, tokens, latency percentiles, retry count), and scoring
  outcome counts (ok/timeout/inf, latency percentiles).

``call_llm`` and ``score`` find the active accumulator via a ``contextvars``
ContextVar set by the ``RunMetrics`` context manager, so we don't have to
thread a metrics handle through every call site.

Persistence shape per generation (one line in metrics.jsonl):

    {
      "gen": 2,
      "stage_times": {"generate_models": 117.3, "score": 622.7, ...},
      "llm_calls": {
        "model":     {"n": 48, "ok": 47, "retried": 5, "in_tokens_total": 142000,
                      "out_tokens_total": 38500, "models": ["claude-sonnet-4-6"],
                      "latency_ms": {"p50": 4100, "p90": 9800, "max": 22000, "mean": 5300}},
        "param_est": {...},
        "jax":       {...}
      },
      "scoring": {"n": 48, "ok": 41, "timeout": 5, "inf": 2,
                  "latency_ms": {"p50": 8200, "p90": 22000, "max": 60000, "mean": 11000}}
    }
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, TYPE_CHECKING

from .status import atomic_write_text, write_status

if TYPE_CHECKING:
    from .logging import RunLog


_active_metrics: contextvars.ContextVar["RunMetrics | None"] = contextvars.ContextVar(
    "_active_metrics",
    default=None,
)
"""The currently active RunMetrics instance.

`call_llm` and `score` read from this ContextVar to find the current metrics
accumulator. It is set by the `RunMetrics` context manager. It is None outside
a run (e.g., during unit tests).
"""

METRICS_FILENAME = "metrics.jsonl"
"""The filename for storing generation metrics."""


@dataclass
class _LLMCall:
    """Represents a single LLM call event.

    Attributes:
        role: The role of the LLM call (e.g., "model", "param_est", "jax").
        model: The name of the LLM model used.
        latency_ms: The latency of the LLM call in milliseconds.
        in_tokens: The number of input tokens used.
        out_tokens: The number of output tokens generated.
        finish_reason: The reason the LLM call finished (e.g., "stop", "length").
        retries: The number of retries attempted for this call.
        ok: True if the LLM call was successful, False otherwise.
    """

    role: str
    model: str
    latency_ms: float
    in_tokens: int
    out_tokens: int
    finish_reason: str | None
    retries: int
    ok: bool


@dataclass
class _ScoreResult:
    """Represents the outcome of a single program scoring attempt.

    Attributes:
        idx: The global index of the program that was scored.
        ms: The time taken for scoring in milliseconds.
        outcome: The outcome of the scoring process, one of "ok", "timeout",
            "inf" (infinite loss), or "banned".
    """

    idx: int
    ms: float
    outcome: Literal["ok", "timeout", "inf", "banned"]


@dataclass
class RunMetrics:
    """Accumulates timing and counts for an EDGAR run.

    Per-generation buckets (`_stage_times`, `_llm_calls`, `_score_results`)
    are reset at the start of each generation. Cumulative rows for past
    generations (`_gen_rows`) are persisted to `metrics.jsonl` upon completion
    of each generation.

    Usage:
        To install `RunMetrics` as the active accumulator:

        ```python
        with RunMetrics(output_dir, run_log, n_gens, started_at) as metrics:
            with stage_timer(metrics, "translate_seeds"):
                await translate_programs(...)
            for gen in range(n_gens):
                metrics.start_generation(gen)
                with stage_timer(metrics, "generate_models", n_items=48):
                    await generate_models(...)
                # ... other stages ...
                metrics.finish_generation()
        ```

    Attributes:
        output_dir: The base directory for storing run artifacts.
        run_log: An instance of `RunLog` for writing log messages, or None.
        n_gens: The total number of generations configured for the run.
        started_at: The timestamp (time.monotonic()) when the run started.
        current_gen: The current generation number. -1 indicates the seed phase.
        current_stage: The name of the currently active pipeline stage,
            potentially with a progress suffix (e.g., "score (23/48)").
        _stage_times: A dictionary mapping stage names to their elapsed time
            (in seconds) for the current generation.
        _llm_calls: A list of `_LLMCall` objects for the current generation.
        _score_results: A list of `_ScoreResult` objects for the current generation.
        _stage_progress: A dictionary tracking the (completed, total) count
            for stages with asynchronous tasks, keyed by stage name.
        _stage_progress_n: A dictionary storing the total number of items
            for stages, keyed by stage name.
        _gen_rows: A list of dictionaries, where each dictionary represents
            the aggregated metrics for a completed generation.
        _token: A `contextvars.Token` used to manage the `_active_metrics` ContextVar.
    """

    output_dir: Path
    run_log: "RunLog | None"
    n_gens: int
    started_at: float

    current_gen: int = -1  # -1 = seed phase
    current_stage: str | None = None
    _stage_times: dict[str, float] = field(default_factory=dict)
    _llm_calls: list[_LLMCall] = field(default_factory=list)
    _score_results: list[_ScoreResult] = field(default_factory=list)

    # Live progress counters for stages whose tasks complete asynchronously.
    # Keyed by stage name, value is (k_completed, n_total). Updated by
    # record_llm_call / record_score_result.
    _stage_progress: dict[str, tuple[int, int]] = field(default_factory=dict)
    _stage_progress_n: dict[str, int] = field(default_factory=dict)

    # Past finished gens.
    _gen_rows: list[dict] = field(default_factory=list)

    _token: Any = None  # contextvar reset token

    def __enter__(self) -> "RunMetrics":
        """Installs this `RunMetrics` instance as the active metrics accumulator."""
        self._token = _active_metrics.set(self)
        return self

    def __exit__(self, *exc: Any) -> bool:
        """Resets the active metrics accumulator."""
        _active_metrics.reset(self._token)
        return False

    def record_llm_call(
        self,
        role: str,
        model: str,
        latency_ms: float,
        in_tokens: int,
        out_tokens: int,
        finish_reason: str | None,
        retries: int,
        ok: bool,
    ) -> None:
        """Appends one LLM call record and updates the current stage's progress.

        The progress counter is updated if the stage was initiated with an
        `n_items` hint using `stage_timer`.

        Args:
            role: The role of the LLM call (e.g., "model", "param_est", "jax").
            model: The name of the LLM model used.
            latency_ms: The latency of the LLM call in milliseconds.
            in_tokens: The number of input tokens used.
            out_tokens: The number of output tokens generated.
            finish_reason: The reason the LLM call finished (e.g., "stop", "length").
            retries: The number of retries attempted for this call.
            ok: True if the LLM call was successful, False otherwise.
        """
        self._llm_calls.append(
            _LLMCall(
                role=role,
                model=model,
                latency_ms=latency_ms,
                in_tokens=in_tokens,
                out_tokens=out_tokens,
                finish_reason=finish_reason,
                retries=retries,
                ok=ok,
            )
        )
        self._tick_stage_progress(self.current_stage_root())

    def record_score_result(self, idx: int, ms: float, outcome: str) -> None:
        """Appends one scoring outcome record and updates the scoring stage's progress.

        Args:
            idx: The global index of the program that was scored.
            ms: The time taken for scoring in milliseconds.
            outcome: The outcome of the scoring process, one of "ok", "timeout",
                "inf", or "banned".
        """
        self._score_results.append(_ScoreResult(idx=idx, ms=ms, outcome=outcome))
        self._tick_stage_progress("score")

    def set_current_stage(self, stage: str | None, n_items: int | None = None) -> None:
        """Sets the current pipeline stage and updates the `status.json` file.

        If `n_items` is provided, a progress counter for the stage is initialized
        and tracked. This operation is best-effort and will not raise exceptions
        if writing to disk fails.

        Args:
            stage: The name of the current stage (e.g., "generate_models").
            n_items: The total number of items to process in this stage. If provided,
                a `(k/n)` progress counter will be shown in `status.json`.
        """
        self.current_stage = stage
        if stage is not None and n_items is not None:
            self._stage_progress[stage] = (0, n_items)
            self._stage_progress_n[stage] = n_items
        self._write_status()

    def current_stage_root(self) -> str | None:
        """Returns the base name of the current stage without progress suffixes.

        For example, if `current_stage` is "score (23/48)", this method returns "score".

        Returns:
            The root name of the current stage, or None if no stage is active.
        """
        s = self.current_stage
        if s is None:
            return None
        return s.split(" (", 1)[0]

    def _tick_stage_progress(self, stage: str | None) -> None:
        """Increments the progress counter for a given stage and updates `status.json`.

        This method is called by `record_llm_call` and `record_score_result`
        to update the user-visible progress. The update to `status.json` is
        an atomic file write, which is relatively fast.

        Args:
            stage: The name of the stage whose progress should be incremented.
        """
        if stage is None:
            return
        if stage not in self._stage_progress_n:
            return
        n_total = self._stage_progress_n[stage]
        k_done = self._stage_progress.get(stage, (0, n_total))[0] + 1
        self._stage_progress[stage] = (k_done, n_total)
        # Update the user-visible label and flush to status.json. This is cheap:
        # atomic file write is ~100 µs, even at 200 calls/gen.
        self.current_stage = f"{stage} ({k_done}/{n_total})"
        self._write_status()

    def _write_status(self) -> None:
        """Atomically writes the current run status to `status.json`.

        This includes the current stage, generation, and the last completed
        generation's metrics row. Disk hiccups are caught and never fail the run.
        """
        try:
            write_status(
                self.output_dir,
                state="running",
                n_gens=self.n_gens,
                current_gen=(self.current_gen if self.current_gen >= 0 else None),
                started_at=self.started_at,
                current_stage=self.current_stage,
                last_metrics=(self._gen_rows[-1] if self._gen_rows else None),
            )
        except Exception:
            # Disk hiccups must never fail the run.
            pass

    def start_generation(self, gen: int) -> None:
        """Resets per-generation metric buckets and sets the current generation number.

        Args:
            gen: The current generation number to start.
        """
        self.current_gen = gen
        self._stage_times = {}
        self._llm_calls = []
        self._score_results = []
        self._stage_progress = {}
        self._stage_progress_n = {}

    def finish_generation(self) -> dict:
        """Snapshots the current generation's metrics, appends a row to `metrics.jsonl`,
        and returns the aggregated metrics row.

        The `metrics.jsonl` file is atomically rewritten with all accumulated
        generation rows to ensure data integrity.

        Returns:
            A dictionary containing the aggregated metrics for the just-finished
            generation.
        """
        row = self._build_gen_row()
        self._gen_rows.append(row)
        _write_metrics_jsonl(self.output_dir / METRICS_FILENAME, self._gen_rows)
        return row

    def _build_gen_row(self) -> dict:
        """Aggregates the current generation's raw metric events into a summary dictionary.

        This method processes `_llm_calls` and `_score_results` to produce
        summary statistics such as counts, token usage, and latency percentiles
        for LLM calls, and outcome counts and latencies for scoring.

        Returns:
            A dictionary representing the summarized metrics for the current generation,
            suitable for serialization to `metrics.jsonl`.
        """
        by_role: dict[str, list[_LLMCall]] = {}
        for call in self._llm_calls:
            by_role.setdefault(call.role, []).append(call)

        llm_summary: dict[str, dict] = {}
        for role, calls in by_role.items():
            llm_summary[role] = {
                "n": len(calls),
                "ok": sum(1 for c in calls if c.ok),
                "retried": sum(1 for c in calls if c.retries > 0),
                "in_tokens_total": sum(c.in_tokens for c in calls),
                "out_tokens_total": sum(c.out_tokens for c in calls),
                "models": sorted({c.model for c in calls if c.model}),
                "latency_ms": _percentiles([c.latency_ms for c in calls]),
            }

        score_summary = {
            "n": len(self._score_results),
            "ok": sum(1 for r in self._score_results if r.outcome == "ok"),
            "timeout": sum(1 for r in self._score_results if r.outcome == "timeout"),
            "inf": sum(1 for r in self._score_results if r.outcome == "inf"),
            "banned": sum(1 for r in self._score_results if r.outcome == "banned"),
            "latency_ms": _percentiles([r.ms for r in self._score_results]),
        }

        return {
            "gen": self.current_gen,
            "stage_times": dict(self._stage_times),
            "llm_calls": llm_summary,
            "scoring": score_summary,
        }


def get_active_metrics() -> RunMetrics | None:
    """Returns the currently active `RunMetrics` instance.

    This function allows various parts of the EDGAR system (e.g., `call_llm`, `score`)
    to retrieve the `RunMetrics` accumulator without explicitly threading a metrics
    handle through every function call.

    Returns:
        The `RunMetrics` instance if a run is active, otherwise None.
    """
    return _active_metrics.get()


@contextmanager
def stage_timer(
    metrics: RunMetrics | None,
    name: str,
    n_items: int | None = None,
    quiet: bool = False,
) -> Iterator[None]:
    """Times a single stage of the EDGAR pipeline.

    This context manager performs the following:
    *   Streams start and end lines to `run.log`.
    *   Updates `status.json` with the current stage name on entry.
    *   Records the duration of the stage in the current generation's
        `stage_times` bucket upon exit.

    It is a safe no-op if `metrics` is None, which is useful for tests that
    run the pipeline without a full metrics setup.

    Args:
        metrics: The active metrics accumulator, or None.
        name: The name of the stage being timed (e.g., "generate_models", "score").
        n_items: Optional. If provided, `set_current_stage` will track a
            `(k/n)` progress counter that is updated by `record_llm_call`
            and `record_score_result` as individual items within the stage complete.
        quiet: If True, suppresses writing start/end lines to `run.log`.
            Useful for very fast stages that would otherwise generate excessive log spam.
    """
    if metrics is None:
        yield
        return

    t0 = time.monotonic()
    gen_label = _gen_label(metrics.current_gen)
    n_part = f" ({n_items} items)" if n_items is not None else ""
    if not quiet:
        _write_line(metrics.run_log, f"  [{gen_label}] {name}: starting{n_part}")
    metrics.set_current_stage(name, n_items=n_items)

    try:
        yield
    finally:
        dt = time.monotonic() - t0
        metrics._stage_times[name] = round(dt, 3)
        if not quiet:
            _write_line(metrics.run_log, f"  [{gen_label}] {name}: done in {dt:.1f}s")


def timed(name: str, quiet: bool = False) -> Callable:
    """Wraps a function such that each call is timed as a pipeline stage.

    This decorator resolves the active `RunMetrics` instance via
    `get_active_metrics()`, eliminating the need to thread a metrics handle
    through function calls. The wrapper extracts an optional `n_items` keyword
    argument from the decorated function's call and forwards it to `stage_timer`,
    enabling per-stage progress tracking.

    Args:
        name: The name of the pipeline stage to associate with the timed function.
        quiet: If True, suppresses writing start/end lines for this stage to `run.log`.

    Returns:
        A decorator that can be applied to either synchronous or asynchronous functions.

    Usage (alias form, as in run.py)::

        t_score = timed("score")(score)
        await t_score(population, ..., n_items=48)
    """

    def decorator(fn: Callable) -> Callable:
        if asyncio.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                n_items = kwargs.pop("n_items", None)
                with stage_timer(
                    get_active_metrics(), name, n_items=n_items, quiet=quiet
                ):
                    return await fn(*args, **kwargs)

            return async_wrapper

        @wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            n_items = kwargs.pop("n_items", None)
            with stage_timer(get_active_metrics(), name, n_items=n_items, quiet=quiet):
                return fn(*args, **kwargs)

        return sync_wrapper

    return decorator


def stream_line(metrics: RunMetrics | None, msg: str) -> None:
    """Prints a message to the console and streams it to the `run.log` file.

    This function acts as a safe no-op if no `RunMetrics` instance is active.

    Args:
        metrics: The active `RunMetrics` instance, or None.
        msg: The string message to print and log.
    """
    if metrics is None:
        print(msg, flush=True)
        return
    _write_line(metrics.run_log, msg)


def read_metrics(run_dir: Path) -> list[dict]:
    """Reads the `metrics.jsonl` file from a specified run directory.

    This function parses each line of `metrics.jsonl` as a JSON object,
    returning a list of dictionaries, each representing the aggregated
    metrics for a generation. It handles cases where the file is absent,
    unreadable, or contains partial writes due to concurrency.

    Args:
        run_dir: The path to the EDGAR run directory.

    Returns:
        A list of dictionaries, where each dictionary contains the metrics
        for one generation. Returns an empty list if the file is not found
        or cannot be read.
    """
    path = run_dir / METRICS_FILENAME
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # Partial-write race: return what we have so far.
                    return out
    except OSError:
        return []
    return out


def _gen_label(gen: int) -> str:
    """Generates a human-readable label for a given generation number.

    Args:
        gen: The generation number. -1 represents the seed phase.

    Returns:
        A string label, either "seed" or "gen N" (where N is the generation number).
    """
    return "seed" if gen < 0 else f"gen {gen}"


def _percentiles(xs: list[float]) -> dict:
    """Calculates statistical percentiles (p50, p90, max, mean) for a list of floats.

    Args:
        xs: A list of floating-point numbers.

    Returns:
        A dictionary containing the median (p50), 90th percentile (p90), maximum,
        and mean of the input list. Returns None for these values if the input
        list is empty.
    """
    if not xs:
        return {"p50": None, "p90": None, "max": None, "mean": None}
    xs_sorted = sorted(xs)
    p90_idx = max(0, min(len(xs_sorted) - 1, int(round(0.9 * (len(xs_sorted) - 1)))))
    return {
        "p50": float(statistics.median(xs_sorted)),
        "p90": float(xs_sorted[p90_idx]),
        "max": float(xs_sorted[-1]),
        "mean": float(sum(xs_sorted) / len(xs_sorted)),
    }


def _write_line(run_log: Any, msg: str) -> None:
    """Streams a single line message to the console and, if available, to a `RunLog` file.

    This helper avoids a direct import of `edgar.io.logging` to keep this module
    independent.

    Args:
        run_log: An object with a `file` attribute (like `RunLog.file`) that supports
            writing, or None.
        msg: The message string to write.
    """
    print(msg, flush=True)
    if run_log is None:
        return
    try:
        run_log.file.write(msg + "\n")
        run_log.file.flush()
    except Exception:
        pass


def _write_metrics_jsonl(path: Path, rows: list[dict]) -> None:
    """Atomically rewrites the entire `metrics.jsonl` file from a list of rows.

    This approach ensures data integrity: a reader will never see a partially
    written or corrupted file, as the new content is written to a temporary file
    and then atomically renamed. This is suitable for `metrics.jsonl` because
    it remains relatively small (typically a few dozen rows per run), making the
    cost of rewriting trivial.

    Args:
        path: The path to the `metrics.jsonl` file.
        rows: A list of dictionaries, where each dictionary represents the metrics
            for a generation.
    """
    payload = "".join(json.dumps(r) + "\n" for r in rows)
    atomic_write_text(path, payload)
"""