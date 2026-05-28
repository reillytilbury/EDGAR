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

import contextvars
import json
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, TYPE_CHECKING

from .status import atomic_write_text, write_status

if TYPE_CHECKING:
    from .logging import RunLog


# Active metrics handle. call_llm / score read from this to find the current
# accumulator. None outside a run (e.g. unit tests hitting call_llm directly).
_active_metrics: contextvars.ContextVar["RunMetrics | None"] = contextvars.ContextVar(
    "_active_metrics",
    default=None,
)

METRICS_FILENAME = "metrics.jsonl"


# ── per-event records ──


@dataclass
class _LLMCall:
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
    idx: int
    ms: float
    outcome: Literal["ok", "timeout", "inf"]


# ── accumulator ──


@dataclass
class RunMetrics:
    """Accumulates timing + counts for a run. Per-generation buckets are reset
    every ``start_generation()``; cumulative rows live in ``_gen_rows`` and
    are persisted to ``metrics.jsonl`` on each ``finish_generation()``.

    Usage:

        with RunMetrics(output_dir, run_log, n_gens, started_at) as metrics:
            with stage_timer(metrics, "translate_seeds"):
                await translate_programs(...)
            for gen in range(n_gens):
                metrics.start_generation(gen)
                with stage_timer(metrics, "generate_models", n_items=48):
                    await generate_models(...)
                ...
                metrics.finish_generation()
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

    # ── context manager: install as the active metrics ──

    def __enter__(self) -> "RunMetrics":
        self._token = _active_metrics.set(self)
        return self

    def __exit__(self, *exc: Any) -> bool:
        _active_metrics.reset(self._token)
        return False

    # ── recorders ──

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
        """Append one LLM-call record and bump the current stage's progress
        counter if the stage was opened with an n_items hint.
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
        """Append one scoring outcome and bump the score stage progress."""
        self._score_results.append(_ScoreResult(idx=idx, ms=ms, outcome=outcome))
        self._tick_stage_progress("score")

    # ── stage progress + status.json ──

    def set_current_stage(self, stage: str | None, n_items: int | None = None) -> None:
        """Push current_stage into status.json. Best-effort; never raises."""
        self.current_stage = stage
        if stage is not None and n_items is not None:
            self._stage_progress[stage] = (0, n_items)
            self._stage_progress_n[stage] = n_items
        self._write_status()

    def current_stage_root(self) -> str | None:
        """Strip the '(k/n)' suffix to get the bare stage name."""
        s = self.current_stage
        if s is None:
            return None
        return s.split(" (", 1)[0]

    def _tick_stage_progress(self, stage: str | None) -> None:
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

    # ── per-gen lifecycle ──

    def start_generation(self, gen: int) -> None:
        """Reset per-gen buckets and bump the generation label."""
        self.current_gen = gen
        self._stage_times = {}
        self._llm_calls = []
        self._score_results = []
        self._stage_progress = {}
        self._stage_progress_n = {}

    def finish_generation(self) -> dict:
        """Snapshot the current gen, append a row to ``metrics.jsonl``,
        and return the row.
        """
        row = self._build_gen_row()
        self._gen_rows.append(row)
        _write_metrics_jsonl(self.output_dir / METRICS_FILENAME, self._gen_rows)
        return row

    def _build_gen_row(self) -> dict:
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
            "latency_ms": _percentiles([r.ms for r in self._score_results]),
        }

        return {
            "gen": self.current_gen,
            "stage_times": dict(self._stage_times),
            "llm_calls": llm_summary,
            "scoring": score_summary,
        }


# ── public API ──


def get_active_metrics() -> RunMetrics | None:
    """Return the currently-active ``RunMetrics``, or None outside a run."""
    return _active_metrics.get()


@contextmanager
def stage_timer(
    metrics: RunMetrics | None,
    name: str,
    n_items: int | None = None,
    quiet: bool = False,
) -> Iterator[None]:
    """Time one stage of the pipeline.

    Streams start/end lines to ``run.log``, updates ``status.json``
    ``current_stage`` on entry, and records the duration in the current gen's
    ``stage_times`` bucket on exit. Safe no-op when ``metrics`` is None (used
    by tests that drive the runner without setting up an accumulator).

    Args:
        metrics: the active accumulator, or None.
        name: stage name, e.g. ``"generate_models"`` or ``"score"``.
        n_items: optional. If provided, ``set_current_stage`` will track a
            (k/n) progress counter that's updated by ``record_llm_call`` /
            ``record_score_result`` as each item completes.
        quiet: don't write start/end lines (useful for sub-millisecond stages
            like ``spawn`` that would otherwise spam the log).
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


def stream_line(metrics: RunMetrics | None, msg: str) -> None:
    """Print one line to console + the run.log file. Safe no-op without metrics."""
    if metrics is None:
        print(msg, flush=True)
        return
    _write_line(metrics.run_log, msg)


def read_metrics(run_dir: Path) -> list[dict]:
    """Read metrics.jsonl from disk. Returns [] if file is absent or unreadable."""
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


# ── helpers ──


def _gen_label(gen: int) -> str:
    return "seed" if gen < 0 else f"gen {gen}"


def _percentiles(xs: list[float]) -> dict:
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
    """Stream one line to console + a RunLog file handle. Avoids importing
    ``edgar.io.logging`` to keep this module dependency-free."""
    print(msg, flush=True)
    if run_log is None:
        return
    try:
        run_log.file.write(msg + "\n")
        run_log.file.flush()
    except Exception:
        pass


def _write_metrics_jsonl(path: Path, rows: list[dict]) -> None:
    """Atomically rewrite the full metrics.jsonl from rows.

    Rewriting is simpler than open-append + truncate-on-failure, and JSONL is
    small (≤ a few dozen rows per run), so the cost is trivial.
    """
    payload = "".join(json.dumps(r) + "\n" for r in rows)
    atomic_write_text(path, payload)
