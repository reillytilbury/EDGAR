"""Tests for edgar/io/metrics.py and the dashboard's metrics surfacing.

Two layers:
- Unit-test the RunMetrics accumulator: stage timing, LLM-call recording,
  scoring-result recording, percentile math, jsonl round-trip.
- Contract-test the dashboard API: a synthesised metrics.jsonl + status.json
  results in /api/state surfacing the totals + current_stage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from edgar.dashboard import data as dd
from edgar.dashboard.server import build_app
from edgar.io.metrics import (
    METRICS_FILENAME,
    RunMetrics,
    get_active_metrics,
    read_metrics,
    stage_timer,
)
from edgar.io.status import read_status, write_status


# ── unit: RunMetrics ──


def test_stage_timer_writes_jsonl_row_with_stage_times(tmp_path: Path):
    log_path = tmp_path / "run.log"

    class _DummyLog:
        def __init__(self, path: Path):
            self.file = open(path, "w")

    log = _DummyLog(log_path)
    with RunMetrics(output_dir=tmp_path, run_log=log, n_gens=2, started_at=0.0) as m:
        m.start_generation(0)
        with stage_timer(m, "generate_models", n_items=4):
            for _ in range(4):
                m.record_llm_call(
                    role="model",
                    model="claude-haiku-4-5",
                    latency_ms=100.0,
                    in_tokens=200,
                    out_tokens=50,
                    finish_reason="STOP",
                    retries=0,
                    ok=True,
                )
        with stage_timer(m, "score", n_items=3):
            m.record_score_result(0, ms=8000.0, outcome="ok")
            m.record_score_result(1, ms=15000.0, outcome="timeout")
            m.record_score_result(2, ms=10.0, outcome="banned")
        row = m.finish_generation()
    log.file.close()

    assert row["gen"] == 0
    assert "generate_models" in row["stage_times"]
    assert "score" in row["stage_times"]
    assert row["llm_calls"]["model"]["n"] == 4
    assert row["llm_calls"]["model"]["ok"] == 4
    assert row["llm_calls"]["model"]["in_tokens_total"] == 800
    assert row["llm_calls"]["model"]["out_tokens_total"] == 200
    assert row["scoring"]["n"] == 3
    assert row["scoring"]["ok"] == 1
    assert row["scoring"]["timeout"] == 1
    assert row["scoring"]["banned"] == 1
    assert row["scoring"]["latency_ms"]["max"] == 15000.0

    # Disk: metrics.jsonl is written atomically and round-trips.
    rows = read_metrics(tmp_path)
    assert len(rows) == 1
    assert rows[0]["gen"] == 0
    assert rows[0]["llm_calls"]["model"]["n"] == 4


def test_get_active_metrics_returns_handle_inside_context(tmp_path: Path):
    assert get_active_metrics() is None  # outside any run
    with RunMetrics(output_dir=tmp_path, run_log=None, n_gens=1, started_at=0.0) as m:
        assert get_active_metrics() is m
    assert get_active_metrics() is None


def test_stage_timer_progress_updates_current_stage(tmp_path: Path):
    """As LLM calls complete inside a stage, current_stage in status.json
    should tick (k/n)."""
    with RunMetrics(output_dir=tmp_path, run_log=None, n_gens=1, started_at=0.0) as m:
        m.start_generation(0)
        with stage_timer(m, "generate_models", n_items=3):
            m.record_llm_call(
                role="model",
                model="m",
                latency_ms=10,
                in_tokens=1,
                out_tokens=1,
                finish_reason=None,
                retries=0,
                ok=True,
            )
            s = read_status(tmp_path)
            assert s["current_stage"] == "generate_models (1/3)"
            m.record_llm_call(
                role="model",
                model="m",
                latency_ms=10,
                in_tokens=1,
                out_tokens=1,
                finish_reason=None,
                retries=0,
                ok=True,
            )
            s = read_status(tmp_path)
            assert s["current_stage"] == "generate_models (2/3)"
            m.record_llm_call(
                role="model",
                model="m",
                latency_ms=10,
                in_tokens=1,
                out_tokens=1,
                finish_reason=None,
                retries=0,
                ok=True,
            )
            s = read_status(tmp_path)
            assert s["current_stage"] == "generate_models (3/3)"


def test_finish_generation_records_retry_and_failure_counts(tmp_path: Path):
    with RunMetrics(output_dir=tmp_path, run_log=None, n_gens=1, started_at=0.0) as m:
        m.start_generation(0)
        m.record_llm_call(
            role="param_est",
            model="m",
            latency_ms=10,
            in_tokens=1,
            out_tokens=1,
            finish_reason=None,
            retries=2,
            ok=True,
        )
        m.record_llm_call(
            role="param_est",
            model="m",
            latency_ms=10,
            in_tokens=1,
            out_tokens=1,
            finish_reason=None,
            retries=0,
            ok=False,
        )
        row = m.finish_generation()
    st = row["llm_calls"]["param_est"]
    assert st["n"] == 2
    assert st["ok"] == 1
    assert st["retried"] == 1


# ── contract: dashboard API ──


def _write_taskspec(run_dir: Path) -> None:
    (run_dir / "task_spec.yaml").write_text(
        "task_name: t\n"
        "evolution: {n_generations: 2, n_islands: 2, batch_size: 2}\n"
        "llms: {model_llm: gemini-2.5-flash, param_est_llm: gemini-2.5-flash, jax_model_translator_llm: gemini-2.5-flash}\n"
        "scoring: {param_penalty_weight: 0.01}\n"
        "io: {data_path: x, save_path: y}\n"
        "project_params: {}\n"
        "prompt_schemas: {model: {base: X}, param_est: {base: Y}, jax_model: {base: Z}}\n"
    )


def test_dashboard_state_surfaces_metrics_and_totals(tmp_path: Path):
    run_dir = tmp_path / "05-26" / "00-00-00"
    run_dir.mkdir(parents=True)
    _write_taskspec(run_dir)

    rows = [
        {
            "gen": 0,
            "stage_times": {"generate_models": 50.0, "score": 100.0},
            "llm_calls": {
                "model": {
                    "n": 4,
                    "ok": 4,
                    "retried": 0,
                    "in_tokens_total": 1000,
                    "out_tokens_total": 500,
                    "models": ["claude-haiku-4-5"],
                    "latency_ms": {"p50": 1000, "p90": 1500, "max": 2000, "mean": 1100},
                },
            },
            "scoring": {
                "n": 4,
                "ok": 3,
                "timeout": 1,
                "inf": 0,
                "banned": 0,
                "latency_ms": {"p50": 8000, "p90": 12000, "max": 15000, "mean": 9000},
            },
        },
        {
            "gen": 1,
            "stage_times": {"generate_models": 60.0, "score": 110.0},
            "llm_calls": {
                "model": {
                    "n": 4,
                    "ok": 3,
                    "retried": 1,
                    "in_tokens_total": 1200,
                    "out_tokens_total": 600,
                    "models": ["claude-haiku-4-5"],
                    "latency_ms": {"p50": 1200, "p90": 1800, "max": 2200, "mean": 1300},
                },
            },
            "scoring": {
                "n": 4,
                "ok": 4,
                "timeout": 0,
                "inf": 0,
                "banned": 1,
                "latency_ms": {"p50": 7500, "p90": 11000, "max": 14000, "mean": 8500},
            },
        },
    ]
    (run_dir / METRICS_FILENAME).write_text("".join(json.dumps(r) + "\n" for r in rows))

    write_status(
        run_dir,
        state="running",
        n_gens=2,
        current_gen=1,
        started_at=__import__("time").time(),
        current_stage="score (3/4)",
        last_metrics=rows[-1],
    )

    app = build_app([tmp_path])
    dd._POP_CACHE.clear()
    dd._CENSUS_CACHE.clear()
    dd._METRICS_CACHE.clear()
    with TestClient(app) as client:
        rid = dd._run_id(run_dir)
        r = client.get(f"/api/runs/{rid}/state")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["current_stage"] == "score (3/4)"
        assert body["last_metrics"]["gen"] == 1
        assert len(body["metrics"]) == 2

        totals = body["totals"]
        assert totals["in_tokens"] == 2200
        assert totals["out_tokens"] == 1100
        assert totals["n_llm_calls"] == 8
        assert totals["n_llm_retried"] == 1
        assert totals["n_scored"] == 8
        assert totals["n_ok"] == 7
        assert totals["n_timeout"] == 1
        assert totals["n_inf"] == 0
        assert totals["n_banned"] == 1
        # LLM and scoring seconds are derived from mean × n
        assert totals["llm_seconds"] == pytest.approx((1100 * 4 + 1300 * 4) / 1000)
        assert totals["score_seconds"] == pytest.approx((9000 * 4 + 8500 * 4) / 1000)


def test_dashboard_summary_also_exposes_totals(tmp_path: Path):
    run_dir = tmp_path / "05-26" / "11-11-11"
    run_dir.mkdir(parents=True)
    _write_taskspec(run_dir)
    rows = [
        {
            "gen": 0,
            "stage_times": {},
            "llm_calls": {
                "model": {
                    "n": 1,
                    "ok": 1,
                    "retried": 0,
                    "in_tokens_total": 7,
                    "out_tokens_total": 3,
                    "models": ["m"],
                    "latency_ms": {"p50": 1, "p90": 1, "max": 1, "mean": 1},
                }
            },
            "scoring": {
                "n": 0,
                "ok": 0,
                "timeout": 0,
                "inf": 0,
                "banned": 0,
                "latency_ms": {"p50": None, "p90": None, "max": None, "mean": None},
            },
        }
    ]
    (run_dir / METRICS_FILENAME).write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_status(
        run_dir,
        state="complete",
        n_gens=1,
        current_gen=0,
        started_at=__import__("time").time(),
    )

    app = build_app([tmp_path])
    dd._POP_CACHE.clear()
    dd._CENSUS_CACHE.clear()
    dd._METRICS_CACHE.clear()
    with TestClient(app) as client:
        rid = dd._run_id(run_dir)
        r = client.get(f"/api/runs/{rid}/summary")
        assert r.status_code == 200
        body = r.json()
        assert body["totals"]["in_tokens"] == 7
        assert body["totals"]["out_tokens"] == 3
