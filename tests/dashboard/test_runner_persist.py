"""Verify edgar/run.py writes per-generation snapshots + a correctly-transitioning
status.json. Drives the fake-LLM pipeline so it doesn't hit any real API.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Iterator

import pytest

from edgar.io.status import read_status
from edgar.run import run
from tests.system.fake_runner import build_fake_spec


@pytest.fixture
def fake_output_dir(tmp_path: Path) -> Iterator[Path]:
    out = tmp_path / "fake_run"
    yield out
    # tmp_path teardown cleans up


def test_fake_run_writes_status_and_per_gen_snapshots(fake_output_dir: Path):
    """Sanity check end-to-end: status.json + per-gen snapshots land, and
    every generation's snapshot has at least as many lines as the previous.

    Because the fake runner finishes in a few seconds we sample by:
      - starting a watcher thread that polls the run_dir
      - running the pipeline synchronously
      - asserting the watcher saw incremental growth
    """
    spec = build_fake_spec(fake_output_dir)
    run_dir = Path(spec.output_dir)

    samples: list[dict] = []
    stop = threading.Event()

    def watcher():
        while not stop.is_set():
            status_doc = read_status(run_dir)
            pop = run_dir / "population.jsonl"
            census = run_dir / "island_census.jsonl"
            n_lines = sum(1 for _ in open(pop)) if pop.exists() else 0
            samples.append(
                {
                    "t": time.time(),
                    "state": (status_doc or {}).get("state"),
                    "current_gen": (status_doc or {}).get("current_gen"),
                    "pop_lines": n_lines,
                    "census_exists": census.exists(),
                }
            )
            time.sleep(0.2)

    t = threading.Thread(target=watcher, daemon=True)
    t.start()
    try:
        asyncio.run(run(spec))
    finally:
        stop.set()
        t.join(timeout=2)
        # Race avoidance: the watcher could be sleeping at the moment the
        # `finally` block of run() writes state=complete. Take one explicit
        # post-run sample so the trace always ends with the terminal state.
        final = read_status(run_dir) or {}
        samples.append(
            {
                "t": time.time(),
                "state": final.get("state"),
                "current_gen": final.get("current_gen"),
                "pop_lines": sum(1 for _ in open(run_dir / "population.jsonl")),
                "census_exists": (run_dir / "island_census.jsonl").exists(),
            }
        )

    # ── invariants ──

    states = [s["state"] for s in samples if s["state"]]
    assert "starting" in states or "running" in states, f"states observed: {states}"
    assert states[-1] == "complete", f"final state was {states[-1]}: {states[-5:]}"

    pop_lines_seen = [s["pop_lines"] for s in samples]
    assert max(pop_lines_seen) > 0, "population.jsonl never had any lines"
    # Monotonic growth (a write may overwrite identical content but never shrink).
    for prev, cur in zip(pop_lines_seen, pop_lines_seen[1:]):
        assert cur >= prev, f"population.jsonl shrank: {prev} -> {cur}"

    final_status = read_status(run_dir)
    assert final_status is not None
    assert final_status["state"] == "complete"
    assert final_status["current_gen"] == spec.evolution["n_generations"] - 1

    # Final files all present
    assert (run_dir / "population.jsonl").exists()
    assert (run_dir / "island_census.jsonl").exists()
    assert (run_dir / "task_spec.yaml").exists()


def test_runner_failure_marks_status_failed(fake_output_dir: Path, monkeypatch):
    """Inject a deliberate exception inside the loop; assert status flips to
    'failed' and we still persisted whatever was complete."""
    spec = build_fake_spec(fake_output_dir)
    run_dir = Path(spec.output_dir)

    # Patch the name in edgar.run (where it's imported into module scope) rather
    # than in edgar.scoring.scoring (where run.py already imported a reference
    # at module load).
    import edgar.run as run_mod

    original_score = run_mod.score
    call_counter = {"n": 0}

    def exploding_score(*args, **kwargs):
        # First score call is the seed-phase score; let that pass.
        # Second call is inside the generation loop — blow up.
        call_counter["n"] += 1
        if call_counter["n"] >= 2:
            raise RuntimeError("synthetic scoring failure")
        return original_score(*args, **kwargs)

    monkeypatch.setattr(run_mod, "score", exploding_score)

    with pytest.raises(RuntimeError, match="synthetic scoring failure"):
        asyncio.run(run(spec))

    s = read_status(run_dir)
    assert s is not None, "status.json should exist after a failure"
    assert s["state"] == "failed", f"expected failed, got {s}"
    assert s["error"] and "synthetic scoring failure" in s["error"], s["error"]
    # Seed phase persisted population.jsonl, so it should exist
    assert (run_dir / "population.jsonl").exists()
