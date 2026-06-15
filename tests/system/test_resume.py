"""
System test for the resume-from-checkpoint feature.

Strategy:
1. Build a fake-LLM spec (2 gens × 2 islands × 3 batch = 12 evolved + 2 seed).
2. Inject a crash mid-gen 1 by raising on the 2nd `score` call (gen 0 scoring
   succeeds, gen 1 scoring blows up). The runner's `finally` block persists
   population.jsonl + island_census.jsonl + status.json.
3. Snapshot disk state: census has 1 entry (gen 0 complete), population has
   2 seed + 6 evolved = 8 programs.
4. Build a fresh spec (new fake LLMs — counters reset) and call
   `run(spec, resume_from=output_dir)`.
5. Assert the resumed run completes, census grows to 2, population reaches
   the full 14, the seed/gen-0 programs aren't re-scored (their loss values
   are preserved across the resume), and gen 0 metric rows are preserved
   in metrics.jsonl.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from edgar.run import run
from tests.system.fake_runner import build_fake_spec


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_census(path: Path) -> list[list[list[int]]]:
    with open(path) as f:
        return json.load(f)


def test_resume_after_midrun_crash(tmp_path):
    # ── 1. First spec: will crash mid-gen 1 ──
    spec = build_fake_spec(tmp_path / "crash_run")
    output_dir = Path(spec.output_dir)

    from edgar.scoring.scoring import score as original_score

    call_count = {"n": 0}

    def crashing_score(*args, **kwargs):
        # Seeds (1) + gen 0 (1) succeed; gen 1 raises.
        kwargs.pop("n_items", None)  # pop n_items added by timed() decorator
        call_count["n"] += 1
        if call_count["n"] >= 3:
            raise RuntimeError("injected mid-gen-1 crash")
        return original_score(*args, **kwargs)

    with (
        patch("edgar.run.t_score", side_effect=crashing_score),
        patch("edgar.run.t_score_seeds", side_effect=crashing_score),
    ):
        with pytest.raises(RuntimeError, match="injected mid-gen-1 crash"):
            asyncio.run(run(spec))

    # ── 2. Verify disk state ──
    # Crash happened during gen 1's score(), AFTER spawn had added 6 shells.
    # So census has 1 entry (only gen 0 was appended) but population has
    # 2 seed + 6 gen-0 (scored) + 6 gen-1 (stale, unscored) = 14 programs.
    census = _read_census(output_dir / "island_census.jsonl")
    assert len(census) == 1, f"expected 1 completed gen in census, got {len(census)}"

    pop_before = _read_jsonl(output_dir / "population.jsonl")
    n_seed = 2
    n_per_gen = spec.evolution["n_islands"] * spec.evolution["batch_size"]
    n_gens = spec.evolution["n_generations"]
    assert len(pop_before) == n_seed + 2 * n_per_gen, (
        f"expected {n_seed + 2 * n_per_gen} programs after crash "
        f"(2 seed + 6 scored gen-0 + 6 stale gen-1 shells), got {len(pop_before)}"
    )

    # The trailing programs should be unscored stale shells.
    n_trailing_unscored = 0
    for p in reversed(pop_before):
        if p["program_losses"]["discover"]["final"] is None:
            n_trailing_unscored += 1
        else:
            break
    assert n_trailing_unscored == n_per_gen, (
        f"expected {n_per_gen} trailing unscored shells, got {n_trailing_unscored}"
    )

    # ── 3. Build a fresh spec and resume ──
    # Fresh build re-uses the same project config + creates fresh fake LLMs.
    # `resume_from` restamps the spec so writes land back in output_dir.
    resumed_spec = build_fake_spec(tmp_path / "throwaway")
    asyncio.run(run(resumed_spec, resume_from=output_dir))

    # ── 4. Final state assertions ──
    census_after = _read_census(output_dir / "island_census.jsonl")
    assert len(census_after) == n_gens, (
        f"expected census to reach n_generations={n_gens}, got {len(census_after)}"
    )

    # After resume: 6 stale shells dropped, 6 new gen-1 programs added.
    # Final total: 2 seed + 6 gen-0 + 6 fresh gen-1 = 14.
    pop_after = _read_jsonl(output_dir / "population.jsonl")
    expected_total = n_seed + n_gens * n_per_gen
    assert len(pop_after) == expected_total, (
        f"expected {expected_total} programs after resume, got {len(pop_after)}"
    )

    # Pre-crash scored programs (seed + gen 0) survived with their losses
    # intact (not re-scored). Stale shells at idx 8..13 were dropped, then
    # replaced by fresh gen-1 programs at the same indices.
    n_preserved = n_seed + n_per_gen  # 2 seed + 6 gen-0
    for i in range(n_preserved):
        assert (
            pop_before[i]["program_losses"]["discover"]["final"]
            == pop_after[i]["program_losses"]["discover"]["final"]
        ), f"pre-crash discover.final loss was overwritten at idx {i}"

    # status flipped to complete.
    status = json.loads((output_dir / "status.json").read_text())
    assert status["state"] == "complete", (
        f"expected state=complete, got {status['state']!r}"
    )

    # metrics.jsonl contains gen 0 history (preserved from before the crash).
    metrics = _read_jsonl(output_dir / "metrics.jsonl")
    gens_logged = {row["gen"] for row in metrics}
    assert 0 in gens_logged, f"gen 0 missing from metrics.jsonl: {gens_logged}"
    assert 1 in gens_logged, (
        f"gen 1 missing from metrics.jsonl (post-resume): {gens_logged}"
    )

    # Resumed run.log has the RESUMED banner.
    log_text = (output_dir / "run.log").read_text()
    assert "RESUMED" in log_text, "run.log missing RESUMED banner after resume"


def test_resume_refuses_completed_run(tmp_path):
    """A clean, fully-completed run should refuse to resume."""
    spec = build_fake_spec(tmp_path / "clean_run")
    output_dir = Path(spec.output_dir)
    asyncio.run(run(spec))

    resumed = build_fake_spec(tmp_path / "throwaway")
    with pytest.raises(ValueError, match="already has .* completed gens"):
        asyncio.run(run(resumed, resume_from=output_dir))


def test_resume_refuses_missing_dir(tmp_path):
    """A nonexistent resume_from path should fail loudly at prep time."""
    spec = build_fake_spec(tmp_path / "throwaway")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        asyncio.run(run(spec, resume_from=tmp_path / "does_not_exist"))


def test_resume_refuses_non_run_dir(tmp_path):
    """A real dir without task_spec.yaml should fail with a clear message."""
    fake_dir = tmp_path / "empty_dir"
    fake_dir.mkdir()
    spec = build_fake_spec(tmp_path / "throwaway")
    with pytest.raises(FileNotFoundError, match="task_spec.yaml"):
        asyncio.run(run(spec, resume_from=fake_dir))
