"""Contract tests for the dashboard HTTP API.

Anchors against the largest finished run at program_databases/05-24/17-17-45/.
If that fixture is missing the run-anchored tests skip; the synthesised-legacy
and atomic-write tests still run.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from edgar.dashboard import data as dd
from edgar.dashboard.server import build_app
from edgar.io.status import write_status


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_RUN = REPO_ROOT / "program_databases" / "05-24" / "17-17-45"
PDB_ROOT = REPO_ROOT / "program_databases"


# ── shared fixtures ──


@pytest.fixture
def fixture_run() -> Path:
    if not (FIXTURE_RUN / "task_spec.yaml").exists():
        pytest.skip(f"fixture run missing: {FIXTURE_RUN}")
    return FIXTURE_RUN


@pytest.fixture
def client_for_pdb() -> Iterator[TestClient]:
    if not PDB_ROOT.exists():
        pytest.skip(f"no program_databases at {PDB_ROOT}")
    app = build_app([PDB_ROOT])
    # Clear data-layer caches so each test sees fresh state
    dd._POP_CACHE.clear()
    dd._CENSUS_CACHE.clear()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def legacy_run(tmp_path: Path) -> Path:
    """Synthesise a legacy-style run dir: task_spec.yaml + population.jsonl only.

    Crucial: no status.json, no island_census.jsonl, no run.log.
    """
    src = FIXTURE_RUN
    if not (src / "task_spec.yaml").exists():
        pytest.skip(f"fixture run missing: {src}")
    dst = tmp_path / "05-24" / "00-00-00"
    dst.mkdir(parents=True)
    shutil.copy(src / "task_spec.yaml", dst / "task_spec.yaml")
    shutil.copy(src / "population.jsonl", dst / "population.jsonl")
    return dst


@pytest.fixture
def client_for_legacy(legacy_run: Path) -> Iterator[TestClient]:
    app = build_app([legacy_run.parent.parent])
    dd._POP_CACHE.clear()
    dd._CENSUS_CACHE.clear()
    with TestClient(app) as c:
        yield c


# ── /api/runs ──


def test_runs_list_includes_fixture(
    fixture_run: Path, client_for_pdb: TestClient
) -> None:
    r = client_for_pdb.get("/api/runs")
    assert r.status_code == 200
    runs = r.json()
    target = dd._run_id(fixture_run)
    found = [run for run in runs if run["run_id"] == target]
    assert found, f"expected run {target} in /api/runs"
    item = found[0]
    for k in ("run_id", "task_name", "started_at", "status", "n_programs", "n_islands"):
        assert k in item, f"missing key {k}"


# ── /api/runs/{id}/summary ──


def test_summary_shape(fixture_run: Path, client_for_pdb: TestClient) -> None:
    rid = dd._run_id(fixture_run)
    r = client_for_pdb.get(f"/api/runs/{rid}/summary")
    assert r.status_code == 200
    s = r.json()
    assert s["task_name"] == "orientation_tuning"
    assert s["n_islands"] == 2
    assert s["n_generations"] == 1
    assert s["n_programs"] > 0
    assert isinstance(s["prompt"], dict)
    assert s["prompt"].get("base"), "expected non-empty prompt.base"
    assert s["llms"]["model"]
    assert s["status"] == "complete"


def test_summary_legacy_run_treated_as_complete(
    legacy_run: Path, client_for_legacy: TestClient
) -> None:
    rid = dd._run_id(legacy_run)
    r = client_for_legacy.get(f"/api/runs/{rid}/summary")
    assert r.status_code == 200
    s = r.json()
    assert s["status"] == "complete"
    assert s["n_alive"] == 0  # no census means we can't infer alive set
    assert s["n_programs"] > 0


# ── /api/runs/{id}/state ──


def test_state_shape(fixture_run: Path, client_for_pdb: TestClient) -> None:
    rid = dd._run_id(fixture_run)
    r = client_for_pdb.get(f"/api/runs/{rid}/state")
    assert r.status_code == 200
    s = r.json()
    assert s["status"] == "complete"
    assert s["n_islands"] == 2
    assert len(s["islands"]) == 2
    for row in s["islands"]:
        assert "idx" in row and "size_alive" in row and "programs" in row
        for p in row["programs"]:
            for k in ("idx", "name", "gen", "island", "mode", "loss_discover"):
                assert k in p
    assert isinstance(s["best_per_gen"], list)


# ── /api/runs/{id}/programs ──


def test_programs_list_complete(fixture_run: Path, client_for_pdb: TestClient) -> None:
    rid = dd._run_id(fixture_run)
    r = client_for_pdb.get(f"/api/runs/{rid}/programs")
    assert r.status_code == 200
    progs = r.json()
    # match line count of population.jsonl
    with open(fixture_run / "population.jsonl") as f:
        n_lines = sum(1 for _ in f)
    assert len(progs) == n_lines
    ranked = [p for p in progs if p.get("rank") is not None]
    assert ranked, "expected at least one ranked program"
    # rank-1 should sort first under the data-layer sort key
    assert progs[0]["rank"] == 1


# ── /api/runs/{id}/programs/{idx} ──


def test_program_detail_winner(fixture_run: Path, client_for_pdb: TestClient) -> None:
    rid = dd._run_id(fixture_run)
    plist = client_for_pdb.get(f"/api/runs/{rid}/programs").json()
    winner = plist[0]
    r = client_for_pdb.get(f"/api/runs/{rid}/programs/{winner['idx']}")
    assert r.status_code == 200
    d = r.json()
    assert d["code"]["model"], "winner must have non-empty model source"
    assert d["code"]["model_jax"], "winner must have non-empty JAX source"
    assert d["losses"]["validate"]["final"] is None or isinstance(
        d["losses"]["validate"]["final"], float
    )
    assert isinstance(d["children"], list)
    assert isinstance(d["parents_detail"], list)


def test_program_detail_404(fixture_run: Path, client_for_pdb: TestClient) -> None:
    rid = dd._run_id(fixture_run)
    r = client_for_pdb.get(f"/api/runs/{rid}/programs/99999")
    assert r.status_code == 404


def test_stale_running_run_reported_as_failed(tmp_path: Path) -> None:
    """A run with status='running' and an updated_at older than the stale
    threshold should be surfaced as 'failed' (with is_stale=True). This
    handles SIGKILL'd / OOM'd runs where the finally block never wrote a
    terminal state.
    """
    import time as time_mod

    # Synthesise a stale run dir
    run_dir = tmp_path / "05-24" / "11-11-11"
    run_dir.mkdir(parents=True)
    # task_spec.yaml: minimal, just enough for _load_task_spec to return data
    (run_dir / "task_spec.yaml").write_text(
        "task_name: stale_test\n"
        "evolution: {n_generations: 3, n_islands: 2, batch_size: 2}\n"
        "llms: {model_llm: claude-haiku-4-5, param_est_llm: claude-haiku-4-5, jax_model_translator_llm: claude-haiku-4-5}\n"
        "scoring: {param_penalty_weight: 0.01}\n"
        "io: {data_path: x, save_path: y}\n"
        "project_params: {}\n"
        "prompt_schemas: {model: {base: 'X'}, param_est: {base: 'Y'}, jax_model: {base: 'Z'}}\n"
    )
    # Write a 'running' status with updated_at well past the stale threshold
    write_status(
        run_dir,
        state="running",
        n_gens=3,
        current_gen=1,
        started_at=time_mod.time() - 9999,
    )
    # Hack the file's updated_at back in time
    s_path = run_dir / "status.json"
    import json as _json

    s = _json.loads(s_path.read_text())
    s["updated_at"] = time_mod.time() - 9999
    s_path.write_text(_json.dumps(s))

    app = build_app([tmp_path])
    dd._POP_CACHE.clear()
    dd._CENSUS_CACHE.clear()
    with TestClient(app) as client:
        rid = dd._run_id(run_dir)
        r = client.get(f"/api/runs/{rid}/state")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "failed", body
        assert body["raw_status"] == "running"
        assert body["is_stale"] is True
        assert "stall" in (body["error"] or "").lower()


# ── atomic-write invariant ──


def test_atomic_write_invariant(tmp_path: Path) -> None:
    """Writer rewrites a large JSON file in a tight loop; the reader hammers
    it concurrently. With os.replace + tmp pattern, no JSONDecodeError is ever
    observed even though the file is large.
    """
    from edgar.io.status import atomic_write_text

    path = tmp_path / "big.jsonl"
    payload_lines = [json.dumps({"i": i, "blob": "x" * 500}) for i in range(2000)]
    payload = "\n".join(payload_lines) + "\n"
    atomic_write_text(path, payload)

    stop = threading.Event()
    errors: list[Exception] = []
    read_count = 0
    write_count = 0

    def writer():
        nonlocal write_count
        while not stop.is_set():
            atomic_write_text(path, payload)
            write_count += 1

    def reader():
        nonlocal read_count
        while not stop.is_set():
            try:
                with open(path) as f:
                    text = f.read()
                for line in text.splitlines():
                    if line:
                        json.loads(line)
                read_count += 1
            except Exception as e:
                errors.append(e)
                return

    threads = [threading.Thread(target=writer) for _ in range(2)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(1.5)
    stop.set()
    for t in threads:
        t.join(timeout=5)

    assert not errors, f"torn read observed: {errors[:3]}"
    assert read_count > 50, f"reader didn't actually run (count={read_count})"
    assert write_count > 5, f"writer didn't actually run (count={write_count})"


# ── LaTeX cache behaviour ──


def test_latex_cache_first_call_writes_second_call_reuses(
    fixture_run: Path, client_for_pdb: TestClient, monkeypatch
) -> None:
    """First POST hits the LLM (stubbed), writes cache; second returns cached
    without calling the LLM again. We monkey-patch call_llm in the module
    namespace where it's looked up.
    """
    from edgar.dashboard import latex_cache as lc

    rid = dd._run_id(fixture_run)
    # Pick the rank-1 winner (any program with non-empty code works)
    progs = client_for_pdb.get(f"/api/runs/{rid}/programs").json()
    winner_idx = progs[0]["idx"]

    # Clear any pre-existing cache file
    cache_path = lc._cache_path(fixture_run, winner_idx)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cache_path.unlink()

    call_counter = {"n": 0}

    async def fake_call_llm(*args, **kwargs):
        call_counter["n"] += 1
        return "$$y = f(x)$$"

    # Patch the import site inside latex_cache.get_or_generate_latex
    # We need to intercept `from ..llm.llm_calling import call_llm` — do that
    # by monkey-patching edgar.llm.llm_calling.call_llm.
    import edgar.llm.llm_calling as llm_mod

    monkeypatch.setattr(llm_mod, "call_llm", fake_call_llm)

    r1 = client_for_pdb.post(f"/api/runs/{rid}/programs/{winner_idx}/latex")
    assert r1.status_code == 200, r1.text
    j1 = r1.json()
    assert j1["cached"] is False
    assert "y = f(x)" in j1["latex"]
    assert call_counter["n"] == 1
    assert cache_path.exists(), "cache file should be written"

    r2 = client_for_pdb.post(f"/api/runs/{rid}/programs/{winner_idx}/latex")
    assert r2.status_code == 200
    j2 = r2.json()
    assert j2["cached"] is True
    assert call_counter["n"] == 1, "should not have re-called the LLM"

    # Force regeneration
    r3 = client_for_pdb.post(f"/api/runs/{rid}/programs/{winner_idx}/latex?force=true")
    assert r3.status_code == 200
    assert r3.json()["cached"] is False
    assert call_counter["n"] == 2

    # Cleanup so we don't leave junk in the canonical fixture run dir
    cache_dir = cache_path.parent
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
