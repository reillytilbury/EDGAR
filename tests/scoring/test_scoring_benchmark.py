"""Opt-in end-to-end benchmark: EDGAR_RUN_SCORING_BENCHMARK=1 uv run pytest -s."""

import os
import statistics
import time

import jax.numpy as jnp
import numpy as np
import pytest

from edgar.evolution.population import Population
from edgar.process import available_cpu_count
from edgar.scoring.scoring import score
from tests.scoring.test_scoring import (
    BASE_CONFIG,
    FAST_MODEL_CODE,
    _make_data,
    _make_program,
    loss_fn,
)


@pytest.mark.skipif(
    os.environ.get("EDGAR_RUN_SCORING_BENCHMARK") != "1",
    reason="set EDGAR_RUN_SCORING_BENCHMARK=1 to run the performance benchmark",
)
def test_parallel_scoring_benchmark():
    cpus = available_cpu_count()
    if cpus < 2:
        pytest.skip("requires at least two CPUs available to the process")

    data = _make_data(n_samples=32)
    n_programs = min(8, 2 * cpus)
    timings = {1: [], None: []}
    final_losses = {}
    # Repeated fresh populations prevent the already-scored fast path. A longer
    # optimization workload amortizes subprocess startup and measures real score().
    for workers in (1, None, None, 1):
        population = Population()
        for _ in range(n_programs):
            population.add(_make_program(FAST_MODEL_CODE))
        config = {
            **BASE_CONFIG,
            "max_workers": workers,
            "optimizer_config": {"max_iter": 300, "learning_rate": 0.01},
        }
        started = time.perf_counter()
        score(population, (data, data), None, config, loss_fn, split="discover")
        timings[workers].append(time.perf_counter() - started)
        values = [p.program_losses.discover.final for p in population]
        assert all(jnp.isfinite(value) for value in values)
        if workers not in final_losses:
            final_losses[workers] = values
        else:
            assert np.allclose(values, final_losses[workers], atol=1e-5)

    serial = statistics.median(timings[1])
    parallel = statistics.median(timings[None])
    speedup = serial / parallel
    print(
        f"CPU scoring benchmark: CPUs={cpus}, programs={n_programs}, "
        f"serial_median={serial:.2f}s, parallel_median={parallel:.2f}s, "
        f"speedup={speedup:.2f}x"
    )
    # This is an opt-in benchmark, not ordinary CI: allow normal noise and
    # shared-runner CPU quotas while still catching substantial regressions.
    assert speedup > 0.8, "parallel scoring regressed by more than 20%"
