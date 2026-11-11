"""Mocked stage orchestration checks scoring boundaries, persistence and timing.

Scoring calculations are tested separately; this module verifies the code and
data supplied to scoring/generation, retaining real seed-translation timing.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from edgar.run import run
from tests.system.fake_runner import build_fake_spec


MODEL = "import jax.numpy as jnp\ndef model(data, params):\n    return params['a'] * data['x']"
TRANSLATED = MODEL + "\n# translated"
ESTIMATOR = "def parameter_estimator(data):\n    return {'a': 1.0}"


@pytest.mark.parametrize(
    "stages",
    [
        ["generate_models"],
        ["generate_models", "generate_param_ests"],
        ["generate_models", "translate_programs"],
        ["generate_models", "generate_param_ests", "translate_programs"],
    ],
)
def test_stage_order_scoring_boundaries_persistence_and_timing(tmp_path, stages):
    spec = build_fake_spec(tmp_path / "out")
    spec.run["generation_stages"] = stages
    spec.evolution.update(
        n_generations=1, n_islands=1, batch_size=1, topology=[0], n_migrants=0
    )
    for seed in spec.seed_programs:
        seed.code.model = MODEL
    events = []
    translation = "translate_programs" in stages
    expected_discover, _, _ = spec.load_data_fn(
        data_path=spec.io["data_path"], **spec.project_params
    )
    expected_feedback = expected_discover[1]
    seed_estimators = {
        p.birth.batch_index: p.code.param_est for p in spec.seed_programs
    }

    async def generate(population, *args, **kwargs):
        events.append("generate_models")
        assert kwargs["data"].keys() == expected_feedback.keys()
        for key, expected in expected_feedback.items():
            np.testing.assert_array_equal(kwargs["data"][key], expected)
        for p in population:
            if p.code.model is None:
                p.code.model = MODEL
                p.default_params = {"a": 1.0}

    async def estimators(population, *args, **kwargs):
        events.append("generate_param_ests")
        for p in population:
            if p.birth.generation == 0:
                p.code.param_est = [ESTIMATOR]

    async def translate_seed_model(program, *args, **kwargs):
        # Keep the real seed translation function and its timed alias in place.
        assert program.birth.generation == -1
        if "translate_seeds" not in events:
            events.append("translate_seeds")
        program.code.model_jax = TRANSLATED

    async def translate(population, *args, **kwargs):
        events.append("translate_programs")
        for p in population:
            if p.code.model is not None and p.code.model_jax is None:
                p.code.model_jax = TRANSLATED

    def score(population, *args, **kwargs):
        for p in population:
            if p.status != "alive":
                continue
            assert p.code.model_jax == (TRANSLATED if translation else MODEL)
            if not translation:
                # Exercise the real model loader and JAX tracer, not just field presence.
                assert (
                    float(
                        jax.jit(p.compile_model())(
                            {"x": jnp.array(2.0)}, {"a": jnp.array(3.0)}
                        )
                    )
                    == 6.0
                )
            if p.birth.generation == -1:
                assert p.code.param_est == seed_estimators[p.birth.batch_index]
            elif "generate_param_ests" in stages:
                assert p.code.param_est == [ESTIMATOR]
            else:
                assert p.code.param_est is None
            p.program_losses.discover.final = 1.0

    def score_seeds(population, *args, **kwargs):
        events.append("score_seeds")
        score(population, *args, **kwargs)

    def score_generated(population, *args, **kwargs):
        events.append("score_generated")
        score(population, *args, **kwargs)

    with (
        patch("edgar.llm.generate.generate_models", generate),
        patch("edgar.llm.generate.generate_param_ests", estimators),
        patch("edgar.llm.generate._translate_one_model", translate_seed_model),
        patch("edgar.llm.generate.translate_programs", translate),
        patch("edgar.run.t_score_seeds", score_seeds),
        patch("edgar.run.t_score", score_generated),
        patch("edgar.run.t_score_validate"),
        patch("edgar.run.rank"),
        patch("edgar.run.t_fits_seeds"),
        patch("edgar.run.t_fits"),
        patch("edgar.run.t_deduplicate"),
        patch("edgar.run.t_prune"),
        patch("edgar.run.t_migrate"),
    ):
        asyncio.run(run(spec))

    assert events == (["translate_seeds"] if translation else []) + [
        "score_seeds",
        *stages,
        "score_generated",
    ]
    output = Path(spec.output_dir)
    rows = [
        json.loads(line)
        for line in (output / "population.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3
    assert all(
        row["code"]["model_jax"] == (TRANSLATED if translation else MODEL)
        for row in rows
    )
    for row in rows:
        birth = row["birth"]
        if birth["generation"] == -1:
            expected_estimator = seed_estimators[birth["batch_index"]]
        else:
            expected_estimator = (
                [ESTIMATOR] if "generate_param_ests" in stages else None
            )
        assert row["code"]["param_est"] == expected_estimator
    assert json.loads((output / "status.json").read_text())["state"] == "complete"
    metric_rows = [
        json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    metrics = {row["gen"]: row for row in metric_rows}
    all_generation_stages = {
        "generate_models",
        "generate_param_ests",
        "translate_programs",
    }
    stage_times = metrics[0]["stage_times"]
    assert all_generation_stages.intersection(stage_times) == set(stages)
    assert all(stage_times[name] >= 0 for name in stages)
    assert "translate_seeds" not in stage_times
    seed_times = metrics[-1]["stage_times"]
    assert ("translate_seeds" in seed_times) == translation
    assert not all_generation_stages.intersection(seed_times)
