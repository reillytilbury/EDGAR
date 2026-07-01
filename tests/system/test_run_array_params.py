"""
System test for dynamic default_params using fake LLM responses.
"""

import json
import yaml
import pytest
import numpy as np
from pathlib import Path
from tests.system.array_params_runner import run_test_array_params, CONFIG_PATH

TEST_OUTPUT_DIR = Path(__file__).parents[2] / "test_output_array_params"


def load_programs(output_dir):
    """Helper to load from population.jsonl"""
    programs = []
    log = next(output_dir.glob("**/population.jsonl"))
    with open(log, "r") as f:
        for line in f:
            programs.append(json.loads(line))
    return programs


def load_config():
    """Helper to load config.yaml"""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture(scope="session")
def array_params_fake_run():
    """Run the full pipeline with array-parameter programs once per test session.
    Programs cycled through are ProgramArrayParams and ProgramArrayParamsFallback, which have default_params that are callables.
    ProgramArrayParams has a parameter_estimator that returns params as arrays, while ProgramArrayParamsFallback has a parameter_estimator which raises and falls back to its default_params callable.
    """
    return run_test_array_params(TEST_OUTPUT_DIR)


def test_run_completes(array_params_fake_run):
    pass


def test_output_population_size(array_params_fake_run):
    """Verify that the expected number of programs are generated and saved."""
    config = load_config()
    n_gen = config["evolution"]["n_generations"]
    n_per_gen = config["evolution"]["batch_size"] * config["evolution"]["n_islands"]
    n_seed = config["llms"]["num_parents"]
    expected_total = n_seed + n_gen * n_per_gen
    programs = load_programs(array_params_fake_run)
    assert len(programs) == expected_total, (
        f"Expected {expected_total} programs, found {len(programs)}"
    )


def test_array_default_params_resolution(array_params_fake_run):
    """Verify that default_params are resolved into dictionaries in the saved output."""
    programs = load_programs(array_params_fake_run)

    for p in programs[2:]:  # skip seed programs
        # In JSON, it should be a dictionary
        assert isinstance(p["_default_params"], dict), (
            f"Program {p['idx']} default_params not resolved to dict"
        )
        # Check dicts are as we expect from tests/llm/programs.py
        assert np.allclose(p["_default_params"]["a"], np.ones(5)), (
            f"Program {p['idx']} default_params 'a' not as expected"
        )
        assert np.allclose(p["_default_params"]["b"], 0.1), (
            f"Program {p['idx']} default_params 'b' not as expected"
        )


def test_population_has_finite_scores(array_params_fake_run):
    """Verify that all programs have a score."""
    programs = load_programs(array_params_fake_run)
    for p in programs[2:]:  # skip seed programs
        assert isinstance(
            p["program_losses"]["discover"]["init"], float
        ) and np.isfinite(p["program_losses"]["discover"]["init"]), (
            f"Program {p['idx']} has non-finite or non-float score"
        )
        assert isinstance(
            p["program_losses"]["discover"]["final"], float
        ) and np.isfinite(p["program_losses"]["discover"]["final"]), (
            f"Program {p['idx']} has non-finite or non-float score"
        )
