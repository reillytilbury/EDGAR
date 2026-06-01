import asyncio
import os
import shutil

import pytest
from pathlib import Path
import json
import yaml
from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from edgar.run import run

CONFIG_PATH = Path(__file__).parent / "test_task" / "config.yaml"
TEST_OUTPUT_DIR = Path(__file__).parents[2] / "test_output"


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
def real_run():
    """
    Run the full pipeline with real LLM responses for simple synthetic data problem.
    """
    config = Config.from_yaml(CONFIG_PATH)
    spec = TaskSpec.from_config(config)
    shutil.rmtree(TEST_OUTPUT_DIR, ignore_errors=True)  # delete any existing output
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    spec.io["save_path"] = str(TEST_OUTPUT_DIR)
    asyncio.run(run(spec))
    return Path(spec.output_dir)


@pytest.mark.slow
@pytest.mark.live
def test_run_completes(real_run):
    """Passes if the input completes without error"""
    pass


@pytest.mark.slow
@pytest.mark.live
def test_program_log_written(real_run):
    log = next(real_run.glob("**/population.jsonl"))
    assert log.exists(), "population.jsonl not written to output directory"


@pytest.mark.slow
@pytest.mark.live
def test_total_programs(real_run):
    """Verifies expected number of programs generated and logged"""
    programs = load_programs(real_run)
    config = load_config()
    assert (
        len(programs)
        == config["evolution"]["n_islands"]
        * config["evolution"]["n_generations"]
        * config["evolution"]["batch_size"]
        + 2
    ), (
        f"Expected {config['evolution']['n_islands'] * config['evolution']['n_generations'] * config['evolution']['batch_size']} +2 programs, found {len(programs)}"
    )


@pytest.mark.slow
@pytest.mark.live
def test_n_ranked_programs(real_run):
    """Programs only ranked if they have validate_loss, i.e survived to the end.
    Expected number = n_islands*(critical_population_size - n_migrants)"""
    programs = load_programs(real_run)
    config = load_config()
    ranked_programs = [p for p in programs if p["rank"] is not None]
    assert len(ranked_programs) == config["evolution"]["n_islands"] * (
        config["evolution"]["critical_population_size"]
        - config["evolution"]["n_migrants"]
    ), (
        f"Expected {config['evolution']['n_islands'] * (config['evolution']['critical_population_size'] - config['evolution']['n_migrants'])} programs with test_loss, found {len(ranked_programs)}"
    )
