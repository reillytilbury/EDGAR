"""
System test using fake LLM responses.

Covers: end-to-end run with fake LLM calls injected via TestModel instances.
See fakellm.py and programs.py for the predetermined programs used.
"""
import asyncio
import json

import yaml
import pytest
from pathlib import Path
import shutil
import os 
from src.run import run
from src.io.config import Config
from src.io.task_spec import TaskSpec
from tests.llm.fakellm import FakeLLM, SeedFakeLLM, CyclingModel
from tests.llm.programs import ProgramSolution

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
    """ Helper to load config.yaml"""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture(scope="session")
def fake_run():
    """Run the full pipeline with fake LLM responses once per test session."""
    config = Config.from_yaml(CONFIG_PATH)
    spec = TaskSpec.from_config(config)
    shutil.rmtree(TEST_OUTPUT_DIR, ignore_errors=True) #delete any existing output
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    spec.io["save_path"] = str(TEST_OUTPUT_DIR)

    n_gen = spec.evolution["n_generations"]
    n_per_gen = spec.evolution["batch_size"] * spec.evolution["n_islands"]
    n_seed = len(spec.seed_programs)
    fake = FakeLLM()
    seed_fake = SeedFakeLLM()

    spec.llms["model_llm"] = CyclingModel([fake.gen_model() for _ in range(n_gen * n_per_gen)])
    spec.llms["param_est_llm"] = CyclingModel([fake.gen_param_est() for _ in range(n_gen * n_per_gen)])
    spec.llms["jax_model_translator_llm"] = CyclingModel([seed_fake.gen_model_translation() for _ in range(n_seed)] + [fake.gen_model_translation() for _ in range(n_gen * n_per_gen)])

    asyncio.run(run(spec))
    return Path(spec.output_dir)

def test_run_completes(fake_run):
    pass

def test_population_written(fake_run):
    log = next(fake_run.glob("**/population.jsonl"))
    assert log.exists(), "population.jsonl not written to output directory"

def test_total_programs(fake_run):
    """ Verifies expected number of programs generated and logged"""
    programs= load_programs(fake_run)
    config = load_config()
    assert len(programs) == config["evolution"]["n_islands"] * config["evolution"]["n_generations"] * config["evolution"]["batch_size"] + 2, f"Expected {config['evolution']['n_islands'] * config['evolution']['n_generations'] * config['evolution']['batch_size']} +2 programs, found {len(programs)}"

def test_n_ranked_programs(fake_run):
    """ Programs only ranked if they have validate_loss, i.e survived to the end.
        Expected number = n_islands*(critical_population_size - n_migrants) """
    programs = load_programs(fake_run)
    config = load_config()
    ranked_programs = [p for p in programs if p["rank"] is not None]
    assert len(ranked_programs) == config["evolution"]["n_islands"] * (config["evolution"]["critical_population_size"] - config["evolution"]["n_migrants"]), f"Expected {config['evolution']['n_islands'] * (config['evolution']['critical_population_size'] - config['evolution']['n_migrants'])} programs with test_loss, found {len(ranked_programs)}"

def test_seed_losses(fake_run):
    """ Verify losses of seed programs match expected values.
        Their discover losses are independent of evolution algorithm, but whether validate losses are present depends on evolution details
    """
    programs = load_programs(fake_run)
    seed_programs = [p for p in programs if p["birth"]["generation"] == -1]
    assert len(seed_programs) == 2, f"Expected exactly 2 seed programs, found {len(seed_programs)}"
    assert seed_programs[0]["name"] == "Seed Model 1", f"Expected first seed program to be 'Seed Model 1', found {seed_programs[0]['name']}"
    assert seed_programs[1]["name"] == "Seed Model 2", f"Expected second seed program to be 'Seed Model 2', found {seed_programs[1]['name']}"
    #Seed 1 losses
    assert round(seed_programs[0]["program_losses"]['discover']['init'], 4) == 1.0176
    assert round(seed_programs[0]["program_losses"]['discover']['final'], 4) == 1.0176
    assert round(seed_programs[0]["program_losses"]['validate']['final'], 4) == 8.7638

    #Seed 2 losses
    assert round(seed_programs[1]["program_losses"]['discover']['init'], 4) == 0.7270
    assert round(seed_programs[1]["program_losses"]['discover']['final'], 4) == 0.7270
    assert seed_programs[1]["program_losses"]['validate']['final'] == 'NOTVALIDATED'

def test_winning_program(fake_run):
    """ Verify correct program is winner"""
    programs = load_programs(fake_run)
    winning_program = [p for p in programs if p["rank"] == 1]
    assert len(winning_program) == 1, "Expected exactly 1 winning program"
    winning_birth_details = (winning_program[0]["birth"]['generation'], winning_program[0]["birth"]["island"], winning_program[0]["birth"]["batch_index"])
    assert winning_birth_details == (0,0,2), f"Expected winning program born at (iteration 0, island 0, batch index 2), found {winning_birth_details}"
    assert winning_program[0]["birth"]["parent_indices"] == [0, 1], f"Expected winning program parent indices [0, 1], found {winning_program[0]['birth']['parent_indices']}"
    assert winning_program[0]["name"] == "Fake Model 2"

def test_winning_losses(fake_run):
    """ Verify losses of winning program match expected values"""
    programs = load_programs(fake_run)
    winning_program = [p for p in programs if p["rank"] == 1][0]
    print("Winning program losses:")
    assert round(winning_program["program_losses"]['discover']['init'], 4) == 0.8431, f"Expected winning program validate_loss of 0.8431, found {winning_program['program_losses']['discover']['init']}"
    assert round(winning_program["program_losses"]['discover']['final'], 4) == 0.0767, f"Expected winning program validate_loss of 0.0767, found {winning_program['program_losses']['discover']['final']}"
    assert round(winning_program["program_losses"]['validate']['init'], 4) == 3.1975, f"Expected winning program validate_loss of 3.1975, found {winning_program['program_losses']['validate']['init']}"
    assert round(winning_program["program_losses"]['validate']['final'], 4) == 0.1327, f"Expected winning program validate_loss of 0.1327, found {winning_program['program_losses']['validate']['final']}"

