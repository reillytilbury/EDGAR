"""
System test using fake LLM responses.

Covers: end-to-end run with fake LLM generating Program1, Program2,
and ProgramSolution (see programs.py and fakellm.py). Verifies the run
completes without error, writes program_generation_log.jsonl, and verifies number of programs generated, winning program details and losses.
"""
import json

import pytest
import asyncio
from pathlib import Path
import yaml

from run import _run_many

CONFIG_PATH = str(Path(__file__).parent / "config.yaml")
# OUTPUT_PATH = Path(__file__).parent / "output"

def load_programs(output_dir):
    """Helper to load from program_generation_log.jsonl"""
    programs = []
    log = next(output_dir.glob("**/program_generation_log.jsonl"))
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
def fake_run(tmp_path_factory):
    """
        Run the system test with fake LLM responses and return the output directory for inspection by other tests.
        As it is a fixture with session scope, it runs once per test session.
    """
    output_dir = tmp_path_factory.mktemp("outputdir")                                   
    asyncio.run(                                                                            
        _run_many(                                                                          
            config_path=CONFIG_PATH,                                             
            output_dir=str(output_dir),                                                     
            use_fake_llm=True,                                                    
            spec_module_path="tests.system.spec",                                           
        )                                                                                   
    )                                                                             
    return output_dir

def test_run_completes(fake_run):
    """ Passes if the input completes without error"""
    pass

def test_program_log_written(fake_run):
    log = next(fake_run.glob("**/program_generation_log.jsonl"))
    assert log.exists(), "program_generation_log.jsonl not written to output directory"

def test_total_programs(fake_run):
    """ Verifies expected number of programs generated and logged"""
    programs= load_programs(fake_run)
    config = load_config()
    assert len(programs) == config["experiment_params"]["n_islands"] * config["experiment_params"]["n_iterations"] * config["experiment_params"]["batch_size"] + 2, f"Expected {config['experiment_params']['n_islands'] * config['experiment_params']['n_iterations'] * config['experiment_params']['batch_size']} +2 programs, found {len(programs)}"

def test_n_programs_with_testloss(fake_run):
    """ Test loss only computed on programs remaining at end of run, verify we have the correct number, which is n_islands*(critical_population_size - n_migrants)"""
    programs = load_programs(fake_run)
    config = load_config()
    programs_with_testloss = [p for p in programs if "test_loss" in p]
    assert len(programs_with_testloss) == config["experiment_params"]["n_islands"] * (config["experiment_params"]["critical_population_size"] - config["experiment_params"]["n_migrants"]), f"Expected {config['experiment_params']['n_islands'] * (config['experiment_params']['critical_population_size'] - config['experiment_params']['n_migrants'])} programs with test_loss, found {len(programs_with_testloss)}"

def test_correct_winner(fake_run):
    """ Verify the final winning program is the (0,0,2) program, which is the ProgramSolution without offset"""
    programs = load_programs(fake_run)
    winning_programs = [p for p in programs if p.get("is_winner")]
    assert len(winning_programs) == 1, "Expected exactly 1 winning program"
    winning_birth_details = (winning_programs[0]["iteration_number"], winning_programs[0]["birth_island"], winning_programs[0]["batch_index"])
    assert winning_birth_details == (0,0,2), f"Expected winning program born at (iteration 0, island 0, batch index 2), found {winning_birth_details}"

def test_winning_loss(fake_run):
    """ Verify the winning program has anticipated train and test loss"""
    programs = load_programs(fake_run)
    winning_program = [p for p in programs if p.get("is_winner")][0]
    #Check the loss on held out x values, used to score programs within evolution
    assert round(winning_program["initial_loss"],3) == 0.681, f"Expected winning program initial_loss (without gd) 0.681, found {winning_program['initial_loss']:.3f}"
    assert round(winning_program["train_loss"],3) == 0.056, f"Expected winning program train_loss 0.056, found {winning_program['train_loss']:.3f}"
    #Check the loss computed on held out samples (i.e parameter sets for the target function)
    assert round(winning_program["test_loss"],3) == 0.141, f"Expected winning program test_loss 0.141, found {winning_program['test_loss']:.3f}"