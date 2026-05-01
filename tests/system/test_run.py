"""
System test using real LLM responses

Covers: end-to-end run. 
Verifies the run completes without error, writes program_generation_log.jsonl, and verifies number of programs generated. 
"""
import json

import pytest
import asyncio
from pathlib import Path
import yaml

from run import _run_many

CONFIG_PATH = str(Path(__file__).parent / "config2.yaml")

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
def run(tmp_path_factory):
    """
        Run the system test with real LLM responses and return the output directory for inspection by other tests.
        As it is a fixture with session scope, it runs once per test session.
    """
    output_dir = tmp_path_factory.mktemp("outputdir")                                   
    asyncio.run(                                                                            
        _run_many(                                                                          
            config_path=CONFIG_PATH,                                             
            output_dir=str(output_dir),                                                     
            use_fake_llm=False,                                                    
            spec_module_path="tests.system.spec",                                           
        )                                                                                   
    )                                                                             
    return output_dir

def test_run_completes(run):
    """ Passes if the input completes without error"""
    pass

def test_program_log_written(run):
    log = next(run.glob("**/program_generation_log.jsonl"))
    assert log.exists(), "program_generation_log.jsonl not written to output directory"

def test_total_programs(run):
    """ Verifies expected number of programs generated and logged"""
    programs= load_programs(run)
    config = load_config()
    assert len(programs) == config["experiment_params"]["n_islands"] * config["experiment_params"]["n_iterations"] * config["experiment_params"]["batch_size"] + 2, f"Expected {config['experiment_params']['n_islands'] * config['experiment_params']['n_iterations'] * config['experiment_params']['batch_size']} +2 programs, found {len(programs)}"

def test_n_programs_with_testloss(run):
    """ Test loss only computed on programs remaining at end of run, verify we have the correct number, which is n_islands*(critical_population_size - n_migrants)"""
    programs = load_programs(run)
    config = load_config()
    programs_with_testloss = [p for p in programs if "test_loss" in p]
    assert len(programs_with_testloss) == config["experiment_params"]["n_islands"] * (config["experiment_params"]["critical_population_size"] - config["experiment_params"]["n_migrants"]), f"Expected {config['experiment_params']['n_islands'] * (config['experiment_params']['critical_population_size'] - config['experiment_params']['n_migrants'])} programs with test_loss, found {len(programs_with_testloss)}"