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

CONFIG_PATH = str(Path(__file__).parent / "config.yaml")

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
            test_mode = True,                                                                          
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
    """ Verifies expected number of programs generated and logged, remember test_mode=True overrides parameters in config.yaml, to:
        n_iterations = 1, n_islands = 2, batch_size = 2, so we expect 2*1*2 + 2 (initial programs) = 6 total programs."""
    programs= load_programs(run)
    assert len(programs) == 6, f"Expected 6 programs, found {len(programs)}"

def test_n_programs_with_testloss(run):
    """ Test loss only computed on programs remaining at end of run, verify we have the correct number, which is n_islands*(critical_population_size).
        With test_mode=True, we have n_islands=2, critical_population_size=2, so expect 4 programs with test_loss."""
    programs = load_programs(run)
    programs_with_testloss = [p for p in programs if "test_loss" in p]
    assert len(programs_with_testloss) == 4, f"Expected 4 programs with test_loss, found {len(programs_with_testloss)}"