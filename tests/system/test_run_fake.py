"""
System test using fake LLM responses.

Covers: end-to-end run with fake LLM calls injected via TestModel instances.
See fakellm.py and programs.py for the predetermined programs used.
"""
import asyncio
import pytest
from pathlib import Path

from src.run import run
from src.io.config import Config
from src.io.task_spec import TaskSpec
from tests.llm.fakellm import FakeLLM, SeedFakeLLM, CyclingModel

CONFIG_PATH = Path(__file__).parent / "test_task" / "config.yaml"
TEST_OUTPUT_DIR = Path(__file__).parents[2] / "test_output"

@pytest.fixture(scope="session")
def fake_run(tmp_path_factory):
    """Run the full pipeline with fake LLM responses once per test session."""
    config = Config.from_yaml(CONFIG_PATH)
    spec = TaskSpec.from_config(config)
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    spec.io["save_path"] = str(TEST_OUTPUT_DIR)

    n_gen = spec.evolution["n_generations"]
    n_per_gen = spec.evolution["batch_size"] * spec.evolution["n_islands"]
    n_seed = len(spec.seed_programs)
    fake = FakeLLM()
    seed_fake = SeedFakeLLM()

    spec.llms["model_llm"] = CyclingModel([fake.gen_model() for _ in range(n_gen * n_per_gen)])
    spec.llms["param_est_llm"] = CyclingModel([fake.gen_param_est() for _ in range(n_gen * n_per_gen)])
    spec.llms["jax_model_translator_llm"] = CyclingModel([seed_fake.gen_model_translation() for _ in range(n_seed)] + [fake.gen_model_translation() for _ in range(n_gen * n_per_gen)])
    spec.llms["jax_param_est_translator_llm"] = CyclingModel([seed_fake.gen_param_est_translation() for _ in range(n_seed)] + [fake.gen_param_est_translation() for _ in range(n_gen * n_per_gen)])

    asyncio.run(run(spec))
    return spec.output_dir

def test_run_completes(fake_run):
    pass