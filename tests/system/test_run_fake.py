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
from tests.llm.fakellm import FakeLLM, SeedFakeLLM

CONFIG_PATH = Path(__file__).parent / "test_task" / "config.yaml"

@pytest.fixture(scope="session")
def fake_run(tmp_path_factory):
    """Run the full pipeline with fake LLM responses once per test session."""
    config = Config.from_yaml(CONFIG_PATH)
    spec = TaskSpec.from_config(config)
    spec.io["save_path"] = str(tmp_path_factory.mktemp("output"))

    fake = FakeLLM()
    seed_fake = SeedFakeLLM()
    n_gen = spec.evolution["n_generations"]
    spec.llms["model_llm"] = [fake.gen_model() for _ in range(n_gen)]
    spec.llms["param_est_llm"] = [fake.gen_param_est() for _ in range(n_gen)]
    spec.llms["jax_translator_llm"] = [seed_fake.gen_model_jax() for _ in range(2)]+ [fake.gen_model_jax() for _ in range(n_gen-2)]

    asyncio.run(run(spec))
    return spec.output_dir

def test_run_completes(fake_run):
    pass
