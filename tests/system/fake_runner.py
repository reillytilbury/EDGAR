"""
Fake-LLM runner for end-to-end pipeline testing without real API calls.

Used by both the pytest suite (tests/system/test_run_fake.py) and the
`edgar test-fake` CLI command.

Example usage:
    from tests.system.fake_runner import run_test_fake
    run_test_fake()
"""
import asyncio
import os
import shutil
from pathlib import Path

from src.io.config import Config
from src.io.task_spec import TaskSpec
from src.run import run
from tests.llm.fakellm import CyclingModel, FakeLLM, SeedFakeLLM

CONFIG_PATH = Path(__file__).parent / "test_task" / "config.yaml"
DEFAULT_OUTPUT_DIR = Path(__file__).parents[2] / "test_output"


def build_fake_spec(output_dir: Path) -> TaskSpec:
    """Build a TaskSpec wired with fake LLMs, writing outputs to output_dir."""
    config = Config.from_yaml(CONFIG_PATH)
    spec = TaskSpec.from_config(config)
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    spec.io["save_path"] = str(output_dir)

    n_gen = spec.evolution["n_generations"]
    n_per_gen = spec.evolution["batch_size"] * spec.evolution["n_islands"]
    n_seed = len(spec.seed_programs)
    fake = FakeLLM()
    seed_fake = SeedFakeLLM()

    spec.llms["model_llm"] = CyclingModel([fake.gen_model() for _ in range(n_gen * n_per_gen)])
    spec.llms["param_est_llm"] = CyclingModel([fake.gen_param_est() for _ in range(n_gen * n_per_gen)])
    spec.llms["jax_model_translator_llm"] = CyclingModel(
        [seed_fake.gen_model_translation() for _ in range(n_seed)]
        + [fake.gen_model_translation() for _ in range(n_gen * n_per_gen)]
    )
    return spec


def run_test_fake(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Run the fake-LLM pipeline and return the output directory path."""
    spec = build_fake_spec(output_dir)
    asyncio.run(run(spec))
    return Path(spec.output_dir)
