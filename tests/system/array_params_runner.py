"""
Fake-LLM runner for testing dynamic default_params without real API calls.
"""

import asyncio
import os
import shutil
from pathlib import Path

from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from edgar.run import run
from tests.llm.fakellm import CyclingModel, FakeLLM, SeedFakeLLM
from tests.llm.programs import PARAMETER_ARRAY_FAKE_PROGRAMS

CONFIG_PATH = (
    Path(__file__).parents[1] / "io" / "test_task_array_params" / "config.yaml"
)
DEFAULT_OUTPUT_DIR = Path(__file__).parents[2] / "test_output_array_params"


def build_array_params_fake_spec(output_dir: Path) -> TaskSpec:
    """Build a TaskSpec wired with fake LLMs using array-parameter programs."""
    config = Config.from_yaml(CONFIG_PATH)
    spec = TaskSpec.from_config(config)
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    spec.io["save_path"] = str(output_dir)

    n_gen = spec.evolution["n_generations"]
    n_per_gen = spec.evolution["batch_size"] * spec.evolution["n_islands"]
    n_seed = len(spec.seed_programs)
    fake = FakeLLM(PARAMETER_ARRAY_FAKE_PROGRAMS)
    seed_fake = SeedFakeLLM()

    spec.llms["model_llm"] = CyclingModel(
        [fake.gen_model() for _ in range(n_gen * n_per_gen)]
    )
    spec.llms["param_est_llm"] = CyclingModel(
        [fake.gen_param_est() for _ in range(n_gen * n_per_gen)]
    )
    spec.llms["jax_model_translator_llm"] = CyclingModel(
        [seed_fake.gen_model_translation() for _ in range(n_seed)]
        + [fake.gen_model_translation() for _ in range(n_gen * n_per_gen)]
    )
    return spec


def run_test_array_params(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Run the array-parameter fake-LLM pipeline and return the output directory path."""
    spec = build_array_params_fake_spec(output_dir)
    asyncio.run(run(spec))
    return Path(spec.output_dir)
