# Investigate different ways of normalizing the BZ015 data.
from pathlib import Path
from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from edgar.llm.utils import translate_to_jax
from edgar.scoring.scoring import score
from edgar.evolution.population import Population
from edgar.evolution.island import seed
import sys
import os
import numpy as np

if not hasattr(sys.modules["__main__"], "__spec__"):
    sys.modules["__main__"].__spec__ = None
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_command_buffer=" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_enable_command_buffer=").strip()


def score_seeds(project_name: str):
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "projects" / project_name / "config.yaml"
    print(f"Loading config from: {path}")
    config = Config.from_yaml(path)
    spec = TaskSpec.from_config(config)
    print("Loading data...")
    X_discover, X_validate, X_eval = spec.load_data_fn(
        data_path=spec.io["data_path"], **spec.project_params
    )

    print(
        f"Max (X_discover): {np.max(X_discover[1]['response'])}, Min (X_discover): {np.min(X_discover[1]['response'])}"
    )
    print(
        f"Max (X_validate): {np.max(X_validate[1]['response'])}, Min (X_validate): {np.min(X_validate[1]['response'])}"
    )

    # Do naive jax translation
    population = Population()
    for program in spec.seed_programs:
        program.code.model_jax = translate_to_jax(program.code.model)

    islands = seed(population, spec.seed_programs, n_islands=1)

    # Scoring
    # Discover scoring
    score(population, X_discover, X_eval, spec.scoring, spec.loss_fn, split="discover")
    # Validate scoring
    population.prepare_validation_scoring(islands=islands)
    score(population, X_validate, None, spec.scoring, spec.loss_fn, split="validate")
    print("Scoring complete\n --------- \n ")
    for i, program in enumerate(population):
        print(f"Seed {i + 1}:")
        print(f"Number of Parameters: {program.n_params}\n")
        print(f"Discover Score: {program.program_losses.discover.final}")
        print(f"Validate Score: {program.program_losses.validate.final}")
        print(f"Number of Parameters: {program.n_params}\n")


if __name__ == "__main__":
    project_name = sys.argv[1]
    score_seeds(project_name)
