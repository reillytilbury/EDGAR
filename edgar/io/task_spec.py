"""
TaskSpec: A frozen bundle containing all configuration and callables needed to run an EDGAR experiment.

TaskSpec serves two purposes:
1. Load project callables and seed programs from a Config into ready-to-use fields.
2. Save a record of exact settings used so runs can be reproduced.

Construction:
    from_config(config)  — accepts a Config object built from Config.from_yaml or Config.from_taskspec

Example usage:
--------------
    # New run from a config file:
    spec = TaskSpec.from_config(Config.from_yaml("projects/my_task/config.yaml"))

    # Reproduce a previous run from its saved task_spec:
    spec = TaskSpec.from_config(Config.from_taskspec("runs/03-15/10-30-45/task_spec.yaml"))

    mode, temp, llms = spec.schedule(generation=0)
    spec.save(run_dir)
"""

from __future__ import annotations

import os
import stat
from collections import namedtuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Callable

import numpy as np
import yaml


from ..evolution.program import Program, BirthCertificate, Code
from ..llm.code_loading import load_function_from_source
from ..llm.prompt_schema import PromptSchema
from .config import Config
from .config import REPO_ROOT


LLMs = namedtuple("LLMs", ["model", "param_est", "model_jax"])
PromptSchemas = namedtuple("PromptSchemas", ["model", "param_est", "jax_model"])


def _git_state() -> tuple[str, bool]:
    """Returns the current Git HEAD SHA and a boolean indicating if the worktree is dirty.

    This function is used to capture the exact state of the repository at the time
    a `TaskSpec` is constructed.

    Returns:
        tuple[str, bool]: A tuple containing:
            - sha (str): The full SHA of the Git HEAD commit. Returns "unknown" if
              Git command fails.
            - dirty (bool): True if there are uncommitted changes in the worktree;
              False otherwise. Returns True if Git status command fails.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        sha = "unknown"
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
            ).strip()
        )
    except Exception:
        dirty = True
    return sha, dirty


@dataclass(frozen=True)
class TaskSpec:
    """Frozen bundle of everything needed to run (or re-run) an EDGAR experiment.

    TaskSpec serves two primary purposes:
    1.  **Initialization**: Loads and merges all configuration parameters, project-specific
        callable functions (e.g., data loader, loss function, plot function), and
        seed programs into a single, ready-to-use object.
    2.  **Reproducibility**: Saves a complete record of the exact settings and
        source code references used, allowing for precise reproduction of past
        experimental runs.

    This dataclass is immutable (`frozen=True`) to ensure that the configuration
    and state of an experiment remain consistent once initialized.

    Attributes:
        task_name (str): Human-readable name of the project (e.g., "orientation_tuning").
        git_sha (str): Full SHA of the git HEAD commit at TaskSpec construction time.
            Ensures exact code version traceability.
        git_dirty (bool): True if the worktree had uncommitted changes when the
            TaskSpec was built, indicating the SHA alone is not sufficient for
            reproduction.
        project_dir (Path): Absolute path to the project directory, which is the
            source of custom callables and seed programs.
        io (dict): Dictionary of I/O configuration parameters, controlling aspects
            like save paths.
        evolution (dict): Dictionary of evolutionary algorithm configuration parameters,
            such as number of generations, population sizes, etc.
        llms (dict): Dictionary of Large Language Model configuration parameters,
            including model names and potentially API settings.
        scoring (dict): Dictionary of scoring configuration parameters, such as
            gradient descent settings or complexity penalties.
        run (dict): Dictionary of general run-time configuration parameters,
            including the random seed.
        project_params (dict): Dictionary of project-specific parameters passed to
            functions like `load_data_fn`.
        model_prompt_schema (PromptSchema): Schema defining the prompt structure for
            generating new candidate models.
        param_est_prompt_schema (PromptSchema): Schema defining the prompt structure for
            generating parameter estimator functions.
        jax_model_prompt_schema (PromptSchema): Schema defining the prompt structure for
            translating NumPy models to JAX.
        load_data_fn (Callable): Function to load training, validation, and evaluation
            data for the project. Signature:
            `load_data(data_path, **project_params) -> (X_disc, X_val, X_eval)`.
        loss_fn (Callable): Project-specific loss function used by the scoring
            sandbox to evaluate model predictions against held-out data.
        plot_fn (Callable | None): Optional function to render model-fit images for
            LLM image-feedback prompts. None if the project does not provide
            `image_feedback/plot.py`.
        creation_timestamp (str): Timestamp set at construction, used to create the
            hierarchical on-disk layout `<save_path>/MM-DD/HH-MM-SS/`.
        seed_programs (list[Program]): Hand-written seed programs (typically 2) that
            bootstrap the initial population. These programs are loaded from
            `<project_dir>/seed_programs/modelN.py + param_estN.py` pairs.
        rng (np.random.Generator): A single seeded NumPy random number generator
            for the entire run. Ensures reproducibility of stochastic processes
            like spawning, migration, and Boltzmann sampling based on `run.random_seed`.
    """

    task_name: str

    git_sha: str

    git_dirty: bool

    project_dir: Path

    io: dict

    evolution: dict

    llms: dict

    scoring: dict

    run: dict

    project_params: dict

    model_prompt_schema: PromptSchema

    param_est_prompt_schema: PromptSchema

    jax_model_prompt_schema: PromptSchema

    load_data_fn: Callable

    loss_fn: Callable

    plot_fn: Callable | None

    creation_timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%m-%d/%H-%M-%S")
    )

    seed_programs: list[Program] = field(default_factory=list)

    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    # ── constructors ──

    @classmethod
    def from_config(cls, config: Config) -> TaskSpec:
        """Build a TaskSpec object from a given `Config` object.

        This factory method loads all necessary callables (data loader, loss function,
        optional plotting function) from the project directory, extracts Git state,
        and initializes seed programs. It serves as the primary way to create a
        `TaskSpec` for either a new run or to reproduce a past run.

        Args:
            config: A `Config` object, typically built from `Config.from_yaml` for
                a new run or `Config.from_taskspec` to reproduce a past run.

        Returns:
            TaskSpec: A fully initialized and frozen `TaskSpec` object ready for an
                EDGAR experiment.

        Raises:
            ValueError: If `load_data.py` does not define `load_data()` or `loss_fn()`.
        """
        task_name = config.task_name

        data_loader_path = config.project_dir / "data_loader" / "load_data.py"
        load_data_fn = load_function_from_source(
            data_loader_path.read_text(), "load_data"
        )
        if load_data_fn is None:
            raise ValueError(f"{data_loader_path} must define callable load_data()")
        loss_fn = load_function_from_source(data_loader_path.read_text(), "loss_fn")
        if loss_fn is None:
            raise ValueError(f"{data_loader_path} must define callable loss_fn()")

        plot_path = config.project_dir / "image_feedback" / "plot.py"
        plot_fn = (
            load_function_from_source(plot_path.read_text(), "plot_model_fits")
            if plot_path.exists()
            else None
        )

        git_sha, git_dirty = _git_state()

        seed_dir = config.project_dir / "seed_programs"
        seed_programs = []
        for batch_idx, model_path in enumerate(sorted(seed_dir.glob("model*.py"))):
            model_num = model_path.stem.replace("model", "")
            param_est_path = seed_dir / f"param_est{model_num}.py"
            seed_programs.append(
                Program(
                    birth=BirthCertificate(
                        generation=-1, island=-1, batch_index=batch_idx, mode="seed"
                    ),
                    code=Code(
                        model=model_path.read_text(),
                        param_est=param_est_path.read_text(),
                    ),
                    name=f"Seed Model {model_num}",
                    _default_params=cls._extract_default_params(model_path.read_text()),
                )
            )

        return cls(
            task_name=task_name,
            git_sha=git_sha,
            git_dirty=git_dirty,
            project_dir=config.project_dir,
            io=config.io.model_dump(),
            evolution=config.evolution.model_dump(),
            llms=config.llms.model_dump(),
            scoring=config.scoring.model_dump(),
            run=config.run.model_dump(),
            project_params=config.project_params,
            model_prompt_schema=config.prompts.model,
            param_est_prompt_schema=config.prompts.parameter_estimator,
            jax_model_prompt_schema=config.prompts.jax_translator_model,
            load_data_fn=load_data_fn,
            loss_fn=loss_fn,
            plot_fn=plot_fn,
            seed_programs=seed_programs,
            rng=np.random.default_rng(config.run.random_seed),
        )

    # ── persistence ──

    def save(self, run_dir: Path) -> Path:
        """Writes the current `TaskSpec` to a `task_spec.yaml` file.

        This method serializes the `TaskSpec`'s configuration, Git state, creation
        timestamp, project directory, seed programs, and prompt schemas into a YAML
        file. After writing, it strips write permissions from the file to prevent
        accidental modification, ensuring the saved specification accurately reflects
        the experiment that produced it.

        Args:
            run_dir: The directory where the `task_spec.yaml` file should be saved.

        Returns:
            Path: The path to the newly saved `task_spec.yaml` file.
        """
        record = {
            "task_name": self.task_name,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project_dir": str(self.project_dir),
            "io": self.io,
            "evolution": self.evolution,
            "llms": self.llms,
            "scoring": self.scoring,
            "run": self.run,
            "project_params": self.project_params,
            "seed_programs": [
                {
                    "batch_index": p.birth.batch_index,
                    "model_code": p.code.model,
                    "param_est_code": p.code.param_est,
                }
                for p in self.seed_programs
            ],
            "prompt_schemas": {
                "model": self.model_prompt_schema.model_dump(),
                "param_est": self.param_est_prompt_schema.model_dump(),
                "jax_model": self.jax_model_prompt_schema.model_dump(),
            },
        }

        path = Path(os.path.join(run_dir, "task_spec.yaml"))
        with open(path, "w") as f:
            yaml.dump(record, f, default_flow_style=False, sort_keys=False)
        # Strip write permission so the saved spec can't drift from the run that
        # actually produced it. Anyone trying to "fix up" a stored spec will hit
        # a PermissionError instead of silently breaking reproducibility.
        os.chmod(path, stat.S_IREAD)
        return path

    # ── generation schedule ──

    def schedule(self, generation: int) -> tuple[str, float, LLMs]:
        """Returns the operational mode, temperature, and specific LLMs for a given generation.

        The `temperature` for LLM generation follows a decaying schedule:
        `temperature = 1 + exp(-generation / n_generations)`. This results in a decay
        from 2.0 (at generation 0) towards 1.37 (as `generation` approaches
        `n_generations`). This range [1.37, 2.0] is considered the **Gemini scale**.
        For LLM providers like Anthropic, which typically accept temperatures in the
        range [0, 1], the `call_llm` function in `llm_calling.py` handles the necessary
        rescaling (e.g., mapping [1.37, 2.0] to [0.685, 1.0]) to ensure consistent
        behavior.

        The `mode` transitions from "explore" (first half of generations) to "exploit"
        (second half), guiding the LLM's behavior towards either novelty or refinement.
        LLMs for model generation, parameter estimation, and JAX translation can be
        specified as single models or as lists to cycle through per generation.

        Args:
            generation: The current generation number (0-indexed) of the evolutionary algorithm.

        Returns:
            tuple[str, float, LLMs]: A tuple containing:
                - mode (str): "explore" if the generation is in the first half of the run,
                  "exploit" otherwise.
                - temperature (float): The Gemini-scale temperature ([1.37, 2.0]) for
                  LLM generation. This value will be rescaled at the call site for
                  LLMs with different accepted temperature ranges (e.g., Anthropic).
                - llms (LLMs): A namedtuple providing the specific LLM models to be
                  used for `model` generation, `param_est` generation, and `model_jax`
                  translation in this generation.
        """
        import numpy as np

        n_generations = self.evolution["n_generations"]

        mode = "explore" if generation < n_generations // 2 else "exploit"
        temperature = 1 + np.exp(-generation / n_generations)

        model_llm = (
            self.llms["model_llm"][generation % len(self.llms["model_llm"])]
            if isinstance(self.llms["model_llm"], list)
            else self.llms["model_llm"]
        )

        llms = LLMs(
            model=model_llm,
            param_est=self.llms["param_est_llm"],
            model_jax=self.llms["jax_model_translator_llm"],
        )

        return mode, temperature, llms

    # ── config ──

    @property
    def output_dir(self) -> str:
        """Returns the full path to the run's output directory.

        This path is constructed by combining the base save path from `io`
        configuration and the unique `creation_timestamp` of this `TaskSpec`.

        Returns:
            str: The absolute path to the run's output directory.
        """
        return os.path.join(self.io["save_path"], self.creation_timestamp)

    @property
    def flat_config(self) -> dict:
        """Merges relevant configuration sections into a single dictionary for prompt variable lookup.

        This property flattens configuration parameters from the `evolution`, `llms`,
        and `scoring` sections into a single dictionary. This is particularly useful
        for prompt templating, where prompts declare variables by name (e.g.,
        `num_parents`, `max_lines`) without needing to know which specific
        configuration section they belong to.

        Returns:
            dict: A merged dictionary containing configuration parameters from the
                `evolution`, `llms`, and `scoring` sections.
        """
        return {**self.evolution, **self.llms, **self.scoring}

    # ── prompt schemas ──

    @property
    def prompt_schemas(self) -> PromptSchemas:
        """Retrieves all prompt schemas as a namedtuple for convenient access.

        This property provides an ergonomic way to access the `PromptSchema` objects
        for model generation, parameter estimator generation, and JAX model translation
        using attribute-style access (e.g., `spec.prompt_schemas.model`).

        Returns:
            PromptSchemas: A namedtuple containing `PromptSchema` objects for
                `model`, `param_est`, and `jax_model` generation.
        """
        return PromptSchemas(
            model=self.model_prompt_schema,
            param_est=self.param_est_prompt_schema,
            jax_model=self.jax_model_prompt_schema,
        )

    @staticmethod
    def _extract_default_params(model_code: str) -> dict:
        """Reads the `DEFAULT_PARAMS` dictionary attached to a model function's source code.

        By convention, seed model files can attach a `DEFAULT_PARAMS` dictionary
        as an attribute to their `model` function (often via a decorator). These
        parameters serve as the initial guess for gradient-descent parameter
        fitting before a program is scored. This method safely loads the model
        function from its source code and attempts to retrieve this attribute.

        Args:
            model_code: A string containing the Python source code of a model.

        Returns:
            dict: The `DEFAULT_PARAMS` dictionary if found, otherwise None. If the
                `model_code` is invalid or `model` function cannot be loaded,
                `None` is returned.
        """
        func = load_function_from_source(model_code, "model")
        default_params = getattr(func, "DEFAULT_PARAMS", None)
        return default_params
"""