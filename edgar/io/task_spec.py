"""
TaskSpec: A frozen bundle containing all configuration and callables needed to run an EDGAR experiment.

TaskSpec serves two purposes:
1. Load project callables and seed programs from a Config into ready-to-use fields
2. Save a record of exact settings used so runs can be reproduced

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
from .config import PROJECT_ROOT


# Lightweight ergonomic wrappers: callers can write `llms.model` and
# `schemas.param_est` instead of `dict["model_llm"]` / attribute soup.
LLMs = namedtuple("LLMs", ["model", "param_est", "model_jax"])
PromptSchemas = namedtuple("PromptSchemas", ["model", "param_est", "jax_model"])


def _git_state() -> tuple[str, bool]:
    """Return (sha, dirty) for the current git HEAD.

    Captured at TaskSpec construction so saved task_spec.yaml records exactly
    which commit produced a run. `dirty=True` means there were uncommitted
    changes, so the sha alone is not enough to reproduce the run.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        sha = "unknown"
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
            ).strip()
        )
    except Exception:
        dirty = True
    return sha, dirty


@dataclass
class TaskSpec:
    """
    Frozen bundle of everything needed to run (or re-run) an experiment.

    Job 1: load and merge all config + project code into ready-to-use fields.
    Job 2: save a record of the exact settings used so the run can be reproduced.

    Constructed via from_config (live run) or from_record (reproduce a past run).
    """

    # ── identity ──
    # Human-readable name of the project (e.g. "orientation_tuning").
    task_name: str

    # Full SHA of the git HEAD commit at TaskSpec construction time.
    git_sha: str

    # True if the worktree had uncommitted changes when the TaskSpec was built.
    git_dirty: bool

    # Absolute path to the project directory (source of callables and seed programs).
    project_dir: Path

    # ── config subsections — plain dicts, passed through to the functions that need them
    # TODO: Put documentation for these in io/config.py
    io: dict

    evolution: dict

    llms: dict

    scoring: dict

    run: dict

    project_params: dict

    # ── prompt schemas — one per LLM role, built from merged prompt yamls ──
    # Schema for the "generate a new candidate model" prompt.
    model_prompt_schema: PromptSchema

    # Schema for the "generate a parameter estimator function" prompt.
    param_est_prompt_schema: PromptSchema

    # Schema for the "translate a numpy model to JAX" prompt.
    jax_model_prompt_schema: PromptSchema

    # ── project callables — loaded from <project_dir>/data_loader/ and image_feedback/ ──
    # Loads training/validation/eval data for this project.
    # Signature: load_data(data_path, **project_params) ->
    #   ((X_disc_train, X_disc_test), (X_val_train, X_val_test), X_eval)
    load_data_fn: Callable

    # Project-specific loss function used by the scoring sandbox to score
    # predictions against held-out data.
    loss_fn: Callable

    # Optional: renders model-fit images for image-feedback prompts. None if the
    # project has no `image_feedback/plot.py`.
    plot_fn: Callable | None

    # ── runtime state ──
    # Timestamp set at construction, produces on-disk layout `<save_path>/MM-DD/HH-MM-SS/`.
    creation_timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%m-%d/%H-%M-%S")
    )

    # Hand-written seed programs (typically 2) that bootstrap the population.
    # Loaded from <project_dir>/seed_programs/modelN.py + param_estN.py pairs.
    seed_programs: list[Program] = field(default_factory=list)

    # Single seeded RNG for the whole run. Use this instead of `np.random` so
    # spawning, migration, and Boltzmann sampling are reproducible from the
    # `run.random_seed` config value.
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    # ── constructors ──

    @classmethod
    def from_config(cls, config: Config) -> TaskSpec:
        """
        Build a TaskSpec from a Config object.

        Callables, git state, and creation_timestamp are always fresh.
        Use Config.from_yaml for a new run, Config.from_taskspec to reproduce a past run.

        Args:
            config: A Config object built from Config.from_yaml or Config.from_taskspec.
        """
        task_name = config.task_name

        # Project callables are loaded from .py source rather than imported, so the
        # same machinery (load_function_from_source) handles seed code, LLM-generated
        # code, and project code on one path. The cost is that import errors surface
        # as `None` returns instead of ImportError, hence the explicit None checks.
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

        # Seed programs are paired by filename suffix: model1.py with param_est1.py,
        # model2.py with param_est2.py, ... sorted alphanumerically. They get
        # sentinel birth fields (generation=-1, island=-1) so downstream code can
        # cheaply distinguish hand-written seeds from LLM-evolved descendants.
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
        """
        Write task_spec.yaml for reproducibility.

        Args:
            run_dir: Directory to save task_spec.yaml into

        Returns:
            Path to the saved task_spec.yaml file
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
        """
        Return (mode, temperature, llms) for a given generation.

        temperature = `1 + exp(-generation / n_generations)`, so decay is from 2 -> 1.37.
        The [1.37, 2.0] range is the **Gemini scale** (range [0, 2]).
        Anthropic only accepts [0, 1]; when the resolved model is an
        AnthropicModel, call_llm rescales by /2 to map [1.37, 2.0] → [0.685, 1.0].
        See `src/llm/llm_calling.py:_build_model` + the rescale guard right after.

        Args:
            generation: Generation number (0-indexed)

        Returns:
            tuple: (mode, temperature, llms) where:
                - mode: "explore" for first half of generations, "exploit" for second half
                - temperature: Gemini-scale [1.37, 2.0]; rescaled at the call site for Anthropic
                - llms: namedtuple with llm.model, llm.param_est and llm.model_jax specifying the LLM to be used this generation
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
        return os.path.join(self.io["save_path"], self.creation_timestamp)

    @property
    def flat_config(self) -> dict:
        """
        Merge all config sections into a single dict for prompt variable lookup.

        Prompts declare their `config_vars` by name (e.g. `num_parents`, `max_lines`,
        `swear_words`) without specifying which section the var lives in. Returning a
        flat merged dict lets the prompt-building code look up any var without
        plumbing the section through.

        Returns:
            dict combining evolution, llms, and scoring config sections
        """
        return {**self.evolution, **self.llms, **self.scoring}

    # ── prompt schemas ──

    @property
    def prompt_schemas(self) -> PromptSchemas:
        """
        Get all prompt schemas as a namedtuple.

        Returns:
            PromptSchemas with model, param_est, and jax PromptSchema objects
        """
        return PromptSchemas(
            model=self.model_prompt_schema,
            param_est=self.param_est_prompt_schema,
            jax_model=self.jax_model_prompt_schema,
        )

    @staticmethod
    def _extract_default_params(model_code: str) -> dict:
        """Read DEFAULT_PARAMS attached to a model function.

        By convention, seed model files attach a DEFAULT_PARAMS dict to the `model`
        function (typically via a decorator). The pipeline uses these as the
        starting point for gradient-descent parameter fitting before scoring; if
        the attribute is missing, returns None and the program is treated as
        having no provided defaults.
        """
        func = load_function_from_source(model_code, "model")
        default_params = getattr(func, "DEFAULT_PARAMS", None)
        return default_params
