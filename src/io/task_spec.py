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
    spec.save_task_spec(run_dir)
"""
from __future__ import annotations

import os
import stat
from collections import namedtuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from pydantic_ai.models import Model

from ..evolution.program import Program, BirthCertificate, Code
from ..llm.code_loading import load_function_from_source
from ..llm.prompt_schema import PromptSchema


LLMs = namedtuple("LLMs", ["model", "param_est", "model_jax", "param_est_jax"])
PromptSchemas = namedtuple("PromptSchemas", ["model", "param_est", "jax_model", "jax_param_est"])


@dataclass
class TaskSpec:
    """
    Frozen bundle of everything needed to run (or re-run) an experiment.

    Job 1: load and merge all config + project code into ready-to-use fields.
    Job 2: save a record of the exact settings used so the run can be reproduced.

    Constructed via from_config (live run) or from_record (reproduce a past run).
    """

    # identity
    task_name: str
    git_sha: str
    git_dirty: bool

    # config subsections — plain dicts, passed through to the functions that need them
    io: dict
    evolution: dict
    llms: dict
    scoring: dict

    # project-specific knobs — kwargs unpacked into load_data_fn, also visible to other project callables
    project_params: dict

    # prompt schemas — one per LLM task, built from merged prompt yamls
    model_prompt_schema: PromptSchema
    param_est_prompt_schema: PromptSchema
    jax_model_prompt_schema: PromptSchema
    jax_param_est_prompt_schema: PromptSchema

    # project callables
    load_data_fn: Callable
    loss_fn: Callable
    plot_fn: Callable | None

    # timestamp set at construction — used to derive output_dir
    creation_timestamp: str = field(default_factory=lambda: datetime.now().strftime("%m-%d/%H-%M-%S"))

    # seed programs — 2 Programs with numpy model_code + param_est_code
    seed_programs: list[Program] = field(default_factory=list)

    # seeded RNG — use this instead of np.random directly for reproducibility
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    # ── constructors ──

    @classmethod
    def from_config(cls, config: "Config") -> TaskSpec:
        """
        Build a TaskSpec from a Config object.

        Callables, git state, and creation_timestamp are always fresh.
        Use Config.from_yaml for a new run, Config.from_taskspec to reproduce a past run.

        Args:
            config: A Config object built from Config.from_yaml or Config.from_taskspec.
        """
        from .config import _git_state, PROJECT_ROOT

        task_name = config.task_name
        prompts = config.prompts
        data_loader_path = config.project_dir / "data_loader" / "load_data.py"
        load_data_fn = load_function_from_source(data_loader_path.read_text(), "load_data")
        if load_data_fn is None:
            raise ValueError(f"{data_loader_path} must define callable load_data()")
        loss_fn = load_function_from_source(data_loader_path.read_text(), "loss_fn")
        if loss_fn is None:
            raise ValueError(f"{data_loader_path} must define callable loss_fn()")

        plot_path = config.project_dir / "image_feedback" / "plot.py"
        plot_fn = load_function_from_source(plot_path.read_text(), "plot_model_fits") if plot_path.exists() else None

        git_sha, git_dirty = _git_state()

        seed_dir = config.project_dir / "seed_programs"
        seed_programs = []
        for batch_idx, model_path in enumerate(sorted(seed_dir.glob("model*.py"))):
            model_num = model_path.stem.replace("model", "")
            param_est_path = seed_dir / f"param_est{model_num}.py"
            seed_programs.append(Program(
                birth=BirthCertificate(generation=-1, island=-1, batch_index=batch_idx, mode="seed"),
                code=Code(model=model_path.read_text(), param_est=param_est_path.read_text()),
                name=f"Seed Model {model_num}",
                _default_params = cls._extract_default_params(model_path.read_text())
            ))

        return cls(
            task_name=task_name,
            git_sha=git_sha,
            git_dirty=git_dirty,
            io=config.io,
            evolution=config.evolution,
            llms=config.llms,
            scoring=config.scoring,
            project_params=config.project_params,
            model_prompt_schema=PromptSchema(**prompts["model"]),
            param_est_prompt_schema=PromptSchema(**prompts["parameter_estimator"]),
            jax_model_prompt_schema=PromptSchema(**prompts["jax_translator_model"]),
            jax_param_est_prompt_schema=PromptSchema(**prompts["jax_translator_param_est"]),
            load_data_fn=load_data_fn,
            loss_fn=loss_fn,
            plot_fn=plot_fn,
            seed_programs=seed_programs,
            rng=np.random.default_rng(config.project_params.get("random_seed")),
        )

    # ── persistence ──

    def save_task_spec(self, run_dir: Path) -> Path:
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
            "io": self.io,
            "evolution": self.evolution,
            "llms": self.llms,
            "scoring": self.scoring,
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
                "jax_param_est": self.jax_param_est_prompt_schema.model_dump(),
            },
        }

        path = Path(os.path.join(run_dir, "task_spec.yaml"))
        with open(path, "w") as f:
            yaml.dump(record, f, default_flow_style=False, sort_keys=False)
        os.chmod(path, stat.S_IREAD)
        return path

    # ── generation schedule ──

    def schedule(self, generation: int) -> tuple[str, float, LLMs]:
        """
        Return (mode, temperature, llms) for a given generation.

        Args:
            generation: Generation number (0-indexed)

        Returns:
            tuple: (mode, temperature, llms) where:
                - mode: "explore" for first half of generations, "exploit" for second half
                - temperature: decays exponentially from ~2.0 to ~1.37
                - llms: LLMs namedtuple with model, param_est, jax (cycles through configured sequences)
        """
        import numpy as np

        n_generations = self.evolution["n_generations"]

        mode = "explore" if generation < n_generations // 2 else "exploit"
        temperature = 1 + np.exp(-generation / n_generations)

        llms = LLMs(
            model=self.llms["model_llm"],
            param_est=self.llms["param_est_llm"],
            model_jax=self.llms["jax_model_translator_llm"],
            param_est_jax=self.llms["jax_param_est_translator_llm"],
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
            jax_param_est=self.jax_param_est_prompt_schema,
        )

    # Load default_params from model code
    @staticmethod
    def _extract_default_params(model_code: str) -> dict:
        func = load_function_from_source(model_code, "model")
        default_params = getattr(func, "DEFAULT_PARAMS", None)
        return default_params