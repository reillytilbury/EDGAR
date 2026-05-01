"""
TaskSpec: A frozen bundle containing all configuration and callables needed to run an EDGAR experiment.

TaskSpec serves two purposes:
1. Load and merge the task and default configs + project code into ready-to-use fields
2. Save a record of exact settings used so runs can be reproduced

Construction:
    from_config(path)  — accepts either a config.yaml or a task_spec.yaml from a previous run

Example usage:
--------------
    spec = TaskSpec.from_config("projects/my_task/config.yaml")
    spec = TaskSpec.from_config("runs/03-15/10-30-45/task_spec.yaml")
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

import yaml

from ..evolution.program import Program, BirthCertificate, Code
from ..llm.code_loading import load_function_from_source
from ..llm.prompt_schema import PromptSchema


LLMs = namedtuple("LLMs", ["model", "param_est", "jax"])
PromptSchemas = namedtuple("PromptSchemas", ["model", "param_est", "jax"])


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
    jax_prompt_schema: PromptSchema

    # project callables
    load_data_fn: Callable
    loss_fn: Callable
    plot_fn: Callable | None

    # timestamp set at construction — used to derive output_dir
    creation_timestamp: str = field(default_factory=lambda: datetime.now().strftime("%m-%d/%H-%M-%S"))

    # seed programs — 2 Programs with numpy model_code + param_est_code
    seed_programs: list[Program] = field(default_factory=list)

    # ── constructors ──

    @classmethod
    def from_config(cls, path: Path) -> TaskSpec:
        """
        Build a TaskSpec from a config.yaml or a task_spec.yaml from a previous run.

        If passed a task_spec.yaml, extracts the config dicts directly from the record.
        If passed a config.yaml, loads and merges with defaults.
        Either way, callables, git state, and creation_timestamp are always fresh.

        Args:
            path: Path to a config.yaml or task_spec.yaml.
        """
        from .config import _deep_merge, _git_state, PROJECT_ROOT

        path = Path(path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if path.name == "task_spec.yaml":
            record = yaml.safe_load(path.read_text())
            task_name = record["task_name"]
            config = {k: record[k] for k in ("io", "evolution", "llms", "scoring", "project_params")}
            schemas = record["prompt_schemas"]
            prompts = {
                "model": schemas["model"],
                "parameter_estimator": schemas["param_est"],
                "jax_translator": schemas["jax"],
            }
        else:
            task_name = path.parent.name
            default_config = yaml.safe_load((PROJECT_ROOT / "projects" / "config_default.yaml").read_text()) or {}
            config = _deep_merge(default_config, yaml.safe_load(path.read_text()) or {})
            default_prompts = yaml.safe_load((PROJECT_ROOT / "projects" / "prompt_defaults.yaml").read_text()) or {}
            task_prompt_path = PROJECT_ROOT / "projects" / task_name / "prompts.yaml"
            task_prompts = yaml.safe_load(task_prompt_path.read_text()) if task_prompt_path.exists() else {}
            prompts = _deep_merge(default_prompts, task_prompts)

        data_loader_path = PROJECT_ROOT / "projects" / task_name / "data_loader" / "load_data.py"
        load_data_fn = load_function_from_source(data_loader_path.read_text(), "load_data")
        if load_data_fn is None:
            raise ValueError(f"{data_loader_path} must define callable load_data()")
        loss_fn = load_function_from_source(data_loader_path.read_text(), "loss_fn")
        if loss_fn is None:
            raise ValueError(f"{data_loader_path} must define callable loss_fn()")

        plot_path = PROJECT_ROOT / "projects" / task_name / "image_feedback" / "plot.py"
        plot_fn = load_function_from_source(plot_path.read_text(), "plot_model_fits") if plot_path.exists() else None

        git_sha, git_dirty = _git_state()

        seed_dir = PROJECT_ROOT / "projects" / task_name / "seed_programs"
        seed_programs = []
        for batch_idx, model_path in enumerate(sorted(seed_dir.glob("model*.py"))):
            model_num = model_path.stem.replace("model", "")
            param_est_path = seed_dir / f"param_est{model_num}.py"
            seed_programs.append(Program(
                birth=BirthCertificate(generation=-1, island=-1, batch_index=batch_idx, mode="seed"),
                code=Code(model=model_path.read_text(), param_est=param_est_path.read_text()),
            ))

        return cls(
            task_name=task_name,
            git_sha=git_sha,
            git_dirty=git_dirty,
            io=config.get("io", {}),
            evolution=config.get("evolution", {}),
            llms=config.get("llms", {}),
            scoring=config.get("scoring", {}),
            project_params=config.get("project_params", {}),
            model_prompt_schema=PromptSchema(**prompts["model"]),
            param_est_prompt_schema=PromptSchema(**prompts["parameter_estimator"]),
            jax_prompt_schema=PromptSchema(**prompts["jax_translator"]),
            load_data_fn=load_data_fn,
            loss_fn=loss_fn,
            plot_fn=plot_fn,
            seed_programs=seed_programs,
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
                "jax": self.jax_prompt_schema.model_dump(),
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
        exploit_point = self.evolution.get("exploit_point", 0.5)

        mode = "explore" if generation < n_generations * exploit_point else "exploit"
        temperature = 1 + np.exp(-generation / n_generations)

        def _cycle(key):
            seq = self.llms[key]
            if isinstance(seq, str):
                return seq
            return seq[generation % len(seq)]

        llms = LLMs(
            model=_cycle("model_llm"),
            param_est=_cycle("param_est_llm"),
            jax=_cycle("jax_translator_llm"),
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
            jax=self.jax_prompt_schema,
        )
