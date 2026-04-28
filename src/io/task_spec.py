"""
TaskSpec: A frozen bundle containing all configuration and callables needed to run an EDGAR experiment.

TaskSpec serves two purposes:
1. Load and merge the task and default configs + project code into ready-to-use fields
2. Save a record of exact settings used so runs can be reproduced

Construction:
    from_config(config_path)      — Load config, merge defaults, load project code
    from_task_spec(task_spec_path) — Reconstruct from a past run's task_spec.yaml

Example usage:
--------------
    # Live run: load from config
    spec = TaskSpec.from_config("projects/my_task/config.yaml")
    mode, temp, llms = spec.schedule(iteration=0)
    X_discover, X_validate, X_eval = spec.load_data_fn(data_path=spec.io["data_path"], ...)

    # Re-run: reproduce from saved task_spec
    spec = TaskSpec.from_task_spec("runs/2024-03-15/10-30-45/task_spec.yaml")
    # same as above

    # Save task_spec for reproducibility
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

    # seed programs — 2 Programs with numpy model_code + param_est_code
    seed_programs: list[Program] = field(default_factory=list)

    # ── constructors ──

    @classmethod
    def from_config(cls, config_path: Path) -> TaskSpec:
        """
        Load and merge configs, load project code, and capture git state.

        Args:
            config_path: Path to projects/<task_name>/config.yaml

        Returns:
            TaskSpec with all fields populated from merged config and loaded callables
        """
        from .config import build_task_spec_from_config
        return build_task_spec_from_config(config_path)

    @classmethod
    def from_task_spec(cls, task_spec_path: Path) -> TaskSpec:
        """
        Reconstruct a TaskSpec from a saved task_spec.yaml file.

        Args:
            task_spec_path: Path to task_spec.yaml from a previous run

        Returns:
            TaskSpec with identical settings as the original run (requires project code to still exist on disk)

        Raises:
            FileNotFoundError: If task_spec.yaml doesn't exist
            ValueError: If file is malformed or required callables are missing
        """
        with open(task_spec_path) as f:
            record = yaml.safe_load(f)

        task_name = record["task_name"]

        seed_programs = [
            Program(
                birth=BirthCertificate(
                    generation=-1,
                    island=-1,
                    batch_index=s["batch_index"],
                    mode="seed",
                ),
                code=Code(model=s["model_code"], param_est=s["param_est_code"]),
            )
            for s in record["seed_programs"]
        ]

        schemas = record["prompt_schemas"]

        # Re-load project callables from the task folder.
        # This requires the project code to still be on disk.
        project_root = Path(__file__).resolve().parent.parent.parent

        # Load project callables
        data_loader_path = project_root / "projects" / task_name / "data_loader" / "load_data.py"
        load_data_fn = load_function_from_source(data_loader_path.read_text(), "load_and_process_data")
        if load_data_fn is None:
            raise ValueError(f"{data_loader_path} must define callable load_and_process_data()")

        loss_fn = load_function_from_source(data_loader_path.read_text(), "loss_fn")
        if loss_fn is None:
            raise ValueError(f"{data_loader_path} must define callable loss_fn()")

        plot_path = project_root / "projects" / task_name / "image_feedback" / "plot.py"
        plot_fn = None
        if plot_path.exists():
            plot_fn = load_function_from_source(plot_path.read_text(), "plot_model_fits")

        return cls(
            task_name=task_name,
            git_sha=record["git_sha"],
            git_dirty=record["git_dirty"],
            io=record["io"],
            evolution=record["evolution"],
            llms=record["llms"],
            scoring=record["scoring"],
            project_params=record["project_params"],
            model_prompt_schema=PromptSchema(**schemas["model"]),
            param_est_prompt_schema=PromptSchema(**schemas["param_est"]),
            jax_prompt_schema=PromptSchema(**schemas["jax"]),
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

        path = Path(run_dir) / "task_spec.yaml"
        with open(path, "w") as f:
            yaml.dump(record, f, default_flow_style=False, sort_keys=False)
        os.chmod(path, stat.S_IREAD)
        return path

    # ── iteration schedule ──

    def schedule(self, iteration: int) -> tuple[str, float, LLMs]:
        """
        Return (mode, temperature, llms) for a given iteration.

        Args:
            iteration: Iteration number (0-indexed)

        Returns:
            tuple: (mode, temperature, llms) where:
                - mode: "explore" for first half of iterations, "exploit" for second half
                - temperature: decays exponentially from ~2.0 to ~1.37
                - llms: LLMs namedtuple with model, param_est, jax (cycles through configured sequences)
        """
        import numpy as np

        n_iterations = self.evolution["n_iterations"]
        exploit_point = self.evolution.get("exploit_point", 0.5)

        mode = "explore" if iteration < n_iterations * exploit_point else "exploit"
        temperature = 1 + np.exp(-iteration / n_iterations)

        def _cycle(key):
            seq = self.llms[key]
            if isinstance(seq, str):
                return seq
            return seq[iteration % len(seq)]

        llms = LLMs(
            model=_cycle("model_llm"),
            param_est=_cycle("param_est_llm"),
            jax=_cycle("jax_translator_llm"),
        )

        return mode, temperature, llms

    # ── config ──

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
