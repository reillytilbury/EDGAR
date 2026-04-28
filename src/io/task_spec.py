"""
TaskSpec: A frozen bundle containing all configuration and callables needed to run an EDGAR experiment.

TaskSpec serves two purposes:
1. Load and merge all configs + project code into ready-to-use fields
2. Save a record of exact settings used so runs can be reproduced

Construction:
    from_config(config_path)  — Load config, merge defaults, import project code
    from_record(record_path)  — Reconstruct from a past run's record

Example usage:
--------------
    # Live run: load from config
    spec = TaskSpec.from_config("projects/my_task/config.yaml")
    mode, temp, llms = spec.schedule(iteration=0)
    X_discover, X_validate, X_eval = spec.load_data_fn(data_path=spec.io["data_path"], ...)

    # Re-run: reproduce from saved record
    spec = TaskSpec.from_record("runs/2024-03-15/10-30-45/task_spec_record.yaml")
    # same as above

    # Save record for reproducibility
    spec.save_record(run_dir)
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..evolution.program import Program
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
        Load and merge configs, import project modules, capture git state.

        Args:
            config_path: Path to projects/<task_name>/config.yaml

        Returns:
            TaskSpec with all fields populated from merged config and project imports
        """
        from .config import build_task_spec_from_config
        return build_task_spec_from_config(config_path)

    @classmethod
    def from_record(cls, record_path: Path) -> TaskSpec:
        """
        Reconstruct a TaskSpec from a saved record file.

        Args:
            record_path: Path to task_spec_record.yaml from a previous run

        Returns:
            TaskSpec with identical settings as the original run (requires project code to still exist on disk)
        """
        from .record import load_record
        return load_record(record_path)

    # ── record ──

    def save_record(self, run_dir: Path) -> Path:
        """
        Write a record that from_record can read back.

        Args:
            run_dir: Directory to save record.yaml into

        Returns:
            Path to the saved record file
        """
        from .record import save_record
        return save_record(self, run_dir)

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
