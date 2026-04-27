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
        """Load + merge configs, import project modules, capture git state."""
        from .config import build_task_spec_from_config
        return build_task_spec_from_config(config_path)

    @classmethod
    def from_record(cls, record_path: Path) -> TaskSpec:
        """Reconstruct a TaskSpec from a saved record."""
        from .record import load_record
        return load_record(record_path)

    # ── record ──

    def save_record(self, run_dir: Path) -> Path:
        """Write a record that from_record can read back."""
        from .record import save_record
        return save_record(self, run_dir)

    # ── iteration schedule ──

    def schedule(self, iteration: int) -> tuple[str, float, LLMs]:
        """
        Returns (mode, temperature, llms) for a given iteration.

        mode: "explore" for the first half, "exploit" for the second half
        temperature: starts near 2.0 and decays toward ~1.37
        llms: which model to use for each LLM task, cycling through sequences
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
        """Merge all config sections into a single dict for prompt variable lookup."""
        return {**self.evolution, **self.llms, **self.scoring}

    # ── prompt schemas ──

    @property
    def prompt_schemas(self) -> PromptSchemas:
        return PromptSchemas(
            model=self.model_prompt_schema,
            param_est=self.param_est_prompt_schema,
            jax=self.jax_prompt_schema,
        )
