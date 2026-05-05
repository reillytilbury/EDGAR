"""
program.py

A Program is one evolved candidate. Top-level fields:
- birth: BirthCertificate — lineage and generation-time metadata
- code: Code — numpy source for model and param_est
- code_jax: Code — JAX-translated source for model and param_est
- name: descriptive model name from the LLM (e.g. "Double Gaussian Model")
- program_losses: Losses — per-split init/final scalar losses (include complexity penalty)
- sample_losses: per-sample cross-validated losses (no penalty), shape (n_samples,)
- n_params: total number of model parameters, set by count_params()
- eval_fingerprint: array of model outputs used for deduplication
- idx: global Population index, set automatically when added to a Population

Methods:
- compile: parse source strings into callable (model_fn, param_est_fn)
- count_params: compile model, count parameters, cache in self.n_params

Save and load is handled by Population, not Program directly.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from ..llm.code_loading import load_function_from_source

MODEL_ENTRYPOINT     = "model"
PARAM_EST_ENTRYPOINT = "parameter_estimator"


@dataclass
class BirthCertificate:
    generation: int
    island: int
    batch_index: int
    mode: str | None = None
    temperature: float | None = None
    parent_indices: list[int] = field(default_factory=list)
    llm_name: str | None = None


@dataclass
class Code:
    model: str | None = None
    param_est: str | None = None


@dataclass
class LossPair:
    init: float | None = None
    final: float | None = None


@dataclass
class Losses:
    discover: LossPair = field(default_factory=LossPair)
    validate: LossPair = field(default_factory=lambda: LossPair(init=None, final=float("inf")))


@dataclass
class Program:
    birth:            BirthCertificate
    code:             Code = field(default_factory=Code)
    code_jax:         Code = field(default_factory=Code)
    name:             str | None = None
    program_losses:   Losses = field(default_factory=Losses)
    n_params:         int | None = None
    eval_fingerprint: np.ndarray | None = field(default=None, repr=False)
    params:           dict | None = field(default=None, repr=False)
    sample_losses:    np.ndarray | None = field(default=None, repr=False)
    image_path:       str | None = None
    idx:              int | None = field(default=None, init=False)

    def compile(self) -> tuple[Callable, Callable]:
        """Compile JAX source into callable (model_fn, param_est_fn)."""
        model_fn = load_function_from_source(self.code_jax.model, MODEL_ENTRYPOINT)
        param_est_fn = load_function_from_source(self.code_jax.param_est, PARAM_EST_ENTRYPOINT)
        if model_fn is None:
            raise ValueError(f"{self.birth}: could not load '{MODEL_ENTRYPOINT}'")
        if param_est_fn is None:
            raise ValueError(f"{self.birth}: could not load '{PARAM_EST_ENTRYPOINT}'")
        return model_fn, param_est_fn

    def count_params(self) -> int:
        model_fn, _ = self.compile()
        self.n_params = sum(np.asarray(v).size for v in model_fn.DEFAULT_PARAMS.values())
        return self.n_params

    # ── prompt template properties ──
    # These match the parent_vars used in prompt_defaults.yaml so that
    # getattr(program, var_name) works in PromptSchema.build_prompt.

    @property
    def descriptive_name(self) -> str:
        return self.name or ""

    @property
    def loss_discover(self) -> float | str:
        v = self.program_losses.discover.final
        return v if v is not None else "not yet scored"

    @property
    def model_code(self) -> str:
        return self.code.model or ""

    @property
    def param_est_code(self) -> str:
        return self.code.param_est or ""
