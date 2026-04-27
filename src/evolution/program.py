"""
program.py

A Program is one evolved candidate. Fields:
- uid: unique identifier (iteration, birth_island, batch_index)
- idx: global Population index, set automatically when added to a Population (None until then)
- model_code: numpy source for the model function (dataset dict, params dict) -> output
- param_est_code: numpy source for the parameter estimator (dataset dict) -> params dict
- model_code_jax: JAX-translated source for the model function
- param_est_code_jax: JAX-translated source for the parameter estimator
- parent_ids: list of uids of this program's parents
- llm_name: name of the LLM that generated this program
- loss_discover: loss on the discovery data (default: inf)
- loss_validate: loss on the held-out validation data (default: inf)
- initial_loss: loss of initial params before gradient descent (default: inf)
- eval_fingerprint: array of model outputs used for deduplication
- n_params: total number of model parameters, set by calling count_params()
- mode: LLM generation mode, e.g. "explore" or "exploit"
- temperature: LLM sampling temperature used during generation
- removal_reason: dict describing why this program was removed from an island (None if still active)

Methods:
- compile: parse source strings into callable (model_fn, param_est_fn)
- count_params: compile model, count parameters, cache in self.n_params

Save and load is handled by Population, not Program directly.

Example usage:
--------------
    p = Program(
        uid=(0, 0, 0),
        model_code="...",
        param_est_code="...",
        parent_ids=[(0, 0, 0), (0, 0, 1)],
        llm_name="claude-sonnet-4-6",
    )
    # p.idx is None until added to a Population
    # p.loss_discover == inf, p.loss_validate == inf

    model_fn, param_est_fn = p.compile()
    params = param_est_fn(dataset)
    output = model_fn(dataset, params)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from ..llm.code_loading import load_function_from_source

MODEL_ENTRYPOINT     = "model"
PARAM_EST_ENTRYPOINT = "parameter_estimator"

@dataclass
class Program:
    uid:            tuple[int, int, int]     # (iteration, island, batch_index)
    idx:            int | None = field(default=None, init=False)
    model_code:         str | None
    param_est_code:     str | None
    model_code_jax:     str | None = None
    param_est_code_jax: str | None = None
    descriptive_name:   str | None = None
    parent_ids:     list[tuple[int, int, int]] = field(default_factory=list)
    llm_name:       str | None = None
    loss_discover:     float = float("inf")
    loss_validate:     float = float("inf")
    initial_loss:      float = float("inf")
    eval_fingerprint:  np.ndarray | None = field(default=None, repr=False)
    n_params:          int | None = None
    mode:              str | None = None
    temperature:       float | None = None
    removal_reason:    dict | None = None

    def compile(self) -> tuple[Callable, Callable]:
        """Compile JAX source into callable (model_fn, param_est_fn).

        Uses the JAX-translated code if available, falls back to numpy source.
        """
        model_src = self.model_code_jax or self.model_code
        param_est_src = self.param_est_code_jax or self.param_est_code
        model_fn = load_function_from_source(model_src, MODEL_ENTRYPOINT)
        param_est_fn = load_function_from_source(param_est_src, PARAM_EST_ENTRYPOINT)
        if model_fn is None:
            raise ValueError(f"{self.uid}: could not load '{MODEL_ENTRYPOINT}'")
        if param_est_fn is None:
            raise ValueError(f"{self.uid}: could not load '{PARAM_EST_ENTRYPOINT}'")
        return model_fn, param_est_fn

    def count_params(self) -> int:
        model_fn, _ = self.compile()
        self.n_params = sum(np.asarray(v).size for v in model_fn.DEFAULT_PARAMS.values())
        return self.n_params
