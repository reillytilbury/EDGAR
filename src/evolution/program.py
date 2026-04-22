"""
program.py

A Program is one evolved candidate. Fields:
- uid: unique identifier (iteration, birth_island, batch_index)
- idx: global Population index, set automatically when added to a Population (None until then)
- model_code: JAX source for the model function (dataset dict, params dict) -> output
- param_est_code: JAX source for the parameter estimator (dataset dict) -> params dict
- parent_ids: list of uids of this program's parents
- llm_name: name of the LLM that generated this program
- train_sample_loss: cross-validated loss on the training sample (default: inf)
- test_sample_loss: cross-validated loss on the test sample (default: inf)
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
    # p.train_sample_loss == inf, p.test_sample_loss == inf

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
    model_code:     str | None
    param_est_code: str | None
    parent_ids:     list[tuple[int, int, int]] = field(default_factory=list)
    llm_name:       str | None = None
    train_sample_loss: float = float("inf")
    test_sample_loss:  float = float("inf")
    eval_fingerprint:  np.ndarray | None = field(default=None, repr=False)
    n_params:          int | None = None
    mode:              str | None = None
    temperature:       float | None = None
    removal_reason:    dict | None = None

    def compile(self) -> tuple[Callable, Callable]:
        model_fn     = load_function_from_source(self.model_code,     MODEL_ENTRYPOINT)
        param_est_fn = load_function_from_source(self.param_est_code, PARAM_EST_ENTRYPOINT)
        if model_fn is None:
            raise ValueError(f"{self.uid}: could not load '{MODEL_ENTRYPOINT}'")
        if param_est_fn is None:
            raise ValueError(f"{self.uid}: could not load '{PARAM_EST_ENTRYPOINT}'")
        return model_fn, param_est_fn

    def count_params(self) -> int:
        model_fn, _ = self.compile()
        self.n_params = sum(np.asarray(v).size for v in model_fn.DEFAULT_PARAMS.values())
        return self.n_params
