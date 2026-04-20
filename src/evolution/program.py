"""
program.py

A Program is one evolved candidate: two JAX source strings (model + param
estimator), where it came from, and its scores.

Scoring protocol
----------------
Params are initialised by param_est_code, then refined by gradient descent
on train trials.  The resulting params are scored on test trials.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Callable
from ..llm.code_loading import load_function_from_source
MODEL_ENTRYPOINT     = "model"
PARAM_EST_ENTRYPOINT = "parameter_estimator"


@dataclass
class Program:
    uid:            tuple[int, int, int]   # (iteration, island, batch_index)
    parent_ids: list[tuple[int, int, int]] = field(default_factory=list)
    model_code:     str | None
    param_est_code: str | None
    llm_name:   str | None = None
    train_sample_loss: float | None = None
    test_sample_loss: float | None = None

    def compile(self) -> tuple[Callable, Callable]:
        """Turn the source strings into callable functions."""
        model_fn     = load_function_from_source(self.model_code,     MODEL_ENTRYPOINT)
        param_est_fn = load_function_from_source(self.param_est_code, PARAM_EST_ENTRYPOINT)
        if model_fn is None:
            raise ValueError(f"{self.uid}: could not load '{MODEL_ENTRYPOINT}'")
        if param_est_fn is None:
            raise ValueError(f"{self.uid}: could not load '{PARAM_EST_ENTRYPOINT}'")
        return model_fn, param_est_fn

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str) -> Program:
        with open(path) as f:
            d = json.load(f)
        d["uid"]        = tuple(d["uid"])
        d["parent_ids"] = [tuple(p) for p in d["parent_ids"]]
        return cls(**d)