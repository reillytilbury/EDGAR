"""
program.py

A Program is one evolved candidate. Top-level fields:
- birth: BirthCertificate — lineage and generation-time metadata
- code: Code — numpy source for model, param_est and jax model code
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
import warnings
from dataclasses import dataclass, field
from typing import Callable
from ..llm.code_loading import load_function_from_source

MODEL_ENTRYPOINT     = "model"
PARAM_EST_ENTRYPOINT = "parameter_estimator"

class ModelLoadingError(Exception): pass
class ParamEstLoadingError(Exception): pass

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
    model_jax: str | None = None

class NotValidated:
    """ Marker for programs that haven't been prepared for validation scoring.
        See, prepare_validation_scoring"""
    def __repr__(self): return "NOTVALIDATED"
    def __float__(self): raise TypeError("Invalid operation on unset loss")

@dataclass
class LossPair:
    init: float | None = None
    final: float | NotValidated | None = None


@dataclass
class Losses:
    discover: LossPair = field(default_factory=LossPair)
    validate: LossPair = field(default_factory=lambda: LossPair(init=None, final=NotValidated()))

@dataclass
class Program:
    birth:            BirthCertificate
    code:             Code = field(default_factory=Code)
    name:             str | None = None
    program_losses:   Losses = field(default_factory=Losses)
    n_params:         int | None = None
    eval_fingerprint: np.ndarray | None = field(default=None, repr=False)
    params:           dict | None = field(default=None, repr=False)
    sample_losses:    np.ndarray | None = field(default=None, repr=False)
    image_path:       str | None = None
    idx:              int | None = field(default=None, init=False)
    rank:             int | None = None
    _default_params:  dict | None = None

    #Called on initialization after default __init__ 
    def __post_init__(self):
        if self._default_params is not None:
            self.default_params = self._default_params #use the setter to validate and set n_params

    def compile_model(self) -> Callable:
        """Load JAX model callable. Raises ModelLoadingError if source is missing or invalid."""
        model_fn = load_function_from_source(self.code.model_jax, MODEL_ENTRYPOINT)
        if model_fn is None:
            raise ModelLoadingError(f"{self.birth}: could not load '{MODEL_ENTRYPOINT}'")
        return model_fn

    def compile_param_est(self) -> Callable:
        """Load parameter_estimator callable. Raises ParamEstLoadingError if source is missing or invalid."""
        param_est_fn = load_function_from_source(self.code.param_est, PARAM_EST_ENTRYPOINT)
        if param_est_fn is None:
            raise ParamEstLoadingError(f"{self.birth}: could not load '{PARAM_EST_ENTRYPOINT}'")
        return param_est_fn

    # ── prompt template properties ──
    # These match the program_vars used in prompt_defaults.yaml so that
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

    @property
    def default_params(self) -> dict:
        if self._default_params is None:
            warnings.warn(f"Accessing default_params=None of Program #{self.idx}, default_params was not set, or setting failed", UserWarning)
            #We don't raise an error as this program will be assigned infinite loss during scoring
        return self._default_params
    
    @default_params.setter
    def default_params(self, default_params_dict: dict):
        """
            Set the default parameters dictionary for the program.
            Counts the number of parameters and caches it in self.n_params
        """
        try:
            self._default_params = default_params_dict
            self.n_params = sum(np.asarray(v).size for v in default_params_dict.values())
        except Exception as e:
            warnings.warn(f"Failed to set default_params for Program #{self.idx}: {e}", UserWarning)
            #We don't raise an error as this program will be assigned infinite loss during scoring
            self._default_params = None
            self.n_params = None
