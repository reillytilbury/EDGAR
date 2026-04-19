"""
program.py — Central Program data structure for EDGAR.

A Program represents a single generated candidate: its JAX code strings
for both model and parameter estimator, the compiled callables, fitted
parameters, and evaluation scores.

Programs are created during generation (code only), filled in after
scoring (losses + params), then added to the global Population.
All island membership is tracked by index — programs themselves are never
copied or moved.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..llm.code_loading import load_function_from_source


# Entrypoint names the LLM is expected to produce
MODEL_ENTRYPOINT = "model"
PARAM_EST_ENTRYPOINT = "parameter_estimator"


# ---------------------------------------------------------------------------
# Param serialisation helpers
# ---------------------------------------------------------------------------

def _params_to_json(params: Any) -> Any:
    """Convert a JAX/numpy pytree to a JSON-serialisable nested structure."""
    if params is None:
        return None
    return jax.tree_util.tree_map(lambda x: np.asarray(x).tolist(), params)


def _params_from_json(params_json: Any) -> Any:
    """Reconstruct a JAX pytree from the output of _params_to_json."""
    if params_json is None:
        return None
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params_json)


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------

@dataclass
class Program:
    """
    A single evolved model candidate.

    Fields
    ------
    model_code, param_est_code
        JAX source for the model and parameter estimator functions.
        Both are expected to be JAX-compatible from the start.

    model_fn, param_est_fn
        Compiled callables.  Populated by calling compile(); not serialised.

    iteration, birth_island, batch_index
        Coordinates identifying where/when this program was generated.

    parent_ids
        UIDs of programs used as parents during generation.

    train_loss, test_loss
        Scalar losses. train_loss defaults to inf (unscored).

    params, initial_params
        Fitted JAX parameter pytrees (leading sample axis).

    evaluation_matrix
        Model predictions on the shared evaluation grid, used for
        behavioural deduplication.
    """

    # Code (JAX, both components)
    model_code: str | None
    param_est_code: str | None

    # Provenance
    iteration: int = -1
    birth_island: int = -1
    batch_index: int = -1
    parent_ids: list[tuple[int, int, int]] = field(default_factory=list)
    llm_name: str | None = None
    mode: str | None = None

    # Compiled callables — not serialised, populated by compile()
    model_fn: Callable | None = field(default=None, repr=False)
    param_est_fn: Callable | None = field(default=None, repr=False)

    # Scores (filled after evaluation)
    train_loss: float = float("inf")
    test_loss: float | None = None
    initial_loss: float | None = None
    params: Any = field(default=None, repr=False)
    initial_params: Any = field(default=None, repr=False)
    n_params: int | None = None

    # Diagnostics
    evaluation_matrix: Any = field(default=None, repr=False)
    optimization_time_s: float | None = None
    image_prompt_path: str | None = None
    train_fit_image_path: str | None = None
    test_fit_image_path: str | None = None

    # -----------------------------------------------------------------------
    # Identity
    # -----------------------------------------------------------------------

    @property
    def uid(self) -> tuple[int, int, int]:
        """Unique run-scoped identifier: (iteration, birth_island, batch_index)."""
        return (self.iteration, self.birth_island, self.batch_index)

    @property
    def code_hash(self) -> str:
        """Fast hash over both code strings for exact-duplicate detection."""
        src = (self.model_code or "") + (self.param_est_code or "")
        return hashlib.md5(src.encode()).hexdigest()

    # -----------------------------------------------------------------------
    # State queries
    # -----------------------------------------------------------------------

    def is_compiled(self) -> bool:
        return self.model_fn is not None and self.param_est_fn is not None

    def is_scored(self) -> bool:
        return self.train_loss is not None and np.isfinite(float(self.train_loss))

    def is_valid(self) -> bool:
        """Compiled and has a finite train loss."""
        return self.is_compiled() and self.is_scored()

    # -----------------------------------------------------------------------
    # Compilation
    # -----------------------------------------------------------------------

    def compile(self) -> bool:
        """
        Execute code strings and bind the named entrypoints to model_fn / param_est_fn.

        Returns True if both callables were loaded successfully.
        """
        self.model_fn = load_function_from_source(self.model_code, MODEL_ENTRYPOINT)
        self.param_est_fn = load_function_from_source(self.param_est_code, PARAM_EST_ENTRYPOINT)
        if self.model_fn is None:
            logging.info("Program %s: failed to compile model_code.", self.uid)
        if self.param_est_fn is None:
            logging.info("Program %s: failed to compile param_est_code.", self.uid)
        return self.is_compiled()

    # -----------------------------------------------------------------------
    # Behavioural similarity (for deduplication)
    # -----------------------------------------------------------------------

    def is_similar_to(
        self,
        other: Program,
        loss_tol: float = 0.01,
        cosine_tol: float = 0.95,
    ) -> bool:
        """
        True if this program is behaviourally equivalent to *other*.

        Checks, in order:
        1. Identical model code → duplicate.
        2. Different parameter-tree signature → not duplicate.
        3. Losses too far apart → not duplicate (cheap fast path).
        4. Cosine similarity of evaluation matrices ≥ cosine_tol → duplicate.
        """
        if self.model_code and self.model_code == other.model_code:
            return True

        if self.params is not None and other.params is not None:
            from .. import utils
            if utils.params_signature(self.params) != utils.params_signature(other.params):
                return False

        if self.is_scored() and other.is_scored():
            denom = max(abs(self.train_loss), abs(other.train_loss), 1e-6)
            if abs(self.train_loss - other.train_loss) / denom > loss_tol:
                return False

        if self.evaluation_matrix is None or other.evaluation_matrix is None:
            return False

        y_a = jnp.asarray(self.evaluation_matrix)
        y_b = jnp.asarray(other.evaluation_matrix)
        if y_a.shape != y_b.shape:
            return False

        # Ensure 2-D: (n_cells, n_eval_points)
        if y_a.ndim == 1:
            y_a, y_b = y_a.reshape(1, -1), y_b.reshape(1, -1)

        norm_a = jnp.linalg.norm(y_a, axis=1, keepdims=True)
        norm_b = jnp.linalg.norm(y_b, axis=1, keepdims=True)
        norm_a = jnp.where(norm_a == 0, 1.0, norm_a)
        norm_b = jnp.where(norm_b == 0, 1.0, norm_b)

        cosine = jnp.sum((y_a / norm_a) * (y_b / norm_b), axis=1)
        return float(jnp.mean(cosine)) >= cosine_tol

    # -----------------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------------

    def to_record(self) -> dict:
        """
        Return a JSON-safe dict for JSONL logging.

        Callables are omitted — use from_record(..., compile=True) to restore them.
        JAX/numpy arrays are converted to nested Python lists.
        """
        return {
            "iteration": self.iteration,
            "birth_island": self.birth_island,
            "batch_index": self.batch_index,
            "parent_ids": [list(p) for p in self.parent_ids],
            "llm_name": self.llm_name,
            "mode": self.mode,
            "model_code": self.model_code,
            "param_est_code": self.param_est_code,
            "train_loss": float(self.train_loss) if self.train_loss is not None else None,
            "test_loss": float(self.test_loss) if self.test_loss is not None else None,
            "initial_loss": float(self.initial_loss) if self.initial_loss is not None else None,
            "n_params": self.n_params,
            "params": _params_to_json(self.params),
            "initial_params": _params_to_json(self.initial_params),
            "evaluation_matrix": _params_to_json(self.evaluation_matrix),
            "optimization_time_s": self.optimization_time_s,
            "image_prompt_path": self.image_prompt_path,
            "train_fit_image_path": self.train_fit_image_path,
            "test_fit_image_path": self.test_fit_image_path,
        }

    @classmethod
    def from_record(cls, record: dict, compile: bool = True) -> Program:
        """
        Reconstruct a Program from a to_record() dict.

        Args:
            record:  Dict as written by to_record().
            compile: If True, immediately call compile() to restore callables.
        """
        program = cls(
            model_code=record.get("model_code"),
            param_est_code=record.get("param_est_code"),
            iteration=record.get("iteration", -1),
            birth_island=record.get("birth_island", -1),
            batch_index=record.get("batch_index", -1),
            parent_ids=[tuple(p) for p in record.get("parent_ids") or []],
            llm_name=record.get("llm_name"),
            mode=record.get("mode"),
            train_loss=record.get("train_loss", float("inf")),
            test_loss=record.get("test_loss"),
            initial_loss=record.get("initial_loss"),
            n_params=record.get("n_params"),
            params=_params_from_json(record.get("params")),
            initial_params=_params_from_json(record.get("initial_params")),
            evaluation_matrix=_params_from_json(record.get("evaluation_matrix")),
            optimization_time_s=record.get("optimization_time_s"),
            image_prompt_path=record.get("image_prompt_path"),
            train_fit_image_path=record.get("train_fit_image_path"),
            test_fit_image_path=record.get("test_fit_image_path"),
        )
        if compile:
            program.compile()
        return program
