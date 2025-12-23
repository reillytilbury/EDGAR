"""Core domain objects for the AI scientist pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence

import random


ArrayLike = Any


@dataclass
class Program:
    """Container for a single generated model and its metadata."""

    function_code_string: str
    function: Callable
    parameter_estimator_code_string: str
    parameter_estimator: Callable
    generation: int
    birth_island: int
    batch_index: int
    train_loss: float
    params: Optional[ArrayLike] = None
    initial_loss: Optional[float] = None
    initial_params: Optional[ArrayLike] = None
    test_loss: Optional[float] = None
    mean_loss: Optional[float] = None
    llm_name: Optional[str] = None
    parent1_id: Optional[tuple[int, int, int]] = None
    parent2_id: Optional[tuple[int, int, int]] = None
    evaluation_matrix: Optional[ArrayLike] = None
    record_index: Optional[int] = None

    def identifier(self) -> tuple[int, int, int]:
        return self.generation, self.birth_island, self.batch_index

    @property
    def param_count(self) -> int:
        if self.params is None:
            return 0
        if hasattr(self.params, "shape") and len(self.params.shape) > 1:
            return int(self.params.shape[1])
        return 0

    def as_dict(self) -> dict[str, Any]:
        """Return a serialisable mapping for logging/dataframe conversion."""
        return {
            "function_code_string": self.function_code_string,
            "function": self.function,
            "parameter_estimator_code_string": self.parameter_estimator_code_string,
            "parameter_estimator": self.parameter_estimator,
            "generation": self.generation,
            "birth_island": self.birth_island,
            "batch_index": self.batch_index,
            "train_loss": self.train_loss,
            "test_loss": self.test_loss,
            "mean_loss": self.mean_loss,
            "llm_name": self.llm_name,
            "params": self.params,
            "initial_loss": self.initial_loss,
            "initial_params": self.initial_params,
            "parent1_id": self.parent1_id,
            "parent2_id": self.parent2_id,
            "evaluation_matrix": self.evaluation_matrix,
            "record_index": self.record_index,
        }

    def prompt_metadata(self) -> dict[str, Any]:
        """Minimal metadata needed to build LLM prompts."""
        return {
            "train_loss": self.train_loss,
            "function_code_string": self.function_code_string,
            "parameter_estimator_code_string": self.parameter_estimator_code_string,
        }

    def clone(self) -> "Program":
        return Program(
            function_code_string=self.function_code_string,
            function=self.function,
            parameter_estimator_code_string=self.parameter_estimator_code_string,
            parameter_estimator=self.parameter_estimator,
            generation=self.generation,
            birth_island=self.birth_island,
            batch_index=self.batch_index,
            train_loss=self.train_loss,
            params=self.params,
            initial_loss=self.initial_loss,
            initial_params=self.initial_params,
            test_loss=self.test_loss,
            mean_loss=self.mean_loss,
            llm_name=self.llm_name,
            parent1_id=self.parent1_id,
            parent2_id=self.parent2_id,
            evaluation_matrix=self.evaluation_matrix,
            record_index=self.record_index,
        )


class Island:
    """Thin wrapper around a list of programs with convenience helpers."""

    def __init__(self, island_id: int, programs: Optional[Sequence[Program]] = None):
        self.island_id = island_id
        self._programs: List[Program] = list(programs or [])

    def __iter__(self):
        return iter(self._programs)

    def __len__(self) -> int:
        return len(self._programs)

    def __getitem__(self, item):
        return self._programs[item]

    @property
    def programs(self) -> list[Program]:
        return self._programs

    def add(self, program: Program) -> None:
        self._programs.append(program)

    def extend(self, programs: Iterable[Program]) -> None:
        self._programs.extend(programs)

    def sample(self, k: int, replace: bool = False) -> list[Program]:
        if not self._programs:
            return []
        if replace:
            return [random.choice(self._programs) for _ in range(k)]
        k = min(k, len(self._programs))
        return random.sample(self._programs, k)

    def sort_by(self, attr: str = "train_loss") -> None:
        self._programs.sort(key=lambda p: getattr(p, attr))

    def drop_indices(self, indices: Sequence[int]) -> None:
        drop_set = set(indices)
        self._programs = [p for idx, p in enumerate(self._programs) if idx not in drop_set]

    def to_records(self) -> list[dict[str, Any]]:
        return [p.as_dict() for p in self._programs]

    def to_prompt_records(self) -> list[dict[str, Any]]:
        return [p.prompt_metadata() for p in self._programs]

@dataclass
class ProgramSnapshot:
    """Serializable view of a program for census/history tracking."""

    program_index: int
    generation: int
    birth_island: int
    batch_index: int
    train_loss: float
    test_loss: Optional[float]
    llm_name: Optional[str]
    parent1_id: Optional[tuple[int, int, int]]
    parent2_id: Optional[tuple[int, int, int]]
    timestamp: float
    param_count: int
    function_code_string: str
    parameter_estimator_code_string: str
    evaluation_matrix: Optional[ArrayLike]
    is_seed: bool = False
    notes: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_index": self.program_index,
            "generation": self.generation,
            "birth_island": self.birth_island,
            "batch_index": self.batch_index,
            "train_loss": self.train_loss,
            "test_loss": self.test_loss,
            "llm_name": self.llm_name,
            "parent1_id": self.parent1_id,
            "parent2_id": self.parent2_id,
            "timestamp": self.timestamp,
            "param_count": self.param_count,
            "function_code_string": self.function_code_string,
            "parameter_estimator_code_string": self.parameter_estimator_code_string,
            "evaluation_matrix": self.evaluation_matrix,
            "is_seed": self.is_seed,
            "notes": self.notes,
        }

    @classmethod
    def from_program(
        cls,
        program: Program,
        program_index: int,
        timestamp: float,
        is_seed: bool = False,
        notes: Optional[str] = None,
    ) -> "ProgramSnapshot":
        return cls(
            program_index=program_index,
            generation=program.generation,
            birth_island=program.birth_island,
            batch_index=program.batch_index,
            train_loss=float(program.train_loss),
            test_loss=float(program.test_loss) if program.test_loss is not None else None,
            llm_name=program.llm_name,
            parent1_id=program.parent1_id,
            parent2_id=program.parent2_id,
            timestamp=timestamp,
            param_count=program.param_count,
            function_code_string=program.function_code_string,
            parameter_estimator_code_string=program.parameter_estimator_code_string,
            evaluation_matrix=program.evaluation_matrix,
            is_seed=is_seed,
            notes=notes,
        )
