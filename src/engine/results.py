from dataclasses import dataclass, field
from typing import Any, Callable


class ModelGenerationResult:
    numpy_code: str | None
    prompt: str | None
    llm_response: str | None
    jax_code: str | None = None
    jax_callable: Callable | None = None
    jax_prompt: str | None = None
    jax_raw_response: str | None = None


class ParamEstimatorGenerationResult:
    code: str | None
    callable_obj: Callable | None
    metadata: dict[str, Any] = field(default_factory=dict)


class CandidateGenerationResult:
    model: ModelGenerationResult
    param_estimator: ParamEstimatorGenerationResult


