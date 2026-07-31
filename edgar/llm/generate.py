"""
LLM generation functions.

This module orchestrates the generation of new program components (model code, parameter estimators,
and JAX translations) using Large Language Models. It identifies programs within a population
that require specific code generation and dispatches asynchronous LLM calls to fill in their fields.

Prompt variable mapping and dynamic prompt construction are handled by `PromptSchema` and are not
directly implemented in this module. This module focuses on the execution of the generation tasks
and the integration of results back into the `Program` objects.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any, TYPE_CHECKING, Callable
import numpy as np

from pydantic_ai.models import Model
from pydantic import BaseModel

from ..evolution.program import Program
from ..evolution.population import Population
from ..scoring.utils import _safe_loss
from ..llm.prompt_schema import PromptSchema
from ..llm.llm_calling import call_llm
from ..io.config import RetryConfig
from ..llm.response_schema import ModelSchema, ParamEstSchema, TranslationSchema
from ..llm.code_loading import load_function_from_source
from ..io.plotting import generate_feedback_image

if TYPE_CHECKING:
    from ..io.task_spec import TaskSpec


# ---------------------------------------------------------------------------
# Filters — which programs need work at each stage
# ---------------------------------------------------------------------------
def _needs_model_code(program: Program) -> bool:
    """Checks if a program requires its numpy model code to be generated.

    Args:
        program: The program to check.

    Returns:
        True if the program's model code is None, False otherwise.
    """
    return program.code.model is None


def _needs_param_est_code(program: Program) -> bool:
    """Checks if a program requires its numpy parameter estimator code to be generated.

    Args:
        program: The program to check.

    Returns:
        True if the program has model code but no parameter estimator code, False otherwise.
    """
    return program.code.model is not None and program.code.param_est is None


def _needs_model_translation(program: Program) -> bool:
    """Checks if a program requires its numpy model code to be translated to JAX.

    Args:
        program: The program to check.

    Returns:
        True if the program has numpy model code but no JAX model code, False otherwise.
    """
    return program.code.model is not None and program.code.model_jax is None


def _filter_programs(
    population: Population, filter_rule: Callable[[Program], bool]
) -> list[Program]:
    """Filters a population of programs based on a given rule.

    Args:
        population: The population of programs to filter.
        filter_rule: A callable that takes a Program and returns True if it
            should be included in the filtered list.

    Returns:
        A list of programs that satisfy the filter rule.
    """
    return [p for p in population if filter_rule(p)]


def _resolve_parents(population: Population, program: Program) -> list[Program]:
    """Resolves and sorts the parent programs for a given program.

    Parents are sorted by their `discover.final` loss in descending order (highest loss first).
    Programs with `None` or non-float losses are treated as having infinite loss.

    Args:
        population: The entire population of programs, used to retrieve parent programs by index.
        program: The program whose parents need to be resolved.

    Returns:
        A list of parent programs, sorted by discover.final loss from highest to lowest.
    """
    parents = [population[i] for i in program.birth.parent_indices]

    def _loss(p: Program) -> float:
        return _safe_loss(p.program_losses.discover.final)

    return sorted(parents, key=_loss, reverse=True)


# ---------------------------------------------------------------------------
# Model code generation
# ---------------------------------------------------------------------------


async def _generate_one_model(
    program: Program,
    parents: list[Program],
    prompt_schema: PromptSchema,
    llm: str | Model,
    mode: str,
    temperature: float,
    config: dict[str, Any] | None = None,
    spec: TaskSpec | None = None,
    data: dict | None = None,
    output_schema: type[BaseModel] = ModelSchema,
) -> None:
    """Generates the numpy model code for a single program using an LLM.

    This function constructs a prompt using the provided `prompt_schema`, potentially including
    image-based feedback, calls the specified LLM, and then updates the program's model code,
    default parameters, and descriptive name based on the LLM's response.

    Args:
        program: The program for which to generate the model code. This program object
            will be mutated with the generated code and metadata.
        parents: A list of parent programs used for contextualizing the LLM prompt.
        prompt_schema: The schema used to build the LLM prompt.
        llm: The LLM model name (str) or a pre-configured PydanticAI Model instance.
        mode: The generation mode (e.g., "explore" or "exploit"), affecting prompt content.
        temperature: The sampling temperature for the LLM.
        config: Optional configuration dictionary containing LLM call parameters like
            `log_raw_llm_response`, `max_tokens`, and `retry_config`.
        spec: The `TaskSpec` object, required for generating image feedback.
        data: The data dictionary, required for generating image feedback.

    Returns:
        None. The `program` object is mutated in-place.

    Raises:
        UserWarning: If `call_llm` returns None, indicating a failure in LLM interaction,
            and the program's model code generation is skipped.
    """
    image_bytes = generate_feedback_image(spec, data, parents, program)
    cfg = dict(config or {})
    program.birth.ideas = prompt_schema.select_ideas(cfg, spec.rng)
    prompt = prompt_schema.build_prompt(mode, parents, cfg)
    result = await call_llm(
        prompt=prompt,
        llm_model=llm,
        output_type=output_schema,
        temperature=temperature,
        image_bytes=image_bytes,
        log_raw_llm_response=cfg.get("log_raw_llm_response", False),
        max_tokens=cfg.get("max_tokens"),
        retry_config=cfg.get("retry_config"),
        role="model",
    )
    if result is None:
        warnings.warn(
            f"[generate] Skipping model code for program #{program.idx}: call_llm returned None"
        )
        return
    header = f'"""\n{result.thought_process}\n"""\n\n'
    program.code.model = header + result.code
    default_params = result.default_params
    if isinstance(default_params, str):
        try:
            default_params = eval(default_params, {"np": np})
            # Debatch data for default_params resolution, since model expects data of shape (n1, n2, ...) not (n_samples, n1, n2, ...)
            program.data = (
                {k: v[0] for k, v in data.items()} if data is not None else None
            )
        except Exception as e:
            warnings.warn(
                f"Failed to evaluate default_params for Program #{program.idx}: {e}",
                UserWarning,
            )
            default_params = None

    program.default_params = default_params
    program.name = result.descriptive_name
    program.birth.llm_name = llm


async def generate_models(
    population: Population,
    prompt_schema: PromptSchema,
    llm: str | Model,
    mode: str,
    temperature: float,
    config: dict[str, Any] | None = None,
    spec: TaskSpec | None = None,
    data: dict | None = None,
    output_schema: type[BaseModel] = ModelSchema,
) -> None:
    """Asynchronously generates numpy model code for all programs that currently lack it.

    This function identifies programs requiring model code, resolves their parents, and
    then dispatches concurrent LLM calls via `_generate_one_model`.

    Args:
        population: The entire population of programs. Programs within this population
            that satisfy `_needs_model_code` will be updated.
        prompt_schema: The schema used to build the LLM prompt for model generation.
        llm: The LLM model name (str) or a pre-configured PydanticAI Model instance.
        mode: The generation mode (e.g., "explore" or "exploit").
        temperature: The sampling temperature for the LLM.
        config: Optional configuration dictionary for LLM calls.
        spec: The `TaskSpec` object, used for generating prompt images and accessing
            the output directory.
        data: The data dictionary, used for generating prompt images.

    Returns:
        None. The `program` objects within the `population` are mutated in-place:
        `program.code.model`, `program.name`, `program.birth.llm_name`,
        and `program.image_path` (if image feedback is enabled) are updated.
    """
    programs = _filter_programs(population, _needs_model_code)
    await asyncio.gather(
        *[
            _generate_one_model(
                p,
                _resolve_parents(population, p),
                prompt_schema,
                llm,
                mode,
                temperature,
                config,
                spec,
                data,
                output_schema=output_schema,
            )
            for p in programs
        ]
    )


# ---------------------------------------------------------------------------
# Parameter estimator code generation
# ---------------------------------------------------------------------------


async def _generate_one_param_est(
    program: Program,
    parents: list[Program],
    prompt_schema: PromptSchema,
    llm: str | Model,
    config: dict[str, Any] | None = None,
    output_schema: type[BaseModel] = ParamEstSchema,
) -> None:
    """Generates the numpy parameter estimator code for a single program using an LLM.

    This function builds a prompt, calls the specified LLM, and then updates the
    program's parameter estimator code based on the LLM's response. The temperature
    for this LLM call is fixed at 1.0.

    Args:
        program: The program for which to generate the parameter estimator code.
            This program object will be mutated with the generated code.
        parents: A list of parent programs used for contextualizing the LLM prompt.
        prompt_schema: The schema used to build the LLM prompt.
        llm: The LLM model name (str) or a pre-configured PydanticAI Model instance.
        config: Optional configuration dictionary containing LLM call parameters like
            `log_raw_llm_response`, `max_tokens`, and `retry_config`.

    Returns:
        None. The `program` object is mutated in-place.

    Raises:
        UserWarning: If `call_llm` returns None, indicating a failure in LLM interaction,
            and the program's parameter estimator code generation is skipped.
    """
    cfg = config or {}
    prompt = prompt_schema.build_prompt(
        "explore", parents, config, current_program=program
    )
    result = await call_llm(
        prompt=prompt,
        llm_model=llm,
        output_type=output_schema,
        temperature=1.0,
        log_raw_llm_response=cfg.get("log_raw_llm_response", False),
        max_tokens=cfg.get("max_tokens"),
        retry_config=cfg.get("retry_config"),
        role="param_est",
    )
    if result is None:
        warnings.warn(
            f"[generate] Skipping param_est for program #{program.idx}: call_llm returned None"
        )
        return
    program.code.param_est = result.code


async def generate_param_ests(
    population: Population,
    prompt_schema: PromptSchema,
    llm: str | Model,
    config: dict[str, Any] | None = None,
    output_schema: type[BaseModel] = ParamEstSchema,
) -> None:
    """Asynchronously generates numpy parameter estimator code for programs that need it.

    This function identifies programs with existing model code but no parameter estimator,
    resolves their parents, and then dispatches concurrent LLM calls via `_generate_one_param_est`.

    Args:
        population: The entire population of programs. Programs within this population
            that satisfy `_needs_param_est_code` will be updated.
        prompt_schema: The schema used to build the LLM prompt for parameter estimator generation.
        llm: The LLM model name (str) or a pre-configured PydanticAI Model instance.
        config: Optional configuration dictionary for LLM calls.

    Returns:
        None. The `program.code.param_est` field of eligible programs in the `population`
        is mutated in-place.
    """
    programs = _filter_programs(population, _needs_param_est_code)
    await asyncio.gather(
        *[
            _generate_one_param_est(
                p,
                _resolve_parents(population, p),
                prompt_schema,
                llm,
                config,
                output_schema=output_schema,
            )
            for p in programs
        ]
    )


# ---------------------------------------------------------------------------
# JAX translation
# ---------------------------------------------------------------------------
async def _translate_one_model(
    program: Program,
    model_prompt_schema: PromptSchema,
    llm: str | Model,
    retry_config: RetryConfig | None = None,
    max_tokens: int | None = None,
    output_schema: type[BaseModel] = TranslationSchema,
) -> None:
    """Translates the numpy model code of a single program into JAX-compatible code using an LLM.

    This function builds a translation prompt, calls the specified LLM, and if the
    translation is successful and verifiable (i.e., `load_function_from_source`
    can load a 'model' function from it), updates the program's JAX model code.
    The temperature for this LLM call is fixed at 1.0.

    Args:
        program: The program whose numpy model code needs translation. This program object
            will be mutated with the translated JAX code.
        model_prompt_schema: The schema used to build the LLM prompt for JAX translation.
        llm: The LLM model name (str) or a pre-configured PydanticAI Model instance.
        retry_config: Optional retry configuration for the LLM call.
        max_tokens: Optional maximum number of tokens for the LLM response.

    Returns:
        None. The `program` object is mutated in-place if translation is successful.
    """
    model_prompt = model_prompt_schema.build_prompt("explore", current_program=program)
    model_result = await call_llm(
        prompt=model_prompt,
        llm_model=llm,
        output_type=output_schema,
        temperature=1.0,
        retry_config=retry_config,
        max_tokens=max_tokens,
        role="jax",
    )
    # Validate the generated JAX code by attempting to load the function from its source.
    # Only assign if the LLM returned a result and the code is valid.
    if (
        model_result is not None
        and load_function_from_source(model_result.code, "model") is not None
    ):
        program.code.model_jax = model_result.code


async def translate_programs(
    population: Population,
    model_prompt_schema: PromptSchema,
    llm: str | Model,
    retry_config: RetryConfig | None = None,
    max_tokens: int | None = None,
    output_schema: type[BaseModel] = TranslationSchema,
) -> None:
    """Asynchronously translates all untranslated numpy model code to JAX-compatible code.

    This function identifies programs requiring JAX translation and then dispatches
    concurrent LLM calls via `_translate_one_model`.

    Args:
        population: The entire population of programs. Programs within this population
            that satisfy `_needs_model_translation` will be updated.
        model_prompt_schema: The schema used to build the LLM prompt for JAX translation.
        llm: The LLM model name (str) or a pre-configured PydanticAI Model instance.
        retry_config: Optional retry configuration for the LLM calls.
        max_tokens: Optional maximum number of tokens for the LLM responses.

    Returns:
        None. The `program.code.model_jax` field of eligible programs in the `population`
        is mutated in-place.
    """
    programs = _filter_programs(population, _needs_model_translation)
    await asyncio.gather(
        *[
            _translate_one_model(
                p,
                model_prompt_schema,
                llm,
                retry_config,
                max_tokens,
                output_schema=output_schema,
            )
            for p in programs
        ]
    )
