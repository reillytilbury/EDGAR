"""
LLM generation functions.

These operate on a Population, finding programs that need code generated
and filling in their fields via LLM calls.

Prompt variable mapping is handled by PromptSchema, not here.
"""
from __future__ import annotations

import asyncio
from typing import Any

from ..evolution.program import Program
from ..evolution.population import Population
from ..llm.prompt_schema import PromptSchema
from ..llm.llm_calling import call_llm
from ..llm.response_schema import ModelSchema, ParamEstSchema, TranslationSchema
from ..llm.code_loading import load_function_from_source


# ---------------------------------------------------------------------------
# Filters — which programs need work at each stage
# ---------------------------------------------------------------------------

def _needs_model_code(population: Population) -> list[Program]:
    """Programs that have been spawned but don't have model code yet."""
    return [population[i] for i in range(len(population))
            if population[i].code.model is None]


def _needs_param_est_code(population: Population) -> list[Program]:
    """Programs with model code but no parameter estimator code."""
    return [population[i] for i in range(len(population))
            if population[i].code.model is not None
            and population[i].code.param_est is None]


def _needs_jax_translation(population: Population) -> list[Program]:
    """Programs with numpy code that hasn't been translated to JAX yet."""
    return [population[i] for i in range(len(population))
            if population[i].code.model is not None
            and population[i].code_jax.model is None]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_parents(population: Population, program: Program) -> list[Program]:
    """Return the list of parent Program objects for a given program."""
    return [population[i] for i in program.birth.parent_indices]


# ---------------------------------------------------------------------------
# Model code generation
# ---------------------------------------------------------------------------

async def _generate_one_model(
    program: Program,
    parents: list[Program],
    prompt_schema: PromptSchema,
    llm: str,
    mode: str,
    temperature: float,
    config: dict[str, Any] | None = None,
) -> None:
    """Call the LLM to generate numpy model code for a single program."""
    prompt = prompt_schema.build_prompt(mode, parents, config)
    result = await call_llm(prompt=prompt, llm_model=llm, output_type=ModelSchema, temperature=temperature)
    program.code.model = result.code
    program.name = result.descriptive_name
    program.birth.llm_name = llm


async def generate_model_code(
    population: Population,
    prompt_schema: PromptSchema,
    llm: str,
    mode: str,
    temperature: float,
    config: dict[str, Any] | None = None,
) -> None:
    """Generate numpy model code for all programs that don't have it yet.

    Mutates: program.code.model, program.name, program.birth.llm_name
    """
    programs_to_modify = _needs_model_code(population)
    await asyncio.gather(*[
        _generate_one_model(p, _resolve_parents(population, p), prompt_schema, llm, mode, temperature, config)
        for p in programs_to_modify
    ], return_exceptions=True)


# ---------------------------------------------------------------------------
# Parameter estimator code generation
# ---------------------------------------------------------------------------

async def _generate_one_param_est(
    program: Program,
    parents: list[Program],
    prompt_schema: PromptSchema,
    llm: str,
    config: dict[str, Any] | None = None,
) -> None:
    """Call the LLM to generate a numpy parameter estimator for a single program."""
    prompt = prompt_schema.build_prompt("explore", parents, config)
    result = await call_llm(prompt=prompt, llm_model=llm, output_type=ParamEstSchema, temperature=1.0)
    program.code.param_est = result.code


async def generate_param_est_code(
    population: Population,
    prompt_schema: PromptSchema,
    llm: str,
    config: dict[str, Any] | None = None,
) -> None:
    """Generate numpy parameter estimator code for programs that have model code but no estimator.

    Mutates: program.code.param_est
    """
    programs_to_modify = _needs_param_est_code(population)
    await asyncio.gather(*[
        _generate_one_param_est(p, _resolve_parents(population, p), prompt_schema, llm, config)
        for p in programs_to_modify
    ], return_exceptions=True)


# ---------------------------------------------------------------------------
# JAX translation
# ---------------------------------------------------------------------------

async def _translate_one_program(
    program: Program,
    prompt_schema: PromptSchema,
    llm: str,
) -> None:
    """Translate a single program's model_code and param_est_code from numpy to JAX."""
    prompt = prompt_schema.build_prompt("explore", [program])
    try:
        result = await call_llm(prompt=prompt, llm_model=llm, output_type=TranslationSchema, temperature=0.3)
        if load_function_from_source(result.model_code, "model") is not None:
            program.code_jax.model = result.model_code
        if load_function_from_source(result.param_est_code, "parameter_estimator") is not None:
            program.code_jax.param_est = result.param_est_code
    except Exception:
        pass


async def translate_to_jax(
    population: Population,
    prompt_schema: PromptSchema,
    llm: str,
) -> None:
    """Translate all untranslated programs from numpy to JAX.

    Mutates: program.code_jax.model, program.code_jax.param_est
    """
    programs_to_modify = _needs_jax_translation(population)
    await asyncio.gather(*[
        _translate_one_program(p, prompt_schema, llm)
        for p in programs_to_modify
    ], return_exceptions=True)
