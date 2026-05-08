"""
LLM generation functions.

These operate on a Population, finding programs that need code generated
and filling in their fields via LLM calls.

Prompt variable mapping is handled by PromptSchema, not here.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, TYPE_CHECKING
import traceback
import warnings

from pydantic_ai.models import Model

from ..evolution.program import Program
from ..evolution.population import Population
from ..llm.prompt_schema import PromptSchema
from ..llm.llm_calling import call_llm
from ..llm.response_schema import ModelSchema, ParamEstSchema, TranslationSchema
from ..llm.code_loading import load_function_from_source

if TYPE_CHECKING:
    from ..io.task_spec import TaskSpec


# ---------------------------------------------------------------------------
# Filters — which programs need work at each stage
# ---------------------------------------------------------------------------

def _needs_model_code(population: Population) -> list[Program]:
    return [population[i] for i in range(len(population))
            if population[i].code.model is None]


def _needs_param_est_code(population: Population) -> list[Program]:
    return [population[i] for i in range(len(population))
            if population[i].code.model is not None
            and population[i].code.param_est is None]


def _needs_jax_translation(population: Population) -> list[Program]:
    return [population[i] for i in range(len(population))
            if population[i].code.model is not None
            and population[i].code_jax.model is None]


def _resolve_parents(population: Population, program: Program) -> list[Program]:
    return [population[i] for i in program.birth.parent_indices]


def _prompt_image_bytes(spec: TaskSpec, data: dict, parents: list[Program], program: Program) -> bytes | None:
    if spec is None or spec.plot_fn is None or data is None:
        return None
    b = program.birth
    img_path = os.path.join(spec.output_dir, "image_feedback",
                            f"gen_{b.generation:03d}",
                            f"island_{b.island:03d}",
                            f"batch_{b.batch_index:03d}",
                            "image.png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    try:
        spec.plot_fn(data, parents, save_path=img_path)
        program.image_path = img_path
        return open(img_path, "rb").read()
    except Exception as e:
        warnings.warn(f"[generate] plot_fn failed for program #{program.idx}: {e}")
        return None


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
) -> None:
    image_bytes = _prompt_image_bytes(spec, data, parents, program)

    prompt = prompt_schema.build_prompt(mode, parents, config)
    result = await call_llm(
        prompt=prompt,
        llm_model=llm,
        output_type=ModelSchema,
        temperature=temperature,
        image_bytes=image_bytes,
    )
    header = f'"""\n{result.thought_process}\n\n{result.latex_equations}\n"""\n\n'
    program.code.model = header + result.code + f'\n\nmodel.DEFAULT_PARAMS = {result.default_params!r}'
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
) -> None:
    """Generate numpy model code for all programs that don't have it yet.

    Pass spec and data to also generate prompt images at
    spec.output_dir/image_feedback/gen_NNN/island_NNN/batch_NNN/image.png.

    Mutates: program.code.model, program.name, program.birth.llm_name, program.image_path
    """
    programs = _needs_model_code(population)
    await asyncio.gather(*[
        _generate_one_model(
            p, _resolve_parents(population, p), prompt_schema, llm, mode, temperature,
            config, spec, data,
        )
        for p in programs
    ], return_exceptions=True)


# ---------------------------------------------------------------------------
# Parameter estimator code generation
# ---------------------------------------------------------------------------

async def _generate_one_param_est(
    program: Program,
    parents: list[Program],
    prompt_schema: PromptSchema,
    llm: str | Model,
    config: dict[str, Any] | None = None,
) -> None:
    prompt = prompt_schema.build_prompt("explore", parents, config)
    result = await call_llm(
        prompt=prompt,
        llm_model=llm,
        output_type=ParamEstSchema,
        temperature=1.0,
    )
    header = f'"""\n{result.thought_process}\n"""\n\n'
    program.code.param_est = header + result.code


async def generate_param_ests(
    population: Population,
    prompt_schema: PromptSchema,
    llm: str | Model,
    config: dict[str, Any] | None = None,
) -> None:
    """Generate numpy parameter estimator code for programs that have model code but no estimator.

    Mutates: program.code.param_est
    """
    programs = _needs_param_est_code(population)
    await asyncio.gather(*[
        _generate_one_param_est(p, _resolve_parents(population, p), prompt_schema, llm, config)
        for p in programs
    ], return_exceptions=True)


# ---------------------------------------------------------------------------
# JAX translation
# ---------------------------------------------------------------------------

async def _translate_one_program(
    program: Program,
    prompt_schema: PromptSchema,
    llm: str | Model,
) -> None:
    prompt = prompt_schema.build_prompt("explore", [program])
    try:
        result = await call_llm(prompt=prompt, llm_model=llm, output_type=TranslationSchema, temperature=1.0)
        if load_function_from_source(result.model_code, "model") is not None:
            program.code_jax.model = result.model_code
        if load_function_from_source(result.param_est_code, "parameter_estimator") is not None:
            program.code_jax.param_est = result.param_est_code
    except Exception as e:
        warnings.warn(
            f"[translate] program #{program.idx} JAX translation failed: {e}\n"
            f"{traceback.format_exc()}"
        )


async def translate_programs(
    population: Population,
    prompt_schema: PromptSchema,
    llm: str | Model,
) -> None:
    """Translate all untranslated programs from numpy to JAX.

    Mutates: program.code_jax.model, program.code_jax.param_est
    """
    programs = _needs_jax_translation(population)
    await asyncio.gather(*[
        _translate_one_program(p, prompt_schema, llm)
        for p in programs
    ], return_exceptions=True)
