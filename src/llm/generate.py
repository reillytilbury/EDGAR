"""
LLM generation functions.

These operate on a Population, finding programs that need code generated
and filling in their fields via LLM calls.

Prompt variable mapping is handled by PromptSchema, not here.
"""
from __future__ import annotations

import asyncio
import os
import warnings
from typing import Any, TYPE_CHECKING, Callable
import itertools

from pydantic_ai.models import Model

from ..evolution.program import Program
from ..evolution.population import Population
from ..llm.prompt_schema import PromptSchema
from ..llm.llm_calling import call_llm
from ..io.config import RetryConfig
from ..llm.response_schema import ModelSchema, ParamEstSchema, TranslationSchema
from ..llm.code_loading import load_function_from_source

if TYPE_CHECKING:
    from ..io.task_spec import TaskSpec


# ---------------------------------------------------------------------------
# Filters — which programs need work at each stage
# ---------------------------------------------------------------------------
def _needs_model_code(program: Program) -> bool:
    return program.code.model is None

def _needs_param_est_code(program: Program) -> bool:
    return program.code.model is not None and program.code.param_est is None

def _needs_model_translation(program: Program) -> bool:
    return program.code.model is not None and program.code.model_jax is None

def _filter_programs(population: Population, filter_rule: Callable[[Program], bool]) -> list[Program]:
    return [p for p in population if filter_rule(p)]

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
    cfg = config or {}
    prompt = prompt_schema.build_prompt(mode, parents, config)
    result = await call_llm(
        prompt=prompt,
        llm_model=llm,
        output_type=ModelSchema,
        temperature=temperature,
        image_bytes=image_bytes,
        log_raw_llm_response=cfg.get("log_raw_llm_response", False),
        max_tokens=cfg.get("max_tokens"),
        retry_config=cfg.get("retry_config"),
    )
    if result is None:
        warnings.warn(f"[generate] Skipping model code for program #{program.idx}: call_llm returned None")
        return
    header = f'"""\n{result.thought_process}\n\n{result.latex_equations}\n"""\n\n'
    program.code.model = header + result.code
    program.default_params = result.default_params
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
    programs = _filter_programs(population, _needs_model_code)
    await asyncio.gather(*[
        _generate_one_model(
            p, _resolve_parents(population, p), prompt_schema, llm, mode, temperature,
            config, spec, data,
        )
        for p in programs
    ])


# ---------------------------------------------------------------------------
# Parameter estimator code generation
# ---------------------------------------------------------------------------

async def _generate_one_param_est(
    program: Program,
    prompt_schema: PromptSchema,
    llm: str | Model,
    config: dict[str, Any] | None = None,
) -> None:
    cfg = config or {}
    prompt = prompt_schema.build_prompt("explore", [program], config)
    result = await call_llm(
        prompt=prompt,
        llm_model=llm,
        output_type=ParamEstSchema,
        temperature=1.0,
        log_raw_llm_response=cfg.get("log_raw_llm_response", False),
        max_tokens=cfg.get("max_tokens"),
        retry_config=cfg.get("retry_config"),
    )
    if result is None:
        warnings.warn(f"[generate] Skipping param_est for program #{program.idx}: call_llm returned None")
        return
    program.code.param_est = result.code


async def generate_param_ests(
    population: Population,
    prompt_schema: PromptSchema,
    llm: str | Model,
    config: dict[str, Any] | None = None,
) -> None:
    """Generate numpy parameter estimator code for programs that have model code but no estimator.

    Mutates: program.code.param_est
    """
    programs = _filter_programs(population, _needs_param_est_code)
    await asyncio.gather(*[
        _generate_one_param_est(p, prompt_schema, llm, config)
        for p in programs
    ])


# ---------------------------------------------------------------------------
# JAX translation
# ---------------------------------------------------------------------------
async def _translate_one_model(
    program: Program,
    model_prompt_schema: PromptSchema,
    llm: str | Model,
    retry_config: RetryConfig | None = None,
    max_tokens: int | None = None,
) -> None:
    model_prompt = model_prompt_schema.build_prompt("explore", [program])
    model_result = await call_llm(prompt=model_prompt, llm_model=llm, output_type=TranslationSchema, temperature=1.0, retry_config=retry_config, max_tokens=max_tokens)
    if model_result is not None and load_function_from_source(model_result.code, "model") is not None:
        program.code.model_jax = model_result.code

async def translate_programs(
    population: Population,
    model_prompt_schema: PromptSchema,
    llm: str | Model,
    retry_config: RetryConfig | None = None,
    max_tokens: int | None = None,
) -> None:
    """Translate all untranslated model code from numpy to JAX.

    Mutates: program.code.model_jax
    """
    programs = _filter_programs(population, _needs_model_translation)
    await asyncio.gather(*[
        _translate_one_model(p, model_prompt_schema, llm, retry_config, max_tokens)
        for p in programs
    ])
