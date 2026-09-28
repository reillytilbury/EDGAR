"""
LLM-based analysis tools for inspecting the output of EDGAR runs.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

from pydantic import BaseModel, Field
from pydantic_ai.models import Model

from edgar.evolution.population import Population
from edgar.evolution.program import Program
from edgar.llm.llm_calling import call_llm


class ClassificationSchema(BaseModel):
    """Schema for classifying a scientific model program into categories."""

    assigned_categories: list[str] = Field(
        description="The subset of provided categories that apply to this model program."
    )


async def categorize_population(
    population: Population | Sequence[Program],
    categories: list[str] | dict[str, str],
    llm_model: str | Model = "gemini-2.5-flash-lite",
    concurrency_limit: int = 10,
) -> dict[int, list[str]]:
    """Analyzes each model program in a population and assigns it to categories using an LLM.

    Args:
        population: The Population object or list of Programs to categorize.
        categories: A list of category names, or a dictionary mapping category names to their descriptions.
        llm_model: The LLM model to use for classification. Defaults to "gemini-2.5-flash-lite".
        concurrency_limit: Maximum number of concurrent LLM calls. Defaults to 10.

    Returns:
        A dictionary mapping program global index `idx` to its list of assigned categories:
        {
            program_idx: ["cat1", "cat2"]
        }
    """
    programs = (
        list(population) if not isinstance(population, Population) else population
    )

    # Format the categories list/dictionary for the prompt
    if isinstance(categories, list):
        categories_str = "\n".join(f"- {cat}" for cat in categories)
        allowed_set = set(categories)
    elif isinstance(categories, dict):
        categories_str = "\n".join(
            f"- {cat}: {desc}" for cat, desc in categories.items()
        )
        allowed_set = set(categories.keys())
    else:
        raise TypeError(
            "categories must be a list of strings or a dictionary of string keys and values."
        )

    semaphore = asyncio.Semaphore(concurrency_limit)

    async def classify_one(program: Program) -> tuple[int, list[str]]:
        idx = program.idx if program.idx is not None else 0
        model_code = program.code.model
        model_name = program.name or f"Program {idx}"

        if not model_code:
            return idx, []

        prompt = (
            "You are an expert scientific code analyzer. Your task is to analyze a Python function "
            "representing a mathematical/scientific model and assign it to one or more of the provided categories.\n\n"
            f"Model Name: {model_name}\n\n"
            "Model Code:\n"
            "```python\n"
            f"{model_code}\n"
            "```\n\n"
            "Available Categories:\n"
            f"{categories_str}\n\n"
            "Analyze the code carefully and determine which of the available categories best describe "
            "the model's structure, behavior, or scientific assumptions. If none of the categories apply, "
            "do not assign any. Select the matching categories."
        )

        async with semaphore:
            result = await call_llm(
                prompt=prompt,
                llm_model=llm_model,
                output_type=ClassificationSchema,
                role="analyzer",
            )

        if result is None:
            return idx, []

        # Filter categories to ensure they are strictly from the supplied list
        assigned = [cat for cat in result.assigned_categories if cat in allowed_set]

        return idx, assigned

    tasks = [classify_one(p) for p in programs]
    results = await asyncio.gather(*tasks)

    return dict(results)
