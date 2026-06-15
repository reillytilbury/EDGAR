"""Pydantic schemas for enforcing structured output from Large Language Models.

This module defines data models that ensure LLM-generated code and metadata
conform to expected formats. These schemas are used by `edgar.llm.llm_calling.call_llm`
to validate and parse LLM responses.
"""

from pydantic import BaseModel, Field


class ModelSchema(BaseModel):
    """Schema for the structured output of LLM model generation.

    This schema defines the expected fields for a new scientific model generated
    by an LLM, including its code, descriptive name, mathematical representation,
    and initial parameter values.
    """

    thought_process: str = Field(
        description=(
            "Summary of reasoning: "
            "(1) what each parent model does and where it falls short, "
            "(2) what specific changes you are making and why they should reduce loss, "
            "(3) the mathematical and scientific justification for your approach."
            "Refer to the parent models by their descriptive names, not as Model 1 and Model 2."
        )
    )
    descriptive_name: str = Field(
        description=(
            "A concise, descriptive name for the model (e.g., 'Double Gaussian Model')."
        )
    )
    latex_equations: str = Field(
        description=(
            "The complete equation for the model in LaTeX, defining all free parameters and variables. "
            "Should be a single self-contained expression."
        )
    )
    code: str = Field(
        description=(
            "The complete, self-contained Python implementation of the model. "
            "Must include imports, any helper functions, and the model function definition."
        )
    )
    default_params: dict = Field(
        description=(
            "A dict of sensible numeric initial values for every free parameter used by the model. "
            "Keys must exactly match those used in `params` inside the model function. "
            "Values must be numeric scalars or plain lists (no numpy arrays). "
            "Example: {'amplitude': 1.0, 'decay_rate': 0.1, 'offset': 0.0}"
        )
    )


class ParamEstSchema(BaseModel):
    """Schema for the structured output of LLM parameter estimator generation."""

    code: str = Field(
        description=(
            "The complete, self-contained Python implementation of the parameter estimator. "
            "Must include imports, any helper functions, and the estimator function definition."
        )
    )


class TranslationSchema(BaseModel):
    """Schema for the structured output of LLM JAX translation."""

    code: str = Field(
        description="The translated JAX-compatible code, including imports."
    )
