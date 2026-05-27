from pydantic import BaseModel, Field


class ModelSchema(BaseModel):
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
            "A concise, descriptive name for the model "
            "(e.g., 'Double Gaussian Model')."
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
    code: str = Field(
        description=(
            "The complete, self-contained Python implementation of the parameter estimator. "
            "Must include imports, any helper functions, and the estimator function definition."
        )
    )


class TranslationSchema(BaseModel):
    code: str = Field(
        description="The translated JAX-compatible code, including imports."
    )
