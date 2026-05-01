from pydantic import BaseModel, Field


class ModelSchema(BaseModel):
    thought_process: str = Field(
        description=(
            "Step-by-step reasoning: (1) what each parent model does and where it falls short, "
            "(2) what specific changes you are making and why they should reduce loss, "
            "(3) the mathematical or algorithmic justification for your approach."
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


class ParamEstSchema(BaseModel):
    thought_process: str = Field(
        description=(
            "Step-by-step reasoning: (1) the mathematical structure of the current model and what each "
            "parameter represents, (2) which statistical properties of the data each parameter maps to, "
            "(3) how this estimator improves on the parent estimators."
        )
    )
    code: str = Field(
        description=(
            "The complete, self-contained Python implementation of the parameter estimator. "
            "Must include imports, any helper functions, and the estimator function definition."
        )
    )


class TranslationSchema(BaseModel):
    model_code: str = Field(
        description="The model function translated to JAX-compatible code, including imports."
    )
    param_est_code: str = Field(
        description="The parameter estimator function translated to JAX-compatible code, including imports."
    )
