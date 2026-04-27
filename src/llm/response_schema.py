from pydantic import BaseModel, Field


class ModelSchema(BaseModel):
    thought_process: str = Field(
        description="Detailed mathematical or algorithmic reasoning for this code."
    )
    descriptive_name: str = Field(
        description=(
            "A concise, descriptive name for the model "
            "(e.g., 'Double Gaussian Model')."
        )
    )
    latex_equations: str = Field(
        description=(
            "The equation for the model formatted in LaTeX. "
            "Should be a single equation that defines the model, "
            "including all variables and parameters."
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
        description="Detailed mathematical or algorithmic reasoning for this code."
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
