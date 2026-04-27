from pydantic import BaseModel, Field


class CodeOnlySchema(BaseModel):
    code: str = Field(
        description=(
            "The complete, self-contained Python implementation. "
            "Must include imports, any helper functions, and the primary function definition."
        )
    )


class ReasonedCodeSchema(CodeOnlySchema):
    thought_process: str = Field(
        description="Detailed mathematical or algorithmic reasoning for this code."
    )


class ModelSchema(ReasonedCodeSchema):
    descriptive_name: str = Field(
        description=(
            "A concise, descriptive name for the model or function being implemented "
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


class ParamEstSchema(ReasonedCodeSchema):
    pass


class TranslationSchema(BaseModel):
    model_code: str = Field(
        description="The model function translated to JAX-compatible code, including imports."
    )
    param_est_code: str = Field(
        description="The parameter estimator function translated to JAX-compatible code, including imports."
    )