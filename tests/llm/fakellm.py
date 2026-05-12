"""
Fake LLM classes for deterministic system testing.

FakeLLM returns TestModel instances with predetermined code for model generation,
parameter estimation, and JAX translation — cycling through Program1, InvalidProgram,
and ProgramSolution with an incrementing offset applied to model code.

SeedFakeLLM returns TestModel instances for JAX-only seeding, cycling through two
simple JAX models.

CyclingModel wraps a list of TestModel instances and delegates each request() call to
the next model in the list, allowing a single Model object to produce different outputs
per program without needing to pass per-program LLM lists through the pipeline.

Example usage:
    fake = FakeLLM()
    cycling = CyclingModel([fake.gen_model() for _ in range(6)])
    result = await call_llm("prompt", llm_model=cycling, output_type=ModelSchema)
"""
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.settings import ModelSettings

from .programs import Program1, Program2, InvalidProgram, ProgramSolution, SeedPrograms


class FakeLLM:
    """Returns TestModel instances with predetermined program code instead of calling a real LLM."""

    def __init__(self, offset: float = 0.1):
        self._programs = (Program1, InvalidProgram, ProgramSolution)
        self._model_counter = [0, 0, 0]
        self._model_jax_counter = [0, 0, 0]
        self._param_est_counter = [0, 0, 0]
        self._param_est_jax_counter = [0, 0, 0]
        self.offset = offset

    @staticmethod
    def _add_offset(code: str, offset: float) -> str:
        return code + f" + {offset:.3f}\n"

    def gen_model(self) -> TestModel:
        """Return a TestModel which outputs a model matching ModelSchema for the next program in rotation."""
        idx = self._model_counter.index(min(self._model_counter))
        code = self._add_offset(
            self._programs[idx].model,
            self.offset * self._model_counter[idx],
        )
        latex_equation = self._programs[idx].latex_equation
        default_params = self._programs[idx].default_params
        
        self._model_counter[idx] += 1
        return TestModel(custom_output_args={
            "thought_process": "fake thought process",
            "descriptive_name": f"Fake Model {idx}",
            "latex_equations": latex_equation,
            "code": code,
            "default_params": default_params
        })

    def gen_param_est(self) -> TestModel:
        """Return a TestModel which outputs a parameter estimator matching ParamEstSchema for the next program in rotation."""
        idx = self._param_est_counter.index(min(self._param_est_counter))
        code = self._programs[idx].param_est
        self._param_est_counter[idx] += 1
        return TestModel(custom_output_args={
            "thought_process": "fake thought process",
            "code": code,
        })

    def gen_model_translation(self) -> TestModel:
        """Return a TestModel which outputs a jax-translated model matching TranslationSchema for the next program in rotation."""
        idx = self._model_jax_counter.index(min(self._model_jax_counter))
        model_code = self._add_offset(
            self._programs[idx].model_jax,
            self.offset * self._model_jax_counter[idx],
        )
        self._model_jax_counter[idx] += 1
        return TestModel(custom_output_args={
            "code": model_code,
        })
    
    def gen_param_est_translation(self) -> TestModel:
        """Return a TestModel which outputs a jax-translated parameter estimator matching TranslationSchema for the next program in rotation."""
        idx = self._param_est_jax_counter.index(min(self._param_est_jax_counter))
        code = self._programs[idx].param_est
        self._param_est_jax_counter[idx] += 1
        return TestModel(custom_output_args={
            "code": code,
        })

    def gen_translation(self) -> TestModel:
        """Return a TestModel which outputs combined JAX model + param_est code for the next program in rotation."""
        idx = self._model_jax_counter.index(min(self._model_jax_counter))
        code = self._programs[idx].model_jax + "\n\n" + self._programs[idx].param_est
        self._model_jax_counter[idx] += 1
        return TestModel(custom_output_args={"code": code})


class SeedFakeLLM:
    """Returns TestModel instances for JAX seeding, cycling through two simple models."""

    _seed_models = (SeedPrograms.model_v1_jax, SeedPrograms.model_v2_jax)

    def __init__(self):
        self._counter = 0

    def gen_model_translation(self) -> TestModel:
        """Return a TestModel whose output matches TranslationSchema for the next seed model."""
        model_code = self._seed_models[self._counter%len(self._seed_models)]
        self._counter += 1
        return TestModel(custom_output_args={
            "code": model_code + "\n"
        })
    
    def gen_param_est_translation(self) -> TestModel:
        """Return a TestModel whose output matches TranslationSchema for the next seed model."""
        return TestModel(custom_output_args={
            "code": SeedPrograms.param_est + "\n"
        })


class CyclingModel(Model):
    """Wraps a list of TestModel instances, delegating each request to the next in sequence.

    Use this instead of passing per-program LLM lists through the pipeline — a single
    CyclingModel instance naturally produces different outputs for each call_llm invocation.
    """

    def __init__(self, models: list[TestModel]):
        self._models = models
        self._counter = 0

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        m = self._models[self._counter % len(self._models)]
        self._counter += 1
        return await m.request(messages, model_settings, model_request_parameters)

    @property
    def model_name(self) -> str:
        return "cycling-test-model"

    @property
    def system(self) -> str:
        return "test"
