"""
Fake LLM classes for deterministic system testing.

FakeLLM returns TestModel instances with predetermined code for model generation,
parameter estimation, and JAX translation — cycling through Program1, InvalidProgram,
and ProgramSolution with an incrementing offset applied to model code.

SeedFakeLLM returns TestModel instances for JAX-only seeding, cycling through two
simple JAX models.

Example usage:
    fake = FakeLLM()
    result = await call_llm("prompt", llm_model=fake.gen_model(), output_type=ModelSchema)
"""
from pydantic_ai.models.test import TestModel

from .programs import Program1, Program2, InvalidProgram, ProgramSolution, SeedPrograms


class FakeLLM:
    """Returns TestModel instances with predetermined program code instead of calling a real LLM."""

    def __init__(self, offset: float = 0.1):
        self._programs = (Program1, InvalidProgram, ProgramSolution)
        self._model_counter = [0, 0, 0]
        self._model_jax_counter = [0, 0, 0]
        self._param_est_counter = [0, 0, 0]
        self.offset = offset

    @staticmethod
    def _add_offset(code: str, offset: float) -> str:
        return code + f" + {offset:.3f}\n"

    def gen_model(self) -> TestModel:
        """Return a TestModel whose output matches ModelSchema for the next program in rotation."""
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
        """Return a TestModel whose output matches ParamEstSchema for the next program in rotation."""
        idx = self._param_est_counter.index(min(self._param_est_counter))
        code = self._programs[idx].param_est
        self._param_est_counter[idx] += 1
        return TestModel(custom_output_args={
            "thought_process": "fake thought process",
            "code": code,
        })

    def gen_translation(self) -> TestModel:
        """Return a TestModel whose output matches TranslationSchema for the next program in rotation."""
        idx = self._model_jax_counter.index(min(self._model_jax_counter))
        model_code = self._add_offset(
            self._programs[idx].model_jax,
            self.offset * self._model_jax_counter[idx],
        )
        param_est_code = self._programs[idx].param_est
        self._model_jax_counter[idx] += 1
        return TestModel(custom_output_args={
            "model_code": model_code,
            "param_est_code": param_est_code,
        })


class SeedFakeLLM:
    """Returns TestModel instances for JAX seeding, cycling through two simple models."""

    _seed_models = (SeedPrograms.model_v1_jax, SeedPrograms.model_v2_jax)

    def __init__(self):
        self._counter = 0

    def gen_model_jax(self) -> TestModel:
        """Return a TestModel whose output matches TranslationSchema for the next seed model."""
        model_code = self._seed_models[self._counter]
        self._counter += 1
        return TestModel(custom_output_args={
            "model_code": model_code,
            "param_est_code": SeedPrograms.param_est,
        })
