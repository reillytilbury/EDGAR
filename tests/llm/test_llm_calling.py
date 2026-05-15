import pytest
import asyncio
from tests.llm.fakellm import FakeLLM
from tests.llm.programs import Program1
from tests.llm.utils import run_model_code, run_param_est_code
from src.llm.llm_calling import call_llm
from src.llm.response_schema import ModelSchema, ParamEstSchema, TranslationSchema
import numpy as np

@pytest.mark.asyncio
async def test_call_llm_with_fake_model():
    llm = FakeLLM()
    model = llm.gen_model()
    result = await call_llm(
        prompt = "Fake prompt",
        llm_model = model,
        output_type = ModelSchema
    )

    assert isinstance(result, ModelSchema)
    assert result.thought_process == "fake thought process"
    assert result.descriptive_name == "Fake Model 0"
    assert result.latex_equations == Program1.latex_equation
    assert result.code.startswith(Program1.model) #code added an offset, so check it starts with same
    assert result.default_params == Program1.default_params
    #Try running the code and check it produces expected output
    output = run_model_code(result.code, {"x": np.array([0,1.0,2.0])}, result.default_params)
    expected_output = np.array([0,1.1,4.2]) #y = x^2 +0.1x
    assert np.allclose(output, expected_output)

@pytest.mark.asyncio
async def test_call_llm_with_fake_param_est():
    llm = FakeLLM()
    param_est = llm.gen_param_est()
    result = await call_llm(
        prompt = "Fake prompt",
        llm_model = param_est,
        output_type = ParamEstSchema
    )

    assert isinstance(result, ParamEstSchema)
    assert result.thought_process == "fake thought process"
    assert result.code == Program1.param_est
    #Try running the code and check it produces expected output
    output = run_param_est_code(result.code, {"x": np.array([0,1.0,2.0])})
    expected_output = Program1.default_params #In the programs we just return the default params as the estimate
    assert output == expected_output

@pytest.mark.asyncio
async def test_call_llm_with_fake_model_translation():
    llm = FakeLLM()
    translation = llm.gen_model_translation()
    result = await call_llm(
        prompt = "Fake prompt",
        llm_model = translation,
        output_type = TranslationSchema
    )

    assert isinstance(result, TranslationSchema)
    assert result.code.startswith(Program1.model_jax)
    #Try running the code and check it produces expected output
    output = run_model_code(result.code, {"x": np.array([0,1.0,2.0])}, Program1.default_params)
    expected_output = np.array([0,1.1,4.2]) #y = x^2 +0.1x
    assert np.allclose(output, expected_output)