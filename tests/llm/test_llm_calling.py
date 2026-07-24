import os
import pytest
from tests.llm.fakellm import FakeLLM
from tests.llm.programs import Program1, DEFAULT_FAKE_PROGRAMS
from tests.llm.utils import run_model_code, run_param_est_code, generate_image_bytes
from edgar.llm.llm_calling import call_llm
from edgar.llm.response_schema import (
    ModelSchema,
    ModelSchemaDynamicDefaultParams,
    ParamEstSchema,
    TranslationSchema,
)
import numpy as np

LLM_MODEL = "gemini-2.5-flash-lite"  # used for real LLM calls

# (model_name, required_env_var). Used by the per-provider ping test below.
PROVIDER_PING_MODELS = [
    ("gemini-2.5-flash-lite", "GOOGLE_API_KEY"),
    ("claude-haiku-4-5", "ANTHROPIC_API_KEY"),
]


@pytest.mark.asyncio
async def test_call_llm_with_fake_model():
    llm = FakeLLM(DEFAULT_FAKE_PROGRAMS)
    model = llm.gen_model()
    result = await call_llm(
        prompt="Fake prompt", llm_model=model, output_type=ModelSchema
    )

    assert isinstance(result, ModelSchema)
    assert result.thought_process == "fake thought process"
    assert result.descriptive_name == "Fake Model 0"
    assert result.code.startswith(
        Program1.model
    )  # code added an offset, so check it starts with same
    assert result.default_params == Program1.default_params
    # Try running the code and check it produces expected output
    output = run_model_code(
        result.code, {"x": np.array([0, 1.0, 2.0])}, result.default_params
    )
    expected_output = np.array([0, 1.1, 4.2])  # y = x^2 +0.1x
    assert np.allclose(output, expected_output)


@pytest.mark.asyncio
async def test_call_llm_with_fake_param_est():
    llm = FakeLLM(DEFAULT_FAKE_PROGRAMS)
    param_est = llm.gen_param_est()
    result = await call_llm(
        prompt="Fake prompt", llm_model=param_est, output_type=ParamEstSchema
    )

    assert isinstance(result, ParamEstSchema)
    assert result.code == Program1.param_est
    # Try running the code and check it produces expected output
    output = run_param_est_code(result.code, {"x": np.array([0, 1.0, 2.0])})
    expected_output = (
        Program1.default_params
    )  # In the programs we just return the default params as the estimate
    assert output == expected_output


@pytest.mark.asyncio
async def test_call_llm_with_fake_model_translation():
    llm = FakeLLM(DEFAULT_FAKE_PROGRAMS)
    translation = llm.gen_model_translation()
    result = await call_llm(
        prompt="Fake prompt", llm_model=translation, output_type=TranslationSchema
    )

    assert isinstance(result, TranslationSchema)
    assert result.code.startswith(Program1.model_jax)
    # Try running the code and check it produces expected output
    output = run_model_code(
        result.code, {"x": np.array([0, 1.0, 2.0])}, Program1.default_params
    )
    expected_output = np.array([0, 1.1, 4.2])  # y = x^2 +0.1x
    assert np.allclose(output, expected_output)


# Real LLM calls


@pytest.mark.live
@pytest.mark.asyncio
async def test_call_llm_live_model_schema():
    prompt = (
        "Write a simple quadratic numpy model.\n"
        "- data is a dict with key 'x' (1D float array)\n"
        "- params is a dict with keys 'a' and 'b'\n"
        "- returns a * x**2 + b * x\n"
        "- function must be named `model`\n"
        "- include numpy import"
    )
    result = await call_llm(prompt=prompt, llm_model=LLM_MODEL, output_type=ModelSchema)
    print("LLM output: ", result)
    assert result is not None
    assert isinstance(result, ModelSchema)
    assert isinstance(result.thought_process, str) and result.thought_process
    assert isinstance(result.descriptive_name, str) and result.descriptive_name
    assert isinstance(result.default_params, dict)
    assert all(
        isinstance(v, (int, float, list)) for v in result.default_params.values()
    )
    compile(result.code, "<ModelSchema.code>", "exec")
    output = run_model_code(
        result.code, {"x": np.array([0.0, 1.0, 2.0])}, result.default_params
    )
    assert output is not None


@pytest.mark.live
@pytest.mark.asyncio
async def test_call_llm_live_model_schema_dynamic_default_params():
    prompt = (
        "Write a simple quadratic numpy model.\n"
        "- data is a dict with key 'x' (1D float array)\n"
        "- params is a dict with keys 'a' and 'b' where a and b are 1D float arrays of the same size as data['x']\n"
        "- returns a * x**2 + b * x\n"
        "- function must be named `model`\n"
        "- include numpy import"
    )
    result = await call_llm(
        prompt=prompt, llm_model=LLM_MODEL, output_type=ModelSchemaDynamicDefaultParams
    )
    print("LLM output: ", result)
    assert result is not None
    assert isinstance(result, ModelSchemaDynamicDefaultParams)
    assert isinstance(result.thought_process, str) and result.thought_process
    assert isinstance(result.descriptive_name, str) and result.descriptive_name
    assert isinstance(result.default_params, str)  # function returned as string
    print(result.default_params)
    default_params = eval(result.default_params)
    assert default_params.__name__ == "<lambda>"
    data = {"x": np.array([0.0, 1.0, 2.0])}
    default_params_dict = default_params(data)
    assert isinstance(default_params_dict, dict)
    assert default_params_dict["a"].shape == data["x"].shape
    assert default_params_dict["b"].shape == data["x"].shape
    compile(result.code, "<ModelSchema.code>", "exec")
    output = run_model_code(result.code, data, default_params_dict)
    print(output)
    assert output is not None
    assert np.shape(output) == data["x"].shape


@pytest.mark.live
@pytest.mark.asyncio
async def test_call_llm_live_model_schema_with_image():
    prompt = (
        "Write a simple quadratic numpy model.\n"
        "- data is a dict with key 'x' (1D float array)\n"
        "- params is a dict with keys 'a' and 'b'\n"
        "- returns a * x**2 + b * x\n"
        "- function must be named `model`\n"
        "- include numpy import\n"
        "Ignore the attached image"
    )
    img_bytes = generate_image_bytes()
    result = await call_llm(
        prompt=prompt,
        llm_model=LLM_MODEL,
        output_type=ModelSchema,
        image_bytes=img_bytes,
    )
    print("LLM output: ", result)
    assert result is not None
    assert isinstance(result, ModelSchema)
    assert isinstance(result.thought_process, str) and result.thought_process
    assert isinstance(result.descriptive_name, str) and result.descriptive_name
    assert isinstance(result.default_params, dict)
    assert all(
        isinstance(v, (int, float, list)) for v in result.default_params.values()
    )
    compile(result.code, "<ModelSchema.code>", "exec")
    output = run_model_code(
        result.code, {"x": np.array([0.0, 1.0, 2.0])}, result.default_params
    )
    assert output is not None


@pytest.mark.live
@pytest.mark.asyncio
async def test_call_llm_live_param_est_schema():
    prompt = (
        "Write a parameter estimator for this model:\n\n"
        "\tdef model(data, params):\n"
        "\t\tx = data['x']\n"
        "\t\treturn params['a'] * x**2 + params['b'] * x\n\n"
        "- data is a dict with keys 'x' and 'y' (1D float arrays)\n"
        "- return a dict with keys 'a' and 'b' as floats\n"
        "- function must be named `parameter_estimator`"
    )
    result = await call_llm(
        prompt=prompt, llm_model=LLM_MODEL, output_type=ParamEstSchema
    )
    print("LLM output: ", result)
    assert result is not None
    assert isinstance(result, ParamEstSchema)
    compile(result.code, "<ParamEstSchema.code>", "exec")
    output = run_param_est_code(
        result.code, {"x": np.array([0.0, 1.0, 2.0]), "y": np.array([0.0, 1.1, 4.2])}
    )
    assert output is not None


@pytest.mark.live
@pytest.mark.asyncio
async def test_call_llm_live_translation_schema():
    prompt = (
        f"Translate this numpy model to JAX by replacing numpy with jax.numpy.\n"
        f"Keep the same function signature and logic. Function must be named `model`.\n\n"
        f"{Program1.model}"
    )
    result = await call_llm(
        prompt=prompt, llm_model=LLM_MODEL, output_type=TranslationSchema
    )
    print("LLM output: ", result)
    assert result is not None
    assert isinstance(result, TranslationSchema)
    compile(result.code, "<TranslationSchema.code>", "exec")
    output = run_model_code(
        result.code, {"x": np.array([0.0, 1.0, 2.0])}, Program1.default_params
    )
    assert output is not None


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name,env_var", PROVIDER_PING_MODELS)
async def test_call_llm_provider_ping(model_name, env_var):
    """Hits each provider with a tiny real call. Skips if the provider's key is unset.

    This is the canonical "is the API key wired correctly" check — covers auth,
    model name validity, and network path in one shot. Cheap (≤10 output tokens).
    """
    if not os.getenv(env_var):
        if os.getenv("CI"):
            pytest.fail(
                f"{env_var} is not set — configure it as a GitHub Actions secret."
            )
        pytest.skip(f"{env_var} not set; skipping {model_name} ping.")
    result = await call_llm(
        prompt="Reply with exactly the token 12345 and nothing else.",
        llm_model=model_name,
        output_type=str,
        max_tokens=20,
    )
    assert result is not None, f"{model_name} returned None"
    assert "12345" in result, f"{model_name} response missing 12345: {result!r}"
