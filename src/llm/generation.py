import json

from .llm_calling import call_llm
from .response_schema import ModelSchema, ParamEstSchema, TranslationSchema


def structured_response_json(response) -> str:
    if hasattr(response, "model_dump_json"):
        return response.model_dump_json()
    return json.dumps(response, default=str)


def append_entrypoint_instruction(prompt: str, entrypoint_name: str) -> str:
    if entrypoint_name == "model":
        instruction = (
            "Return structured output whose `code` field contains a complete Python module. "
            "The module may include imports, constants, helper functions, and helper classes, "
            "but it must expose the public entrypoint `def model(data, params): ...`."
        )
    elif entrypoint_name == "parameter_estimator":
        instruction = (
            "Return structured output whose `code` field contains a complete Python module. "
            "The module may include imports, constants, helper functions, and helper classes, "
            "but it must expose the public entrypoint `def parameter_estimator(data): ...`."
        )
    else:
        instruction = (
            "Return structured output whose `code` field contains a complete Python module. "
            f"The module must expose the public entrypoint `{entrypoint_name}`."
        )
    return f"{prompt}\n\n{instruction}"


async def request_model_module(
    prompt: str,
    model_name: str,
    image_bytes: bytes | None = None,
    temperature: float = 1.0,
    thinking: bool | str | None = None,
) -> tuple[str | None, str]:
    response = await call_llm(
        prompt=append_entrypoint_instruction(prompt, "model"),
        model_name=model_name,
        output_type=ModelSchema,
        image_bytes=image_bytes,
        temperature=temperature,
        thinking=thinking,
    )
    return response.code, structured_response_json(response)


async def request_parameter_estimator_module(
    prompt: str,
    model_name: str,
    image_bytes: bytes | None = None,
    temperature: float = 1.0,
    thinking: bool | str | None = "low",
) -> tuple[str | None, str]:
    response = await call_llm(
        prompt=append_entrypoint_instruction(prompt, "parameter_estimator"),
        model_name=model_name,
        output_type=ParamEstSchema,
        image_bytes=image_bytes,
        temperature=temperature,
        thinking=thinking,
    )
    return response.code, structured_response_json(response)


async def request_jax_translation(
    prompt: str,
    model_name: str,
    entrypoint_name: str,
    temperature: float = 0.0,
) -> tuple[str | None, str]:
    full_prompt = (
        f"{prompt}\n\n"
        "Translate the entire module, including all imports, constants, helper "
        "functions, and helper classes needed by the public entrypoint. "
        f"The translated code must define `def {entrypoint_name}(...):` with "
        "the same signature and must not include markdown fences or explanatory text."
    )
    response = await call_llm(
        prompt=full_prompt,
        model_name=model_name,
        output_type=TranslationSchema,
        temperature=temperature,
    )
    return response.code, structured_response_json(response)

