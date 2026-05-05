from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.settings import ModelSettings


_DOTENV_LOADED = False


def _ensure_dotenv_loaded() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    _DOTENV_LOADED = True


async def call_llm(
    prompt: str,
    llm_model: str,
    output_type=str,
    image_bytes: bytes | None = None,
    temperature: float = 1.0,
    thinking: bool | str | None = None,
):
    """
    Call an LLM through PydanticAI and return the parsed output.

    Args:
        prompt: The text prompt to send to the model.
        llm_model: The PydanticAI model specifier, e.g. "google-gla:gemini-2.5-flash".
        output_type: The expected output type. Use `str` for plain text or a Pydantic model
            for structured output.
        image_bytes: Optional PNG image bytes to include alongside the text prompt.
        temperature: Sampling temperature for the model.
        thinking: Optional reasoning effort setting. Can be `True`, `False`, or one of
            "minimal", "low", "medium", "high", "xhigh".

    Returns:
        The model output, either as a string or as an instance of `output_type`.
    """
    _ensure_dotenv_loaded()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise UserError(
            "GOOGLE_API_KEY is not set. Export it in your shell or place it in the repo .env file."
        )
    model = GoogleModel(
        llm_model,
        provider=GoogleProvider(api_key=api_key),
    )
    agent = Agent(model, output_type=output_type)

    user_input = (
        [prompt, BinaryContent(data=image_bytes, media_type="image/png")]
        if image_bytes is not None
        else prompt
    )

    result = await agent.run(
        user_input,
        model_settings=ModelSettings(
            temperature=temperature,
            thinking=thinking,
        ),
    )
    return result.output