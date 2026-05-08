import os
from dotenv import load_dotenv
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models import Model

load_dotenv()

async def call_llm(
    prompt: str,
    llm_model: str | Model,
    output_type=str,
    image_bytes: bytes | None = None,
    temperature: float = 1.0,
    thinking: bool | str | None = None,
    max_tokens: int | None = 32_000,
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
        max_tokens: Maximum number of tokens to generate.

    Returns:
        The model output, either as a string or as an instance of `output_type`.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise UserError(
            "GOOGLE_API_KEY is not set. Export it in your shell or place it in the repo .env file."
        )
    if isinstance(llm_model, str):
        model = GoogleModel(
            llm_model,
            provider=GoogleProvider(api_key=api_key),
        )
    elif isinstance(llm_model, Model):
        model = llm_model
    else:
        raise TypeError("llm_model must be a string or a PydanticAI Model instance.")
    
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
            max_tokens=max_tokens,
        ),
    )
    return result.output