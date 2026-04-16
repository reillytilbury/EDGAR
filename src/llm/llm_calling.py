from pydantic_ai import Agent, BinaryContent
from pydantic_ai.settings import ModelSettings


def normalize_model_name(model_name: str) -> str:
    """
    Convert legacy bare Gemini model names to PydanticAI model specifiers.
    """
    if ":" in model_name:
        return model_name
    if model_name.startswith("gemini-"):
        return f"google-gla:{model_name}"
    return model_name


async def call_llm(
    prompt: str,
    model_name: str,
    output_type=str,
    image_bytes: bytes | None = None,
    temperature: float = 1.0,
    thinking: bool | str | None = None,
):
    """
    Call an LLM through PydanticAI and return the parsed output.

    Args:
        prompt: The text prompt to send to the model.
        model_name: The PydanticAI model specifier, e.g. "google-gla:gemini-2.0-flash".
        output_type: The expected output type. Use `str` for plain text or a Pydantic model
            for structured output.
        image_bytes: Optional PNG image bytes to include alongside the text prompt.
        temperature: Sampling temperature for the model.
        thinking: Optional reasoning effort setting. Can be `True`, `False`, or one of
            "minimal", "low", "medium", "high", "xhigh".

    Returns:
        The model output, either as a string or as an instance of `output_type`.
    """
    model_name = normalize_model_name(model_name)
    agent = Agent(model_name, output_type=output_type)

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
