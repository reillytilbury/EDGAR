import asyncio
import os
import random
import warnings

from dotenv import load_dotenv
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior, UserError
from pydantic_ai.models import Model
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

# 500 - Unexpected error on Google's servers, 503 - service temporarily down — all transient per Google's retry guidance.
# 429 Rate-limited - because the number of requests we send concurrently is fixed, we should not ignore this error 
# 400 Bad Request, 401 Unauthorized, 403 Forbidden → hard failures, do not retry.
# Error guidelines : https://ai.google.dev/gemini-api/docs/troubleshooting
_RETRYABLE_STATUS_CODES = frozenset({500, 503})
_MAX_RETRIES = 3
_INITIAL_DELAY = 1.0
_BACKOFF_MULTIPLIER = 2.0
_MAX_DELAY = 60.0


async def call_llm(
    prompt: str,
    llm_model: str | Model,
    output_type=str,
    image_bytes: bytes | None = None,
    temperature: float = 1.0,
    thinking: bool | str | None = None,
    max_tokens: int | None = 10_000,
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

    Raises:
        UnexpectedModelBehavior: The model responded but its output could not be parsed
            into `output_type`. Likely a prompt/schema mismatch — not retried.
        ModelHTTPError: A non-retryable HTTP error (e.g. 400, 401, 403).
        Other AgentRunError subclasses (UsageLimitExceeded, ModelAPIError, etc.) propagate
            immediately as they indicate persistent configuration or quota problems.
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

    model_settings = ModelSettings(
        temperature=temperature,
        thinking=thinking,
        max_tokens=max_tokens,
    )

    delay = _INITIAL_DELAY
    for attempt in range(_MAX_RETRIES):
        try:
            result = await agent.run(user_input, model_settings=model_settings)
            return result.output

        except UnexpectedModelBehavior as e:
            type_name = getattr(output_type, "__name__", repr(output_type))
            raise UnexpectedModelBehavior(
                f"LLM output could not be parsed as {type_name!r}. "
                f"Check that your prompt instructs the model to return the correct structure. "
                f"Raw body: {e.body}. Original error: {e.message}"
            ) from e

        except ModelHTTPError as e:
            if e.status_code not in _RETRYABLE_STATUS_CODES:
                raise
            if attempt == _MAX_RETRIES - 1:
                raise
            jitter = random.uniform(0, 1)
            wait = min(delay + jitter, _MAX_DELAY)
            warnings.warn(
                f"[call_llm] HTTP {e.status_code} on attempt {attempt + 1}/{_MAX_RETRIES}, "
                f"retrying in {wait:.1f}s."
            )
            await asyncio.sleep(wait)
            delay = min(delay * _BACKOFF_MULTIPLIER, _MAX_DELAY)

        # ModelAPIError (non-HTTP network failure), UsageLimitExceeded, UserError → propagate immediately
