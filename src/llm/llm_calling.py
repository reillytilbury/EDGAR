import asyncio
import os
import random
import warnings
from typing import Union, TypeAlias

from dotenv import load_dotenv
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior, UserError, ModelAPIError, UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

from ..io.config import RetryConfig
from .response_schema import ModelSchema, ParamEstSchema, TranslationSchema

LLMOutputTypes: TypeAlias = Union[str, ModelSchema, ParamEstSchema, TranslationSchema]

load_dotenv()


# Provider dispatch is by model-name prefix. The model string is the single source
# of truth for which API gets called; no separate `provider:` field in config. Add
# a new prefix here when adding a new provider.
_PROVIDER_PREFIXES = ("gemini-", "claude-")


def _build_model(model_name: str) -> Model:
    """Construct a PydanticAI Model from a model-name string, dispatching by prefix.

    Reads the matching provider's API key from the environment; raises UserError
    with a clear message if it's missing or the prefix is unknown.
    """
    if model_name.startswith("gemini-"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise UserError(
                "GOOGLE_API_KEY is not set. Export it in your shell or place it in the repo .env file."
            )
        return GoogleModel(model_name, provider=GoogleProvider(api_key=api_key))

    if model_name.startswith("claude-"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise UserError(
                "ANTHROPIC_API_KEY is not set. Export it in your shell or place it in the repo .env file."
            )
        return AnthropicModel(model_name, provider=AnthropicProvider(api_key=api_key))

    raise UserError(
        f"Unknown LLM model {model_name!r}; expected a name starting with one of {_PROVIDER_PREFIXES}."
    )


class _LogRawResponseCapability(AbstractCapability):
    """Prints the raw model response parts after every model call, before any parsing."""

    async def after_model_request(
        self,
        ctx: RunContext,
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        parts_summary = []
        for part in response.parts:
            if isinstance(part, TextPart):
                parts_summary.append(f"  TextPart: {part.content!r}")
            elif isinstance(part, ToolCallPart):
                parts_summary.append(f"  ToolCallPart tool_name={part.tool_name!r} args={part.args!r}")
            else:
                parts_summary.append(f"  {type(part).__name__}: {part!r}")
        lines = [f"[call_llm] raw model response (model={response.model_name}):"]
        lines.append("\n".join(parts_summary) if parts_summary else "  (empty)")
        lines.append(f"  usage: input_tokens={response.usage.input_tokens} output_tokens={response.usage.output_tokens}")
        if response.provider_details:
            lines.append(f"  provider_details: {response.provider_details}")
        print("\n".join(lines))
        return response



class _WarnOnMaxTokensCapability(AbstractCapability):
    """Warns if the model stopped because it hit the max_tokens limit."""

    async def after_model_request(
        self,
        ctx: RunContext,
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        finish_reason = (response.provider_details or {}).get("finish_reason")
        if finish_reason == "MAX_TOKENS":
            warnings.warn(
                f"[call_llm] Response truncated: model hit max_tokens limit "
                f"(output_tokens={response.usage.output_tokens}, model={response.model_name}). "
            )
        elif finish_reason == "MALFORMED_FUNCTION_CALL":
            warnings.warn(
                f"[call_llm] Malformed function call from {response.model_name} "
                f"(output_tokens={response.usage.output_tokens}). "
                f"The response was likely truncated — try increasing max_tokens."
            )
        return response


async def call_llm(
    prompt: str,
    llm_model: str | Model,
    output_type: LLMOutputTypes = str,
    image_bytes: bytes | None = None,
    temperature: float = 1.0,
    thinking: bool | str | None = None,
    max_tokens: int | None = 10_000,
    log_raw_llm_response: bool = False,
    retry_config: RetryConfig | None = None,
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
    if isinstance(llm_model, str):
        model = _build_model(llm_model)
    elif isinstance(llm_model, Model):
        model = llm_model
    else:
        raise TypeError("llm_model must be a string or a PydanticAI Model instance.")

    # Provider temperature ranges differ: Google accepts [0, 2], Anthropic [0, 1].
    # The schedule in task_spec.schedule() emits temp in [~1.37, 2.0] (Gemini-scale).
    # For Anthropic, rescale by /2 to map [1.37, 2.0] -> [0.685, 1.0], preserving
    # the schedule's relative decay shape inside Anthropic's valid range. Only
    # rescales when the supplied temperature exceeds 1.0 — explicitly-passed
    # in-range values (e.g. 0.7 from tests) are left alone.
    if isinstance(model, AnthropicModel) and temperature > 1.0:
        temperature = temperature / 2.0

    # Anthropic's API treats max_tokens as REQUIRED (Google's allows None and uses
    # its own server-side default). Caller bugs that leak None through here crash
    # deep inside the anthropic SDK with a confusing TypeError. Fall back to the
    # call_llm default rather than failing — log if we hit this path so the bug
    # can be fixed upstream.
    if isinstance(model, AnthropicModel) and max_tokens is None:
        max_tokens = 10_000
        warnings.warn(
            "[call_llm] max_tokens was None for an Anthropic call; falling back to 10000. "
            "Pass max_tokens explicitly upstream to silence this warning."
        )

    rc = retry_config or RetryConfig()
    capabilities = [_WarnOnMaxTokensCapability()]
    if log_raw_llm_response:
        capabilities.append(_LogRawResponseCapability())
    agent = Agent(model, output_type=output_type, output_retries=rc.max_retries, capabilities=capabilities)

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

    delay = rc.initial_delay
    for attempt in range(rc.max_retries):
        try:
            result = await agent.run(user_input, model_settings=model_settings)
            return result.output

        except UnexpectedModelBehavior as e:
            type_name = getattr(output_type, "__name__", repr(output_type))
            warnings.warn(
                f"[call_llm] LLM output could not be parsed as {type_name!r} after exhausting retries — skipping. "
                f"{e.message}"
            )
            return None

        except ModelHTTPError as e:
            if e.status_code not in rc.retryable_status_codes:
                warnings.warn(f"[call_llm] Non-retryable HTTP {e.status_code} error: {e}")
                raise
            if attempt == rc.max_retries - 1:
                warnings.warn(
                    f"[call_llm] HTTP {e.status_code} on final attempt {attempt + 1}/{rc.max_retries}. No more retries left. Returning None."
                )
                return None
            jitter = random.uniform(0, 1)
            wait = min(delay + jitter, rc.max_delay)
            warnings.warn(
                f"[call_llm] HTTP {e.status_code} on attempt {attempt + 1}/{rc.max_retries}, "
                f"retrying in {wait:.1f}s."
            )
            await asyncio.sleep(wait)
            delay = min(delay * rc.backoff_multiplier, rc.max_delay)

        # ModelAPIError (non-HTTP network failure), UsageLimitExceeded, UserError → propagate immediately
        except (ModelAPIError, UsageLimitExceeded, UserError) as e:
            warnings.warn(f"[call_llm] {type(e).__name__} encountered: {e}")
            raise