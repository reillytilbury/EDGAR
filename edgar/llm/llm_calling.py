"""Provides robust, retryable, and structured interaction with various Large Language Models.

This module serves as the interface for EDGAR to communicate with LLM providers
like Google (Gemini) and Anthropic (Claude). It dynamically builds PydanticAI Model
instances, handles provider-specific configurations (e.g., API keys, temperature
rescaling for Anthropic), and implements retry mechanisms for transient HTTP errors.
Additionally, it supports multimodal input (text and images) and structured output
parsing using Pydantic schemas, ensuring that LLM responses conform to expected formats.
"""

import asyncio
import os
import random
import time
import warnings
from typing import Union, TypeAlias

from dotenv import load_dotenv
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.exceptions import (
    ModelHTTPError,
    UnexpectedModelBehavior,
    UserError,
    ModelAPIError,
    UsageLimitExceeded,
)
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

from ..io.config import RetryConfig
from ..io.metrics import get_active_metrics
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

    Args:
        model_name: The string identifier for the LLM, e.g., "gemini-pro" or "claude-3-sonnet".

    Returns:
        An instance of `pydantic_ai.models.Model` configured for the specified LLM.

    Raises:
        UserError: If the required API key for the model provider is not found
            in the environment, or if the model name prefix is unrecognized.
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
    """A PydanticAI capability that prints the raw LLM response after each call.

    This capability intercepts the model response after a request has been made
    but before any parsing or structured output extraction. It's useful for
    debugging and understanding the exact content returned by the LLM, including
    text parts, tool calls, and usage statistics.
    """

    async def after_model_request(
        self,
        ctx: RunContext,
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Logs the raw model response details to the console.

        Args:
            ctx: The PydanticAI run context.
            request_context: The context of the model request.
            response: The raw `ModelResponse` received from the LLM.

        Returns:
            The `ModelResponse` unchanged, allowing it to pass through the
            capability chain.
        """
        parts_summary = []
        for part in response.parts:
            if isinstance(part, TextPart):
                parts_summary.append(f"  TextPart: {part.content!r}")
            elif isinstance(part, ToolCallPart):
                parts_summary.append(
                    f"  ToolCallPart tool_name={part.tool_name!r} args={part.args!r}"
                )
            else:
                parts_summary.append(f"  {type(part).__name__}: {part!r}")
        lines = [f"[call_llm] raw model response (model={response.model_name}):"]
        lines.append("\n".join(parts_summary) if parts_summary else "  (empty)")
        lines.append(
            f"  usage: input_tokens={response.usage.input_tokens} output_tokens={response.usage.output_tokens}"
        )
        if response.provider_details:
            lines.append(f"  provider_details: {response.provider_details}")
        print("\n".join(lines))
        return response


class _WarnOnMaxTokensCapability(AbstractCapability):
    """A PydanticAI capability that warns if the LLM response was truncated.

    This capability checks the `finish_reason` in the model's provider details.
    If the model stopped due to hitting the `max_tokens` limit or if it returned
    a malformed function call (often indicative of truncation), a warning is issued.
    This helps in identifying cases where the LLM might not have completed its
    response, which can affect the quality of generated code.
    """

    async def after_model_request(
        self,
        ctx: RunContext,
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Checks for truncation and malformed function calls in the LLM response.

        Args:
            ctx: The PydanticAI run context.
            request_context: The context of the model request.
            response: The `ModelResponse` received from the LLM.

        Returns:
            The `ModelResponse` unchanged.
        """
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


class _RecordCallCapability(AbstractCapability):
    """Stash usage + finish_reason + model name from each HTTP model response.

    Always installed by call_llm so we can hand the numbers to ``RunMetrics``
    after the agent run completes (success or failure). Updates per-response —
    on retry, the values reflect the most recent HTTP exchange.
    """

    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.finish_reason: str | None = None
        self.model_name: str | None = None

    async def after_model_request(
        self,
        ctx: RunContext,
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        usage = response.usage
        if usage is not None:
            self.input_tokens = int(usage.input_tokens or 0)
            self.output_tokens = int(usage.output_tokens or 0)
        self.finish_reason = (response.provider_details or {}).get("finish_reason")
        self.model_name = response.model_name
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
    role: str | None = None,
) -> LLMOutputTypes | None:
    """Calls an LLM through PydanticAI, handling provider-specific settings, retries, and structured output.

    This asynchronous function provides a unified interface for interacting with different
    LLMs (e.g., Gemini, Claude). It manages the construction of
    the PydanticAI `Agent`, applies provider-specific temperature rescaling, handles
    `max_tokens` defaults, and implements a retry mechanism for transient
    HTTP errors. It also supports multimodal input by optionally including an image with the text prompt.

    Args:
        prompt: The text prompt to send to the model.
        llm_model: The PydanticAI model specifier (e.g., "gemini-2.5-flash")
            or an already constructed `pydantic_ai.models.Model` instance.
        output_type: The expected output type. Use `str` for plain text or a Pydantic
            model (`ModelSchema`, `ParamEstSchema`, `TranslationSchema`)
            for structured output.
        image_bytes: Optional PNG image bytes to include alongside the text prompt
            for multimodal LLM calls.
        temperature: Sampling temperature for the model. Values are on a Gemini-scale
            ([0, 2]). If an Anthropic model is used and the temperature exceeds 1.0,
            it will be rescaled to fit Anthropic's [0, 1] range (e.g., [1.37, 2.0]
            becomes [0.685, 1.0]).
        thinking: Optional reasoning effort setting, influencing the LLM's response style.
            Can be `True`, `False`, or one of "minimal", "low", "medium", "high", "xhigh".
        max_tokens: Maximum number of tokens to generate in the model's response.
            For Anthropic models, this is explicitly set to 10,000 if `None` is provided,
            as their API requires it.
        log_raw_llm_response: If `True`, the raw LLM response (before parsing)
            will be printed to the console, useful for debugging.
        retry_config: An optional `RetryConfig` object specifying the retry
            strategy for transient HTTP errors. If `None`, a default configuration is used.
        role: Optional role name for the LLM call.

    Returns:
        The model's output, either as a string, an instance of the specified `output_type`,
        or `None` if an `UnexpectedModelBehavior` error occurs after all retries,
        or if non-retryable errors are encountered during the retry loop.

    Raises:
        TypeError: If `llm_model` is not a string or a `PydanticAI Model` instance.
        UserError: If the required API key for the model provider is not found
            in the environment.
        ModelHTTPError: A non-retryable HTTP error (e.g., 400, 401, 403) from the LLM provider.
        ModelAPIError: Indicates a non-HTTP network failure or a generic API issue.
        UsageLimitExceeded: The LLM provider's usage limits have been hit.
        Other AgentRunError subclasses: Propagate immediately as they indicate
            persistent configuration or quota problems.
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
    recorder = _RecordCallCapability()
    capabilities = [_WarnOnMaxTokensCapability(), recorder]
    if log_raw_llm_response:
        capabilities.append(_LogRawResponseCapability())
    agent = Agent(
        model,
        output_type=output_type,
        output_retries=rc.max_retries,
        capabilities=capabilities,
    )

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

    # Track per-call_llm wall clock + attempt count for the metrics recorder.
    # Wall clock includes sleep-between-retries on purpose: that time is what
    # the run loop actually waits for.
    t_call_start = time.monotonic()
    attempt_count = 0
    ok = False
    try:
        delay = rc.initial_delay
        for attempt in range(rc.max_retries):
            attempt_count = attempt + 1
            try:
                result = await agent.run(user_input, model_settings=model_settings)
                ok = True
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
                    warnings.warn(
                        f"[call_llm] Non-retryable HTTP {e.status_code} error: {e}"
                    )
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
    finally:
        # Record once per call_llm invocation, success or failure. If the run
        # has no active RunMetrics (tests calling call_llm directly), this is a
        # no-op.
        metrics = get_active_metrics()
        if metrics is not None:
            latency_ms = (time.monotonic() - t_call_start) * 1000.0
            resolved_model = recorder.model_name or (
                llm_model
                if isinstance(llm_model, str)
                else getattr(llm_model, "model_name", "unknown")
            )
            metrics.record_llm_call(
                role=role or "unknown",
                model=str(resolved_model),
                latency_ms=latency_ms,
                in_tokens=recorder.input_tokens,
                out_tokens=recorder.output_tokens,
                finish_reason=recorder.finish_reason,
                retries=max(0, attempt_count - 1),
                ok=ok,
            )
