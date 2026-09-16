"""Headless Claude Code adapter — a pydantic_ai Model that shells out to the
local ``claude`` CLI instead of hitting Anthropic's API.
Useful when the operator has a Claude Code CLI available but no API budget
(subscription-plan users, Claude for Enterprise seat-holders, offline sandboxes
with a local Claude proxy, etc.).
Invocation shape (per LLM call):
    claude -p --output-format json [--json-schema '<schema>']
           [--max-turns 1] [--bare] '<prompt>'
The CLI returns a JSON envelope; we extract the result string, and — when a
structured output tool is expected — parse it back into the tool call's
arg dict. When no output tool is expected, we return the string verbatim as
a ``TextPart``.
Registered in ``llm_calling._build_model`` under the ``claude-code-`` name
prefix; ``ValidLLMs`` also carries a matching entry.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage


@dataclass
class _ClaudeHeadlessConfig:
    """Behavior knobs; sensible defaults for the EDGAR use case."""

    binary: str = "claude"
    """Executable name (or absolute path) — resolved via ``$PATH``."""

    bare: bool = False
    """Whether to pass ``--bare`` (skips hooks / auto-memory / CLAUDE.md, etc.).
    Defaults to False because ``--bare`` disables OAuth/keychain auth — it
    strictly requires ``ANTHROPIC_API_KEY``, which defeats the "use my logged-in
    Claude Code session" motivation. Turn it on only if you have an API key set
    AND want the leaner behavior.
    """

    timeout_s: float = 300.0
    """Kill the subprocess after this many seconds (headless Claude sometimes
    hangs when the schema is very complex). Roomy default for our small
    prompts; caller can override via env var if needed."""


class ClaudeHeadlessModel(Model):
    """A pydantic_ai ``Model`` that delegates each request to a ``claude -p`` subprocess.
    Handles two response shapes:
    * **Structured output** (``output_tools`` non-empty): sends
      ``--json-schema`` derived from the first output tool's parameters schema,
      parses the JSON reply, wraps it in a ``ToolCallPart`` so pydantic_ai's
      Agent extracts it as the structured type.
    * **Free text** (no output tools): returns the raw string as a ``TextPart``.
    Function tools (mid-conversation tool calling) are not supported — EDGAR
    doesn't use them, and translating them to Claude Code's own tool loop is
    a much bigger project.
    """

    def __init__(self, config: _ClaudeHeadlessConfig | None = None) -> None:
        self._config = config or _ClaudeHeadlessConfig()
        if shutil.which(self._config.binary) is None:
            raise RuntimeError(
                f"claude CLI not found on $PATH (looked for {self._config.binary!r}). "
                "Install Claude Code (https://docs.claude.com/claude-code) or "
                "set the correct binary in the ClaudeHeadlessModel config."
            )

    # ── pydantic_ai Model interface ──

    @property
    def model_name(self) -> str:
        return "claude-code-headless"

    @property
    def system(self) -> str:
        return "anthropic"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        if model_request_parameters.function_tools:
            raise NotImplementedError(
                "ClaudeHeadlessModel does not support in-conversation function tools; "
                "EDGAR only uses output_tools (structured output), which IS supported."
            )
        if model_request_parameters.native_tools:
            raise NotImplementedError(
                "ClaudeHeadlessModel does not support native/built-in tools."
            )

        prompt = _flatten_prompt(messages)
        output_tools = model_request_parameters.output_tools

        argv = [self._config.binary, "-p", "--output-format", "json"]
        if self._config.bare:
            argv.append("--bare")

        if output_tools:
            # Ask Claude to produce JSON matching the first output tool's schema.
            # EDGAR calls only ever register exactly one output tool.
            schema = output_tools[0].parameters_json_schema
            argv.extend(["--json-schema", json.dumps(schema)])

        argv.append(prompt)

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=self._config.timeout_s
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(
                f"claude subprocess timed out after {self._config.timeout_s}s "
                f"for a {'structured' if output_tools else 'text'} request."
            )

        if proc.returncode != 0:
            raise RuntimeError(
                f"claude subprocess failed with exit code {proc.returncode}:\n"
                f"stderr:\n{stderr_b.decode(errors='replace')}"
            )

        envelope = _parse_json_envelope(stdout_b.decode(errors="replace"))
        input_tokens, output_tokens = _extract_usage(envelope)

        if envelope.get("is_error"):
            raise RuntimeError(
                f"claude reported an error: {envelope.get('result', '(no message)')}\n"
                f"api_error_status={envelope.get('api_error_status')}, "
                f"stop_reason={envelope.get('stop_reason')}"
            )

        if output_tools:
            # Claude Code puts the schema-conformant JSON in ``structured_output``;
            # the ``result`` field is a human-readable summary ("Provided.").
            args_dict = envelope.get("structured_output")
            if not isinstance(args_dict, dict):
                # Fallback: try to parse ``result`` as JSON, in case a future CLI
                # version stops emitting ``structured_output``.
                args_dict = _parse_structured_result(
                    _extract_result_text(envelope), output_tools[0]
                )
            part = ToolCallPart(
                tool_name=output_tools[0].name,
                args=args_dict,
                tool_call_id=f"claude_code_headless__{output_tools[0].name}",
            )
        else:
            part = TextPart(content=_extract_result_text(envelope))

        return ModelResponse(
            parts=[part],
            model_name=self.model_name,
            usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        )


# ── helpers ──


def _flatten_prompt(messages: list[ModelMessage]) -> str:
    """Concatenate System + User parts across the message list into a single string.
    EDGAR always passes the prompt as one ``UserPromptPart``, sometimes preceded
    by a system prompt part; we serialize everything to a single string that
    ``claude -p`` accepts as the positional prompt argument.
    """
    chunks: list[str] = []
    for msg in messages:
        if not isinstance(msg, ModelRequest):
            continue
        for part in msg.parts:
            if isinstance(part, SystemPromptPart):
                chunks.append(f"SYSTEM:\n{part.content}\n")
            elif isinstance(part, UserPromptPart):
                if isinstance(part.content, str):
                    chunks.append(part.content)
                else:
                    # Multimodal (image bytes) not supported in the CLI path yet;
                    # keep only text parts and warn via the exception message on
                    # empty prompts.
                    for item in part.content:
                        if isinstance(item, str):
                            chunks.append(item)
    if not chunks:
        raise ValueError(
            "No text prompt could be extracted from messages — "
            "ClaudeHeadlessModel does not currently accept image-only inputs."
        )
    return "\n".join(chunks)


def _parse_json_envelope(stdout: str) -> dict[str, Any]:
    """Parse ``claude -p --output-format json`` output.
    Claude Code emits either a single JSON object per invocation (the default
    for ``--output-format json``), or one JSON object per line if it decides
    to stream. Take the last non-empty line and parse; if that fails, try the
    full stdout.
    """
    stdout = stdout.strip()
    if not stdout:
        raise RuntimeError("claude produced empty stdout.")
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    for candidate in (lines[-1], stdout):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"claude stdout was not JSON:\n{stdout[:500]}")


def _extract_result_text(envelope: dict[str, Any]) -> str:
    """Pull the assistant's response text out of the ``claude -p`` JSON envelope.
    Known shape: ``{"result": "...", "usage": {...}, ...}``. Some Claude Code
    versions nest it under ``"response"`` or embed it in ``"messages"[-1]``;
    handle those fallbacks so we're not brittle to CLI upgrades.
    """
    for key in ("result", "response", "text", "output"):
        val = envelope.get(key)
        if isinstance(val, str) and val.strip():
            return val
    messages = envelope.get("messages")
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict):
            content = last.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list) and content:
                # anthropic-style content list of {type, text}
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "")
    raise RuntimeError(
        f"could not find result text in claude JSON envelope keys={list(envelope.keys())}"
    )


def _extract_usage(envelope: dict[str, Any]) -> tuple[int, int]:
    """Best-effort token counts. Zero if the CLI didn't include usage."""
    usage = envelope.get("usage") or envelope.get("total_usage") or {}
    in_tokens = int(usage.get("input_tokens", 0) or 0)
    out_tokens = int(usage.get("output_tokens", 0) or 0)
    return in_tokens, out_tokens


def _parse_structured_result(result_text: str, tool: Any) -> dict[str, Any]:
    """Extract the JSON object that matches the tool's parameters schema.
    ``claude --json-schema`` forces the CLI to produce valid JSON matching the
    schema. In practice the result field is that JSON as a string. If the model
    surrounded it with markdown fences, strip them.
    """
    text = result_text.strip()
    if text.startswith("```"):
        # strip markdown fences: ```json ... ``` or ``` ... ```
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"claude structured output was not JSON for tool {tool.name!r}: {e}\n"
            f"raw text (first 500 chars):\n{result_text[:500]}"
        )
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"claude structured output for tool {tool.name!r} must be an object, "
            f"got {type(parsed).__name__}"
        )
    return parsed


def build_claude_headless_model(_model_name: str) -> ClaudeHeadlessModel:
    """Constructor used by ``llm_calling._build_model`` for the ``claude-code-`` prefix.
    Reads optional env-var overrides:
      * ``CLAUDE_HEADLESS_BINARY``  — path/name of the ``claude`` executable
      * ``CLAUDE_HEADLESS_TIMEOUT`` — subprocess timeout in seconds
      * ``CLAUDE_HEADLESS_BARE``    — set to ``"0"`` to disable ``--bare``
    """
    cfg = _ClaudeHeadlessConfig(
        binary=os.getenv("CLAUDE_HEADLESS_BINARY", "claude"),
        timeout_s=float(os.getenv("CLAUDE_HEADLESS_TIMEOUT", "300")),
        bare=os.getenv("CLAUDE_HEADLESS_BARE", "1") != "0",
    )
    return ClaudeHeadlessModel(cfg)