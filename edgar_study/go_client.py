"""Minimal OpenCode Go chat-completions client with a spend ledger and a paid-model guard."""

import asyncio
import base64
import contextvars
import hashlib
import json
import os
import time
from pathlib import Path

import httpx

BASE_URL = "https://opencode.ai/inference/go/openai/v1"
USER_AGENT = "edgar-deepseek/0.1"
# Models priced at zero in the Go model list on 23 Nov 2026.
FREE_MODELS = {"hy4-flash-free", "beam-free", "iceman-contributor-free"}
SPEND_CAP_USD = 9.50

# Set by wrappers around the upstream call sites, so each request records its role.
role = contextvars.ContextVar("role", default="proposal")


class BudgetExhausted(RuntimeError):
    pass


def load_key():
    path = os.environ.get("OPENCODE_API_KEY_FILE")
    if path:
        return Path(path).read_text().strip()
    key = os.environ.get("OPENCODE_API_KEY")
    if not key:
        raise RuntimeError("Set OPENCODE_API_KEY_FILE (preferred) or OPENCODE_API_KEY.")
    return key


class Ledger:
    """Append-only record of list-equivalent spend, shared by every run."""

    def __init__(self, path, cap=SPEND_CAP_USD):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cap = cap
        self.total = 0.0
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                self.total += json.loads(line)["usd"]

    def check(self):
        if self.total >= self.cap:
            raise BudgetExhausted(f"Spend ledger at ${self.total:.4f} reached cap ${self.cap:.2f}.")

    def add(self, usd, **meta):
        self.total += usd
        with self.path.open("a") as stream:
            stream.write(json.dumps({"usd": usd, "time": time.time(), **meta}) + "\n")


def split_model(name):
    """'model#effort' selects a reasoning effort per tier while keeping one model ID."""
    model, _, effort = name.partition("#")
    return model, effort or None


class GoClient:
    def __init__(self, run_dir, ledger, session_id, allow_paid=False, max_concurrency=8,
                 max_tokens=32_000, reasoning_effort=None, temperature_scale=1.0, timeout_s=600):
        self.calls_path = Path(run_dir) / "api_calls.jsonl"
        self.calls_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger = ledger
        self.session_id = session_id
        self.allow_paid = allow_paid
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.temperature_scale = temperature_scale
        self.timeout_s = timeout_s
        self._key = load_key()
        self._http = None

    def _guard(self, model):
        if model not in FREE_MODELS and not self.allow_paid:
            raise BudgetExhausted(f"{model} is not a free model; pass --allow-paid to use it.")
        self.ledger.check()

    async def chat(self, model, prompt, temperature=None, img_bytes=None):
        """Return the reply text, or None on failure (upstream scores None as a failed program)."""
        if prompt is None:
            return None
        model, effort = split_model(model)
        effort = effort or self.reasoning_effort
        self._guard(model)
        content = prompt
        if img_bytes is not None:
            data = base64.b64encode(img_bytes).decode()
            content = [{"type": "text", "text": prompt},
                       {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}]
        body = {"model": model, "messages": [{"role": "user", "content": content}],
                "max_tokens": self.max_tokens}
        if temperature is not None:
            body["temperature"] = round(temperature * self.temperature_scale, 4)
        if effort:
            body["reasoning_effort"] = effort
        headers = {"Authorization": f"Bearer {self._key}", "User-Agent": USER_AGENT,
                   "x-opencode-session": self.session_id}
        if self._http is None:
            self._http = httpx.AsyncClient(base_url=BASE_URL, timeout=self.timeout_s)

        record = {"role": role.get(), "model": model, "reasoning_effort": effort,
                  "temperature_upstream": temperature, "temperature_sent": body.get("temperature"),
                  "image": img_bytes is not None,
                  "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
        text = None
        async with self.semaphore:
            for attempt in (1, 2):
                started = time.time()
                try:
                    response = await self._http.post("/chat/completions", json=body, headers=headers)
                    status, payload = response.status_code, response.json()
                except (httpx.HTTPError, ValueError) as error:
                    status, payload = None, {"error": f"{type(error).__name__}: {error}"}
                usage = payload.get("usage") or {}
                usd = float((usage.get("opencode") or {}).get("list_cost") or usage.get("cost") or 0.0)
                self.ledger.add(usd, model=model, role=record["role"], session=self.session_id)
                choice = (payload.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                retry = status is None or status == 429 and "quota" not in str(payload).lower() or (status or 0) >= 500
                self._log({**record, "attempt": attempt, "status": status,
                           "latency_s": round(time.time() - started, 3),
                           "served_model": payload.get("model"), "usage": usage,
                           "finish_reason": choice.get("finish_reason"),
                           "content_chars": len(message.get("content") or ""),
                           "error": None if status == 200 else str(payload.get("error", payload))[:500]})
                if status == 200:
                    text = message.get("content") or None
                    break
                if not retry:
                    break
                await asyncio.sleep(5)
        return text

    def _log(self, row):
        with self.calls_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    async def close(self):
        if self._http is not None:
            await self._http.aclose()
