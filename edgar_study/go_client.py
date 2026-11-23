"""Minimal OpenCode Go chat-completions client with a spend ledger and a paid-model guard."""

import asyncio
import base64
import contextvars
import json
import os
import time
from pathlib import Path

import httpx

from edgar_study.upstream import code_key

BASE_URL = "https://opencode.ai/inference/go/openai/v1"
USER_AGENT = "edgar-deepseek/0.1"
SPEND_CAP_USD = 9.50

# Set by wrappers around the upstream call sites, so each request records its role.
role = contextvars.ContextVar("role", default="unknown")


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


def headers(key):
    return {"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT}


def fetch_models(key):
    """The Go model list, keyed by model ID."""
    listing = httpx.get(f"{BASE_URL}/models", headers=headers(key), timeout=30).json()
    return {entry["id"]: entry for entry in listing["data"]}


def is_free(entry):
    price = entry["cost"]["text"]
    return price["input"]["price"] == 0 and price["output"]["price"] == 0


def list_cost(usage):
    """List-equivalent USD for one response's `usage` object."""
    return float((usage.get("opencode") or {}).get("list_cost") or usage.get("cost") or 0.0)


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


class GoClient:
    def __init__(self, run_dir, ledger, session_id, free_models, allow_paid=False, max_concurrency=16,
                 max_tokens=32_000, timeout_s=600):
        self.calls_path = Path(run_dir) / "api_calls.jsonl"
        self.calls_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger = ledger
        self.session_id = session_id
        self.free_models = set(free_models)
        self.allow_paid = allow_paid
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_tokens = max_tokens
        self._http = httpx.AsyncClient(base_url=BASE_URL, timeout=timeout_s,
                                       headers={**headers(load_key()), "x-opencode-session": session_id})

    async def chat(self, model, prompt, temperature=None, effort=None, img_bytes=None, **meta):
        """Return the reply text, or None on failure (upstream scores None as a failed program)."""
        if model not in self.free_models and not self.allow_paid:
            raise BudgetExhausted(f"{model} is not a free model; pass --allow-paid to use it.")
        self.ledger.check()
        content = prompt
        if img_bytes is not None:
            data = base64.b64encode(img_bytes).decode()
            content = [{"type": "text", "text": prompt},
                       {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}]
        body = {"model": model, "messages": [{"role": "user", "content": content}],
                "max_tokens": self.max_tokens}
        if temperature is not None:
            body["temperature"] = temperature
        if effort:
            body["reasoning_effort"] = effort
        record = {"role": role.get(), "model": model, "reasoning_effort": effort, "temperature": temperature,
                  **meta, "image": img_bytes is not None, "prompt_sha256": code_key(prompt)}

        for attempt in (1, 2):
            async with self.semaphore:
                started = time.time()
                try:
                    response = await self._http.post("/chat/completions", json=body)
                    status, payload = response.status_code, response.json()
                except (httpx.HTTPError, ValueError) as error:
                    status, payload = None, {"error": f"{type(error).__name__}: {error}"}
            usage = payload.get("usage") or {}
            self.ledger.add(list_cost(usage), model=model, role=record["role"], session=self.session_id)
            choice = (payload.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content") or None
            self._log({**record, "attempt": attempt, "status": status,
                       "latency_s": round(time.time() - started, 3),
                       "served_model": payload.get("model"), "usage": usage,
                       "finish_reason": choice.get("finish_reason"), "content_chars": len(text or ""),
                       "error": None if status == 200 else str(payload.get("error", payload))[:500]})
            if status == 200:
                return text
            retry = status is None or status >= 500 or (status == 429 and "quota" not in str(payload).lower())
            if not retry or attempt == 2:
                return None
            await asyncio.sleep(5)

    def _log(self, row):
        with self.calls_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    async def close(self):
        await self._http.aclose()
