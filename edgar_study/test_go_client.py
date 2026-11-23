"""Offline checks for the Go client, ledger, paid-model guard and tier mapping:
uv run python -m edgar_study.test_go_client"""

import asyncio
import json
import os
import tempfile
import types
from pathlib import Path

import httpx

from edgar_study.engine import install
from edgar_study.go_client import BudgetExhausted, GoClient, Ledger, role


def fake_transport(replies, bodies):
    def handler(request):
        bodies.append(json.loads(request.content))
        status, payload = replies.pop(0)
        return httpx.Response(status, json=payload)
    return httpx.MockTransport(handler)


def ok(text, list_cost=0.0):
    return 200, {"model": "m", "choices": [{"message": {"content": text}}],
                 "usage": {"completion_tokens": 3, "opencode": {"list_cost": list_cost}}}


def fake_upstream():
    """Just enough of the upstream modules for engine.install to patch."""
    async def coroutine(*args, **kwargs):
        return None, None
    utils = types.SimpleNamespace(create_program_prompt=None, create_parameter_estimator_prompt=None,
                                  create_linked_prompt=None)
    engine = types.SimpleNamespace(genai=types.SimpleNamespace(Client=None), translate_to_jax=coroutine,
                                   generate_new_neuron_model=coroutine,
                                   generate_new_parameter_estimator=coroutine)
    return engine, utils


async def main(directory):
    os.environ.pop("OPENCODE_API_KEY_FILE", None)
    os.environ["OPENCODE_API_KEY"] = "test-key"
    ledger = Ledger(directory / "spend.jsonl", cap=1.0)
    go = GoClient(directory, ledger, session_id="test", free_models=["hy4-flash-free"])
    replies = [ok("```python\ndef neuron_model(theta): return theta\n```"),
               (500, {"error": "boom"}), ok("second try"),
               (401, {"error": "bad key"}), ok("")]
    bodies = []
    go._http = httpx.AsyncClient(base_url="https://example.invalid", transport=fake_transport(replies, bodies))

    # Upstream's tiers and temperatures are mapped once, in the patched call_llm_async.
    engine, utils = fake_upstream()
    install(engine, utils, go, default_effort="none", temperature_scale=0.5)
    role.set("translation")
    assert "neuron_model" in await utils.call_llm_async("prompt", model_name="hy4-flash-free", temperature=0)
    assert await utils.call_llm_async("prompt", model_name="hy4-flash-free#high", temperature=1.8,
                                      img_bytes=b"png") == "second try"  # retried 5xx
    assert await go.chat("hy4-flash-free", "prompt") is None  # 401 is not retried
    assert await go.chat("hy4-flash-free", "prompt") is None  # empty content counts as failure
    assert bodies[0]["reasoning_effort"] == "none" and bodies[0]["temperature"] == 0
    assert bodies[1]["model"] == "hy4-flash-free" and bodies[1]["reasoning_effort"] == "high"
    assert bodies[1]["temperature"] == 0.9
    try:
        await go.chat("deepseek-v4.1-flash", "prompt")
        raise AssertionError("paid model should be refused without --allow-paid")
    except BudgetExhausted:
        pass

    rows = [json.loads(line) for line in (directory / "api_calls.jsonl").read_text().splitlines()]
    assert [row["status"] for row in rows] == [200, 500, 200, 401, 200]
    assert rows[0]["role"] == "translation" and rows[1]["image"] is True
    assert rows[1]["temperature_upstream"] == 1.8
    assert "test-key" not in (directory / "api_calls.jsonl").read_text()

    # The ledger refuses new requests once list-equivalent spend reaches the cap, across instances.
    ledger.add(1.0, model="x")
    try:
        Ledger(directory / "spend.jsonl", cap=1.0).check()
        raise AssertionError("ledger should be exhausted")
    except BudgetExhausted:
        pass
    await go.close()
    print("Go client checks passed.")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        asyncio.run(main(Path(temp)))
