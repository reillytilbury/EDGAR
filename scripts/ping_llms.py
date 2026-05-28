"""Ping each LLM provider with a small real API call.

Skips a provider whose API key is not set. Cheap (~10 output tokens per call).
Use this to confirm your .env keys are wired before burning a full smoke run.

Usage:
    python scripts/ping_llms.py

Exit code is non-zero iff a provider that should have worked actually failed
(missing keys are skipped, not failed).
"""
import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from edgar.llm.llm_calling import call_llm  # noqa: E402

PROVIDER_PAIRS = [
    ("gemini-2.5-flash-lite", "GOOGLE_API_KEY"),
    ("claude-haiku-4-5", "ANTHROPIC_API_KEY"),
    ("opus-4-7", "ANTHROPIC_API_KEY"),
]

async def main() -> int:
    n_fail = 0
    for model_name, env_var in PROVIDER_PAIRS:
        if not os.getenv(env_var):
            print(f"[skip] {model_name} ({env_var} not set)")
            continue
        try:
            response = await call_llm(
                prompt="Reply with exactly the token 12345 and something else.",
                llm_model=model_name,
                output_type=str,
                max_tokens=20,
            )
        except Exception as e:
            print(f"[err ] {model_name}: {type(e).__name__}: {e}")
            n_fail += 1
            continue
        ok = "12345" in (response or "")
        marker = "ok  " if ok else "fail"
        print(f"[{marker}] {model_name} ({env_var} set)")
        if not ok:
            n_fail += 1
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
