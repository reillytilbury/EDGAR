#!/usr/bin/env bash
# Checks which LLM API keys are configured and working.
# Reports pass/fail per provider; exits non-zero if any fail.
# Works with both uv and conda-activated environments.
set -e

if command -v uv &> /dev/null; then
    PY="uv run python"
else
    PY="python"
fi

echo "--- Pinging LLM providers ---"
$PY scripts/ping_llms.py
