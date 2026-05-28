#!/usr/bin/env bash
# Validates that edgar is importable and the fake-LLM pipeline runs end-to-end.
# Works with both uv and conda-activated environments.
set -e

if command -v uv &> /dev/null; then
    PY="uv run python"
    EDGAR="uv run edgar"
else
    PY="python"
    EDGAR="edgar"
fi

echo "--- Checking edgar import ---"
$PY -c "import edgar; print('OK: edgar imported')"

echo "--- Running edgar test-fake ---"
$EDGAR test-fake > /dev/null 2>&1 && echo "OK: test-fake passed" || { echo "ERROR: test-fake failed"; exit 1; }

echo "--- Environment OK ---"
