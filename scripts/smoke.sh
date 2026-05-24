#!/usr/bin/env bash
# scripts/smoke.sh — minimal end-to-end pipeline run.
#
# Exercises every layer (config → loader → JAX scoring → LLM → outputs) with
# the smallest possible inputs (1 generation, 2 islands, batch size 2). Use
# this after any non-trivial change to confirm nothing is fundamentally wired
# wrong. Expect ~30s wallclock; will likely hit Gemini's free-tier 429 after a
# few LLM calls — that's the quota, not a failure.
#
# Usage:
#   scripts/smoke.sh                                # uses orientation_tuning
#   scripts/smoke.sh projects/<task>/config.yaml    # any other project
#
# Env:
#   EDGAR_PYTHON  Override the python interpreter (default: the edgar conda env)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${1:-projects/orientation_tuning/config.yaml}"
PYTHON="${EDGAR_PYTHON:-/opt/homebrew/Caskroom/miniforge/base/envs/edgar/bin/python}"

exec "$PYTHON" -m src.cli run "$CONFIG" \
    --evolution.n_generations=1 \
    --evolution.n_islands=2 \
    --evolution.batch_size=2 \
    --evolution.topology='[1,0]' \
    --scoring.timeout_s=60 \
    --scoring.gradient_descent.max_iter=50
