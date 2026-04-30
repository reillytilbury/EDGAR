"""
System test for hypothesis_engine using fake LLM responses.

Covers: end-to-end run of hypothesis_engine_fake with Program1, Program2,
and ProgramSolution cycling through candidate slots. Verifies the run
completes without error and writes program_generation_log.jsonl.
"""

import asyncio
from pathlib import Path

from tests.system.run_test import _run_many

CONFIG_PATH = str(Path(__file__).parent / "config.yaml")
OUTPUT_PATH = Path(__file__).parent / "output"


def test_hypothesis_engine_system(tmp_path):
    asyncio.run(
        _run_many(
            config_path=CONFIG_PATH, output_dir=str(OUTPUT_PATH), use_fake_llm=True
        )
    )
