"""latex_cache.py — on-demand LaTeX equation rendering for evolved programs.

Asks the LLM (using the same model the run was originally executed against) to
read the numpy `model` source and emit only the LaTeX equations it implements.
Result is cached on disk at `<run_dir>/latex_cache/{idx}.json` so subsequent
requests are instant and free.

Reused with light edits from the pattern in tutorials/inspect_outputs.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from ..io.config import RetryConfig
from ..io.status import atomic_write_text


CACHE_DIRNAME = "latex_cache"


def _cache_path(run_dir: Path, idx: int) -> Path:
    return Path(run_dir) / CACHE_DIRNAME / f"{idx}.json"


def read_cached_latex(run_dir: Path, idx: int) -> dict | None:
    path = _cache_path(run_dir, idx)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


async def get_or_generate_latex(
    run_dir: Path,
    idx: int,
    program_detail: dict,
    force: bool = False,
) -> dict:
    """Return cached LaTeX if present, otherwise generate and cache.

    Raises RuntimeError if the LLM call fails (e.g. missing API key, quota).
    """
    run_dir = Path(run_dir)
    if not force:
        cached = read_cached_latex(run_dir, idx)
        if cached is not None:
            return {**cached, "cached": True}

    model_code = (program_detail.get("code") or {}).get("model") or ""
    if not model_code:
        raise RuntimeError(f"program {idx} has no model source")

    name = program_detail.get("name") or f"P{idx}"

    spec_path = run_dir / "task_spec.yaml"
    llm_model = _llm_from_task_spec(spec_path)
    if not llm_model:
        raise RuntimeError(
            "no model_llm found in task_spec.yaml; can't pick an LLM for LaTeX generation"
        )

    prompt = _LATEX_PROMPT.format(name=name, code=model_code)

    try:
        from ..llm.llm_calling import call_llm
    except ModuleNotFoundError as e:
        import sys

        raise RuntimeError(
            f"LLM dependencies are missing in {sys.executable!r} "
            f"(failed to import {e.name!r}). Activate the 'edgar' conda env "
            "(or `pip install -e .` from this repo) and restart the dashboard."
        ) from e
    try:
        retry_config = RetryConfig()
        latex = await call_llm(
            prompt=prompt,
            llm_model=llm_model,
            output_type=str,
            temperature=0.2,
            retry_config=retry_config,
        )
    except Exception as e:  # noqa: BLE001 — surface everything as a clean error
        raise RuntimeError(f"LLM call failed: {type(e).__name__}: {e}") from e

    if not latex:
        raise RuntimeError("LLM returned an empty LaTeX response")

    payload = {
        "idx": idx,
        "name": name,
        "llm": llm_model,
        "latex": latex,
        "generated_at": time.time(),
    }
    cache_path = _cache_path(run_dir, idx)
    atomic_write_text(cache_path, json.dumps(payload, indent=2))
    return {**payload, "cached": False}


def _llm_from_task_spec(spec_path: Path) -> str | None:
    if not spec_path.exists():
        return None
    try:
        with open(spec_path) as f:
            spec = yaml.safe_load(f) or {}
    except yaml.YAMLError:
        return None
    llms = spec.get("llms") or {}
    return llms.get("model_llm")


_LATEX_PROMPT = """\
You are given the numpy source of a parametric model. Output ONLY the LaTeX
equations the code implements - no prose, no explanation, no code fences.
Wrap display equations in $$...$$ so they render in Markdown. Define every
symbol you use in a brief variable-key block after the equations.

Model: {name!r}

```python
{code}
```
"""
