"""latex_cache.py — on-demand LaTeX equation rendering for evolved programs.

This module is responsible for translating the numpy `model` source code of
an evolved program into its mathematical LaTeX representation using an LLM.
The generated LaTeX is aggressively cached on disk to ensure that subsequent
requests are fast and do not incur additional LLM costs.

The process involves:
1. Checking for a cached LaTeX representation for a given program.
2. If not found or forced, extracting the program's numpy model source code.
3. Identifying the specific LLM (the `jax_model_translator_llm` from the run's
   `task_spec.yaml`) that was configured for JAX model translation.
4. Constructing a prompt (`_LATEX_PROMPT`) to instruct the LLM to output
   only the LaTeX equations.
5. Calling the LLM via `llm_calling.call_llm` with a low temperature for
   deterministic output and retries for robustness.
6. Atomically writing the LLM's LaTeX response to a JSON file
   (`run_dir/latex_cache/{idx}.json`) for persistent caching.

This mechanism ensures that the dashboard can display mathematical equations
for evolved models efficiently without re-invoking the LLM unless necessary.

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
    """Constructs the file path for a cached LaTeX equation.

    Args:
        run_dir: The root directory of the EDGAR run.
        idx: The global index of the program.

    Returns:
        The full path to the cached LaTeX JSON file for the given program.
    """
    return Path(run_dir) / CACHE_DIRNAME / f"{idx}.json"


def read_cached_latex(run_dir: Path, idx: int) -> dict | None:
    """Reads a cached LaTeX equation from disk.

    Checks if a cached LaTeX file exists for a given program index within
    the specified run directory and attempts to load its content.

    Args:
        run_dir: The root directory of the EDGAR run.
        idx: The global index of the program.

    Returns:
        A dictionary containing the cached LaTeX data if found and successfully
        parsed, otherwise `None`.
    """
    path = _cache_path(run_dir, idx)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        # If the file is corrupted or not valid JSON, treat as if not cached.
        return None


async def get_or_generate_latex(
    run_dir: Path,
    idx: int,
    program_detail: dict,
    force: bool = False,
) -> dict:
    """Returns cached LaTeX if present, otherwise generates and caches it using an LLM.

    This function first attempts to read the LaTeX representation of a program's
    model from a local cache. If no cached version is found, or if `force` is
    set to `True`, it invokes an LLM to generate the LaTeX from the program's
    numpy model source code. The generated LaTeX is then stored in the cache
    for future requests.

    Args:
        run_dir: The root directory of the EDGAR run.
        idx: The global index of the program for which to get or generate LaTeX.
        program_detail: A dictionary containing the program's details,
            specifically its model source code under `code.model` and its name.
        force: If `True`, forces the regeneration of LaTeX via the LLM,
            bypassing the cache. Defaults to `False`.

    Returns:
        A dictionary containing the LaTeX string and associated metadata,
        including an `idx`, `name`, the `llm` model used, the `latex` string
        itself, `generated_at` timestamp, and a `cached` boolean flag indicating
        if the result was from cache (`True`) or newly generated (`False`).

    Raises:
        RuntimeError: If the program has no model source code, if no
            `jax_model_translator_llm` is found in the `task_spec.yaml`,
            if the LLM call fails (e.g., missing API key, quota issues,
            network errors), or if the LLM returns an empty response.
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
            f"(failed to import {e.name!r}). This is likely due to running the "
            "dashboard from the wrong environment. Activate the 'edgar' conda env, "
            "`pip install -e .` from the repo root, or use the prefix `uv run` "
            "and restart the dashboard."
        ) from e
    try:
        retry_config = RetryConfig()
        latex = await call_llm(
            prompt=prompt,
            llm_model=llm_model,
            output_type=str,
            temperature=0.2,  # Low temperature for deterministic output
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
    """Extracts the LLM model name used for JAX model translation from the `task_spec.yaml`.

    This helper function reads the `task_spec.yaml` file from a run directory
    and retrieves the specific LLM configured for JAX model translation,
    which is then reused for LaTeX generation to maintain consistency.

    Args:
        spec_path: The file path to the `task_spec.yaml` for a given EDGAR run.

    Returns:
        The name of the LLM model (e.g., "gemini-pro") as a string, or `None`
        if the file does not exist, cannot be parsed, or the specific LLM
        configuration is not found.
    """
    if not spec_path.exists():
        return None
    try:
        with open(spec_path) as f:
            spec = yaml.safe_load(f) or {}
    except yaml.YAMLError:
        return None
    llms = spec.get("llms") or {}
    return llms.get("jax_model_translator_llm")


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
