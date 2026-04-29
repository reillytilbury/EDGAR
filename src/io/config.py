"""
Config loading and merging: reads yaml configs, loads project code, captures git state.

Entry point: build_task_spec_from_config(config_path) → TaskSpec

Internal steps:
1. Load default config + task config, deep-merge into single config dict
2. Load default prompts + task prompts, deep-merge into single prompts dict
3. Load project callables from source files (load_data_fn, loss_fn, plot_fn)
4. Capture git state (SHA, dirty)
5. Load seed programs (all model*.py + param_est*.py pairs)
6. Return fully populated TaskSpec

Example usage:
--------------
    spec = build_task_spec_from_config("projects/my_task/config.yaml")
    # Now spec has all config, callables, seed programs, and schemas
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from ..evolution.program import Program, BirthCertificate, Code
from ..llm.prompt_schema import PromptSchema
from ..llm.code_loading import load_function_from_source
from .task_spec import TaskSpec


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge override dict into base dict.

    Args:
        base: Base configuration dict
        override: Override values (takes precedence)

    Returns:
        Merged dict with overrides applied recursively
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _git_state() -> tuple[str, bool]:
    """
    Get current git SHA and working tree dirty state.

    Returns:
        tuple: (sha, dirty) where:
            - sha: Git commit SHA string, or "unknown" if not available
            - dirty: True if working tree has uncommitted changes, False otherwise (True on error)
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except Exception:
        sha = "unknown"

    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            text=True,
        )
        dirty = len(result.strip()) > 0
    except Exception:
        dirty = True

    return sha, dirty


def build_task_spec_from_config(config_path: Path) -> TaskSpec:
    """
    Load config and prompts, merge with defaults, load project code, and return fully populated TaskSpec.

    Args:
        config_path: Path to projects/<task_name>/config.yaml (can be relative or absolute)

    Returns:
        TaskSpec with all fields populated from merged config, loaded callables, and seed programs

    Raises:
        FileNotFoundError: If config or required files don't exist
        ValueError: If required callables (load_data, loss_fn) are missing from source files
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    task_name = config_path.parent.name

    # 1. Load and merge config
    default_config_path = PROJECT_ROOT / "projects" / "config_default.yaml"
    with open(default_config_path) as f:
        default_config = yaml.safe_load(f) or {}
    with open(config_path) as f:
        task_config = yaml.safe_load(f) or {}
    config = _deep_merge(default_config, task_config)

    # 2. Load and merge prompts
    default_prompt_path = PROJECT_ROOT / "projects" / "prompt_defaults.yaml"
    with open(default_prompt_path) as f:
        default_prompts = yaml.safe_load(f) or {}
    task_prompt_path = PROJECT_ROOT / "projects" / task_name / "prompts.yaml"
    if task_prompt_path.exists():
        with open(task_prompt_path) as f:
            task_prompts = yaml.safe_load(f) or {}
    else:
        task_prompts = {}
    prompts = _deep_merge(default_prompts, task_prompts)

    # 3. Load project callables from source files
    data_loader_path = PROJECT_ROOT / "projects" / task_name / "data_loader" / "load_data.py"
    load_data_fn = load_function_from_source(data_loader_path.read_text(), "load_data")
    if load_data_fn is None:
        raise ValueError(f"{data_loader_path} must define callable load_data()")

    loss_fn = load_function_from_source(data_loader_path.read_text(), "loss_fn")
    if loss_fn is None:
        raise ValueError(f"{data_loader_path} must define callable loss_fn()")

    plot_path = PROJECT_ROOT / "projects" / task_name / "image_feedback" / "plot.py"
    plot_fn = None
    if plot_path.exists():
        plot_fn = load_function_from_source(plot_path.read_text(), "plot_model_fits")

    # 4. Capture git state
    git_sha, git_dirty = _git_state()

    # 5. Load seed programs (all model*.py + param_est*.py pairs)
    seed_dir = PROJECT_ROOT / "projects" / task_name / "seed_programs"
    model_files = sorted(seed_dir.glob("model*.py"))
    seed_programs = []
    for batch_idx, model_path in enumerate(model_files):
        model_num = model_path.stem.replace("model", "")
        param_est_path = seed_dir / f"param_est{model_num}.py"
        model_code = model_path.read_text()
        param_est_code = param_est_path.read_text()
        seed_programs.append(Program(
            birth=BirthCertificate(
                generation=-1,
                island=-1,
                batch_index=batch_idx,
                mode="seed",
            ),
            code=Code(model=model_code, param_est=param_est_code),
        ))

    # 6. Return fully populated TaskSpec
    return TaskSpec(
        task_name=task_name,
        git_sha=git_sha,
        git_dirty=git_dirty,
        io=config.get("io", {}),
        evolution=config.get("evolution", {}),
        llms=config.get("llms", {}),
        scoring=config.get("scoring", {}),
        project_params=config.get("project_params", {}),
        model_prompt_schema=PromptSchema(**prompts["model"]),
        param_est_prompt_schema=PromptSchema(**prompts["parameter_estimator"]),
        jax_prompt_schema=PromptSchema(**prompts["jax_translator"]),
        load_data_fn=load_data_fn,
        loss_fn=loss_fn,
        plot_fn=plot_fn,
        seed_programs=seed_programs,
    )
