"""
Config loading and merging: reads yaml configs, imports project modules, captures git state.

Entry point: build_task_spec_from_config(config_path) → TaskSpec

Internal steps:
1. Load default config (projects/config_default.yaml)
2. Load task config (projects/<task_name>/config.yaml)
3. Deep-merge overrides into defaults
4. Load and merge prompt schemas (defaults + task-specific)
5. Import project callables (load_data_fn, loss_fn, plot_fn, seed_programs)
6. Capture git state (SHA, dirty)
7. Return fully populated TaskSpec

Example usage:
--------------
    spec = build_task_spec_from_config("projects/my_task/config.yaml")
    # Now spec has all config, callables, seed programs, and schemas
"""
from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import yaml

from ..evolution.program import Program, BirthCertificate, Code
from ..llm.prompt_schema import PromptSchema
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


def _load_yaml(path: Path) -> dict:
    """
    Load YAML file and return dict (or empty dict if file is empty).

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML dict or empty dict
    """
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_merged_config(config_path: Path) -> dict:
    """
    Load and merge default config with task-specific config.

    Args:
        config_path: Path to projects/<task_name>/config.yaml

    Returns:
        Merged config dict with task overrides applied
    """
    default_path = PROJECT_ROOT / "projects" / "config_default.yaml"
    default = _load_yaml(default_path)
    task = _load_yaml(config_path)
    return _deep_merge(default, task)


def _load_merged_prompts(task_name: str) -> dict:
    """
    Load and merge default prompts with task-specific prompts.

    Args:
        task_name: Task folder name

    Returns:
        Merged prompt dict with task overrides applied (uses defaults if task has no prompts.yaml)
    """
    default_path = PROJECT_ROOT / "projects" / "prompt_defaults.yaml"
    task_path = PROJECT_ROOT / "projects" / task_name / "prompts.yaml"
    default = _load_yaml(default_path)
    if task_path.exists():
        task = _load_yaml(task_path)
        return _deep_merge(default, task)
    return default


def _build_prompt_schema(prompt_dict: dict) -> PromptSchema:
    """
    Build a PromptSchema from a dict.

    Args:
        prompt_dict: Dict with prompt schema keys

    Returns:
        PromptSchema instance
    """
    return PromptSchema(**prompt_dict)


def _import_fn(module_path: str, fn_name: str):
    """
    Dynamically import a function from a module path.

    Args:
        module_path: Module path (e.g., "projects.my_task.data_loader.load_data")
        fn_name: Function name to import

    Returns:
        The imported callable

    Raises:
        ModuleNotFoundError: If module doesn't exist
        ValueError: If function doesn't exist or is not callable
    """
    mod = importlib.import_module(module_path)
    fn = getattr(mod, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"{module_path} must define callable {fn_name}()")
    return fn


def _import_fn_optional(module_path: str, fn_name: str):
    """
    Dynamically import a function, returning None if not found or not callable.

    Args:
        module_path: Module path (e.g., "projects.my_task.image_feedback.plot")
        fn_name: Function name to import

    Returns:
        The imported callable, or None if import fails
    """
    try:
        return _import_fn(module_path, fn_name)
    except (ModuleNotFoundError, ValueError):
        return None


def _git_sha() -> str:
    """
    Get current git commit SHA.

    Returns:
        Git SHA string, or "unknown" if not a git repo or git unavailable
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    """
    Check if working tree has uncommitted changes.

    Returns:
        True if there are uncommitted changes, False otherwise (or True on error)
    """
    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            text=True,
        )
        return len(result.strip()) > 0
    except Exception:
        return True


def _load_seed_programs(task_name: str) -> list[Program]:
    """
    Load two seed programs (model1 + param_est1, model2 + param_est2).

    Args:
        task_name: Task folder name

    Returns:
        List of 2 Program objects with source code captured
    """
    import inspect

    programs = []
    for i in range(1, 3):
        model_fn = _import_fn(f"projects.{task_name}.seed_programs.model{i}", "model")
        param_est_fn = _import_fn(f"projects.{task_name}.seed_programs.param_est{i}", "parameter_estimator")
        model_code = inspect.getsource(inspect.getmodule(model_fn))
        param_est_code = inspect.getsource(inspect.getmodule(param_est_fn))
        programs.append(Program(
            birth=BirthCertificate(
                generation=-1,
                island=-1,
                batch_index=i - 1,
                mode="seed",
            ),
            code=Code(model=model_code, param_est=param_est_code),
        ))
    return programs


def build_task_spec_from_config(config_path: Path) -> TaskSpec:
    """
    Load config, merge with defaults, import project code, and return fully populated TaskSpec.

    Args:
        config_path: Path to projects/<task_name>/config.yaml (can be relative or absolute)

    Returns:
        TaskSpec with all fields populated from merged config, imported callables, and seed programs

    Raises:
        FileNotFoundError: If config or required files don't exist
        ImportError: If project modules can't be imported
        ValueError: If required functions are missing
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = _load_merged_config(config_path)
    # task_name is the folder name: projects/<task_name>/config.yaml
    task_name = config_path.parent.name

    prompts = _load_merged_prompts(task_name)

    return TaskSpec(
        task_name=task_name,
        git_sha=_git_sha(),
        git_dirty=_git_dirty(),
        io=config.get("io", {}),
        evolution=config.get("evolution", {}),
        llms=config.get("llms", {}),
        scoring=config.get("scoring", {}),
        project_params=config.get("project_params", {}),
        model_prompt_schema=_build_prompt_schema(prompts["model"]),
        param_est_prompt_schema=_build_prompt_schema(prompts["parameter_estimator"]),
        jax_prompt_schema=_build_prompt_schema(prompts["jax_translator"]),
        load_data_fn=_import_fn(f"projects.{task_name}.data_loader.load_data", "load_and_process_data"),
        loss_fn=_import_fn(f"projects.{task_name}.data_loader.load_data", "loss_fn"),
        plot_fn=_import_fn_optional(f"projects.{task_name}.image_feedback.plot", "plot_model_fits"),
        seed_programs=_load_seed_programs(task_name),
    )
