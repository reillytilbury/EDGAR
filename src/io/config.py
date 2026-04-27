"""
Config loading: reads yaml configs, imports project modules, captures git state,
and returns a fully populated TaskSpec.
"""
from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import yaml

from ..evolution.program import Program
from ..llm.prompt_schema import PromptSchema
from .task_spec import TaskSpec


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_merged_config(config_path: Path) -> dict:
    default_path = PROJECT_ROOT / "projects" / "config_default.yaml"
    default = _load_yaml(default_path)
    task = _load_yaml(config_path)
    return _deep_merge(default, task)


def _load_merged_prompts(task_name: str) -> dict:
    default_path = PROJECT_ROOT / "projects" / "prompt_defaults.yaml"
    task_path = PROJECT_ROOT / "projects" / task_name / "prompts.yaml"
    default = _load_yaml(default_path)
    if task_path.exists():
        task = _load_yaml(task_path)
        return _deep_merge(default, task)
    return default


def _build_prompt_schema(prompt_dict: dict) -> PromptSchema:
    return PromptSchema(**prompt_dict)


def _import_fn(module_path: str, fn_name: str):
    mod = importlib.import_module(module_path)
    fn = getattr(mod, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"{module_path} must define callable {fn_name}()")
    return fn


def _import_fn_optional(module_path: str, fn_name: str):
    try:
        return _import_fn(module_path, fn_name)
    except (ModuleNotFoundError, ValueError):
        return None


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
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
    import inspect

    programs = []
    for i in range(1, 3):
        model_fn = _import_fn(f"projects.{task_name}.seed_programs.model{i}", "model")
        param_est_fn = _import_fn(f"projects.{task_name}.seed_programs.param_est{i}", "parameter_estimator")
        model_code = inspect.getsource(inspect.getmodule(model_fn))
        param_est_code = inspect.getsource(inspect.getmodule(param_est_fn))
        programs.append(Program(
            uid=(-1, -1, i - 1),
            model_code=model_code,
            param_est_code=param_est_code,
            mode="seed",
        ))
    return programs


def build_task_spec_from_config(config_path: Path) -> TaskSpec:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = _load_merged_config(config_path)
    task_name = config["task"]["data_path"]  # used to derive task name from path
    # task_name is the folder name under projects/
    # infer from config_path: projects/<task_name>/config.yaml
    task_name = config_path.parent.name

    prompts = _load_merged_prompts(task_name)

    return TaskSpec(
        task_name=task_name,
        git_sha=_git_sha(),
        git_dirty=_git_dirty(),
        evolution=config.get("evolution", {}),
        llms=config.get("llms", {}),
        scoring=config.get("scoring", {}),
        model_prompt_schema=_build_prompt_schema(prompts["model"]),
        param_est_prompt_schema=_build_prompt_schema(prompts["parameter_estimator"]),
        jax_prompt_schema=_build_prompt_schema(prompts["jax_translator"]),
        load_data_fn=_import_fn(f"projects.{task_name}.data_loader.load_data", "load_and_process_data"),
        loss_fn=_import_fn(f"projects.{task_name}.data_loader.load_data", "loss_fn"),
        plot_fn=_import_fn_optional(f"projects.{task_name}.image_feedback.plot", "plot_model_fits"),
        data_processing_params=config.get("data_processing_params", {}),
        seed_programs=_load_seed_programs(task_name),
    )
