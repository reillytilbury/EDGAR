"""
Config: a flat, serialisable bundle of all settings needed to build a TaskSpec.

Two constructors:
    Config.from_yaml(path)       — load a config.yaml and merge with project defaults
    Config.from_taskspec(path)   — extract config from a task_spec.yaml saved by a previous run

Private helpers (_deep_merge, _git_state) are also used by TaskSpec.
"""
from __future__ import annotations

import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import yaml


def _warn_unknown_keys(config: dict) -> None:
    default = yaml.safe_load((PROJECT_ROOT / "projects" / "config_default.yaml").read_text()) or {}
    for section, default_section in default.items():
        if section == "project_params" or not isinstance(default_section, dict):
            continue
        known = set(default_section.keys())
        for key in config.get(section, {}):
            if key not in known:
                warnings.warn(
                    f"Unknown config key '{section}.{key}' — it will be ignored. "
                    f"Only project_params accepts arbitrary keys.",
                    stacklevel=4,
                )


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _git_state() -> tuple[str, bool]:
    """Return (sha, dirty) for the current git HEAD."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        sha = "unknown"
    try:
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip())
    except Exception:
        dirty = True
    return sha, dirty


@dataclass
class Config:
    """
    All settings needed to construct a TaskSpec, in plain-dict form.

    Constructed via from_yaml (live run) or from_taskspec (reproduce a past run).
    TaskSpec.from_config(config) turns this into a fully-loaded TaskSpec.
    """
    task_name: str
    project_dir: Path
    io: dict
    evolution: dict
    llms: dict
    scoring: dict
    project_params: dict
    prompts: dict

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        """
        Load a config.yaml and merge with project defaults.

        Args:
            path: Path to a project config.yaml.
        """
        path = Path(path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        task_name = path.parent.name
        project_dir = path.parent
        default_config = yaml.safe_load((PROJECT_ROOT / "projects" / "config_default.yaml").read_text()) or {}
        config = _deep_merge(default_config, yaml.safe_load(path.read_text()) or {})

        default_prompts = yaml.safe_load((PROJECT_ROOT / "projects" / "prompt_defaults.yaml").read_text()) or {}
        task_prompt_path = project_dir / "prompts.yaml"
        task_prompts = yaml.safe_load(task_prompt_path.read_text()) if task_prompt_path.exists() else {}
        prompts = _deep_merge(default_prompts, task_prompts)
        _warn_unknown_keys(config)

        return cls(
            task_name=task_name,
            project_dir=project_dir,
            io=config.get("io", {}),
            evolution=config.get("evolution", {}),
            llms=config.get("llms", {}),
            scoring=config.get("scoring", {}),
            project_params=config.get("project_params", {}),
            prompts=prompts,
        )

    @classmethod
    def from_taskspec(cls, path: Path) -> Config:
        """
        Extract config from a task_spec.yaml saved by a previous run.

        Callables, git state, and creation_timestamp are always regenerated fresh
        when TaskSpec.from_config is later called — only the config dicts are preserved.

        Args:
            path: Path to a task_spec.yaml.
        """
        path = Path(path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        record = yaml.safe_load(path.read_text())
        schemas = record["prompt_schemas"]
        prompts = {
            "model": schemas["model"],
            "parameter_estimator": schemas["param_est"],
            "jax_translator": schemas["jax"],
        }
        return cls(
            task_name=record["task_name"],
            project_dir=PROJECT_ROOT / "projects" / record["task_name"],
            io=record.get("io", {}),
            evolution=record.get("evolution", {}),
            llms=record.get("llms", {}),
            scoring=record.get("scoring", {}),
            project_params=record.get("project_params", {}),
            prompts=prompts,
        )
