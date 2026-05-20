"""
Config: a flat, serialisable bundle of all settings needed to build a TaskSpec.

Two constructors:
    Config.from_yaml(path)       — load a config.yaml and merge with project defaults
    Config.from_taskspec(path)   — extract config from a task_spec.yaml saved by a previous run

Private helpers (_deep_merge, _git_state) are also used by TaskSpec.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal
from dataclasses import field
import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from ..llm.prompt_schema import PromptSchema

ValidLLMs = Literal["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro",
                    "gemini-3-pro-preview", "gemini-3-flash-preview","gemini-3.1-flash-lite", "gemini-3.1-pro-preview"] #List of supported LLMs, update as needed


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

# ── sub-models ──
# Here we define the types of all expected config sections, and define any validation logic. Use this to ensure parameters have desired properties/types

class _LaxModel(BaseModel):
    """ Pydantic BaseModel which raises a warning if there are unexpected fields.
    """
    model_config = ConfigDict(extra="ignore") #Doesnt raise an error if extra fields

    @model_validator(mode="before")
    @classmethod
    def warn_extra_fields(cls, values: dict) -> dict:
        """ Warn if there are any unexpected fields"""
        if isinstance(values, dict):
            extra = set(values.keys()) - set(cls.model_fields.keys())
            if extra:
                warnings.warn(f"Config section '{cls.__name__}' contains unknown fields that will be ignored: {sorted(extra)}. Valid fields are: {sorted(cls.model_fields.keys())}. Project-specific parameters should go under 'project_params'")
        return values

class IOConfig(_LaxModel):
    data_path: str
    save_path: str

class EvolutionConfig(_LaxModel):
    n_generations: int
    time_limit: int | float
    n_islands: int
    batch_size: int
    critical_population_size: int
    n_migrants: int
    topology: list[int]

    @model_validator(mode="after")
    def check_args(self) -> EvolutionConfig:
        if len(self.topology) != self.n_islands:
            raise ValueError(f"topology length ({len(self.topology)}) must equal n_islands ({self.n_islands})")
        if set(self.topology) != set(range(self.n_islands)):
            raise ValueError(f"topology must contain exactly the indices 0 to {self.n_islands - 1}")
        return self
    

class RetryConfig(_LaxModel):
    max_retries: int = 3
    initial_delay: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay: float = 60.0
    retryable_status_codes: list[int] = field(default_factory=lambda: [500, 503])


class LLMsConfig(_LaxModel):
    num_parents: int
    retry: RetryConfig
    model_llm: ValidLLMs | list[ValidLLMs]
    param_est_llm: ValidLLMs | list[ValidLLMs]
    jax_model_translator_llm: ValidLLMs | list[ValidLLMs]
    log_raw_llm_response: bool
    max_lines: int
    swear_words: list[str]
    max_tokens: int

class GradientDescentConfig(_LaxModel):
    max_iter: int
    learning_rate: float


class ScoringConfig(_LaxModel):
    param_penalty_weight: float
    timeout_s: float
    gradient_descent: GradientDescentConfig


class PromptsConfig(_LaxModel):
    model: PromptSchema
    parameter_estimator: PromptSchema
    jax_translator_model: PromptSchema


# ── Config ──

class Config(BaseModel):
    """
    All settings needed to construct a TaskSpec, in plain-dict form.

    Constructed via from_yaml (live run) or from_taskspec (reproduce a past run).
    TaskSpec.from_config(config) turns this into a fully-loaded TaskSpec.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_name: str
    project_dir: Path
    io: IOConfig
    evolution: EvolutionConfig
    llms: LLMsConfig
    scoring: ScoringConfig
    project_params: dict #Not checked for types or otherwise as project-specific
    prompts: PromptsConfig

    @classmethod
    def from_yaml(cls, path: Path | str, default_path: Path | str | None = None) -> Config:
        """
        Load a config.yaml and merge with project defaults.

        Args:
            path: Path to a project config.yaml.
            default_path: Optional path to a default config.yaml to merge with before loading the project config. If None, uses projects/config_default.yaml
        """
        path = Path(path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if default_path is None:
            default_path = PROJECT_ROOT / "projects" / "config_default.yaml"
        default_path = Path(default_path)
        if not default_path.is_absolute():
            default_path = PROJECT_ROOT / default_path

        task_name = path.parent.name
        project_dir = path.parent
        default_dir = default_path.parent

        default_config = yaml.safe_load(default_path.read_text()) or {}
        config = _deep_merge(default_config, yaml.safe_load(path.read_text()) or {})

        default_prompts = yaml.safe_load((default_dir / "prompt_defaults.yaml").read_text()) or {}
        task_prompt_path = project_dir / "prompts.yaml"
        task_prompts = yaml.safe_load(task_prompt_path.read_text()) if task_prompt_path.exists() else {}
        prompts = _deep_merge(default_prompts, task_prompts)

        # Warn about any unexpected top-level fields in config
        io = config.pop("io", {})
        evolution = config.pop("evolution", {})
        llms = config.pop("llms", {})
        scoring = config.pop("scoring", {})
        project_params = config.pop("project_params", {})
        if config:
            warnings.warn(f"Config contains unknown keys: {list(config.keys())}, custom keys should be defined under field 'project_params', these keys will be ignored", stacklevel=2)

        return cls(
            task_name=task_name,
            project_dir=project_dir,
            io=io,
            evolution=evolution,
            llms=llms,
            scoring=scoring,
            project_params=project_params,
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
            "jax_translator_model": schemas["jax_model"],
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
