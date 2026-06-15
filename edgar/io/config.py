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

ValidLLMs = Literal[
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
]
"""Literal type for valid LLM model names.

The provider is inferred from the model name prefix (e.g., 'gemini-' for Google,
'claude-' for Anthropic).
"""


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
"""The root directory of the EDGAR project."""


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, with override taking precedence.

    Example: `base = {"a": 1, "b": 2} and override = {"b": 3, "c": 4}` would produce `{"a": 1, "b": 3, "c": 4}`.

    Args:
        base: The base dictionary.
        override: The dictionary with overriding values.

    Returns:
        A new dictionary representing the merged result.
    """
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
    """Pydantic BaseModel which raises a warning if there are unexpected fields.

    This model is used as a base for configuration sections to allow for
    forward compatibility and to provide helpful warnings when unknown
    configuration fields are encountered, rather than raising errors.
    """

    model_config = ConfigDict(extra="ignore")  # Doesnt raise an error if extra fields

    @model_validator(mode="before")
    @classmethod
    def warn_extra_fields(cls, values: dict) -> dict:
        """Warns if the input dictionary contains any unexpected fields for the model.

        Args:
            values: The dictionary of values to validate.

        Returns:
            The original dictionary of values.
        """
        if isinstance(values, dict):
            extra = set(values.keys()) - set(cls.model_fields.keys())
            if extra:
                warnings.warn(
                    f"Config section '{cls.__name__}' contains unknown fields that will be ignored: {sorted(extra)}. Valid fields are: {sorted(cls.model_fields.keys())}. Project-specific parameters should go under 'project_params'"
                )
        return values


class RunConfig(_LaxModel):
    """Configuration settings related to the overall EDGAR run.

    Attributes:
        random_seed: An optional integer seed for the random number generator
            to ensure reproducibility of runs. If None, a random seed will be used.
    """

    random_seed: int | None


class IOConfig(_LaxModel):
    """Configuration settings for input/output operations.

    Attributes:
        data_path: The path to the directory or file containing the experiment's data.
        save_path: The base path where all run artifacts (logs, programs,
            dashboard data) will be saved.
    """

    data_path: str
    save_path: str


class EvolutionConfig(_LaxModel):
    """Configuration settings for the evolutionary algorithm.

    Attributes:
        n_generations: The total number of evolutionary generations to run.
        n_islands: The number of independent islands (subpopulations) in the
            island evolutionary algorithm.
        batch_size: The number of new programs to generate and score per
            generation on each island.
        critical_population_size: The target population size for each island
            after pruning. This value influences the number of programs retained.
        n_migrants: The number of programs to migrate between islands each generation.
        topology: A list of integers representing the migration topology.
            `topology[i]` specifies the island that island `i` will migrate programs to.
            Must be a permutation of `range(n_islands)`.
    """

    n_generations: int
    n_islands: int
    batch_size: int
    critical_population_size: int
    n_migrants: int
    topology: list[int]

    @model_validator(mode="after")
    def check_args(self) -> EvolutionConfig:
        """Validates the `topology` configuration.

        Ensures that the length of the `topology` list matches `n_islands` and
        that it contains exactly the indices from 0 to `n_islands - 1` (i.e., it's a
        permutation of island indices).

        Raises:
            ValueError: If the `topology` length does not match `n_islands` or
                if it does not contain valid island indices.

        Returns:
            The validated EvolutionConfig instance.
        """
        if len(self.topology) != self.n_islands:
            raise ValueError(
                f"topology length ({len(self.topology)}) must equal n_islands ({self.n_islands})"
            )
        if set(self.topology) != set(range(self.n_islands)):
            raise ValueError(
                f"topology must contain exactly the indices 0 to {self.n_islands - 1}"
            )
        return self


class RetryConfig(_LaxModel):
    """Configuration for retrying failed LLM calls.

    Attributes:
        max_retries: The maximum number of times to retry a failed LLM call.
        initial_delay: The initial delay in seconds before the first retry.
        backoff_multiplier: The multiplier for exponential backoff between retries.
        max_delay: The maximum delay in seconds between retries.
        retryable_status_codes: A list of HTTP status codes that should trigger a retry.
    """

    max_retries: int = 3
    initial_delay: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay: float = 60.0
    retryable_status_codes: list[int] = field(default_factory=lambda: [500, 503])


class LLMsConfig(_LaxModel):
    """Configuration settings for Large Language Model interactions.

    Attributes:
        num_parents: The number of parent programs to include in the prompt
            when generating new programs.
        retry: Configuration for retrying failed LLM API calls.
        model_llm: The LLM model(s) to use for generating new scientific models
            (numpy `model` code and `default_params`). Can be a single LLM or a list for cycling.
        param_est_llm: The LLM model(s) to use for generating `parameter_estimator` code.
            Can be a single LLM or a list for cycling.
        jax_model_translator_llm: The LLM model(s) to use for translating numpy `model`
            code into JAX-compatible code. Can be a single LLM or a list for cycling.
        log_raw_llm_response: If True, logs the raw JSON responses from LLM calls
            for debugging purposes.
        max_lines: The maximum number of lines allowed in generated code snippets.
        swear_words: A list of words to filter out or flag in LLM-generated text.
        max_tokens: The maximum number of tokens to request from the LLM in a response.
    """

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
    """Configuration settings for the gradient descent optimizer.

    Attributes:
        max_iter: The maximum number of iterations for the gradient descent algorithm.
        learning_rate: The learning rate used by the optimizer.
    """

    max_iter: int
    learning_rate: float


class ScoringConfig(_LaxModel):
    """Configuration settings for model scoring and evaluation.

    Attributes:
        param_penalty_weight: A weighting factor for the parameter complexity penalty
            added to the loss function. Higher values penalize more complex models.
        timeout_s: The maximum time in seconds allowed for scoring a single program.
            If exceeded, the program is considered to have infinite loss.
        gradient_descent: Configuration for the gradient descent optimization
            performed during scoring.
    """

    param_penalty_weight: float
    timeout_s: float
    gradient_descent: GradientDescentConfig


class PromptsConfig(_LaxModel):
    """Configuration containing the prompt schemas for different LLM generation tasks.

    Attributes:
        model: The PromptSchema for generating new scientific models.
        parameter_estimator: The PromptSchema for generating parameter estimators.
        jax_translator_model: The PromptSchema for translating numpy models to JAX.
    """

    model: PromptSchema
    parameter_estimator: PromptSchema
    jax_translator_model: PromptSchema


# ── Config ──


class Config(BaseModel):
    """
    All settings needed to construct a TaskSpec, in plain-dict form.

    Constructed via from_yaml (live run) or from_taskspec (reproduce a past run).
    TaskSpec.from_config(config) turns this into a TaskSpec.

    Attributes:
        task_name: The name of the current scientific task or project.
        project_dir: The path to the project directory containing task-specific files.
        io: I/O configuration settings.
        evolution: Evolutionary algorithm configuration settings.
        llms: Large Language Model configuration settings.
        scoring: Model scoring configuration settings.
        run: Run-specific configuration settings.
        project_params: A dictionary for arbitrary project-specific parameters
            that are not covered by other structured config sections.
        prompts: Prompt schemas for various LLM generation tasks.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_name: str
    project_dir: Path
    io: IOConfig
    evolution: EvolutionConfig
    llms: LLMsConfig
    scoring: ScoringConfig
    run: RunConfig
    project_params: dict  # Not checked for types or otherwise as project-specific
    prompts: PromptsConfig

    @classmethod
    def from_yaml(
        cls, path: Path | str, default_path: Path | str | None = None
    ) -> Config:
        """Loads configuration from a YAML file, merging with project defaults.

        This method loads a `config.yaml` specific to a task and merges it with
        a default configuration (`projects/config_default.yaml`). It also loads
        and merges prompt configurations from `prompt_defaults.yaml` and a
        task-specific `prompts.yaml`.

        Args:
            path: Path to the project's `config.yaml` file.
            default_path: Optional path to a default `config.yaml` to merge with
                before loading the project config. If None, it defaults to
                `REPO_ROOT / "projects" / "config_default.yaml"`.

        Returns:
            A `Config` object populated with the merged configuration settings.
        """
        path = Path(path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if default_path is None:
            default_path = REPO_ROOT / "projects" / "config_default.yaml"
        default_path = Path(default_path)
        if not default_path.is_absolute():
            default_path = REPO_ROOT / default_path

        task_name = path.parent.name
        project_dir = path.parent
        default_dir = default_path.parent

        default_config = yaml.safe_load(default_path.read_text()) or {}
        config = _deep_merge(default_config, yaml.safe_load(path.read_text()) or {})

        default_prompts = (
            yaml.safe_load((default_dir / "prompt_defaults.yaml").read_text()) or {}
        )
        task_prompt_path = project_dir / "prompts.yaml"
        task_prompts = (
            yaml.safe_load(task_prompt_path.read_text())
            if task_prompt_path.exists()
            else {}
        )
        prompts = _deep_merge(default_prompts, task_prompts)

        # Warn about any unexpected top-level fields in config
        io = config.pop("io", {})
        evolution = config.pop("evolution", {})
        llms = config.pop("llms", {})
        scoring = config.pop("scoring", {})
        project_params = config.pop("project_params", {})
        run = config.pop("run", {})
        if config:
            warnings.warn(
                f"Config contains unknown keys: {list(config.keys())}, custom keys should be defined under field 'project_params', these keys will be ignored",
                stacklevel=2,
            )

        return cls(
            task_name=task_name,
            project_dir=project_dir,
            io=io,
            evolution=evolution,
            llms=llms,
            scoring=scoring,
            run=run,
            project_params=project_params,
            prompts=prompts,
        )

    @classmethod
    def from_taskspec(cls, path: Path) -> Config:
        """Extracts configuration settings from a previously saved `task_spec.yaml`.

        This method is used to reconstruct the configuration of a past EDGAR run.
        Callables, git state, and creation timestamp
        are regenerated when `TaskSpec.from_config` is later called; only the
        configuration dictionaries are preserved from the `task_spec.yaml`.

        Args:
            path: Path to a `task_spec.yaml` file.

        Returns:
            A `Config` object populated with settings extracted from the `task_spec.yaml`.
        """
        path = Path(path)
        if not path.is_absolute():
            path = REPO_ROOT / path

        record = yaml.safe_load(path.read_text())
        schemas = record["prompt_schemas"]
        prompts = {
            "model": schemas["model"],
            "parameter_estimator": schemas["param_est"],
            "jax_translator_model": schemas["jax_model"],
        }
        return cls(
            task_name=record["task_name"],
            project_dir=Path(record["project_dir"])
            if "project_dir" in record
            else REPO_ROOT / "projects" / record["task_name"],
            io=record.get("io", {}),
            evolution=record.get("evolution", {}),
            llms=record.get("llms", {}),
            scoring=record.get("scoring", {}),
            project_params=record.get("project_params", {}),
            run=record.get("run", {}),
            prompts=prompts,
        )
