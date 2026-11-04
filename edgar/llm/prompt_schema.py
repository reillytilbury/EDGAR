"""
prompt_schema.py

Schema holding the component parts of a prompt.

This module defines the `PromptSchema` Pydantic model for constructing LLM prompts. It enables
modular prompt design by separating various instructions (base, mode-specific, code/docstring guidelines,
image analysis) and supports dynamic variable substitution.

Template variables:
- config variables: Sourced from the global configuration or TaskSpec (e.g. {num_parents}, {max_lines}).
- program attributes: Referenced directly using dotted paths into Program objects (e.g. {name},
  {code.model}, {program_losses.discover.final}, {diagnostics.r2_overall}).
"""

from __future__ import annotations

import string
from pydantic import BaseModel, Field
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..evolution.program import Program
    import numpy as np


def _get_nested_attr(obj: Any, dotted_key: str, default: Any = "") -> Any:
    """Safely retrieves a nested attribute or dictionary value from an object using a dotted key.

    Example: `_get_nested_attr(program, "code.model")` -> `program.code.model`.
    """
    item = obj
    for part in dotted_key.split("."):
        if item is None:
            return default
        if isinstance(item, dict):
            item = item.get(part, default)
        elif hasattr(item, part):
            item = getattr(item, part)
        else:
            return default
    return item if item is not None else default


class SafeProgramFormatter(string.Formatter):
    """Formatter that resolves dotted field names directly against a Program object."""

    def get_field(
        self, field_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[Any, Any]:
        if field_name in kwargs:
            return kwargs[field_name], field_name

        program = kwargs.get("program")
        if program is not None:
            val = _get_nested_attr(program, field_name, default="")
            return val, field_name

        return "", field_name

    def format_field(self, value: Any, format_spec: str) -> str:
        if value == "" and format_spec:
            return ""
        try:
            return super().format_field(value, format_spec)
        except (ValueError, TypeError):
            return str(value)


class PromptSchema(BaseModel):
    """Defines the schema for constructing an LLM prompt."""

    base: str = Field(description="The base instructions for the LLM.")
    explore: Optional[str] = Field(
        None, description="Instructions specific to the 'explore' generation mode."
    )
    exploit: Optional[str] = Field(
        None, description="Instructions specific to the 'exploit' generation mode."
    )
    code_guidelines: str = Field(
        description="Guidelines for the structure and style of generated code."
    )
    docstring_guidelines: str = Field(
        description="Guidelines for the format and content of docstrings in generated code."
    )
    image_analysis_instructions: Optional[str] = Field(
        None,
        description="Instructions for the LLM on how to interpret and use multimodal image feedback.",
    )
    parent_program_template: str = Field(
        description="A template string for formatting parent programs. Dotted paths (e.g. {code.model}, {diagnostics.r2_overall}) are resolved directly from parent programs."
    )
    parent_program_vars: list[str] = Field(
        default_factory=list,
        description="Deprecated: dotted paths are now resolved directly in templates.",
    )
    current_program_template: Optional[str] = Field(
        None,
        description="A template string for formatting the current program. Dotted paths (e.g. {code.model}) are resolved directly from current_program.",
    )
    current_program_vars: list[str] = Field(
        default_factory=list,
        description="Deprecated: dotted paths are now resolved directly in templates.",
    )
    ideas_template: str = Field(
        default="Some ideas you may want to incorporate into your model:\n {ideas-injection-point}",
        description="Template for injecting ideas into the prompt.",
    )
    ideas: list[str] = Field(
        default_factory=list,
        description="A list of ideas to inject into the prompt with probability idea_probability.",
    )

    def build_prompt(
        self,
        mode: str,
        parent_programs: list[Program] | None = None,
        config: dict[str, Any] | None = None,
        current_program: Program | None = None,
    ) -> str:
        """Builds a complete LLM prompt by selecting and formatting schema sections.

        This method combines the base instructions, mode-specific guidance (explore/exploit),
        code and docstring guidelines, and information about parent and current programs
        into a single, coherent prompt string. It substitutes variables from the global
        configuration and program objects into their respective templates.

        Args:
            mode: The current generation mode, either 'explore' or 'exploit'. This
                determines which set of mode-specific instructions to include.
            parent_programs: An optional list of `Program` objects that serve as
                parents for the generation of a new program. Their attributes will
                be included in the prompt via `parent_program_template` and `parent_program_vars`.
            config: An optional dictionary of global configuration variables (e.g., from
                `TaskSpec.flat_config`) to be substituted into the prompt templates, e.g `num_parents`.
            current_program: An optional `Program` object representing the program
                currently being worked on (e.g., for parameter estimation/translation). Its
                attributes will be included in the prompt via `current_program_template` and `current_program_vars`.

        Returns:
            A string containing the fully formatted and substituted LLM prompt.

        Raises:
            ValueError: If the provided `mode` is not 'explore' or 'exploit'.
        """
        if mode not in {"explore", "exploit"}:
            raise ValueError("mode must be 'explore' or 'exploit'")

        parent_programs = parent_programs or []
        config = config or {}

        config_copy = dict(config)
        if "ideas-injection-point" not in config_copy:
            config_copy["ideas-injection-point"] = ""

        sections = [
            self.base,
            getattr(self, mode),
            self.code_guidelines,
            self.docstring_guidelines,
            self.image_analysis_instructions,
        ]

        if config_copy.get("ideas-injection-point"):
            sections.insert(1, self.ideas_template)

        prompt_parts = [s.format(**config_copy) for s in sections if s]

        formatter = SafeProgramFormatter()

        if parent_programs:
            programs_text = [
                formatter.format(
                    self.parent_program_template,
                    program=p,
                    parent_number=i + 1,
                )
                for i, p in enumerate(parent_programs)
            ]
            prompt_parts.append("\n".join(programs_text))

        if current_program is not None and self.current_program_template is not None:
            current_text = formatter.format(
                self.current_program_template,
                program=current_program,
            )
            prompt_parts.append(current_text)

        return "\n\n".join(prompt_parts).strip()

    def select_ideas(
        self,
        cfg: dict[str, Any],
        rng: np.random.Generator,
    ) -> list[str]:
        """Selects a subset of ideas from the ideas pool based on a given probability and updates the configuration.

        Each idea in the `ideas` list is selected with an independent probability `idea_probability`
        (from `cfg`). The selected ideas are then joined by newlines and stored in the `cfg` dictionary
        under the key `'ideas-injection-point'`, to be included in the prompt by `build_prompt`.

        Args:
            cfg: The configuration dictionary to be mutated in-place with selected ideas.
            rng: The random number generator to use for probabilistic selection.
        Returns:
            A list of selected ideas, which may be empty if no ideas are selected.
        """
        idea_probability = cfg.get("idea_probability", 0.0)
        selected_ideas = []
        if self.ideas and idea_probability > 0.0:
            for idea in self.ideas:
                if rng.random() < idea_probability:
                    selected_ideas.append(idea)

        cfg["ideas-injection-point"] = "\n".join(selected_ideas)
        return selected_ideas
