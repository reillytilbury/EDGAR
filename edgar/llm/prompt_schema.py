"""
prompt_schema.py

Schema holding the component parts of a prompt.

This module defines the `PromptSchema` Pydantic model for constructing LLM prompts. It enables
modular prompt design by separating various instructions (base, mode-specific, code/docstring guidelines,
image analysis) and supports dynamic variable substitution.

There are two kinds of template variables:
- config_vars: These are derived from the global configuration or `TaskSpec` (e.g., `k`, `max_lines`,
  `swear_words`). They are sourced directly from the `config` dictionary passed to `build_prompt`.
- program_vars (parent_program_vars, current_program_vars): These are extracted from `Program`
  objects using dotted paths (e.g., "name", "code.model", "program_losses.discover.final").
  Dots in the path are replaced with underscores to form the template variable name, so
  "code.model" is referenced as `{code_model}` in prompt templates.

Example usage:
    schema = PromptSchema(
        base="You are a scientist. Below are {num_parents} models...",
        explore="Be creative...",
        code_guidelines="Function signature must be...",
        docstring_guidelines="Include a brief description...",
        parent_program_template="model: {name}\\nloss: {program_losses_discover_final}\\n{code_model}\\n",
        parent_program_vars=["name", "program_losses.discover.final", "code.model"],
    )

    # config: flat dict from TaskSpec.flat_config (merged evolution + llms + scoring)
    config = {"num_parents": 2, "max_lines": 50, "swear_words": "scipy.optimize, curve_fit"}

    # parent_programs: list of Program objects (parent_program_vars extracted via dotted-path traversal)
    prompt = schema.build_prompt("explore", parent_programs=parents, config=config)
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..evolution.program import Program
    import numpy as np


def _fill_program_vars(program: Any, var_names: list[str]) -> dict[str, Any]:
    """Fills a dictionary with program attributes, converting dotted paths to underscore-separated keys.

    Args:
        program: The Program object from which to extract variables.
        var_names: A list of dotted-path strings representing the program attributes
            to extract (e.g., "name", "code.model").

    Returns:
        A dictionary where keys are underscore-separated variable names (e.g., "code_model")
        and values are the extracted attributes from the program.
    """
    return {x.replace(".", "_"): _get_nested_attr(program, x, "") for x in var_names}


def _get_nested_attr(obj: Any, dotted_key: str, default: Any = None) -> Any:
    """Safely retrieves a nested attribute from an object using a dotted key.

    Example: `_get_nested_attr(program, "code.model")` would return `program.code.model`.

    Args:
        obj: The object from which to retrieve the attribute.
        dotted_key: A string representing the dotted path to the attribute (e.g., "parent.child.grandchild").
        default: The default value to return if any part of the dotted path is not found
            or is None. Defaults to None.

    Returns:
        The value of the nested attribute, or the default value if not found.
    """
    item = obj

    for part in dotted_key.split("."):
        if item is None or not hasattr(item, part):
            return default
        item = getattr(item, part)

    return item


class PromptSchema(BaseModel):
    """Defines the schema for constructing an LLM prompt.

    This Pydantic model structures the various components of a prompt, allowing for
    flexible and contextualized prompt generation for LLM code generation tasks.
    It supports different sections for base instructions, mode-specific guidance,
    code/docstring conventions, and image analysis, alongside templating for
    configuration and program-specific variables.
    """

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
        description="A template string for formatting information about parent programs, variables defined as e.g {code_model} are filled with the corresponding dotted variable from parent_program_vars (e.g code.model)"
    )
    parent_program_vars: list[str] = Field(
        default_factory=list,
        description="A list of dotted-path strings for variables to extract from parent `Program` objects, e.g code.model",
    )
    current_program_template: Optional[str] = Field(
        None,
        description="A template string for formatting information about the program currently being generated/modified, variables defined as e.g {code_model} are filled with the corresponding dotted variable from current_program_vars (e.g code.model)",
    )
    current_program_vars: list[str] = Field(
        default_factory=list,
        description="A list of dotted-path strings for variables to extract from the current `Program` object, e.g code.model",
    )
    ideas: list[str] = Field(
        default_factory=list,
        description="A list of ideas/bits of text to inject into the prompt with probability idea_probability.",
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

        prompt_parts = [s.format(**config_copy) for s in sections if s]

        if parent_programs:
            programs_text = [
                self.parent_program_template.format(
                    parent_number=i + 1,
                    **_fill_program_vars(p, self.parent_program_vars),
                )
                for i, p in enumerate(parent_programs)
            ]
            prompt_parts.append("\n".join(programs_text))

        if current_program is not None and self.current_program_template is not None:
            current_text = self.current_program_template.format(
                **_fill_program_vars(current_program, self.current_program_vars)
            )
            prompt_parts.append(current_text)

        return "\n\n".join(prompt_parts).strip()

    def select_ideas(
        self,
        cfg: dict[str, Any],
        rng: np.random.Generator,
    ) -> list[str]:
        """Selects ideas from the ideas pool with independent probability, returning a list of them and mutates cfg in-place with the 'random-text-injection' key.

        Args:
            cfg: The configuration dictionary to be mutated in-place.
            rng: The random number generator to use.
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
