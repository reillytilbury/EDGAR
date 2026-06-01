"""
prompt_schema.py

Schema holding the component parts of a prompt.

There are two kinds of template variables:
- config_vars: come from config/spec (e.g. k, max_lines, swear_words).
  Only source: the config file / TaskSpec.
- parent_program_vars: dotted paths into Program fields (e.g. "name",
  "code.model", "program_losses.discover.final"). Dots in the path are
  replaced with underscores to form the template variable name, so
  "code.model" is referenced as {code_model} in parent_program_template.

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


def _fill_program_vars(program: Any, var_names: list[str]) -> dict[str, Any]:
    return {x.replace(".", "_"): _get_nested_attr(program, x, "") for x in var_names}


def _get_nested_attr(obj: Any, dotted_key: str, default: Any = None) -> Any:
    item = obj

    for part in dotted_key.split("."):
        if item is None or not hasattr(item, part):
            return default
        item = getattr(item, part)

    return item


class PromptSchema(BaseModel):
    base: str
    explore: Optional[str] = None
    exploit: Optional[str] = None
    code_guidelines: str
    docstring_guidelines: str
    image_analysis_instructions: Optional[str] = None
    parent_program_template: str
    parent_program_vars: list[str] = Field(default_factory=list)
    current_program_template: Optional[str] = None
    current_program_vars: list[str] = Field(default_factory=list)

    def build_prompt(
        self,
        mode: str,
        parent_programs: list[Program] | None = None,
        config: dict[str, Any] | None = None,
        current_program: Program | None = None,
    ) -> str:
        """Build a prompt by selecting and formatting schema sections."""
        if mode not in {"explore", "exploit"}:
            raise ValueError("mode must be 'explore' or 'exploit'")

        parent_programs = parent_programs or []
        config = config or {}

        sections = [
            self.base,
            getattr(self, mode),
            self.code_guidelines,
            self.docstring_guidelines,
            self.image_analysis_instructions,
        ]

        prompt_parts = [s.format(**config) for s in sections if s]

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
