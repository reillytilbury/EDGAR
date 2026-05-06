"""
prompt_schema.py

Schema holding the component parts of a prompt.

There are two kinds of template variables:
- config_vars: come from config/spec (e.g. k, max_lines, swear_words).
  Only source: the config file / TaskSpec.
- parent_vars: dotted paths into Program fields (e.g. "name",
  "code.model", "program_losses.discover.final"). Dots in the path are
  replaced with underscores to form the template variable name, so
  "code.model" is referenced as {code_model} in parent_detail_template.

Example usage:
    schema = PromptSchema(
        base="You are a scientist. Below are {k_max} models...",
        explore="Be creative...",
        code_guidelines="Function signature must be...",
        docstring_guidelines="Include a brief description...",
        parent_detail_template="model: {name}\\nloss: {program_losses_discover_final}\\n{code_model}\\n",
        config_vars=["k_max"],
        parent_vars=["name", "program_losses.discover.final", "code.model"],
    )

    # config: flat dict from TaskSpec.flat_config (merged evolution + llms + scoring)
    config = {"k_max": 2, "max_lines": 50, "swear_words": "scipy.optimize, curve_fit"}

    # parents: list of Program objects (parent_vars extracted via dotted-path traversal)
    prompt = schema.build_prompt("explore", parents, config)
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..evolution.program import Program


class PromptSchema(BaseModel):
    base: str
    explore: Optional[str] = None
    exploit: Optional[str] = None
    code_guidelines: str
    docstring_guidelines: str
    image_analysis_instructions: Optional[str] = None
    parent_detail_template: str
    config_vars: list[str] = Field(default_factory=list)
    parent_vars: list[str] = Field(default_factory=list)

    def build_prompt(
        self,
        mode: str,
        parents: list[Program] | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Build a prompt by selecting and formatting schema sections."""
        if mode not in {"explore", "exploit"}:
            raise ValueError("mode must be 'explore' or 'exploit'")

        parents = parents or []
        config = config or {}

        sections = [
            self.base,
            getattr(self, mode),
            self.code_guidelines,
            self.docstring_guidelines,
            self.image_analysis_instructions,
        ]

        prompt_parts = [s.format(**config) for s in sections if s]

        if parents:
            parents_text = [
                self.parent_detail_template.format(
                    parent_number=i + 1,
                    **{x.replace(".", "_"): getattr(p, x, "") for x in self.parent_vars}
                )
                for i, p in enumerate(parents)
            ]
            prompt_parts.append("\n".join(parents_text))

        return "\n\n".join(prompt_parts).strip()