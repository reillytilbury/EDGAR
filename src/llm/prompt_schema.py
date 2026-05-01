"""
prompt_schema.py

Schema holding the component parts of a prompt.

There are two kinds of template variables:
- config_vars: come from config/spec (e.g. k, max_lines, swear_words).
  Only source: the config file / TaskSpec.
- parent_vars: come from Program fields (e.g. descriptive_name, loss_discover,
  model_code). Extracted from parent Programs by the caller.

Example usage:
    schema = PromptSchema(
        base="You are a scientist. Below are {k_max} models...",
        explore="Be creative...",
        code_guidelines="Function signature must be...",
        docstring_guidelines="Include a brief description...",
        parent_detail_template="model: {descriptive_name}\\nloss: {loss_discover}\\n{model_code}\\n",
        config_vars=["k_max"],
        parent_vars=["descriptive_name", "loss_discover", "model_code"],
    )

    # config: flat dict from TaskSpec.flat_config (merged evolution + llms + scoring)
    config = {"k_max": 2, "max_lines": 50, "swear_words": "scipy.optimize, curve_fit"}

    # parents: list of Program objects (parent_vars extracted via getattr)
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
                    **{x: getattr(p, x, "") or "" for x in self.parent_vars}
                )
                for i, p in enumerate(parents)
            ]
            prompt_parts.append("\n".join(parents_text))

        return "\n\n".join(prompt_parts).strip()