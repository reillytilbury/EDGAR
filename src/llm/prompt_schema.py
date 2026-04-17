from pydantic import BaseModel, Field
from typing import Optional, Any

class PromptSchema(BaseModel):
    """
    Schema holding the component parts of a prompt.
    """
    base: str
    explore: Optional[str] = None
    exploit: Optional[str] = None
    code_guidelines: str
    docstring_guidelines: str
    image_analysis_instructions: Optional[str] = None
    parent_detail_template: str
    global_vars: list[str] = Field(default_factory=list)
    parent_vars: list[str] = Field(default_factory=list)

    def build_prompt(
        self,
        mode: str,
        global_variables: dict[str, Any] | None = None,
        parent_variables: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Build a prompt by selecting and formatting schema sections.
        """
        if mode not in {"explore", "exploit"}:
            raise ValueError("mode must be 'explore' or 'exploit'")

        global_variables = global_variables or {}
        parent_variables = parent_variables or []

        sections = [
            self.base,
            getattr(self, mode),
            self.code_guidelines,
            self.docstring_guidelines,
            self.image_analysis_instructions,
        ]

        prompt_parts = [s.format(**global_variables) for s in sections if s]

        if parent_variables:
            parents_text = [
                self.parent_detail_template.format(**p)
                for p in parent_variables
            ]
            prompt_parts.append("\n".join(parents_text))

        return "\n\n".join(prompt_parts).strip()