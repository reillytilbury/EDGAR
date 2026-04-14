from pydantic import BaseModel, Field
from typing import List, Optional

class PromptComponent(BaseModel):
    text: str = ""
    allowed_variables: List[str] = []

class PromptSchema(BaseModel):
    # Core Instruction
    base: PromptComponent
    
    # Evolutionary strategy components (can be empty for JAX/ParamEst)
    explore: Optional[PromptComponent] = None
    exploit: Optional[PromptComponent] = None
    
    # Guidelines
    code_guidelines: PromptComponent
    docstring_guidelines: PromptComponent
    
    # Data/Context injection
    image_analysis_instructions: Optional[PromptComponent] = None
    parent_detail_template: PromptComponent = Field(
        description="The template used to format each parent in the prompt"
    )