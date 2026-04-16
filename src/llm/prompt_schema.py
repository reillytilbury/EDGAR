from pydantic import BaseModel
from typing import Optional, List

class PromptSchema(BaseModel):
    """
    Schema to hold component parts of all different prompts. 

    Each section can be formatted with variables: 
    - global variables (e.g. dataset name, evaluation metric)  
    - parent variables (e.g. model name, loss, code string).

    The parent variables are formatted into the `parent_detail_template` for each parent model, and the resulting strings are appended to the end of the prompt.
    Typically, the parent variables differ between prompts while the global variables will be the same across prompts for a given project.
    """
    base: str
    explore: Optional[str] = None
    exploit: Optional[str] = None
    code_guidelines: str
    docstring_guidelines: str
    image_analysis_instructions: Optional[str] = None
    parent_detail_template: str
    global_vars: List[str] = []
    parent_vars: List[str] = []

def build_prompt(schema: PromptSchema, mode: str, global_variables: dict = {}, parent_variables: List[dict] = []) -> str:
    """
    Build a prompt by selecting and formatting the appropriate sections of the schema based on the mode and provided variables.
    Args:
        schema (PromptSchema): The prompt schema containing all sections and templates.
        mode (str): The mode of the prompt, either "explore" or "exploit".
        global_variables (dict, optional): A dictionary of global variables to format into the prompt sections.
        parent_variables (List[dict], optional): A list of dictionaries, each containing variables for each parent.
    """
    assert mode in ["explore", "exploit"], "Mode must be either 'explore' or 'exploit'"

    # 1. Select the text blocks to include
    sections = [
        schema.base, 
        getattr(schema, mode), 
        schema.code_guidelines, 
        schema.docstring_guidelines, 
        schema.image_analysis_instructions
    ]
    
    # 2. Format blocks with variables (Python ignores unused keys in the dict)
    prompt_parts = [s.format(**global_variables) for s in sections if s]
    
    # 3. Add the repeating parent models if they exist
    parents_text = [schema.parent_detail_template.format(**p) for p in parent_variables]
    prompt_parts.append("\n".join(parents_text))
        
    return "\n\n".join(prompt_parts).strip()