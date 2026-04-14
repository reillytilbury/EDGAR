from pydantic import BaseModel, Field

class ResponseSchema(BaseModel):
    thought_process: str = Field(
        description="Detailed mathematical or algorithmic reasoning for this code."
    )
    code: str = Field(
        description="The complete, self-contained Python implementation. Must include imports, any helper functions, and the primary function definition."
    )
