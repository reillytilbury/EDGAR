from src.llm.prompt_schema import PromptSchema
from tests.llm.programs import Program1, Program2


def test_build_model_prompt():
    prompt_schema = PromptSchema(
        base="This is a fake prompt for {num_parents} programs",
        explore="Be creative ...",
        code_guidelines="**Code Generation Guidelines**:\n ... \n",
        docstring_guidelines="**Docstring Guidelines**:\n ... \n",
        image_analysis_instructions="When analyzing images, do X, Y, Z",
        program_detail_template="How to format the details of programs included in the prompt\n"
        "Model {parent_number}:\n"
        "{model}\n",
        program_vars=["model"],
    )

    prompt = prompt_schema.build_prompt(
        mode="explore", programs=[Program1, Program2], config={"num_parents": 2}
    )

    assert "This is a fake prompt for 2 programs" in prompt
    assert "Be creative ..." in prompt
    assert "**Code Generation Guidelines**:\n ... \n" in prompt
    assert "**Docstring Guidelines**:\n ... \n" in prompt
    assert "When analyzing images, do X, Y, Z" in prompt
    assert "Model 1:" in prompt
    assert Program1.model in prompt
    assert "Model 2:" in prompt
    assert Program2.model in prompt

    print(prompt)  # call pytest with -s to view prompt output in console
