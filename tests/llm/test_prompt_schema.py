from edgar.llm.prompt_schema import PromptSchema
from edgar.evolution.program import Program, BirthCertificate, Code, Losses, LossPair
from tests.llm.programs import Program1, Program2


def _make_schema(**kwargs) -> PromptSchema:
    defaults = dict(
        base="This is a fake prompt for {num_parents} programs",
        explore="Be creative ...",
        code_guidelines="**Code Generation Guidelines**:\n ... \n",
        docstring_guidelines="**Docstring Guidelines**:\n ... \n",
        parent_program_template="Model {parent_number}:\n{code_model}\n",
        parent_program_vars=["code.model"],
    )
    return PromptSchema(**{**defaults, **kwargs})


def _make_program(model: str, loss: float | None = None) -> Program:
    p = Program(birth=BirthCertificate(generation=0, island=0, batch_index=0))
    p.code = Code(model=model)
    if loss is not None:
        p.program_losses = Losses(discover=LossPair(final=loss))
    return p


def test_build_model_prompt():
    schema = _make_schema(
        image_analysis_instructions="When analyzing images, do X, Y, Z"
    )
    p1 = _make_program(Program1.model)
    p2 = _make_program(Program2.model)

    prompt = schema.build_prompt(
        mode="explore", parent_programs=[p1, p2], config={"num_parents": 2}
    )

    assert "This is a fake prompt for 2 programs" in prompt
    assert "Be creative ..." in prompt
    assert "**Code Generation Guidelines**" in prompt
    assert "**Docstring Guidelines**" in prompt
    assert "When analyzing images, do X, Y, Z" in prompt
    assert "Model 1:" in prompt
    assert Program1.model in prompt
    assert "Model 2:" in prompt
    assert Program2.model in prompt

    print(prompt)  # call pytest with -s to view prompt output in console


def test_build_prompt_no_parents():
    schema = _make_schema()
    prompt = schema.build_prompt(mode="explore", config={"num_parents": 0})

    assert "This is a fake prompt for 0 programs" in prompt
    assert "Be creative ..." in prompt
    assert Program1.model not in prompt


def test_build_prompt_with_current_program():
    schema = _make_schema(
        parent_program_template="Parent {parent_number}:\n{code_model}\n",
        parent_program_vars=["code.model"],
        current_program_template="New model:\n{code_model}\n",
        current_program_vars=["code.model"],
    )
    parent = _make_program(Program1.model)
    current = _make_program(Program2.model)

    prompt = schema.build_prompt(
        mode="explore",
        parent_programs=[parent],
        config={"num_parents": 1},
        current_program=current,
    )

    assert "Parent 1:" in prompt
    assert Program1.model in prompt
    assert "New model:" in prompt
    assert Program2.model in prompt
    # Current program appears after parent programs
    assert prompt.index("New model:") > prompt.index("Parent 1:")


def test_current_program_not_rendered_without_template():
    """current_program is silently ignored when current_program_template is None."""
    schema = _make_schema()  # no current_program_template
    parent = _make_program(Program1.model)
    current = _make_program(Program2.model)

    prompt = schema.build_prompt(
        mode="explore",
        parent_programs=[parent],
        config={"num_parents": 1},
        current_program=current,
    )

    assert Program1.model in prompt
    assert Program2.model not in prompt


def test_build_prompt_exploit_mode():
    schema = _make_schema(exploit="Be conservative ...")
    prompt = schema.build_prompt(mode="exploit", config={"num_parents": 0})

    assert "Be conservative ..." in prompt
    assert "Be creative ..." not in prompt


def test_build_prompt_ideas_injection():
    # Test that placeholder is correctly replaced if present in base prompt
    schema = _make_schema(base="This has placeholder: {ideas-injection-point}")

    # 1. Without ideas-injection-point in config, should default to empty string
    prompt = schema.build_prompt(mode="explore", config={"num_parents": 0})
    assert "This has placeholder: " in prompt
    assert "{ideas-injection-point}" not in prompt

    # 2. With ideas-injection-point in config, should substitute
    prompt = schema.build_prompt(
        mode="explore",
        config={"num_parents": 0, "ideas-injection-point": "Hello Idea!"},
    )
    assert "This has placeholder: Hello Idea!" in prompt
