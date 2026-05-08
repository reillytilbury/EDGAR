import asyncio
from src.llm.code_loading import load_function_from_source
from src.llm.generate import _generate_one_model, _generate_one_param_est
from src.llm.prompt_schema import PromptSchema
from tests.evolution.utils import make_empty_program
from tests.llm.fakellm import FakeLLM
from tests.llm.programs import InvalidProgram, Program1

def run_model_code(code: str, data: dict, params: dict):
    func = load_function_from_source(code, "model")
    return func(data, params)

def run_param_est_code(code: str, data: dict):
    func = load_function_from_source(code, "parameter_estimator")
    return func(data)


async def generate_one_fake_model():
    program = make_empty_program()
    parents = [Program1(), InvalidProgram()]
    prompt_schema = PromptSchema(
        base="...",
        explore="...",
        code_guidelines='...',
        docstring_guidelines="...",
        program_detail_template="..",
        program_vars=[],
    )
    llm = FakeLLM()
    llm_model = llm.gen_model() #A TestModel with code for Program1
    await _generate_one_model(program, parents, prompt_schema, llm_model, "explore", 1.0)
    return program

async def generate_one_fake_param_est():
    program = await generate_one_fake_model() #The output of _generate_one_model
    prompt_schema = PromptSchema(
        base="...",
        explore="...",
        code_guidelines='...',
        docstring_guidelines="...",
        program_detail_template="..",
        program_vars=[],
    )
    llm = FakeLLM()
    llm_model = llm.gen_param_est() #A TestModel with param_est for Program1
    await _generate_one_param_est(program, prompt_schema, llm_model)
    return program