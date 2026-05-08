import pytest
from src.llm.generate import _generate_one_model, _generate_one_param_est, generate_models
from src.llm.prompt_schema import PromptSchema
from tests.evolution.utils import make_empty_program
from tests.llm.programs import Program1, InvalidProgram, ProgramSolution
from tests.llm.fakellm import FakeLLM
from tests.llm.utils import generate_one_fake_model, generate_one_fake_param_est

#Model generation
@pytest.mark.asyncio
async def test_generate_one_model():
    program = make_empty_program()
    birth = program.birth
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

    header = '"""\nfake thought process\n\n' + Program1.latex_equation + '\n"""\n\n'
    assert program.code.model == header + Program1.model + " + 0.000\n"
    assert program.default_params == Program1.default_params
    assert program.name == "Fake Model 0"
    assert program.birth == birth
    #print(program)

@pytest.mark.asyncio
async def test_generate_same_model():
    """
        Generate three models cycling of just Program1.
        Check the programs are suitably mutated with the expected code, default_params, name, and that birth info is unchanged.
    """
    existing_program = make_empty_program()
    existing_program.code.model = "Existing model code"
    existing_program.default_params = None
    existing_program.name = "Existing Model"

    population = [existing_program] + [make_empty_program() for _ in range(3)] #first program already has model, then 3 Program1s
    birth_info = [p.birth for p in population]
    prompt_schema = PromptSchema(
        base="...",
        explore="...",
        code_guidelines='...',
        docstring_guidelines="...",
        program_detail_template="..",
        program_vars=[],
    )
    llm = FakeLLM()
    llm_model = llm.gen_model() # A TestModel with code for Program1
    await generate_models(population, prompt_schema, llm_model, "explore", 1.0)

    #Check existing program is unchanged
    assert population[0].code.model == "Existing model code"
    assert population[0].default_params is None
    assert population[0].name == "Existing Model"
    assert population[0].birth == birth_info[0]

    #Check mutated programs
    mutated_programs = 3 * [Program1()] #Expected programs generated

    for i, program in enumerate(population[1:]):
        header = '"""\nfake thought process\n\n' + mutated_programs[i].latex_equation + '\n"""\n\n'
        assert program.code.model == header + mutated_programs[i].model + " + 0.000\n"
        assert program.default_params == mutated_programs[i].default_params
        assert program.name == "Fake Model 0"
        assert program.birth == birth_info[i+1]

@pytest.mark.asyncio
async def test_generate_distinct_models():
    """
        Generate three models cycling through fake programs Program1, InvalidProgram, and ProgramSolution.
        Check the programs are suitably mutated with the expected code, default_params, name, and that birth info is unchanged.
    """
    existing_program = make_empty_program()
    existing_program.code.model = "Existing model code"
    existing_program.default_params = None
    existing_program.name = "Existing Model"
    population = [existing_program] + [make_empty_program() for _ in range(9)]
    birth_info = [p.birth for p in population]
    prompt_schema = PromptSchema(
        base="...",
        explore="...",
        code_guidelines='...',
        docstring_guidelines="...",
        program_detail_template="..",
        program_vars=[],
    )
    llm = FakeLLM()
    llm_models = [llm.gen_model() for _ in range(9)] #TestModels with code for Program1, InvalidProgram, and ProgramSolution
    await generate_models(population, prompt_schema, llm_models, "explore", 1.0)

    #Check existing program is unchanged
    assert population[0].code.model == "Existing model code"
    assert population[0].default_params is None
    assert population[0].name == "Existing Model"
    assert population[0].birth == birth_info[0]

    programs = 3 * [Program1(), InvalidProgram(), ProgramSolution()] #Expected programs generated

    for i, program in enumerate(population[1:]):
        header = '"""\nfake thought process\n\n' + programs[i].latex_equation + '\n"""\n\n'
        assert program.code.model == header + programs[i].model + f" + {i//3*llm.offset:.3f}\n"
        assert program.default_params == programs[i].default_params
        assert program.name == f"Fake Model {i%3}"
        assert program.birth == birth_info[i+1]

#Parameter estimator generation
@pytest.mark.asyncio
async def test_generate_one_param_est():
    program = await generate_one_fake_model() #The output of _generate_one_model
    model_code = program.code.model
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

    header = '"""\nfake thought process\n"""\n\n'
    assert program.code.param_est == header + Program1.param_est
    
    assert program.code.model == model_code #Check model code is unchanged

# #Parameter estimator generation
# @pytest.mark.asyncio
# async def test_generate_param_est_doesnt_need_parameter_estimation():
#     program = await generate_one_fake_param_est() #Already has parameter est. for Program1
#     model_code = program.code.model
#     prompt_schema = PromptSchema(
#         base="...",
#         explore="...",
#         code_guidelines='...',
#         docstring_guidelines="...",
#         program_detail_template="..",
#         program_vars=[],
#     )
#     llm = FakeLLM()
#     _ = llm.gen_param_est()
#     llm_model = llm.gen_param_est() #A TestModel with param_est for InvalidProgram
#     await _generate_one_param_est(program, prompt_schema, llm_model)

#     print(program.code.param_est)

#     header = '"""\nfake thought process\n"""\n\n'
#     assert program.code.param_est == header + Program1.param_est #Check param_est code is unchanged
#     assert program.code.model == model_code #Check model code is unchanged