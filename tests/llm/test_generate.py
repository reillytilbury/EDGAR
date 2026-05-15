import pytest
from src.llm.generate import _generate_one_model, _generate_one_param_est, generate_models, generate_param_ests, _translate_one_model, translate_programs
from src.llm.prompt_schema import PromptSchema
from tests.evolution.utils import make_empty_program
from tests.llm.programs import Program1, InvalidProgram, ProgramSolution
from tests.llm.fakellm import FakeLLM, CyclingModel
from tests.llm.utils import generate_one_fake_model, generate_one_fake_param_est, generate_fake_models

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
    existing_program.name = "Existing Model"
    existing_program.default_params = {"fake params": 0.5}

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
    assert population[0].default_params == {"fake params": 0.5}
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
    existing_program.name = "Existing Model"
    existing_program.default_params = {"fake params": 0.5}
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
    llm_models = CyclingModel([llm.gen_model() for _ in range(9)])
    await generate_models(population, prompt_schema, llm_models, "explore", 1.0)

    #Check existing program is unchanged
    assert population[0].code.model == "Existing model code"
    assert population[0].default_params == {"fake params": 0.5}
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

@pytest.mark.asyncio
async def test_generate_param_est():
    """
        Generate param ests for, programs without model code or param est code, programs with model code but no param est code, and programs with both.
        Check the programs which require it are suitably mutated.
    """
    prompt_schema = PromptSchema(
        base="...",
        explore="...",
        code_guidelines='...',
        docstring_guidelines="...",
        program_detail_template="..",
        program_vars=[],
    )
    no_model = make_empty_program() #Needs model and param est
    model_no_param_est = await generate_fake_models(3)
    model_and_param_est = await generate_one_fake_param_est() #Already has model and param est
    llm = FakeLLM()
    llm_models = CyclingModel([llm.gen_param_est() for _ in range(3)])
    population = [no_model] + model_no_param_est + [model_and_param_est]
    await generate_param_ests(population, prompt_schema, llm_models)

    #Check expected solutions
    programs = [None, Program1(), InvalidProgram(), ProgramSolution(), Program1()]
    model_headers = [None] + ['"""\nfake thought process\n\n' + programs[i].latex_equation + '\n"""\n\n' for i in range(1,5)]
    model_footers = [None]+ [" + 0.000\n" for i in range(1,5)]
    models = [None] + [model_headers[i] + programs[i].model + model_footers[i] for i in range(1,5)]
    param_est_headers = [None] + ['"""\nfake thought process\n"""\n\n' for _ in range(1,5)]
    param_ests = [None] + [param_est_headers[i] + programs[i].param_est for i in range(1,5)]
    #Check model codes
    for i, program in enumerate(population):
        assert program.code.model == models[i]

    #Check param est codes
    for i, program in enumerate(population):
        assert program.code.param_est == param_ests[i]

#JAX translation
@pytest.mark.asyncio
async def test_translate_one_model():
    program = await generate_one_fake_param_est() #The output of _generate_one_param_est
    assert program.code.model_jax is None
    prompt_schema = PromptSchema(
        base="...",
        explore="...",
        code_guidelines='...',
        docstring_guidelines="...",
        program_detail_template="..",
        program_vars=[],
    )
    llm = FakeLLM()
    llm_model = llm.gen_model_translation() #A TestModel with code.model_jax for Program1
    await _translate_one_model(program, prompt_schema, llm_model)

    assert program.code.model_jax == Program1.model_jax + " + 0.000\n"
    assert program.code.model == '"""\nfake thought process\n\n' + Program1.latex_equation + '\n"""\n\n' + Program1.model + " + 0.000\n" #Check model code is unchanged
    assert program.code.param_est == '"""\nfake thought process\n"""\n\n' + Program1.param_est #Check param est code is unchanged

@pytest.mark.asyncio
async def test_translate_models():
    """
        Generate model jax translations for programs with model code, with model code but no jax translations, and with model code and jax translations.
        Check the programs which require it are suitably mutated.
    """
    prompt_schema = PromptSchema(
        base="...",
        explore="...",
        code_guidelines='...',
        docstring_guidelines="...",
        program_detail_template="..",
        program_vars=[],
    )
    no_model = make_empty_program() #Needs model and param est
    models_no_jax = await generate_fake_models(3)
    model_jax = await generate_one_fake_model() #Already has model
    model_jax.code.model_jax = "Existing jax model code" #Set existing jax model code to check it is unchanged
    llm = FakeLLM()
    llm_models = CyclingModel([llm.gen_model_translation() for _ in range(3)])
    population = [no_model] + models_no_jax + [model_jax]
    await translate_programs(population, prompt_schema, llm_models)

    #Check expected solutions
    programs = [None, Program1(), InvalidProgram(), ProgramSolution(), Program1()]
    model_footer = " + 0.000\n"
    model_jaxes = [None] + [programs[i].model_jax + model_footer for i in range(1,4)] + ["Existing jax model code"] #Expected jax translations

    #Check model jax code
    for i, program in enumerate(population):
        assert program.code.model_jax == model_jaxes[i]