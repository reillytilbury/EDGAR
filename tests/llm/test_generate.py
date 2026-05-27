import pytest
from edgar.evolution.program import BirthCertificate, Code, LossPair, Losses, Program
from edgar.llm.generate import _generate_one_model, _generate_one_param_est, generate_models, generate_param_ests, _translate_one_model, translate_programs
from edgar.llm.code_loading import load_function_from_source
from edgar.llm.prompt_schema import PromptSchema
from tests.evolution.utils import make_empty_program
from tests.llm.programs import Program1, InvalidProgram, Program2, ProgramSolution
from tests.llm.fakellm import FakeLLM, CyclingModel
from tests.llm.utils import generate_one_fake_model, generate_one_fake_param_est, generate_fake_models, make_fake_spec

LLM_MODEL = "gemini-2.5-flash-lite" #used for real LLM calls 

# --- Fake LLM tests --- 
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
    await _generate_one_model(program, parents, prompt_schema, llm_model, "explore", 1.0, spec=make_fake_spec(output_dir="test_output"), data={})

    header = '"""\nfake thought process\n\n' + Program1.latex_equation + '\n"""\n\n'
    assert program.code.model == header + Program1.model + " + 0.000\n"
    assert program.default_params == Program1.default_params
    assert program.name == "Fake Model 0"
    assert program.birth == birth
    assert program.image_path is not None

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

    assert program.code.param_est == Program1.param_est
    
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
    param_ests = [None] + [programs[i].param_est for i in range(1,5)]
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
    assert program.code.param_est == Program1.param_est #Check param est code is unchanged

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

# --- Real LLM tests ---
@pytest.mark.live
@pytest.mark.asyncio
async def test_generate_one_model_with_real_llm():
    program = make_empty_program()
    birth = program.birth
    prompt_schema = PromptSchema(
        base="You are an AI Scientist. Below are 2 models describing a phenomenon, sorted from worst to best. Your task is to create a new model, that is better than the models below.",
        exploit = "Focus on generating a model which executes, not too disimilar to the existing model",
        code_guidelines = "Import any packages you use." \
        "Model signature must be `def model(data, params):` where `data is a dict of named arrays and `params` is a dict of named scalars.",
        docstring_guidelines = "Include a docstring, with a short descriptive name for the model",
        image_analysis_instructions = "Add a short description in the docstring of the image you see",
        program_detail_template="Model {parent_number}: {name}" \
        "loss: {program_losses_discover_final}" \
        "" \
        "{code_model}",
        program_vars = ["name", "program_losses.discover.final", "code.model"]
    )
    program1 = Program(
        birth = BirthCertificate(generation=0, island=0, batch_index=0),
        code = Code(model=Program1.model),
        name = "Program1",
        program_losses = Losses(discover=LossPair(final=0.5))
    )
    program2 = Program(
        birth = BirthCertificate(generation=0, island=0, batch_index=1),
        code = Code(model=Program2.model),
        name = "Program2",
        program_losses = Losses(discover=LossPair(final=0.1))
    )
    parents = [program1, program2]
    await _generate_one_model(program, parents, prompt_schema, llm = LLM_MODEL,
                              mode = "exploit", temperature = 1.0, spec = make_fake_spec(output_dir="test_output"), data  = {})
    print("Generated model code:\n", program.code.model)
    assert "def model(data, params):" in program.code.model
    print("Generated default params:\n", program.default_params)
    assert isinstance(program.default_params, dict)
    print("Generated model name:\n", program.name)
    assert isinstance(program.name, str)
    print("Generated program birth info:\n", program.birth)
    assert program.birth == birth
    assert program.image_path is not None
    assert load_function_from_source(program.code.model, "model") is not None

@pytest.mark.live
@pytest.mark.asyncio
async def test_generate_one_param_est_with_real_llm():
    program = Program(
        birth=BirthCertificate(generation=0, island=0, batch_index=0),
        code=Code(model=Program1.model),
        name="Program1",
        program_losses=Losses(discover=LossPair(final=0.5))
    )
    model_code = program.code.model
    prompt_schema = PromptSchema(
        base="You are an AI Scientist. Given the model below, write a parameter estimator for it.",
        explore="The estimator should return a sensible initial guess for the model parameters.",
        code_guidelines="Function signature must be `def parameter_estimator(data):` where `data` is a dict of named arrays. Return a dict of named scalar floats.",
        docstring_guidelines="Include a short docstring describing the estimation strategy.",
        program_detail_template="Model: {name}\n\n{code_model}",
        program_vars=["name", "code.model"],
    )
    await _generate_one_param_est(program, prompt_schema, llm=LLM_MODEL)
    print("Generated param est code:\n", program.code.param_est)
    assert "def parameter_estimator(data):" in program.code.param_est
    assert isinstance(program.code.param_est, str)
    assert program.code.model == model_code
    assert load_function_from_source(program.code.param_est, "parameter_estimator") is not None

@pytest.mark.live
@pytest.mark.asyncio
async def test_translate_one_model_with_real_llm():
    program = Program(
        birth=BirthCertificate(generation=0, island=0, batch_index=0),
        code=Code(model=Program1.model),
        name="Program1",
    )
    model_code = program.code.model
    prompt_schema = PromptSchema(
        base="You are an AI Scientist. Translate the numpy model below into JAX.",
        explore="Preserve the logic exactly; only replace numpy with jax.numpy.",
        code_guidelines="Import jax.numpy as jnp. Function signature must be `def model(data, params):` identical to the original.",
        docstring_guidelines="Keep the original docstring unchanged.",
        program_detail_template="Model: {name}\n\n{code_model}",
        program_vars=["name", "code.model"],
    )
    await _translate_one_model(program, prompt_schema, llm=LLM_MODEL)
    print("Generated JAX model code:\n", program.code.model_jax)
    assert "def model(data, params):" in program.code.model_jax
    assert "jnp" in program.code.model_jax
    assert program.code.model == model_code
    assert load_function_from_source(program.code.model_jax, "model") is not None