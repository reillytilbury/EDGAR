import tempfile
from types import SimpleNamespace
import numpy as np
from PIL import Image
from edgar.llm.code_loading import load_function_from_source
from edgar.llm.generate import (
    _generate_one_model,
    _generate_one_param_est,
    generate_models,
)
from edgar.llm.prompt_schema import PromptSchema
from tests.evolution.utils import make_empty_program
from tests.llm.fakellm import FakeLLM, CyclingModel
from tests.llm.programs import InvalidProgram, Program1, DEFAULT_FAKE_PROGRAMS


def make_fake_spec(output_dir: str | None = None) -> SimpleNamespace:
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    def plot_fn(data, parents, save_path, **kwargs):
        with open(save_path, "wb") as f:
            f.write(generate_image_bytes())

    return SimpleNamespace(
        plot_fn=plot_fn, output_dir=output_dir, rng=np.random.default_rng()
    )


def run_model_code(code: str, data: dict, params: dict):
    func = load_function_from_source(code, "model")
    return func(data, params)


def run_param_est_code(code: str, data: dict):
    func = load_function_from_source(code, "parameter_estimator")
    return func(data)


def generate_image_bytes():
    matrix = np.array([[0, 255, 0], [255, 0, 255], [0, 255, 0]], dtype=np.uint8)
    img = Image.fromarray(matrix, mode="L")
    img.save("test_image.png")
    return open("test_image.png", "rb").read()


async def generate_one_fake_model():
    program = make_empty_program()
    parents = [Program1(), InvalidProgram()]
    prompt_schema = PromptSchema(
        base="...",
        explore="...",
        code_guidelines="...",
        docstring_guidelines="...",
        parent_program_template="..",
        parent_program_vars=[],
    )
    llm = FakeLLM(DEFAULT_FAKE_PROGRAMS)
    llm_model = llm.gen_model()  # A TestModel with code for Program1
    await _generate_one_model(
        program,
        parents,
        prompt_schema,
        llm_model,
        "explore",
        1.0,
        spec=make_fake_spec(),
        data={},
    )
    return program


async def generate_fake_models(n: int):
    population = [make_empty_program() for _ in range(n)]
    prompt_schema = PromptSchema(
        base="...",
        explore="...",
        code_guidelines="...",
        docstring_guidelines="...",
        parent_program_template="..",
        parent_program_vars=[],
    )
    llm = FakeLLM(DEFAULT_FAKE_PROGRAMS)
    llm_models = CyclingModel([llm.gen_model() for _ in range(n)])
    await generate_models(
        population,
        prompt_schema,
        llm_models,
        "explore",
        1.0,
        spec=make_fake_spec(),
        data={},
    )
    return population


async def generate_one_fake_param_est():
    program = await generate_one_fake_model()  # The output of _generate_one_model
    prompt_schema = PromptSchema(
        base="...",
        explore="...",
        code_guidelines="...",
        docstring_guidelines="...",
        parent_program_template="..",
        parent_program_vars=[],
    )
    llm = FakeLLM(DEFAULT_FAKE_PROGRAMS)
    llm_model = llm.gen_param_est()  # A TestModel with param_est for Program1
    param_est = await _generate_one_param_est(
        program, [], prompt_schema, llm_model, config={"n_param_ests": 1}
    )
    program.code.param_est = param_est
    return program
