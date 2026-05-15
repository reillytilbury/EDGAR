# Define programs which will be returned by the fake LLM call
from .programs import Program1, Program2, ProgramSolution, InvalidProgram

class FakeLLM:
    """A fake LLM engine that returns pre-configured candidates and seed_jax outputs instead of making actual LLM calls."""

    def __init__(self, offset: float = 0.1):
        self.programs = (Program1, InvalidProgram, ProgramSolution)
        self.model_counter = [0, 0, 0]
        self.model_jax_counter = [0, 0, 0]
        self.param_est_counter = [0, 0, 0]
        self.offset = offset

    @staticmethod
    def add_offset_to_code_str(code_str: str, offset: float):
        return code_str + f" + {offset:.3f}\n"

    # Assume gen_model, gen_model_jax, and gen_param_est are called in order
    def gen_model(self) -> str:
        """
        Return the next candidate program's model function as a code string. Adding an offset depending on how many times the program has been generated.
        """
        next_index = self.model_counter.index(min(self.model_counter))
        code_str = self.add_offset_to_code_str(
            self.programs[next_index].model,
            self.offset * self.model_counter[next_index],
        )
        self.model_counter[next_index] += 1
        return code_str

    def gen_model_jax(self) -> str:
        next_index = self.model_jax_counter.index(min(self.model_jax_counter))
        code_str = self.add_offset_to_code_str(
            self.programs[next_index].model_jax,
            self.offset * self.model_jax_counter[next_index],
        )
        self.model_jax_counter[next_index] += 1
        return code_str

    def gen_param_est(self) -> str:
        next_index = self.param_est_counter.index(min(self.param_est_counter))
        code_str = self.programs[next_index].param_est
        self.param_est_counter[next_index] += 1
        return code_str


class SeedFakeLLM:
    """A fake LLM engine that returns pre-configured model_jax outputs instead of making actual LLM calls."""

    model_v1_jax = (
        "import jax.numpy as jnp\n\n"
        "def model(data, params):\n"
        '\t""" y = a * relu(x - b) """\n'
        '\tx = data["x"]\n'
        '\ta = params["a"]\n'
        '\tb = params["b"]\n'
        "\treturn a * jnp.maximum(0, x - b)"
    )

    model_v2_jax = (
        "import jax.numpy as jnp\n\n"
        "def model(data, params):\n"
        '\t""" y = a * x + b """\n'
        '\tx = data["x"]\n'
        '\ta = params["a"]\n'
        '\tb = params["b"]\n'
        "\treturn a * x + b"
    )

    output_strs = (model_v1_jax, model_v2_jax)

    def __init__(self):
        self.gen_counter = 0

    def gen_model_jax(self) -> str:
        self.gen_counter += 1
        return self.output_strs[self.gen_counter - 1]
