# Define programs which will be returned by the fake LLM call
import inspect
import re
import textwrap
import numpy as np
import jax.numpy as jnp


def _extract_src(func) -> str:
    src = textwrap.dedent(inspect.getsource(func))
    return re.sub(r"@staticmethod\s*\n", "", src)


def make_program(base_cls, offset: float):
    """Return a variant of base_cls whose model adds a constant offset to the output."""
    base_jax_src = _extract_src(base_cls.__dict__["model_jax"].__func__)
    renamed = base_jax_src.replace("def model_jax(", "def _base(", 1)
    jax_src = (
        f"import jax.numpy as jnp\n\n{renamed}\n"
        f"def model_jax(data, params):\n    return _base(data, params) + {offset}\n"
    )

    class _Variant:
        @staticmethod
        def model(data, params):
            return base_cls.model(data, params) + offset

        @staticmethod
        def model_jax(data, params):
            return base_cls.model_jax(data, params) + offset

        model.DEFAULT_PARAMS = base_cls.__dict__["model"].DEFAULT_PARAMS

        @staticmethod
        def param_est(data):
            return base_cls.param_est(data)

    _Variant._jax_src = jax_src
    _Variant.__name__ = f"{base_cls.__name__}_offset{offset}"
    return _Variant


def build_candidates(n_iterations: int, n_islands: int, batch_size: int) -> list:
    """Build a candidate list with offset 2*iteration + 5*island per (iteration, island, batch) slot."""
    base_programs = [Program1, Program2, ProgramSolution]
    return [
        make_program(base_programs[k % len(base_programs)], offset=2 * i + 5 * island)
        for i in range(n_iterations)
        for island in range(n_islands)
        for k in range(batch_size)
    ]


_fake_cfg: dict = {
    "candidates": None,
    "seed_jax": None,
    "_gen_model_counter": 0,
    "_jax_trans_counter": 0,
    "_param_est_counter": 0,
}


def configure_fake(candidates: list, seed_jax: list | None = None) -> None:
    """Configure fake LLM responses for testing. Call before running hypothesis_engine."""
    _fake_cfg["candidates"] = candidates
    _fake_cfg["seed_jax"] = seed_jax or []
    _fake_cfg["_gen_model_counter"] = 0
    _fake_cfg["_jax_trans_counter"] = 0
    _fake_cfg["_param_est_counter"] = 0


def setup_fake_engine(params: dict, model_v1_jax, model_v2_jax) -> None:
    """Build offset candidates and seed_jax, then configure the fake LLM engine."""
    seed_jax = [
        (inspect.getsource(model_v1_jax), model_v1_jax),
        (inspect.getsource(model_v2_jax), model_v2_jax),
    ]
    candidates = build_candidates(
        n_iterations=int(params.get("n_iterations", 1)),
        n_islands=int(params.get("n_islands", 1)),
        batch_size=int(params.get("batch_size", 1)),
    )
    configure_fake(candidates, seed_jax=seed_jax)


class Program1:
    @staticmethod
    def model(data, params):
        """y = ax^2 + bx + c"""
        x = data["x"]
        a = params["a"]
        b = params["b"]
        c = params["c"]
        return a * x**2 + b * x + c

    @staticmethod
    def model_jax(data, params):
        """y = ax^2 + bx + c"""
        x = data["x"]
        a = params["a"]
        b = params["b"]
        c = params["c"]
        return a * x**2 + b * x + c

    model.DEFAULT_PARAMS = {"a": float(1), "b": float(0), "c": float(0)}

    @staticmethod
    def param_est(data):
        return {"a": float(1), "b": float(0), "c": float(0)}


class Program2:
    @staticmethod
    def model(data, params):
        """y = ax^3 + bx^2 + cx + d"""
        x = data["x"]
        a = params["a"]
        b = params["b"]
        c = params["c"]
        d = params["d"]
        return a * x**3 + b * x**2 + c * x + d

    @staticmethod
    def model_jax(data, params):
        """y = ax^3 + bx^2 + cx + d"""
        x = data["x"]
        a = params["a"]
        b = params["b"]
        c = params["c"]
        d = params["d"]
        return a * x**3 + b * x**2 + c * x + d

    model.DEFAULT_PARAMS = {"a": float(1), "b": float(0), "c": float(0), "d": float(0)}

    @staticmethod
    def param_est(data):
        return {"a": float(1), "b": float(0), "c": float(0), "d": float(0)}


class ProgramSolution:
    """Exact solution of the noiseless synthetic data problem: y = (a*x^2 + b*x + c) * sin(k*x + phi_0)"""

    @staticmethod
    def model(data, params):
        x = data["x"]
        a = params["a"]
        b = params["b"]
        c = params["c"]
        k = params["k"]
        phi_0 = params["phi_0"]
        return (a * x**2 + b * x + c) * np.sin(k * x + phi_0)

    @staticmethod
    def model_jax(data, params):
        x = data["x"]
        a = params["a"]
        b = params["b"]
        c = params["c"]
        k = params["k"]
        phi_0 = params["phi_0"]
        return (a * x**2 + b * x + c) * jnp.sin(k * x + phi_0)

    model.DEFAULT_PARAMS = {
        "a": float(1),
        "b": float(0),
        "c": float(0),
        "k": float(1),
        "phi_0": float(0),
    }

    @staticmethod
    def param_est(data):
        return {
            "a": float(1),
            "b": float(0),
            "c": float(0),
            "k": float(1),
            "phi_0": float(0),
        }
