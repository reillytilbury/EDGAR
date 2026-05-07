"""
Tests for Program in src/evolution/program.py.

Covers:
- compile(): returns callable (model_fn, param_est_fn) for valid code strings (not using JAX for simplicty)
- compile(): checks outputs of compiled functions against expected outputs
- compile(): raises ValueError when model or param_est code is missing or has the wrong entrypoint name
- count_params(): counts number of parameters in the program's param_est output and caches it in n_params
"""

# See conftest.py for make_program() fixture and specification of test model and param_est
import numpy as np
import pytest

from src.evolution.program import _NotYetPrepared

class TestCompile:
    def test_compile_returns_callables(self, make_program):
        model_fn, param_est_fn = make_program().compile()
        assert callable(model_fn)
        assert callable(param_est_fn)

    def test_compiled_model_produces_correct_numeric_output(self, make_program):
        data = {"x": np.array([1.0, 2.0])}
        model_fn, param_est_fn = make_program().compile()
        params = param_est_fn(data)
        assert list(params.values()) == pytest.approx(
            [1.0, 2.0]
        )  # a=x_min=1, b=x_max=2
        result = model_fn(data, params)
        assert result.tolist() == pytest.approx([3.0, 4.0])  # y = x + 2

    def test_compile_raises_when_model_code_missing(self, make_program):
        with pytest.raises(ValueError, match="model"):
            make_program(model_code=None).compile()

    def test_compile_raises_when_param_est_code_missing(self, make_program):
        with pytest.raises(ValueError, match="parameter_estimator"):
            make_program(param_est_code=None).compile()

    def test_compile_raises_when_model_entrypoint_wrong(
        self, make_program, wrong_entrypoint_code
    ):
        with pytest.raises(ValueError, match="model"):
            make_program(model_code=wrong_entrypoint_code).compile()

    def test_compile_raises_when_param_est_entrypoint_wrong(
        self, make_program, wrong_entrypoint_code
    ):
        with pytest.raises(ValueError, match="parameter_estimator"):
            make_program(param_est_code=wrong_entrypoint_code).compile()

def test_count_params(make_program):
    program = make_program()
    assert program.n_params is None
    n_params = program.count_params()
    assert n_params == 2  # a and b
    assert program.n_params == n_params  # cached value

def test_count_params_fails_on_invalid_code(make_program):
    program = make_program(model_code="def model(data, params): return 1") #No model.DEFAULT_PARAMS
    with pytest.raises(AttributeError):
        program.count_params()

class TestLossesDefaults:
    def test_discover_final_is_none(self, make_program):
        assert make_program().program_losses.discover.final is None

    def test_discover_init_is_none(self, make_program):
        assert make_program().program_losses.discover.init is None

    def test_validate_final_is_not_yet_prepared(self, make_program):
        assert isinstance(make_program().program_losses.validate.final, _NotYetPrepared)

    def test_validate_init_is_none(self, make_program):
        assert make_program().program_losses.validate.init is None
