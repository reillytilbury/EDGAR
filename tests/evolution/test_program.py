"""
Tests for Program in src/evolution/program.py.

Covers:
- compile_model() / compile_param_est(): return callables for valid code strings
- compile_model() / compile_param_est(): check outputs against expected values
- compile_model(): raises ModelLoadingError when model code is missing or has wrong entrypoint
- compile_param_est(): raises ParamEstLoadingError when param_est code is missing or has wrong entrypoint
- count_params(): counts number of parameters in the program's param_est output and caches it in n_params
"""

# See conftest.py for make_program() fixture and specification of test model and param_est
import numpy as np
import pytest

from edgar.evolution.program import NotValidated
from tests.evolution.utils import make_program, wrong_entrypoint_code

class TestCompile:
    def test_compile_returns_callables(self):
        program = make_program()
        model_fn = program.compile_model()
        param_est_fn = program.compile_param_est()
        assert callable(model_fn)
        assert callable(param_est_fn)

    def test_compiled_model_produces_correct_numeric_output(self):
        data = {"x": np.array([1.0, 2.0])}
        program = make_program()
        model_fn = program.compile_model()
        param_est_fn = program.compile_param_est()
        params = param_est_fn(data)
        assert list(params.values()) == pytest.approx(
            [1.0, 2.0]
        )  # a=x_min=1, b=x_max=2
        result = model_fn(data, params)
        assert result.tolist() == pytest.approx([3.0, 4.0])  # y = x + 2

    def test_compile_raises_when_model_code_missing(self):
        program = make_program(model_code=None)
        with pytest.raises(ModelLoadingError, match="model"):
            program.compile_model()

    def test_compile_raises_when_param_est_code_missing(self):
        program = make_program(param_est_code=None)
        with pytest.raises(ParamEstLoadingError, match="parameter_estimator"):
            program.compile_param_est()

    def test_compile_raises_when_model_entrypoint_wrong(self):
        program = make_program(model_code=wrong_entrypoint_code())
        with pytest.raises(ModelLoadingError, match="model"):
            program.compile_model()

    def test_compile_raises_when_param_est_entrypoint_wrong(self):
        program = make_program(param_est_code=wrong_entrypoint_code())
        with pytest.raises(ParamEstLoadingError, match="parameter_estimator"):
            program.compile_param_est()

def test_no_default_params():
    program = make_program()
    assert program.n_params is None
    with pytest.warns(UserWarning):
        assert program.default_params is None

def test_initializing_with_default_params():
    default_params = {'a': 1.0, 'b': 2.0}
    program = make_program(default_params=default_params)
    assert program.n_params == 2
    assert program.default_params == default_params

def test_setting_default_params_after_initialization():
    program = make_program()
    assert program.n_params is None
    assert program.default_params is None
    default_params = {'a': 1.0, 'b': 2.0, 'c': 3.0}
    program.default_params = default_params
    assert program.n_params == 3
    assert program.default_params == default_params

def test_setting_default_params_with_invalid_input():
    program = make_program()
    with pytest.warns(UserWarning):
        program.default_params = "not a dict"
    assert program.n_params is None
    assert program.default_params is None

class TestLossesDefaults:
    def test_discover_final_is_none(self):
        program = make_program()
        assert program.program_losses.discover.final is None

    def test_discover_init_is_none(self):
        program = make_program()
        assert program.program_losses.discover.init is None

    def test_validate_final_is_not_yet_prepared(self):
        program = make_program()
        assert isinstance(program.program_losses.validate.final, NotValidated) 

    def test_validate_init_is_none(self):
        program = make_program()
        assert program.program_losses.validate.init is None
