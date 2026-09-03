"""
Tests for Program in src/evolution/program.py.
"""

# See conftest.py for make_program() fixture and specification of test model and param_est
import numpy as np
import pytest

from edgar.evolution.program import (
    NotValidated,
    ModelLoadingError,
)
from tests.evolution.utils import make_program, wrong_entrypoint_code


class TestCompile:
    def test_compile_returns_callables(self):
        program = make_program()
        model_fn = program.compile_model()
        param_est_fn = program.compile_param_ests()[0]
        assert callable(model_fn)
        assert callable(param_est_fn)

    def test_compiled_model_produces_correct_numeric_output(self):
        data = {"x": np.array([1.0, 2.0])}
        program = make_program()
        model_fn = program.compile_model()
        param_est_fn = program.compile_param_ests()[0]
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

    def test_compile_param_ests_partial_failure_warns_and_continues(self):
        # Estimators: 0 is healthy, 1 is broken, 2 is healthy
        from tests.evolution.utils import linear_param_est_code

        broken_code = wrong_entrypoint_code()
        healthy_code = linear_param_est_code()

        program = make_program(param_est_code=[healthy_code, broken_code, healthy_code])

        with pytest.warns(UserWarning, match="Failed to compile parameter estimator 1"):
            compiled = program.compile_param_ests()

        assert len(compiled) == 2
        assert callable(compiled[0])
        assert callable(compiled[1])

    def test_compile_raises_when_model_entrypoint_wrong(self):
        program = make_program(model_code=wrong_entrypoint_code())
        with pytest.raises(ModelLoadingError, match="model"):
            program.compile_model()


def test_code_param_est_coercion_scenarios():
    from edgar.evolution.program import Code

    # 1. Assigned as a single string
    c = Code(param_est="def my_estimator(): ...")
    assert c.param_est == ["def my_estimator(): ..."]

    # 2. Assigned as a list of strings
    c = Code(param_est=["est_1", "est_2"])
    assert c.param_est == ["est_1", "est_2"]

    # 3. Assigned as None
    c = Code(param_est=None)
    assert c.param_est is None


def test_param_est_code_property():
    est_1 = "def est_1(): ..."
    program = make_program(param_est_code=[est_1])

    # 1. Before scoring (best_param_est is None): returns empty string
    assert program.param_est_code == ""

    # 2. After scoring (best_param_est is set): resolves directly to the best
    program.code.best_param_est = est_1
    assert program.param_est_code == est_1


def test_no_default_params():
    program = make_program()
    assert program.n_params is None
    with pytest.warns(UserWarning):
        assert program.default_params is None


def test_initializing_with_default_params():
    default_params = {"a": 1.0, "b": 2.0}
    program = make_program(default_params=default_params)
    assert program.n_params == 2
    assert program.default_params == default_params


def test_setting_default_params_after_initialization():
    program = make_program()
    assert program.n_params is None
    assert program.default_params is None
    default_params = {"a": 1.0, "b": 2.0, "c": 3.0}
    program.default_params = default_params
    assert program.n_params == 3
    assert program.default_params == default_params


def test_setting_default_params_with_invalid_input():
    program = make_program()
    with pytest.warns(UserWarning):
        program.default_params = "not a dict"
    assert program.n_params is None
    assert program.default_params is None


def test_setting_default_params_with_callable():
    def default_params_fn(data):
        return {"a": np.zeros(data["x"].shape), "b": np.ones(data["x"].shape)}

    data = {"x": np.array([1.0, 2.0, 3.0])}
    program = make_program(data=data, default_params=default_params_fn)
    assert program.n_params == 6  # a and b each have 3 parameters
    expected_params = {"a": np.zeros(3), "b": np.ones(3)}
    assert all(
        np.array_equal(program.default_params[key], expected_params[key])
        for key in expected_params
    )


def test_setting_default_params_with_callable_after_initialization():
    def default_params_fn(data):
        return {"a": np.zeros(data["x"].shape), "b": np.ones(data["x"].shape)}

    data = {"x": np.array([1.0, 2.0, 3.0])}
    program = make_program(data=data)
    program.default_params = default_params_fn
    assert program.n_params == 6  # a and b each have 3 parameters
    expected_params = {"a": np.zeros(3), "b": np.ones(3)}
    assert all(
        np.array_equal(program.default_params[key], expected_params[key])
        for key in expected_params
    )


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


def test_setting_callable_default_params_resolution_failure():
    def faulty_default_params_fn(data):
        raise ValueError("Simulated resolution error")

    data = {"x": np.array([1.0, 2.0])}
    program = make_program(data=data)
    with pytest.warns(UserWarning, match="Failed to resolve dynamic default_params"):
        with pytest.warns(UserWarning, match="Invalid default_params"):
            program.default_params = faulty_default_params_fn
    assert program.n_params is None
    assert program.default_params is None


def test_setting_callable_default_params_without_data_raises_error():
    def default_params_fn(data):
        return {"a": 1.0}

    program = make_program(data=None)
    with pytest.raises(
        RuntimeError,
        match="Cannot resolve dynamic default_params.*because program.data is None",
    ):
        program.default_params = default_params_fn


def test_status_default_alive():
    program = make_program()
    assert program.status == "alive"
