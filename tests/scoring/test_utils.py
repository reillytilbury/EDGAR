import numpy as np
import jax.numpy as jnp
import pytest
from edgar.scoring.utils import (
    _evaluate_sample_losses,
    _evaluate_scalar_loss,
    _safe_loss,
    _evaluate_model_output,
)
from edgar.evolution.program import NotValidated
from edgar.llm.code_loading import load_function_from_source

# _safe_loss


def test_safe_loss_basic():
    # Valid numeric values
    assert _safe_loss(12.34) == 12.34
    assert _safe_loss(0.0) == 0.0
    assert _safe_loss(-5.2) == -5.2


def test_safe_loss_none_and_not_validated():
    # None maps to inf, NotValidated should raise TypeError
    assert _safe_loss(None) == float("inf")
    with pytest.raises(TypeError):
        _safe_loss(NotValidated())


def test_safe_loss_nan():
    # NaN values should map to float("inf")
    assert _safe_loss(float("nan")) == float("inf")
    assert _safe_loss(np.nan) == float("inf")


def test_safe_loss_invalid_types():
    # Strings that cannot be floats, or arbitrary objects should raise exceptions
    with pytest.raises(ValueError):
        _safe_loss("not_a_float")
    with pytest.raises(TypeError):
        _safe_loss([])
    with pytest.raises(TypeError):
        _safe_loss({})


def test_safe_sorting():
    # Sorting a list containing NaN and None with _safe_loss key
    test_cases = [
        {"idx": 1, "loss": 16.035},
        {"idx": 2, "loss": 16.075},
        {"idx": 3, "loss": float("nan")},
        {"idx": 4, "loss": 15.954},
        {"idx": 5, "loss": None},
        {"idx": 6, "loss": np.nan},
        {"idx": 7, "loss": 16.120},
    ]

    # Sort using _safe_loss key
    sorted_cases = sorted(test_cases, key=lambda x: _safe_loss(x["loss"]))

    # Expected order:
    # 1. idx 4: 15.954
    # 2. idx 1: 16.035
    # 3. idx 2: 16.075
    # 4. idx 7: 16.120
    # Following are infinity: idx 3, 5, 6
    assert sorted_cases[0]["idx"] == 4
    assert sorted_cases[1]["idx"] == 1
    assert sorted_cases[2]["idx"] == 2
    assert sorted_cases[3]["idx"] == 7

    # Ensure NaN/None elements are placed at the end
    inf_indices = {x["idx"] for x in sorted_cases[4:]}
    assert inf_indices == {3, 5, 6}


# _evaluate_model_output
BASIC_MODEL = """
def model(data, params):
    return params["w"] * data["x"]
"""
STRICT_MODEL = """
import jax.numpy as jnp
def model(data, params):
    assert data["x"].shape == (3,)
    assert params["w"].shape == (3,) 
    return jnp.dot(params["w"], data["x"])
"""


def test_evaluate_model_output():
    model_fn = load_function_from_source(BASIC_MODEL, "model")
    # Unbatched params with batched data should raise an error
    params = {"w": 1.0}
    data = {"x": jnp.array([1.0, 2.0])}  # Shape (2,)
    with pytest.raises(ValueError):  # can't vmap mismatched axes
        _evaluate_model_output(model_fn, params, data)

    # Unbatched data with batched params should raise an error
    params = {"w": jnp.array([1.0, 2.0])}  # Shape (2,)
    data = {"x": 1.0}
    with pytest.raises(ValueError):  # can't vmap mismatched axes
        _evaluate_model_output(model_fn, params, data)

    # Both unbatched should raise an error
    params = {"w": 1.0}
    data = {"x": 2.0}
    with pytest.raises(ValueError):  # can't vmap with no batch dimension
        _evaluate_model_output(model_fn, params, data)

    # Both batched
    params = {"w": jnp.array([1.0, 2.0])}  # Shape (2,)
    data = {"x": jnp.array([2.0, 4.0])}  # Shape (2,)
    result = _evaluate_model_output(model_fn, params, data)
    assert result.shape == (2,)  # output shape (2,)
    assert jnp.allclose(result, jnp.array([2.0, 8.0]))

    # Both batched and with strict shape checking in model and array params/data
    model_fn = load_function_from_source(STRICT_MODEL, "model")
    params = {"w": jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}  # Shape (2, 3)
    data = {"x": jnp.array([[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]])}  # Shape (2, 3)
    result = _evaluate_model_output(model_fn, params, data)
    assert result.shape == (2,)
    assert jnp.allclose(result, jnp.array([28.0, 49.0]))


def test_evaluate_sample_losses():
    model_fn = load_function_from_source(BASIC_MODEL, "model")

    def loss_fn(output, data):
        assert output.shape[0] == data["y"].shape[0]
        return jnp.abs(output - data["y"])

    params = {"w": jnp.array([1.0, 2.0])}  # Shape (2,)
    data = {"x": jnp.array([2.0, 4.0]), "y": jnp.array([3.0, 9.0])}  # Shape (2,)
    losses = _evaluate_sample_losses(model_fn, loss_fn, params, data)
    assert losses.shape == (2,)
    expected_losses = jnp.array([1.0, 1.0])  # |(1*2 - 3)| and |(2*4 - 9)|
    assert jnp.allclose(losses, expected_losses)


def test_evaluate_scalar_loss_expected():
    model_fn = load_function_from_source(BASIC_MODEL, "model")

    def loss_fn(output, data):
        assert output.shape[0] == data["y"].shape[0]
        return jnp.abs(output - data["y"])

    params = {"w": jnp.array([1.0, 2.0])}  # Shape (2,)
    data = {"x": jnp.array([2.0, 4.0]), "y": jnp.array([3.0, 9.0])}  # Shape (2,)
    scalar_loss = _evaluate_scalar_loss(model_fn, loss_fn, params, data)
    assert scalar_loss.shape == ()
    expected_scalar_loss = jnp.mean(
        jnp.array([1.0, 1.0])
    )  # Mean of |(1*2 - 3)| and |(2*4 - 9)|
    assert jnp.isclose(scalar_loss, expected_scalar_loss)
