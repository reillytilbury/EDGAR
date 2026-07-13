import numpy as np
import pytest
from edgar.scoring.utils import _safe_loss
from edgar.evolution.program import NotValidated


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
