import sys
import time
from pathlib import Path

import jax.numpy as jnp
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evolution.program import Program
from src.scoring.scoring import score_program, score_with_timeout


# --- Shared model/estimator code ---

FAST_MODEL_CODE = """
import jax.numpy as jnp

def model(data, params):
    return params['w'] * data['x']

model.DEFAULT_PARAMS = {'w': jnp.array(1.0)}
"""

# repeated large XLA matmuls to hang the subprocess during actual XLA execution
SLOW_MODEL_CODE = """
import jax.numpy as jnp

def model(data, params):
    x = jnp.ones((1000, 1000))
    for _ in range(10_000):
        x = jnp.dot(x, x)
    return params['w'] * data['x'] + x[0, 0] * 0.0

model.DEFAULT_PARAMS = {'w': jnp.array(1.0)}
"""

PARAM_EST_CODE = """
import jax.numpy as jnp

def parameter_estimator(data):
    return {'w': jnp.array(1.0)}
"""

BASE_CONFIG = {
    "timeout_s": 30.0,
    "param_penalty_weight": 0.0,
    "gradient_descent": {"max_iter": 20, "learning_rate": 0.01},
}


def _make_program(model_code, n_params=1):
    p = Program(uid=(0, 0, 0), model_code=model_code, param_est_code=PARAM_EST_CODE)
    p.n_params = n_params
    return p


def _make_data(n_samples=3, n_trials=8):
    x = jnp.ones((n_samples, n_trials))
    return {"x": x, "y": x}  # y = x so w=1.0 is the perfect fit


def loss_fn(output, data):
    return jnp.mean((output - data["y"]) ** 2)


# --- Tests ---

def test_score_program_returns_finite_loss():
    program = _make_program(FAST_MODEL_CODE)
    data = (_make_data(), _make_data())
    loss = score_program(program, data, loss_fn, BASE_CONFIG)
    assert jnp.isfinite(loss)
    assert loss >= 0.0


def test_score_program_perfect_fit():
    """w=1.0 with y=x should give near-zero loss after optimization."""
    program = _make_program(FAST_MODEL_CODE)
    data = (_make_data(), _make_data())
    loss = score_program(program, data, loss_fn, BASE_CONFIG)
    assert loss < 1e-4


def test_score_with_timeout_completes():
    program = _make_program(FAST_MODEL_CODE)
    data = (_make_data(), _make_data())
    loss = score_with_timeout(program, data, loss_fn, BASE_CONFIG)
    assert jnp.isfinite(loss)


def test_score_with_timeout_kills_slow_model():
    program = _make_program(SLOW_MODEL_CODE)
    data = (_make_data(), _make_data())
    config = {**BASE_CONFIG, "timeout_s": 2.0}
    t0 = time.time()
    loss = score_with_timeout(program, data, loss_fn, config)
    elapsed = time.time() - t0
    assert loss == float("inf")
    assert elapsed < 10.0  # subprocess was killed, didn't wait the full 100s
