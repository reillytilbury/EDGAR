import sys
import time
from pathlib import Path

import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evolution.program import Program, BirthCertificate, Code
from src.evolution.population import Population
from src.scoring.scoring import _score_one_model, score


# --- shared fixtures ---

FAST_MODEL_CODE = """
import jax.numpy as jnp

def model(data, params):
    return params['w'] * data['x']

model.DEFAULT_PARAMS = {'w': jnp.array(1.0)}
"""

# repeated large XLA matmuls to hang the subprocess during XLA execution
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


def _make_program(model_code):
    return Program(
        birth=BirthCertificate(generation=0, island=0, batch_index=0),
        code_jax=Code(model=model_code, param_est=PARAM_EST_CODE),
    )


def _make_data(n_samples=3, n_trials=8):
    x = jnp.ones((n_samples, n_trials))
    return {"x": x, "y": x}  # y = x so w=1.0 is the perfect fit


def loss_fn(output, data):
    return jnp.mean((output - data["y"]) ** 2, axis=-1)


# --- _score_one_model ---

def test_score_one_model_returns_finite_loss():
    program = _make_program(FAST_MODEL_CODE)
    data = (_make_data(), _make_data())
    final_loss, initial_loss, *_ = _score_one_model(program, data, loss_fn, BASE_CONFIG)
    assert jnp.isfinite(final_loss)
    assert jnp.isfinite(initial_loss)
    assert final_loss >= 0.0
    assert initial_loss >= 0.0


def test_score_one_model_perfect_fit():
    """w=1.0 with y=x should give near-zero loss after optimization."""
    program = _make_program(FAST_MODEL_CODE)
    data = (_make_data(), _make_data())
    final_loss, _, _ = _score_one_model(program, data, loss_fn, BASE_CONFIG)
    assert final_loss < 1e-4


def test_score_one_model_auto_populates_n_params():
    program = _make_program(FAST_MODEL_CODE)
    assert program.n_params is None
    _score_one_model(program, (_make_data(), _make_data()), loss_fn, BASE_CONFIG)
    assert program.n_params == 1


def test_score_one_model_kills_slow_model():
    program = _make_program(SLOW_MODEL_CODE)
    data = (_make_data(), _make_data())
    config = {**BASE_CONFIG, "timeout_s": 2.0}
    t0 = time.time()
    final_loss, initial_loss, *_ = _score_one_model(program, data, loss_fn, config)
    elapsed = time.time() - t0
    assert final_loss == float("inf")
    assert initial_loss == float("inf")
    assert elapsed < 10.0  # subprocess was killed; didn't wait for the loop


# --- score (population) ---

def test_score_writes_back_to_population():
    pop = Population()
    pop.add(_make_program(FAST_MODEL_CODE))
    pop.add(_make_program(FAST_MODEL_CODE))
    data = (_make_data(), _make_data())

    score(pop, data, None, BASE_CONFIG, loss_fn, split="discover")

    for i in range(len(pop)):
        assert jnp.isfinite(pop[i].program_losses.discover.final)
        assert jnp.isfinite(pop[i].program_losses.discover.init)
        assert pop[i].program_losses.discover.final < 1e-4
        assert pop[i].program_losses.validate.final is None
        assert pop[i].n_params == 1


def test_score_skips_already_scored_programs():
    pop = Population()
    pop.add(_make_program(FAST_MODEL_CODE))
    pop[0].program_losses.discover.final = 0.5
    pop[0].program_losses.discover.init = 0.5

    score(pop, (_make_data(), _make_data()), None, BASE_CONFIG, loss_fn, split="discover")

    assert pop[0].program_losses.discover.final == 0.5
    assert pop[0].program_losses.discover.init == 0.5


def test_score_skips_programs_without_code():
    pop = Population()
    pop.add(Program(birth=BirthCertificate(generation=0, island=0, batch_index=0)))

    score(pop, (_make_data(), _make_data()), None, BASE_CONFIG, loss_fn, split="discover")

    assert pop[0].program_losses.discover.final is None
