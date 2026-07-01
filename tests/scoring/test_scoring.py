# ruff: noqa: E402
import sys
import time
from pathlib import Path

import jax.numpy as jnp

from tests.llm.programs import Program1

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edgar.evolution.program import Program, BirthCertificate, Code, NotValidated
from edgar.evolution.population import Population
from edgar.scoring.scoring import _eval_loss, _optimize, _score_one_model, rank, score


# --- shared fixtures ---

FAST_MODEL_CODE = """
import jax.numpy as jnp
def model(data, params):
    return params['w'] * data['x']
"""

SLOW_MODEL_CODE = """
import time
import jax.numpy as jnp

def model(data, params):
    time.sleep(15)
    return params['w'] * data['x']
"""

PARAM_EST_CODE = """
import jax.numpy as jnp

def parameter_estimator(data):
    return {'w': jnp.array([0.9])}
"""

ARRAY_PARAM_EST_CODE = """
import jax.numpy as jnp
def parameter_estimator(data):
    return {'w': jnp.full(data['x'].shape[-1], 0.9)}
"""

FAILING_PARAM_EST_CODE = """
import jax.numpy as jnp

def parameter_estimator(data):
    raise RuntimeError("param_est intentionally broken")
"""

BASE_CONFIG = {
    "timeout_s": 10.0,
    "param_penalty_weight": 0.0,
    "gradient_descent": {"max_iter": 20, "learning_rate": 0.01},
}

BASE_CONFIG_WITH_PARAM_PENALTY = {
    **BASE_CONFIG,
    "param_penalty_weight": 0.01,
}


def _make_program(
    model_code, param_est=PARAM_EST_CODE, default_params={"w": jnp.array(0.5)}
):
    return Program(
        birth=BirthCertificate(generation=0, island=0, batch_index=0),
        code=Code(param_est=param_est, model_jax=model_code),
        _default_params=default_params,
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
    final_loss, initial_loss, _, _, _, _, _ = _score_one_model(
        program, data, loss_fn, BASE_CONFIG
    )
    assert jnp.isfinite(final_loss)
    assert jnp.isfinite(initial_loss)
    assert final_loss >= 0.0
    assert initial_loss >= 0.0


def test_score_one_model_perfect_fit():
    """w=1 with y=x should give near-zero loss after optimization."""
    program = _make_program(FAST_MODEL_CODE)
    data = (_make_data(), _make_data())
    final_loss, _, _, _, _, _, _ = _score_one_model(program, data, loss_fn, BASE_CONFIG)
    assert final_loss < 1e-4


def test_score_one_model_perfect_fit_with_param_penalty():
    """w=1.0 with y=x should give near-param_penalty loss after optimization."""
    program = _make_program(FAST_MODEL_CODE)
    data = (_make_data(), _make_data())
    final_loss, _, _, _, _, _, _ = _score_one_model(
        program, data, loss_fn, BASE_CONFIG_WITH_PARAM_PENALTY
    )
    assert (
        final_loss < BASE_CONFIG_WITH_PARAM_PENALTY["param_penalty_weight"] + 1e-4
    )  # since n_params=1


def test_score_one_model_with_array_params():
    """w=1 with y=x should give near-zero loss after optimization."""
    data = (_make_data(), _make_data())
    program = _make_program(
        FAST_MODEL_CODE,
        param_est=ARRAY_PARAM_EST_CODE,
        default_params={"w": jnp.zeros(data[0]["x"].shape[-1])},
    )
    final_loss, _, _, _, _, _, _ = _score_one_model(program, data, loss_fn, BASE_CONFIG)
    assert final_loss < 1e-4


def test_score_one_gives_infinite_loss_for_program_with_none_default_params():
    program = _make_program(FAST_MODEL_CODE, default_params=None)
    assert program.n_params is None
    final_loss, initial_loss, _, _, _, _, _ = _score_one_model(
        program, (_make_data(), _make_data()), loss_fn, BASE_CONFIG
    )
    assert final_loss == float("inf")
    assert initial_loss == float("inf")


def test_score_one_model_kills_slow_model():
    program = _make_program(SLOW_MODEL_CODE)
    data = (_make_data(), _make_data())
    config = {**BASE_CONFIG, "timeout_s": 2.0}
    t0 = time.time()
    final_loss, initial_loss, _, _, _, _, _ = _score_one_model(
        program, data, loss_fn, config
    )
    elapsed = time.time() - t0
    assert final_loss == float("inf")
    assert initial_loss == float("inf")
    assert elapsed < 10.0  # subprocess was killed; didn't wait for the loop


def test_score_one_model_falls_back_to_default_params():
    """param_est_fn raises → _get_params falls back to default_params, loss still finite."""
    program = Program(
        birth=BirthCertificate(generation=0, island=0, batch_index=0),
        code=Code(param_est=FAILING_PARAM_EST_CODE, model_jax=FAST_MODEL_CODE),
        _default_params={"w": jnp.array(1.0)},
    )
    data = (_make_data(), _make_data())
    final_loss, initial_loss, _, _, _, _, _ = _score_one_model(
        program, data, loss_fn, BASE_CONFIG
    )

    assert jnp.isfinite(final_loss)
    assert jnp.isfinite(initial_loss)


BROKEN_PARAM_EST_CODE = """
def parameter_estimator(data):
    return {'w': 1.0
"""

BROKEN_MODEL_CODE = """
def model(data, params):
    return params['w'] * data['x'
"""


def test_score_one_model_gives_infinite_loss_for_broken_model_syntax():
    """model_jax with syntax error → ModelLoadingError → infinite loss."""
    program = Program(
        birth=BirthCertificate(generation=0, island=0, batch_index=0),
        code=Code(param_est=PARAM_EST_CODE, model_jax=BROKEN_MODEL_CODE),
        _default_params={"w": jnp.array(1.0)},
    )
    data = (_make_data(), _make_data())
    final_loss, initial_loss, _, _, _, _, _ = _score_one_model(
        program, data, loss_fn, BASE_CONFIG
    )

    assert final_loss == float("inf")
    assert initial_loss == float("inf")


def test_score_one_model_falls_back_when_param_est_syntax_error():
    """param_est with syntax error → falls back to default_params, loss still finite."""
    program = Program(
        birth=BirthCertificate(generation=0, island=0, batch_index=0),
        code=Code(param_est=BROKEN_PARAM_EST_CODE, model_jax=FAST_MODEL_CODE),
        _default_params={"w": jnp.array(1.0)},
    )
    data = (_make_data(), _make_data())
    final_loss, initial_loss, _, _, _, _, _ = _score_one_model(
        program, data, loss_fn, BASE_CONFIG
    )

    assert jnp.isfinite(final_loss)
    assert jnp.isfinite(initial_loss)


def test_score_one_model_with_notvalidated_loss():
    """Program with NotValidated validate loss (default state) scores without BrokenPipeError."""
    program = _make_program(FAST_MODEL_CODE)
    assert type(program.program_losses.validate.final).__name__ == "NotValidated"
    data = (_make_data(), _make_data())
    final_loss, initial_loss, _, _, _, _, _ = _score_one_model(
        program, data, loss_fn, BASE_CONFIG
    )

    assert jnp.isfinite(final_loss)
    assert jnp.isfinite(initial_loss)


# --- optimize and _eval_loss ---


def test_optimize_before_convergence():
    """Do a few optimization steps but not enough to converge, checking against expected value which protects against optimization loop errors, e.g params used from wrong iteration."""
    ns = {}
    exec(Program1.model_jax, ns)
    model_fn = ns["model"]

    data_train = _make_data()
    n_samples = data_train["x"].shape[0]
    params_init = {
        k: jnp.stack([jnp.asarray(v)] * n_samples)
        for k, v in Program1.default_params.items()
    }
    gd_config = {"max_iter": 5, "learning_rate": 0.001}
    best_params = _optimize(
        model_fn=model_fn,
        loss_fn=loss_fn,
        params_init=params_init,
        data_train=data_train,
        gd_config=gd_config,
    )
    loss = _eval_loss(model_fn, loss_fn, best_params, data_train)
    print(f"Final loss: {loss:.6f}")
    assert round(loss, 6) == 0.008466


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
        assert isinstance(
            pop[i].program_losses.validate.final, NotValidated
        )  # this is set to NotValidated, so final validation scoring is opt in, see scoring._needs_scoring, population.prepare_validation_scoring
        assert pop[i].n_params == 1


def test_score_skips_already_scored_programs():
    pop = Population()
    pop.add(_make_program(FAST_MODEL_CODE))
    pop[0].program_losses.discover.final = 0.5
    pop[0].program_losses.discover.init = 0.5

    score(
        pop, (_make_data(), _make_data()), None, BASE_CONFIG, loss_fn, split="discover"
    )

    assert pop[0].program_losses.discover.final == 0.5
    assert pop[0].program_losses.discover.init == 0.5


def test_score_skips_programs_without_code():
    pop = Population()
    pop.add(Program(birth=BirthCertificate(generation=0, island=0, batch_index=0)))

    score(
        pop, (_make_data(), _make_data()), None, BASE_CONFIG, loss_fn, split="discover"
    )

    assert pop[0].program_losses.discover.final is None


# --- rank ---
def test_rank():
    pop = Population()
    for i in range(5):
        p = _make_program(FAST_MODEL_CODE)
        p.idx = i
        pop.add(p)

    validate_losses = (NotValidated(), 0.1, 2.1, 0.5, None)
    for i, loss in enumerate(validate_losses):
        pop[i].program_losses.validate.final = loss

    rank(pop)
    expected_rank = (None, 1, 3, 2, 4)
    for i in range(5):
        assert pop[i].rank == expected_rank[i]
