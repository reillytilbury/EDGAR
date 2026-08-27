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
from edgar.scoring.scoring import (
    _eval_loss,
    _optimize,
    _score_one_model,
    rank,
    score,
)


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
    "banned_strings": [],
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
    final_loss, initial_loss, _, _, _, _, _, outcome = _score_one_model(
        program, data, loss_fn, BASE_CONFIG
    )
    assert jnp.isfinite(final_loss)
    assert jnp.isfinite(initial_loss)
    assert final_loss >= 0.0
    assert initial_loss >= 0.0
    assert outcome == "ok"


def test_score_one_model_perfect_fit():
    """w=1 with y=x should give near-zero loss after optimization."""
    program = _make_program(FAST_MODEL_CODE)
    data = (_make_data(), _make_data())
    final_loss, _, _, _, _, _, _, outcome = _score_one_model(
        program, data, loss_fn, BASE_CONFIG
    )
    assert final_loss < 1e-4
    assert outcome == "ok"


def test_score_one_model_perfect_fit_with_param_penalty():
    """w=1.0 with y=x should give near-param_penalty loss after optimization."""
    program = _make_program(FAST_MODEL_CODE)
    data = (_make_data(), _make_data())
    final_loss, _, _, _, _, _, _, outcome = _score_one_model(
        program, data, loss_fn, BASE_CONFIG_WITH_PARAM_PENALTY
    )
    assert (
        final_loss < BASE_CONFIG_WITH_PARAM_PENALTY["param_penalty_weight"] + 1e-4
    )  # since n_params=1
    assert outcome == "ok"


def test_score_one_model_with_array_params():
    """w=1 with y=x should give near-zero loss after optimization."""
    data = (_make_data(), _make_data())
    program = _make_program(
        FAST_MODEL_CODE,
        param_est=ARRAY_PARAM_EST_CODE,
        default_params={"w": jnp.zeros(data[0]["x"].shape[-1])},
    )
    final_loss, _, _, _, _, _, _, outcome = _score_one_model(
        program, data, loss_fn, BASE_CONFIG
    )
    assert final_loss < 1e-4
    assert outcome == "ok"


def test_score_one_gives_infinite_loss_for_program_with_none_default_params():
    program = _make_program(FAST_MODEL_CODE, default_params=None)
    assert program.n_params is None
    final_loss, initial_loss, _, _, _, _, _, outcome = _score_one_model(
        program, (_make_data(), _make_data()), loss_fn, BASE_CONFIG
    )
    assert final_loss == float("inf")
    assert initial_loss == float("inf")
    assert outcome == "inf"


def test_score_one_model_kills_slow_model():
    program = _make_program(SLOW_MODEL_CODE)
    data = (_make_data(), _make_data())
    config = {**BASE_CONFIG, "timeout_s": 2.0}
    t0 = time.time()
    final_loss, initial_loss, _, _, _, _, _, outcome = _score_one_model(
        program, data, loss_fn, config
    )
    elapsed = time.time() - t0
    assert final_loss == float("inf")
    assert initial_loss == float("inf")
    assert elapsed < 10.0  # subprocess was killed; didn't wait for the loop
    assert outcome == "timeout"


def test_score_one_model_falls_back_to_default_params():
    """param_est_fn raises → _get_params falls back to default_params, loss still finite."""
    program = Program(
        birth=BirthCertificate(generation=0, island=0, batch_index=0),
        code=Code(param_est=FAILING_PARAM_EST_CODE, model_jax=FAST_MODEL_CODE),
        _default_params={"w": jnp.array(1.0)},
    )
    data = (_make_data(), _make_data())
    final_loss, initial_loss, _, _, _, _, _, outcome = _score_one_model(
        program, data, loss_fn, BASE_CONFIG
    )

    assert jnp.isfinite(final_loss)
    assert jnp.isfinite(initial_loss)
    assert outcome == "ok"


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
    final_loss, initial_loss, _, _, _, _, _, outcome = _score_one_model(
        program, data, loss_fn, BASE_CONFIG
    )

    assert final_loss == float("inf")
    assert initial_loss == float("inf")
    assert outcome == "inf"


def test_score_one_model_falls_back_when_param_est_syntax_error():
    """param_est with syntax error → falls back to default_params, loss still finite."""
    program = Program(
        birth=BirthCertificate(generation=0, island=0, batch_index=0),
        code=Code(param_est=BROKEN_PARAM_EST_CODE, model_jax=FAST_MODEL_CODE),
        _default_params={"w": jnp.array(1.0)},
    )
    data = (_make_data(), _make_data())
    final_loss, initial_loss, _, _, _, _, _, outcome = _score_one_model(
        program, data, loss_fn, BASE_CONFIG
    )

    assert jnp.isfinite(final_loss)
    assert jnp.isfinite(initial_loss)
    assert outcome == "ok"


def test_score_one_model_with_notvalidated_loss():
    """Program with NotValidated validate loss (default state) scores without BrokenPipeError."""
    program = _make_program(FAST_MODEL_CODE)
    assert type(program.program_losses.validate.final).__name__ == "NotValidated"
    data = (_make_data(), _make_data())
    final_loss, initial_loss, _, _, _, _, _, outcome = _score_one_model(
        program, data, loss_fn, BASE_CONFIG
    )

    assert jnp.isfinite(final_loss)
    assert jnp.isfinite(initial_loss)
    assert outcome == "ok"


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


def test_rank_with_nan():
    pop = Population()
    # Add 6 programs
    for i in range(6):
        p = _make_program(FAST_MODEL_CODE)
        p.idx = i
        pop.add(p)

    # validate losses: contains a NaN and other numbers that would get scrambled if NaN isn't handled
    validate_losses = (NotValidated(), 0.1, 2.1, float("nan"), 0.5, None)
    for i, loss in enumerate(validate_losses):
        pop[i].program_losses.validate.final = loss

    rank(pop)
    # Ranks:
    # Index 0: NotValidated -> rank None (not in validated indices)
    # Index 1 (0.1): rank 1
    # Index 4 (0.5): rank 2
    # Index 2 (2.1): rank 3
    # Index 3 (nan): rank 4 (treated as inf)
    # Index 5 (None): rank 5 (treated as inf)
    expected_rank = (None, 1, 3, 4, 2, 5)
    for i in range(6):
        assert pop[i].rank == expected_rank[i]


def test_score_one_model_with_separate_train_test_loss_fns():
    """
    Tests that scoring uses loss_fn_train for optimization
    and loss_fn_test for evaluating initial and final losses.
    """
    # 1. Setup a standard program
    program = _make_program(FAST_MODEL_CODE)
    data = (_make_data(), _make_data())

    # 2. Define train and test loss functions with different optimization targets
    def loss_fn_train(output, data):
        # Minimized when output is 2.0 * data["x"]
        return jnp.mean((output - 2.0 * data["x"]) ** 2, axis=-1)

    def loss_fn_test(output, data):
        # Minimized when output is 5.0 * data["x"]
        return jnp.mean((output - 5.0 * data["x"]) ** 2, axis=-1)

    # 3. Call scoring with the separate loss functions
    loss_fn_tuple = (loss_fn_train, loss_fn_test)
    custom_config = {
        "timeout_s": 10.0,
        "param_penalty_weight": 0.0,
        "gradient_descent": {"max_iter": 150, "learning_rate": 0.1},
    }
    final_loss, initial_loss, _, params, _, _, _, outcome = _score_one_model(
        program, data, loss_fn_tuple, custom_config
    )

    assert outcome == "ok"

    # 4. Verify parameter was optimized using loss_fn_train (drives w close to 2.0)
    w_opt = params["w"]
    assert jnp.allclose(w_opt, 2.0, atol=1e-2)

    # 5. Verify initial loss was evaluated using loss_fn_test
    # params_init is {'w': [0.9]}, so output is 0.9 * x.
    # loss_fn_test evaluates (0.9 * x - 5.0 * x) ** 2, which has a mean of (0.9 - 5.0) ** 2 = 16.81
    assert jnp.allclose(initial_loss, 16.81, atol=1e-2)

    # 6. Verify final loss was evaluated using loss_fn_test
    # params is close to {'w': 2.0}, so output is 2.0 * x.
    # loss_fn_test evaluates (2.0 * x - 5.0 * x) ** 2, which has a mean of (2.0 - 5.0) ** 2 = 9.0
    expected_final_loss = (w_opt - 5.0) ** 2
    assert jnp.allclose(final_loss, expected_final_loss, atol=1e-2)


def test_score_one_model_banned_string():
    program = _make_program(FAST_MODEL_CODE)
    data = (_make_data(), _make_data())
    # Configure w to be a banned string in scoring configuration dict
    custom_config = {
        "banned_strings": ["params['w']"],
        "timeout_s": 10.0,
        "param_penalty_weight": 0.0,
    }
    final_loss, initial_loss, _, _, _, _, _, outcome = _score_one_model(
        program, data, loss_fn, custom_config
    )
    assert final_loss == float("inf")
    assert initial_loss == float("inf")
    assert outcome == "banned"
    assert program.status == "banned"


def test_score_and_prune_banned_program():
    from edgar.evolution.island import prune
    from tests.llm.programs import BannedProgram

    # Setup population with Program1 and BannedProgram
    pop = Population()

    # Program 1 (Normal)
    p1 = Program(
        birth=BirthCertificate(generation=0, island=0, batch_index=0),
        _default_params=Program1.default_params,
        code=Code(
            model=Program1.model,
            model_jax=Program1.model_jax,
            param_est=Program1.param_est,
        ),
    )
    pop.add(p1)

    # Program 2 (Banned)
    p2 = Program(
        birth=BirthCertificate(generation=0, island=0, batch_index=1),
        _default_params=BannedProgram.default_params,
        code=Code(
            model=BannedProgram.model,
            model_jax=BannedProgram.model_jax,
            param_est=BannedProgram.param_est,
        ),
    )
    pop.add(p2)

    # Set up island containing both programs
    islands = [{0, 1}]

    # Score the population with config banning "THIS_IS_BANNED"
    data = (_make_data(), _make_data())
    custom_config = {**BASE_CONFIG, "banned_strings": ["THIS_IS_BANNED"]}

    score(pop, data, None, custom_config, loss_fn, split="discover")

    # Verify that the normal program got a finite loss and the banned got infinite loss
    assert pop[0].program_losses.discover.final < 1e-4
    assert pop[1].program_losses.discover.final == float("inf")
    assert pop[1].status == "banned"

    # Prune island down to size 1
    evolution_cfg = {
        "critical_population_size": 2,
        "n_migrants": 1,
    }

    prune(islands, pop, evolution_cfg)

    # The normal program (idx 0) should remain on the island, and the banned one (idx 1) pruned from island
    assert islands[0] == {0}
    # Banned program status must NOT be overwritten to "pruned"
    assert pop[1].status == "banned"
