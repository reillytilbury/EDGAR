"""Offline checks that the fast fitters agree with upstream's: uv run python -m edgar_study.test_fitting"""

from pathlib import Path

import jax
import numpy as np

from edgar_study import fitting
from edgar_study.datasets import prepare, synthetic_recording
from edgar_study.upstream import PENALTY, import_upstream


def main():
    upstream = import_upstream(Path(".").resolve())
    engine, seeds = upstream["hypothesis_engine"], upstream["seed_programs"]
    loss = upstream["loss_functions"].quadratic_loss
    x, y, _, _ = prepare(*synthetic_recording(), "paper-fidelity")
    x, y = x[:15], y[:15]  # 15 cells do not divide evenly across devices, so padding is exercised
    model, estimator = seeds.neuron_model_2_jax, seeds.parameter_estimator_2

    def run(fitter):
        objective = engine.objective if fitter == "upstream" else fitting.make_objective(engine.objective, fitter)
        np.random.seed(0)
        _, initial_params, score, params = objective(model, estimator, loss, x, y,
                                                     param_penalty_weight=PENALTY, max_iter=300)
        return score, params, fitting.train_loss(model, loss, params, x, y), initial_params

    reference, reference_params, _, initial = run("upstream")
    adam, adam_params, adam_train, _ = run("adam")
    _, _, lm_train, _ = run("lm")
    assert abs(adam - reference) < 1e-3, (adam, reference)
    assert np.allclose(adam_params, reference_params, atol=1e-3)
    assert lm_train <= fitting.train_loss(model, loss, initial, x, y) + 1e-6  # LM never ends worse than its start
    print(f"{len(jax.devices())} devices; upstream {reference:.5f}, compiled Adam {adam:.5f}; "
          f"train loss Adam {adam_train:.5f}, LM {lm_train:.5f}. Fitting checks passed.")


if __name__ == "__main__":
    main()
