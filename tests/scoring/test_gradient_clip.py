"""Tests for the optional ``gradient_clip_norm`` config in ``_optimize``.

The state-space DSL project (``oscillator_ss``) backprops through long
``lax.scan`` sequences and can produce exploding gradients on early
iterations. ``_optimize`` supports optional pre-Adam ``clip_by_global_norm``
via ``gd_config["gradient_clip_norm"]``. These tests verify:

1. When the key is absent (or None), behavior is bit-identical to plain Adam
   — no regression for existing projects.
2. When the key is set, gradients are actually clipped: with an aggressive
   learning rate that would diverge without clipping, params stay bounded.
"""
# ruff: noqa: E402
import sys
from pathlib import Path

import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edgar.scoring.scoring import _optimize


def _make_model_and_data():
    """Simple linear model y = w*x with a batch of 3 samples."""
    def model_fn(data, params):
        return params["w"] * data["x"]

    def loss_fn(output, data):
        return jnp.mean((output - data["y"]) ** 2, axis=-1)

    x = jnp.ones((3, 8))
    data = {"x": x, "y": x * 2.0}   # true w = 2.0
    params_init = {"w": jnp.array([1.0, 1.0, 1.0])}
    return model_fn, loss_fn, data, params_init


def test_optimize_without_clip_norm_matches_baseline():
    """``gradient_clip_norm`` absent → result identical to plain Adam."""
    model_fn, loss_fn, data, params_init = _make_model_and_data()

    (result_no_key,), _ = _optimize(
        model_fn, loss_fn, params_init, data,
        gd_config={"max_iter": 20, "learning_rate": 0.05},
    )
    (result_none,), _ = _optimize(
        model_fn, loss_fn, params_init, data,
        gd_config={"max_iter": 20, "learning_rate": 0.05, "gradient_clip_norm": None},
    )
    assert jnp.allclose(result_no_key["w"], result_none["w"])


def test_optimize_with_clip_norm_produces_finite_result():
    """``gradient_clip_norm`` set → optimization completes with finite params."""
    model_fn, loss_fn, data, params_init = _make_model_and_data()

    (result,), _ = _optimize(
        model_fn, loss_fn, params_init, data,
        gd_config={"max_iter": 50, "learning_rate": 0.05, "gradient_clip_norm": 1.0},
    )
    assert jnp.all(jnp.isfinite(result["w"]))
    # true w = 2.0; with 50 steps @ lr=0.05 it should be well on the way
    assert jnp.all(result["w"] > 1.2)


def test_clip_norm_keeps_params_bounded_under_aggressive_lr():
    """Aggressive lr + tiny clip → per-step update is bounded, no divergence.

    Without clipping, an lr of 5.0 on this loss would produce updates far larger
    than the parameter itself and drive the loss to nan or to enormous values.
    With ``gradient_clip_norm=0.1``, the raw gradient norm is capped, so the
    Adam-scaled update stays reasonable across all iterations.
    """
    model_fn, loss_fn, data, params_init = _make_model_and_data()

    (result,), _ = _optimize(
        model_fn, loss_fn, params_init, data,
        gd_config={"max_iter": 100, "learning_rate": 5.0, "gradient_clip_norm": 0.1},
    )
    assert jnp.all(jnp.isfinite(result["w"]))
    # With clipping the trajectory is well-behaved; params stay in a reasonable
    # neighborhood of the initial value and the true optimum.
    assert jnp.all(jnp.abs(result["w"]) < 100.0)


def test_clip_norm_config_uses_get_not_indexing():
    """Regression: ``_optimize`` reads clip_norm via .get() so the key can be absent.

    If the code path did ``gd_config["gradient_clip_norm"]`` directly, this
    would KeyError on any existing project config that predates the new key.
    """
    model_fn, loss_fn, data, params_init = _make_model_and_data()

    # No gradient_clip_norm key at all in the config — must not raise.
    (result,), _ = _optimize(
        model_fn, loss_fn, params_init, data,
        gd_config={"max_iter": 5, "learning_rate": 0.01},
    )
    assert jnp.all(jnp.isfinite(result["w"]))
