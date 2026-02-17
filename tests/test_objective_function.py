"""
Tests for objective function implementations.

Tests both scalar (n_targets=1) and vectorized (n_targets>1) cases.
"""
import pytest
import numpy as np
import jax.numpy as jnp
import jax
import sys
from pathlib import Path

# Add src to path for imports
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Also add parent for package imports
parent_path = str(Path(__file__).parent.parent)
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

from src.hypothesis_engine import (
    objective,
    objective_legacy,
    objective_vectorized,
    validate_model_output,
    validate_model_execution,
    _call_trial_split,
)
from src.data_structures import Inputs, Outputs, ensure_inputs, ensure_outputs


# ============================================================================
# Test fixtures and helper models
# ============================================================================

def simple_scalar_model(X, a=1.0, b=0.0):
    """
    Simple linear model for scalar output.
    X: (n_features, n_trials)
    Returns: (n_trials,) - mean over features scaled by a, plus b
    """
    return a * jnp.mean(X, axis=0) + b


def simple_vectorized_model(X, a=1.0, b=0.0):
    """
    Simple linear model for vectorized output (2 targets).
    X: (n_features, n_trials)
    Returns: (2, n_trials) - two different linear combinations
    """
    mean_x = jnp.mean(X, axis=0)  # (n_trials,)
    target_0 = a * mean_x + b
    target_1 = (a / 2) * mean_x - b
    return jnp.stack([target_0, target_1], axis=0)  # (2, n_trials)


def flexible_model(X, a=1.0, b=0.0, n_targets=1):
    """
    Model that can output either 1D or 2D based on n_targets parameter.
    For testing that objective_vectorized handles n_targets=1.
    """
    mean_x = jnp.mean(X, axis=0)  # (n_trials,)
    if n_targets == 1:
        return a * mean_x + b  # (n_trials,)
    else:
        outputs = []
        for i in range(n_targets):
            outputs.append((a / (i + 1)) * mean_x + b * i)
        return jnp.stack(outputs, axis=0)  # (n_targets, n_trials)


def simple_param_estimator(X, y):
    """
    Simple parameter estimator that returns reasonable defaults.
    
    Handles both scalar y (n_trials,) and vectorized y (n_targets, n_trials).
    """
    # For vectorized y, could use mean across targets or any sensible strategy
    # Here we just return defaults regardless of y shape
    return np.array([1.0, 0.0])

def legacy_split(n_trials_x, random_seed=0):
    idx = np.arange(n_trials_x)
    return idx[:n_trials_x // 2], idx[n_trials_x // 2:]

def default_sample_loss_fn(model, x_i, y_i, params):
    """Default per-sample loss: MSE averaged over all outputs and trials.

    Works for both scalar outputs (n_trials,) and vectorized outputs (n_targets, n_trials).
    Equivalent to uniform-weight MSE across all targets and trials.
    """
    pred = model(x_i, *params)
    if pred.ndim == 1:
        pred = pred[None, :]  # normalize to (1, n_trials)
    return jnp.mean((pred - y_i) ** 2)

# ============================================================================
# Test validate_model_output
# ============================================================================

class TestValidateModelOutput:
    """Tests for validate_model_output function."""
    
    def test_scalar_1d_valid(self):
        """Valid 1D output for n_targets=1."""
        output = jnp.zeros((100,))
        is_valid, msg = validate_model_output(output, expected_n_trials=100, expected_n_targets=1)
        assert is_valid, msg
        
    def test_scalar_1d_wrong_trials(self):
        """1D output with wrong number of trials."""
        output = jnp.zeros((50,))
        is_valid, msg = validate_model_output(output, expected_n_trials=100, expected_n_targets=1)
        assert not is_valid
        assert "n_trials" in msg
        
    def test_scalar_2d_valid_when_allowed(self):
        """2D output (1, n_trials) valid for n_targets=1 when allowed."""
        output = jnp.zeros((1, 100))
        is_valid, msg = validate_model_output(
            output, expected_n_trials=100, expected_n_targets=1, allow_1d_for_single_target=True
        )
        assert is_valid, msg
        
    def test_vectorized_2d_valid(self):
        """Valid 2D output for n_targets>1."""
        output = jnp.zeros((5, 100))
        is_valid, msg = validate_model_output(output, expected_n_trials=100, expected_n_targets=5)
        assert is_valid, msg
        
    def test_vectorized_wrong_targets(self):
        """2D output with wrong number of targets."""
        output = jnp.zeros((3, 100))
        is_valid, msg = validate_model_output(output, expected_n_trials=100, expected_n_targets=5)
        assert not is_valid
        assert "n_targets" in msg
        
    def test_vectorized_1d_invalid(self):
        """1D output invalid for n_targets>1."""
        output = jnp.zeros((100,))
        is_valid, msg = validate_model_output(output, expected_n_trials=100, expected_n_targets=5)
        assert not is_valid
        assert "2D" in msg


# ============================================================================
# Test validate_model_execution
# ============================================================================

class TestValidateModelExecution:
    """Tests for validate_model_execution function."""
    
    def test_scalar_model_valid(self):
        """Validate scalar model execution."""
        x_data = jnp.ones((10, 3, 100))  # 10 samples, 3 features, 100 trials
        initial_params = jnp.ones((10, 2))  # 10 samples, 2 params (a, b)
        
        is_valid, msg = validate_model_execution(
            simple_scalar_model, x_data, initial_params, n_samples=10, expected_n_targets=1
        )
        assert is_valid, msg
        
    def test_vectorized_model_valid(self):
        """Validate vectorized model execution."""
        x_data = jnp.ones((10, 3, 100))
        initial_params = jnp.ones((10, 2))
        
        is_valid, msg = validate_model_execution(
            simple_vectorized_model, x_data, initial_params, n_samples=10, expected_n_targets=2
        )
        assert is_valid, msg
        
    def test_wrong_n_targets_fails(self):
        """Mismatch between model output and expected n_targets."""
        x_data = jnp.ones((10, 3, 100))
        initial_params = jnp.ones((10, 2))
        
        # Model outputs 2 targets but we expect 3
        is_valid, msg = validate_model_execution(
            simple_vectorized_model, x_data, initial_params, n_samples=10, expected_n_targets=3
        )
        assert not is_valid


# ============================================================================
# Test objective_legacy (scalar case)
# ============================================================================
class TestObjectiveLegacy:
    """Tests for objective_legacy function (scalar outputs)."""

    def test_basic_execution(self):
        """Basic test that objective_legacy runs without error."""
        np.random.seed(42)
        n_samples, n_features, n_trials = 10, 3, 100
        
        x = jnp.array(np.random.randn(n_samples, n_features, n_trials))
        y = jnp.array(np.random.randn(n_samples, n_trials))
        
        initial_loss, initial_params, final_loss, params = objective_legacy(
            model=simple_scalar_model,
            param_estimator=simple_param_estimator,
            x=x,
            y=y,
            fit_params=True,
            max_iter=100,  # Need >= 51 for warmup schedule
            param_penalty_weight=0.01,
            create_train_test_trial_split_fn=legacy_split,
            sample_loss_fn=default_sample_loss_fn,
        )
        
        assert np.isfinite(initial_loss)
        assert np.isfinite(final_loss)
        assert params.shape == (n_samples, 2)
        
    def test_2d_input_auto_expand(self):
        """Test that 2D inputs are auto-expanded to 3D."""
        np.random.seed(42)
        n_samples, n_trials = 10, 100
        
        # 2D input (should be expanded to 3D with n_features=1)
        x = jnp.array(np.random.randn(n_samples, n_trials))
        y = jnp.array(np.random.randn(n_samples, n_trials))
        
        initial_loss, initial_params, final_loss, params = objective_legacy(
            model=simple_scalar_model,
            param_estimator=simple_param_estimator,
            x=x,
            y=y,
            fit_params=False,  # Skip optimization for speed
            create_train_test_trial_split_fn=legacy_split,
            sample_loss_fn=default_sample_loss_fn,
        )
        
        assert np.isfinite(initial_loss)


# ============================================================================
# Test objective_vectorized
# ============================================================================

class TestObjectiveVectorized:
    """Tests for objective_vectorized function (n_targets >= 1)."""

    def test_single_target(self):
        """Test objective_vectorized with n_targets=1."""
        np.random.seed(42)
        n_samples, n_features, n_trials = 10, 3, 100
        
        x = jnp.array(np.random.randn(n_samples, n_features, n_trials))
        # 3D output with n_targets=1
        y = jnp.array(np.random.randn(n_samples, 1, n_trials))
        
        initial_loss, initial_params, final_loss, params = objective_vectorized(
            model=simple_scalar_model,
            param_estimator=simple_param_estimator,
            x=x,
            y=y,
            fit_params=True,
            max_iter=100,  # Need >= 51 for warmup schedule
            param_penalty_weight=0.01,
            create_train_test_trial_split_fn=legacy_split,
            sample_loss_fn=default_sample_loss_fn,
        )
        
        assert np.isfinite(initial_loss)
        assert np.isfinite(final_loss)
        assert params.shape == (n_samples, 2)
        
    def test_multiple_targets(self):
        """Test objective_vectorized with n_targets=2."""
        np.random.seed(42)
        n_samples, n_features, n_trials = 10, 3, 100
        n_targets = 2
        
        x = jnp.array(np.random.randn(n_samples, n_features, n_trials))
        y = jnp.array(np.random.randn(n_samples, n_targets, n_trials))
        
        initial_loss, initial_params, final_loss, params = objective_vectorized(
            model=simple_vectorized_model,
            param_estimator=simple_param_estimator,
            x=x,
            y=y,
            fit_params=True,
            max_iter=100,  # Need >= 51 for warmup schedule
            param_penalty_weight=0.01,
            create_train_test_trial_split_fn=legacy_split,
            sample_loss_fn=default_sample_loss_fn,
        )
        
        assert np.isfinite(initial_loss)
        assert np.isfinite(final_loss)
        assert params.shape == (n_samples, 2)
        
    def test_multiple_targets_default_loss(self):
        """Test that default MSE loss works correctly with multiple targets."""
        np.random.seed(42)
        n_samples, n_features, n_trials = 5, 2, 50
        n_targets = 2

        x = jnp.array(np.random.randn(n_samples, n_features, n_trials))
        y = jnp.array(np.random.randn(n_samples, n_targets, n_trials))

        _, _, loss, _ = objective_vectorized(
            model=simple_vectorized_model,
            param_estimator=simple_param_estimator,
            x=x,
            y=y,
            fit_params=False,
            create_train_test_trial_split_fn=legacy_split,
            sample_loss_fn=default_sample_loss_fn,
        )

        assert np.isfinite(loss)
        
    def test_custom_sample_loss_fn_changes_loss(self):
        """Test that a custom sample_loss_fn changes the loss value."""
        np.random.seed(42)
        n_samples, n_features, n_trials = 5, 2, 50
        n_targets = 2

        x = jnp.array(np.random.randn(n_samples, n_features, n_trials))
        y = jnp.array(np.random.randn(n_samples, n_targets, n_trials))

        # Default loss: uniform MSE over both targets
        _, _, loss_uniform, _ = objective_vectorized(
            model=simple_vectorized_model,
            param_estimator=simple_param_estimator,
            x=x,
            y=y,
            fit_params=False,
            create_train_test_trial_split_fn=legacy_split,
            sample_loss_fn=default_sample_loss_fn,
        )

        # Custom loss: MSE only on first target (ignores second target)
        def first_target_only(model, x_i, y_i, params):
            pred = model(x_i, *params)  # (n_targets, n_trials)
            return jnp.mean((pred[0] - y_i[0]) ** 2)

        _, _, loss_first_only, _ = objective_vectorized(
            model=simple_vectorized_model,
            param_estimator=simple_param_estimator,
            x=x,
            y=y,
            fit_params=False,
            sample_loss_fn=first_target_only,
            create_train_test_trial_split_fn=legacy_split,
        )

        # Losses should differ
        assert not np.isclose(loss_uniform, loss_first_only, rtol=1e-3)
        
    def test_outputs_object_input(self):
        """Test that Outputs objects are accepted."""
        np.random.seed(42)
        n_samples, n_features, n_trials = 5, 2, 50
        n_targets = 2
        
        x = Inputs.from_array(np.random.randn(n_samples, n_features, n_trials))
        y = Outputs.from_array(np.random.randn(n_samples, n_targets, n_trials))
        
        initial_loss, _, final_loss, params = objective_vectorized(
            model=simple_vectorized_model,
            param_estimator=simple_param_estimator,
            x=x,
            y=y,
            fit_params=False,
            create_train_test_trial_split_fn=legacy_split,
            sample_loss_fn=default_sample_loss_fn,
        )
        
        assert np.isfinite(initial_loss)
        assert np.isfinite(final_loss)


# ============================================================================
# Test objective (main entry point)
# ============================================================================

class TestObjective:
    """Tests for objective function (main entry point)."""

    def test_scalar_output_accepted(self):
        """Test that scalar outputs (2D y) work via the unified objective path."""
        np.random.seed(42)
        n_samples, n_features, n_trials = 5, 2, 50
        
        x = jnp.array(np.random.randn(n_samples, n_features, n_trials))
        y_2d = jnp.array(np.random.randn(n_samples, n_trials))
        
        # Call objective with 2D y (auto-expanded to n_targets=1)
        initial_loss, _, final_loss, params = objective(
            model=simple_scalar_model,
            param_estimator=simple_param_estimator,
            x=x,
            y=y_2d,
            fit_params=False,
            create_train_test_trial_split_fn=legacy_split,
            sample_loss_fn=default_sample_loss_fn,
        )
        
        assert np.isfinite(initial_loss)
        assert np.isfinite(final_loss)
        
    def test_vectorized_output_accepted(self):
        """Test that vectorized outputs (n_targets>1) work via the unified objective path."""
        np.random.seed(42)
        n_samples, n_features, n_trials = 5, 2, 50
        n_targets = 2
        
        x = jnp.array(np.random.randn(n_samples, n_features, n_trials))
        y_3d = jnp.array(np.random.randn(n_samples, n_targets, n_trials))
        
        # Call objective with 3D y (n_targets=2)
        initial_loss, _, final_loss, params = objective(
            model=simple_vectorized_model,
            param_estimator=simple_param_estimator,
            x=x,
            y=y_3d,
            fit_params=False,
            create_train_test_trial_split_fn=legacy_split,
            sample_loss_fn=default_sample_loss_fn,
        )
        
        assert np.isfinite(initial_loss)
        assert np.isfinite(final_loss)
        
    def test_sample_loss_fn_passed_through(self):
        """Test that sample_loss_fn is passed through objective to objective_vectorized."""
        np.random.seed(42)
        n_samples, n_features, n_trials = 5, 2, 50

        x = jnp.array(np.random.randn(n_samples, n_features, n_trials))
        y = jnp.array(np.random.randn(n_samples, n_trials))

        def always_zero_loss(model, x_i, y_i, params):
            return jnp.zeros(())

        _, _, loss, _ = objective(
            model=simple_scalar_model,
            param_estimator=simple_param_estimator,
            x=x,
            y=y,
            sample_loss_fn=always_zero_loss,
            fit_params=False,
            create_train_test_trial_split_fn=legacy_split,
            param_penalty_weight=0.0,
        )

        assert np.isclose(loss, 0.0, atol=1e-6)


# ============================================================================
# Integration tests
# ============================================================================

class TestIntegration:
    """Integration tests comparing legacy and vectorized implementations."""

    def test_single_target_consistency(self):
        """
        Test that objective_vectorized with n_targets=1 gives similar results
        to objective_legacy (they should be equivalent).
        """
        np.random.seed(42)
        n_samples, n_features, n_trials = 10, 3, 100
        
        x = jnp.array(np.random.randn(n_samples, n_features, n_trials))
        y_2d = jnp.array(np.random.randn(n_samples, n_trials))
        y_3d = y_2d[:, None, :]  # Expand to (n_samples, 1, n_trials)
        
        # Legacy implementation
        init_loss_legacy, _, final_loss_legacy, params_legacy = objective_legacy(
            model=simple_scalar_model,
            param_estimator=simple_param_estimator,
            x=x,
            y=y_2d,
            fit_params=False,
            create_train_test_trial_split_fn=legacy_split,
            sample_loss_fn=default_sample_loss_fn,
        )

        # Vectorized with n_targets=1
        init_loss_vec, _, final_loss_vec, params_vec = objective_vectorized(
            model=simple_scalar_model,
            param_estimator=simple_param_estimator,
            x=x,
            y=y_3d,
            fit_params=False,
            create_train_test_trial_split_fn=legacy_split,
            sample_loss_fn=default_sample_loss_fn,
        )
        
        # Losses should be very close (numerical precision differences ok)
        assert np.isclose(init_loss_legacy, init_loss_vec, rtol=1e-4), \
            f"Initial loss mismatch: legacy={init_loss_legacy}, vec={init_loss_vec}"
        assert np.isclose(final_loss_legacy, final_loss_vec, rtol=1e-4), \
            f"Final loss mismatch: legacy={final_loss_legacy}, vec={final_loss_vec}"


# ============================================================================
# Test _call_trial_split
# ============================================================================

class TestCallTrialSplit:
    """Tests for the _call_trial_split wrapper function."""

    def test_legacy_matched_trials(self):
        """Legacy split_fn with matched n_trials returns duplicated 4-tuple."""
        def legacy_split(n_trials, random_seed=0):
            idx = np.arange(n_trials)
            return idx[:n_trials // 2], idx[n_trials // 2:]

        x_tr, x_te, y_tr, y_te = _call_trial_split(legacy_split, 100, 100, 42)
        assert len(x_tr) == 50
        assert len(x_te) == 50
        # When matched, x and y indices should be identical
        np.testing.assert_array_equal(x_tr, y_tr)
        np.testing.assert_array_equal(x_te, y_te)

    def test_legacy_mismatched_trials_raises(self):
        """Legacy split_fn with mismatched n_trials raises ValueError."""
        def legacy_split(n_trials, random_seed=0):
            idx = np.arange(n_trials)
            return idx[:n_trials // 2], idx[n_trials // 2:]

        with pytest.raises(ValueError, match="legacy signature"):
            _call_trial_split(legacy_split, 100, 80, 42)

    def test_generalized_4tuple(self):
        """Generalized split_fn returning 4-tuple works."""
        def generalized_split(n_trials_x, n_trials_y, random_seed=0):
            x_idx = np.arange(n_trials_x)
            y_idx = np.arange(n_trials_y)
            return (x_idx[:n_trials_x // 2], x_idx[n_trials_x // 2:],
                    y_idx[:n_trials_y // 2], y_idx[n_trials_y // 2:])

        x_tr, x_te, y_tr, y_te = _call_trial_split(generalized_split, 100, 80, 42)
        assert len(x_tr) == 50
        assert len(x_te) == 50
        assert len(y_tr) == 40
        assert len(y_te) == 40

    def test_none_returns_identity(self):
        """None split_fn returns a ValueError since we require an explicit split function."""
        with pytest.raises(ValueError, match="Trial split function is None"):
            _call_trial_split(None, 100, 80, 42)


# ============================================================================
# Test sample_loss_fn with mismatched trials
# ============================================================================

class TestMismatchedTrials:
    """Tests for objective with mismatched input/output trial dimensions."""

    def test_objective_with_sample_loss_fn(self):
        """Test objective works with custom sample_loss_fn and mismatched trials."""
        np.random.seed(42)
        n_samples, n_features = 5, 2
        n_trials_x, n_trials_y = 100, 80

        x = jnp.array(np.random.randn(n_samples, n_features, n_trials_x))
        # y has different trial count
        y = jnp.array(np.random.randn(n_samples, 1, n_trials_y))

        def custom_sample_loss(model, x_i, y_i, params):
            """Custom loss: compare mean of predictions to mean of targets."""
            pred = model(x_i, *params)  # (n_trials_x,)
            # Compare means — works regardless of trial count mismatch
            return (jnp.mean(pred) - jnp.mean(y_i)) ** 2

        def generalized_split(n_trials_x, n_trials_y, random_seed=0):
            """Split x and y trials independently."""
            key = jax.random.PRNGKey(random_seed)
            k1, k2 = jax.random.split(key)
            x_idx = jax.random.permutation(k1, jnp.arange(n_trials_x))
            y_idx = jax.random.permutation(k2, jnp.arange(n_trials_y))
            x_half = n_trials_x // 2
            y_half = n_trials_y // 2
            return x_idx[:x_half], x_idx[x_half:], y_idx[:y_half], y_idx[y_half:]

        initial_loss, initial_params, final_loss, params = objective(
            model=simple_scalar_model,
            param_estimator=simple_param_estimator,
            x=x,
            y=y,
            create_train_test_trial_split_fn=generalized_split,
            sample_loss_fn=custom_sample_loss,
            fit_params=True,
            max_iter=50,
            param_penalty_weight=0.01,
        )

        assert np.isfinite(initial_loss), f"Initial loss is not finite: {initial_loss}"
        assert np.isfinite(final_loss), f"Final loss is not finite: {final_loss}"
        assert params.shape == (n_samples, 2)

    def test_matched_trials_default_loss(self):
        """Verify that the default sample_loss_fn (MSE) works for matched trials."""
        np.random.seed(42)
        n_samples, n_features, n_trials = 10, 3, 100

        x = jnp.array(np.random.randn(n_samples, n_features, n_trials))
        y = jnp.array(np.random.randn(n_samples, n_trials))

        # Default sample_loss_fn (MSE)
        initial_loss, initial_params, final_loss, params = objective(
            model=simple_scalar_model,
            param_estimator=simple_param_estimator,
            x=x,
            y=y,
            fit_params=False,
            create_train_test_trial_split_fn=legacy_split,
            sample_loss_fn=default_sample_loss_fn,
        )

        assert np.isfinite(initial_loss)
        assert np.isfinite(final_loss)
        assert params.shape == (n_samples, 2)

    def test_sample_loss_fn_with_matched_trials(self):
        """sample_loss_fn also works when trials happen to match."""
        np.random.seed(42)
        n_samples, n_features, n_trials = 5, 2, 50

        x = jnp.array(np.random.randn(n_samples, n_features, n_trials))
        y = jnp.array(np.random.randn(n_samples, 1, n_trials))

        def custom_sample_loss(model, x_i, y_i, params):
            pred = model(x_i, *params)
            return jnp.mean((pred - y_i[0]) ** 2)

        initial_loss, _, final_loss, params = objective(
            model=simple_scalar_model,
            param_estimator=simple_param_estimator,
            x=x,
            y=y,
            sample_loss_fn=custom_sample_loss,
            create_train_test_trial_split_fn=legacy_split,
            fit_params=False,
        )

        assert np.isfinite(initial_loss)
        assert np.isfinite(final_loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
