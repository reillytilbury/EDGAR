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
)
from src.data_structures import Inputs, Outputs, ensure_inputs, ensure_outputs
from src.loss_functions import quadratic_loss


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
            loss_func=quadratic_loss,
            x=x,
            y=y,
            fit_params=True,
            max_iter=100,  # Need >= 51 for warmup schedule
            param_penalty_weight=0.01,
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
            loss_func=quadratic_loss,
            x=x,
            y=y,
            fit_params=False,  # Skip optimization for speed
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
            loss_func=quadratic_loss,
            x=x,
            y=y,
            fit_params=True,
            max_iter=100,  # Need >= 51 for warmup schedule
            param_penalty_weight=0.01,
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
            loss_func=quadratic_loss,
            x=x,
            y=y,
            fit_params=True,
            max_iter=100,  # Need >= 51 for warmup schedule
            param_penalty_weight=0.01,
        )
        
        assert np.isfinite(initial_loss)
        assert np.isfinite(final_loss)
        assert params.shape == (n_samples, 2)
        
    def test_target_weights_uniform(self):
        """Test that uniform weights give same result as None."""
        np.random.seed(42)
        n_samples, n_features, n_trials = 5, 2, 50
        n_targets = 2
        
        x = jnp.array(np.random.randn(n_samples, n_features, n_trials))
        y = jnp.array(np.random.randn(n_samples, n_targets, n_trials))
        
        # Without weights (defaults to uniform)
        _, _, loss_no_weights, _ = objective_vectorized(
            model=simple_vectorized_model,
            param_estimator=simple_param_estimator,
            loss_func=quadratic_loss,
            x=x,
            y=y,
            fit_params=False,
        )
        
        # With explicit uniform weights
        _, _, loss_uniform, _ = objective_vectorized(
            model=simple_vectorized_model,
            param_estimator=simple_param_estimator,
            loss_func=quadratic_loss,
            x=x,
            y=y,
            fit_params=False,
            target_weights=jnp.array([0.5, 0.5]),
        )
        
        assert np.isclose(loss_no_weights, loss_uniform, rtol=1e-5)
        
    def test_target_weights_custom(self):
        """Test that custom weights change the loss value."""
        np.random.seed(42)
        n_samples, n_features, n_trials = 5, 2, 50
        n_targets = 2
        
        x = jnp.array(np.random.randn(n_samples, n_features, n_trials))
        y = jnp.array(np.random.randn(n_samples, n_targets, n_trials))
        
        # Uniform weights
        _, _, loss_uniform, _ = objective_vectorized(
            model=simple_vectorized_model,
            param_estimator=simple_param_estimator,
            loss_func=quadratic_loss,
            x=x,
            y=y,
            fit_params=False,
            target_weights=jnp.array([0.5, 0.5]),
        )
        
        # Heavily weighted toward first target
        _, _, loss_weighted, _ = objective_vectorized(
            model=simple_vectorized_model,
            param_estimator=simple_param_estimator,
            loss_func=quadratic_loss,
            x=x,
            y=y,
            fit_params=False,
            target_weights=jnp.array([0.9, 0.1]),
        )
        
        # Losses should be different with different weights
        assert not np.isclose(loss_uniform, loss_weighted, rtol=1e-3)
        
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
            loss_func=quadratic_loss,
            x=x,
            y=y,
            fit_params=False,
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
            loss_func=quadratic_loss,
            x=x,
            y=y_2d,
            fit_params=False,
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
            loss_func=quadratic_loss,
            x=x,
            y=y_3d,
            fit_params=False,
        )
        
        assert np.isfinite(initial_loss)
        assert np.isfinite(final_loss)
        
    def test_target_weights_passed_through(self):
        """Test that target_weights is passed to vectorized."""
        np.random.seed(42)
        n_samples, n_features, n_trials = 5, 2, 50
        n_targets = 2
        
        x = jnp.array(np.random.randn(n_samples, n_features, n_trials))
        y = jnp.array(np.random.randn(n_samples, n_targets, n_trials))
        
        # Should not raise - target_weights should be passed through
        _, _, loss, _ = objective(
            model=simple_vectorized_model,
            param_estimator=simple_param_estimator,
            loss_func=quadratic_loss,
            x=x,
            y=y,
            target_weights=jnp.array([0.7, 0.3]),
            fit_params=False,
        )
        
        assert np.isfinite(loss)


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
            loss_func=quadratic_loss,
            x=x,
            y=y_2d,
            fit_params=False,
        )
        
        # Vectorized with n_targets=1
        init_loss_vec, _, final_loss_vec, params_vec = objective_vectorized(
            model=simple_scalar_model,
            param_estimator=simple_param_estimator,
            loss_func=quadratic_loss,
            x=x,
            y=y_3d,
            fit_params=False,
        )
        
        # Losses should be very close (numerical precision differences ok)
        assert np.isclose(init_loss_legacy, init_loss_vec, rtol=1e-4), \
            f"Initial loss mismatch: legacy={init_loss_legacy}, vec={init_loss_vec}"
        assert np.isclose(final_loss_legacy, final_loss_vec, rtol=1e-4), \
            f"Final loss mismatch: legacy={final_loss_legacy}, vec={final_loss_vec}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
