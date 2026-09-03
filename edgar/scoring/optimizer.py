"""Parameter optimization for EDGAR using JAX.

Provides a JIT-compiled gradient descent Optimizer class, using jax.lax.scan to
execute the optimization loop.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import optax

from .utils import _evaluate_scalar_loss


class Optimizer:
    """Gradient descent solver for JAX models.

    Holds the model function, loss function, data, and optimizer state to perform
    parameter optimization across multiple initial parameter values.
    This class is designed to be JIT-compiled and executed entirely on a device
    (e.g., GPU or TPU) via the `run_optimization` method. It enables efficient
    parallel optimization of multiple initial parameter sets.
    The optimized parameters and training loss trajectories are then returned.
    """

    def __init__(
        self,
        model_fn: Callable[[Any, Any], jax.Array],
        loss_fn: Callable[[jax.Array, Any], jax.Array],
        data_train: dict[str, Any],
        gd_config: dict[str, Any],
    ) -> None:
        """Initializes the Optimizer with model, loss, and training details.

        Args:
            model_fn: The JAX-compiled model function that takes parameters and data,
                and returns predictions.
            loss_fn: The loss function that takes model predictions and true data,
                and returns a loss value.
            data_train: A dictionary containing the training data.
            gd_config: Configuration dictionary for gradient descent,
                expected to contain 'learning_rate' and 'max_iter'.
        """
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.data_train = data_train
        self.gd_config = gd_config
        # Initializes the Adam optimizer from Optax with the specified learning rate.
        self.opt = optax.adam(gd_config["learning_rate"])

    def flatten_and_init_params(
        self,
        initial_params: list[dict[str, Any]],
    ):
        """Flatten the initial parameters and initialize the Optax optimizer state.

        This method prepares parameter PyTrees for optimization by linearizing them
        into flat JAX arrays and stacking them for batch processing. It also
        initializes the Optax Adam optimizer state.

        Args:
            initial_params: A list of PyTrees, where each element is a dictionary
                representing the initial parameters for a single optimization run.

        Returns:
            flat_all: A 2D JAX array of shape `(n_opts, flat_dim)` containing the
                flattened initial parameters for `n_opts` parallel optimizations.
            opt_state: The initial Optax optimizer state, configured for the
                flattened parameters.
        """
        # ravel_pytree flattens a PyTree into a 1D array and returns a function
        # to unflatten it back. We store unflatten for later use.
        _, self.unflatten = ravel_pytree(initial_params[0])
        # Stack all flattened initial parameters into a single 2D array.
        flat_all = jnp.stack([ravel_pytree(p)[0] for p in initial_params])
        # Initialize the Optax optimizer state for the batched flattened parameters.
        opt_state = self.opt.init(flat_all)
        return flat_all, opt_state

    def _scalar_loss_single(self, flat_p: jax.Array) -> jax.Array:
        """Computes the mean scalar loss for a single flattened parameter set.

        This internal helper function unflattens the parameters, then calculates
        the mean loss using the model and loss functions provided during
        initialization.

        Args:
            flat_p: A 1D JAX array representing a single set of flattened parameters.

        Returns:
            The scalar mean loss computed for the given parameters and training data.
        """
        # Unflatten the 1D parameter array back into its original PyTree structure.
        p = self.unflatten(flat_p)
        # Evaluate the scalar mean loss using the model, loss function, parameters,
        # and training data.
        return _evaluate_scalar_loss(self.model_fn, self.loss_fn, p, self.data_train)

    def _loss_and_grad_batched(
        self, flat_all: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Computes batched losses and gradients across all parallel optimizations.

        This function uses `jax.vmap` to efficiently vectorize the computation of
        loss and gradient for each set of parameters in `flat_all` in parallel.

        Args:
            flat_all: Stacked flattened parameters as a 2D JAX array
                of shape `(n_opts, flat_dim)`, where `n_opts` is the number of
                parallel optimizations and `flat_dim` is the dimension of the
                flattened parameter space.

        Returns:
            A tuple containing:
                - Batched losses: A 1D JAX array of shape `(n_opts,)`, where each
                  element is the scalar loss for a corresponding parameter set.
                - Batched gradients: A 2D JAX array of shape `(n_opts, flat_dim)`,
                  where each row contains the gradients for a corresponding
                  parameter set.
        """
        # `jax.value_and_grad` computes both the function value and its gradient.
        single_value_and_grad = jax.value_and_grad(self._scalar_loss_single)
        # `jax.vmap` applies `single_value_and_grad` across the leading axis of
        # `flat_all`, effectively batching the computation.
        return jax.vmap(single_value_and_grad)(flat_all)

    def _step_fn(
        self,
        carry: tuple[jax.Array, optax.OptState, jax.Array, jax.Array],
        step_idx: jax.Array,
    ) -> tuple[tuple[jax.Array, optax.OptState, jax.Array, jax.Array], jax.Array]:
        """Standard step function invoked by `jax.lax.scan` at each optimization step.

        This function performs a single step of gradient descent for all parallel
        optimizations, updates parameters, and tracks the best-performing parameters
        and losses seen so far.

        Args:
            carry: A tuple representing the current state of the optimization,
                containing:
                - `flat_all`: A 2D JAX array of shape `(n_opts, flat_dim)`
                  representing the current stacked flattened parameters for all
                  parallel optimizations.
                - `opt_state`: The current Optax optimizer state.
                - `best_losses`: A 1D JAX array of shape `(n_opts,)` storing the
                  lowest loss found so far for each parallel optimization.
                - `best_flats`: A 2D JAX array of shape `(n_opts, flat_dim)` storing
                  the flattened parameters that yielded `best_losses` for each
                  parallel optimization.
            step_idx: The current step index in the optimization loop. This is
                required by `jax.lax.scan` but is not directly used in this function.

        Returns:
            A tuple containing:
                - The updated carry tuple (`new_flats`, `next_opt_state`,
                  `best_losses_next`, `best_flats_next`) after this optimization step.
                - `loss_vals`: A 1D JAX array of shape `(n_opts,)` containing the
                  loss values computed at the current step for each parallel
                  optimization.
        """
        flat_all, opt_state, best_losses, best_flats = carry

        # 1. Compute loss and gradients for all optimizations in parallel.
        loss_vals, grads = self._loss_and_grad_batched(flat_all)

        # 2. Compute optimizer updates for all optimizations.
        updates, next_opt_state = self.opt.update(grads, opt_state, flat_all)
        # Apply the computed updates to the current parameters.
        new_flats = optax.apply_updates(flat_all, updates)

        # 3. Track the best loss and parameters seen so far for each optimization.
        # `is_better` identifies which parallel optimizations have a lower loss
        # at the current step compared to their previously recorded best.
        is_better = loss_vals < best_losses
        # Update `best_losses_next` by conditionally selecting between current
        # `loss_vals` and previous `best_losses`.
        best_losses_next = jnp.where(is_better, loss_vals, best_losses)
        # Update `best_flats_next` by conditionally selecting between current
        # `flat_all` and previous `best_flats`. The `[:, None]` reshapes
        # `is_better` to enable broadcasting for element-wise selection.
        best_flats_next = jnp.where(is_better[:, None], flat_all, best_flats)

        return (new_flats, next_opt_state, best_losses_next, best_flats_next), loss_vals

    @partial(jax.jit, static_argnums=0)
    def run_optimization(
        self, flat_all: jax.Array, opt_state: optax.OptState
    ) -> tuple[list[dict[str, Any]], jax.Array]:
        """Executes the JIT-compiled optimization loop entirely on-device using `jax.lax.scan`.

        This is the main entry point for running the gradient descent optimization.
        It uses `jax.lax.scan` to unroll the optimization steps efficiently on the device.

        Args:
            flat_all: Stacked flattened initial parameters of shape
                `(n_opts, flat_dim)`, where `n_opts` is the number of parallel
                optimizations.
            opt_state: Initial Optax optimizer state, obtained from
                `flatten_and_init_params`.

        Returns:
            A tuple of (optimized_parameters, loss_trajectories):
                - `optimized_parameters`: A list of `n_opts` PyTrees, where each
                  PyTree contains the optimized parameters for one parallel
                  optimization. These are the parameters that yielded the
                  lowest loss during their respective optimization runs.
                - `loss_trajectories`: A 2D JAX array of shape `(max_iter, n_opts)`
                  containing the step-by-step loss values for each parallel
                  optimization throughout the entire optimization process.
        """
        n_opts = flat_all.shape[0]
        # Initialize `best_losses` with infinity to ensure any initial loss is better.
        best_losses = jnp.full((n_opts,), jnp.inf)
        # Initialize `best_flats` with the initial parameters.
        best_flats = jnp.copy(flat_all)

        # `init_carry` holds the initial state for the `jax.lax.scan` loop.
        init_carry = (flat_all, opt_state, best_losses, best_flats)
        # `steps` define the number of iterations for `jax.lax.scan`.
        steps = jnp.arange(self.gd_config["max_iter"])

        # `jax.lax.scan` iteratively applies `self._step_fn`.
        # The first returned value is the final carry state, and the second
        # is the stacked results (loss_vals) from each step.
        (_, _, _, final_best_flats), loss_trajectories = jax.lax.scan(
            self._step_fn, init_carry, steps
        )
        # Unflatten the final best parameters back into their original PyTree structure
        # for each parallel optimization.
        return [self.unflatten(flat) for flat in final_best_flats], loss_trajectories

