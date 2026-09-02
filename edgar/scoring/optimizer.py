"""Parameter optimization for EDGAR using JAX.

Provides a JIT-compiled gradient descent Optimizer class, using jax.lax.scan to execute the optimization loop.
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

    Holds the model function, loss function, data, and optimizer state to perform parameter optimization across multiple initial parameter values.
    This is designed to be JIT-compiled and executed entirely on GPU, via run_optimization.
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
            model_fn: The JAX-compiled model function (callable).
            loss_fn: The loss function (callable).
            data_train: The training data dictionary.
            gd_config: Configuration dictionary for gradient descent.
        """
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.data_train = data_train
        self.gd_config = gd_config
        self.opt = optax.adam(gd_config["learning_rate"])

    def flatten_and_init_params(
        self,
        initial_params: list[dict[str, Any]],
    ):
        """Flatten the initial parameters and initialize the optax optimizer state.

        Args:
            initial_params: A list of PyTrees, each element is the initial parameters for a single optimization.

        Returns:
            flat_all: A 2D JAX array of shape (n_opts, flat_dim) containing the flattened initial parameters
            opt_state: The initial Optax optimizer state
        """
        _, self.unflatten = ravel_pytree(initial_params[0])
        flat_all = jnp.stack([ravel_pytree(p)[0] for p in initial_params])
        opt_state = self.opt.init(flat_all)
        return flat_all, opt_state

    def _scalar_loss_single(self, flat_p: jax.Array) -> jax.Array:
        """Computes the mean loss for a single flattened parameter set.

        Args:
            flat_p: Flattened parameters, matching the shape expected by self.unflatten.

        Returns:
            The scalar mean loss
        """
        p = self.unflatten(flat_p)
        return _evaluate_scalar_loss(self.model_fn, self.loss_fn, p, self.data_train)

    def _loss_and_grad_batched(
        self, flat_all: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Computes batched losses and gradients across all parallel optimizations.

        Args:
            flat_all: Stacked flattened parameters as a 2D JAX array
                of shape (n_opts, flat_dim).

        Returns:
            A tuple containing:
                - Batched losses of shape (n_opts,).
                - Batched gradients of shape (n_opts, flat_dim).
        """
        single_value_and_grad = jax.value_and_grad(self._scalar_loss_single)
        return jax.vmap(single_value_and_grad)(flat_all)

    def _step_fn(
        self,
        carry: tuple[jax.Array, optax.OptState, jax.Array, jax.Array],
        step_idx: jax.Array,
    ) -> tuple[tuple[jax.Array, optax.OptState, jax.Array, jax.Array], jax.Array]:
        """Standard step function invoked by jax.lax.scan at each optimization step.

        Args:
            carry: Current state tuple containing:
                - flat_all: Stacked flattened parameters.
                - opt_state: Stacked Optax optimizer state.
                - best_losses: Best losses tracked on-device so far.
                - best_flats: Best parameter configurations tracked on-device so far.
            step_idx: The step index (unused, but required by jax.lax.scan).

        Returns:
            A tuple containing the next state tuple, and current step loss values.
        """
        flat_all, opt_state, best_losses, best_flats = carry

        # 1. Compute loss and gradients for all optimizations in parallel
        loss_vals, grads = self._loss_and_grad_batched(flat_all)

        # 2. Compute optimizer updates for all optimizations
        updates, next_opt_state = self.opt.update(grads, opt_state, flat_all)
        new_flats = optax.apply_updates(flat_all, updates)

        # 3. Track the best loss and parameters seen so far for each optimization
        is_better = loss_vals < best_losses
        best_losses_next = jnp.where(is_better, loss_vals, best_losses)
        best_flats_next = jnp.where(is_better[:, None], flat_all, best_flats)

        return (new_flats, next_opt_state, best_losses_next, best_flats_next), loss_vals

    @partial(jax.jit, static_argnums=0)
    def run_optimization(
        self, flat_all: jax.Array, opt_state: optax.OptState
    ) -> tuple[jax.Array, jax.Array]:
        """JIT-compiled scan loop executing entirely on-device.

        Args:
            flat_all: Stacked flattened parameters of shape (n_opts, flat_dim)
            opt_state: Initial Optax optimizer state.

        Returns:
            A tuple of (optimized_parameters, loss_trajectories)
                - optimized_parameters: A len(n_opts) list of PyTrees of optimized parameters, one for each optimization.
                - loss_trajectories: A 2D JAX array of shape (max_iter, n_opts) containing step-by-step losses during optimization.
        """
        n_opts = flat_all.shape[0]
        best_losses = jnp.full((n_opts,), jnp.inf)
        best_flats = jnp.copy(flat_all)

        init_carry = (flat_all, opt_state, best_losses, best_flats)
        steps = jnp.arange(self.gd_config["max_iter"])

        (_, _, _, final_best_flats), loss_trajectories = jax.lax.scan(
            self._step_fn, init_carry, steps
        )
        return [self.unflatten(flat) for flat in final_best_flats], loss_trajectories
