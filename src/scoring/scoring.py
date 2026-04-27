import multiprocessing as mp

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree

from ..evolution.program import Program


def _worker(queue, program, data, loss_fn, config, X_eval):
    try:
        result = score_program(program, data, loss_fn, config, X_eval)
    except Exception:
        result = (float("inf"), float("inf"), None)
    queue.put(result)


def score_with_timeout(program, data, loss_fn, config, X_eval=None) -> tuple[float, float, jnp.ndarray | None]:
    """Run score_program in a subprocess, returning (final_loss, initial_loss, eval_fingerprint).

    Uses spawn (not fork) so JAX initialises cleanly in the child process.

    Args:
        program: Candidate program to score. Must have ``n_params`` set.
        data: Tuple of ``(data_train, data_test)``, each a JAX data dict with
            arrays of shape ``(n_samples, ..., n_trials)``.
        loss_fn: Loss function with signature ``(output, data) -> scalar``,
            where output and data both have a leading sample axis. Must be a
            module-level function so it can be pickled across the subprocess
            boundary.
        config: The ``scoring`` config subsection. Keys:
            ``timeout_s``, ``param_penalty_weight``, ``gradient_descent``.
        X_eval: Optional evaluation data dict for computing the deduplication
            fingerprint. If None, eval_fingerprint is not computed.

    Returns:
        Tuple of (final_loss, initial_loss, eval_fingerprint), or
        (inf, inf, None) if timed out or failed.
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(queue, program, data, loss_fn, config, X_eval))
    proc.start()
    proc.join(timeout=config["timeout_s"])
    if proc.is_alive():
        proc.kill()
        proc.join()
        return (float("inf"), float("inf"), None)
    return queue.get()


def _get_params(param_est_fn, default_params, data_train):
    """Estimate initial parameters for all samples via vmapped param estimator.

    Args:
        param_est_fn: JAX parameter estimator with signature
            ``(data_i) -> params`` for a single sample (no batch axis).
            Vmapped over the leading sample axis of ``data_train``.
        default_params: Single-sample fallback pytree used when the estimator
            fails (e.g. ``model_fn.DEFAULT_PARAMS``).
        data_train: JAX data dict with arrays of shape ``(n_samples, ..., n_trials)``.

    Returns:
        Batched parameter pytree with a leading sample axis on every leaf.
    """
    try:
        return jax.vmap(param_est_fn)(data_train)
    except Exception:
        n = next(iter(data_train.values())).shape[0]
        return jax.tree_util.tree_map(lambda x: jnp.stack([x] * n), default_params)


def _optimize(model_fn, loss_fn, params_init, data_train, gd_config):
    """Fit parameters on data_train using Adam, returning the best params found.

    Optimises a flattened parameter vector with ``jax.value_and_grad`` and
    tracks the best (lowest-loss) checkpoint across all steps.

    Args:
        model_fn: JAX model with signature ``(data_i, params) -> output``
            for a single sample. Vmapped over the leading sample axis internally.
        loss_fn: Loss function with signature ``(output, data) -> scalar``,
            called on the full batch after vmapping the model.
        params_init: Initial batched parameter pytree with a leading sample axis.
        data_train: JAX data dict with arrays of shape ``(n_samples, ..., n_trials)``.
        gd_config: The ``scoring.gradient_descent`` config subsection.
            Keys: ``learning_rate`` (float), ``max_iter`` (int).

    Returns:
        Optimized parameter pytree with the same structure as ``params_init``,
        corresponding to the lowest loss seen during training.
    """
    flat, unflatten = ravel_pytree(params_init)

    def total_loss(flat_p):
        p = unflatten(flat_p)
        output = jax.vmap(model_fn, in_axes=(0, 0))(data_train, p)
        return loss_fn(output, data_train)

    loss_and_grad = jax.jit(jax.value_and_grad(total_loss))
    opt = optax.adam(gd_config["learning_rate"])
    opt_state = opt.init(flat)
    best_loss, best_flat = float("inf"), flat

    for step in range(gd_config["max_iter"]):
        loss_val, grad = loss_and_grad(flat)
        if not jnp.isfinite(loss_val):
            break
        if float(loss_val) < best_loss:
            best_loss, best_flat = float(loss_val), flat.copy()
        updates, opt_state = opt.update(grad, opt_state, flat)
        flat = optax.apply_updates(flat, updates)
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss_val:.4f}")

    return unflatten(best_flat)


def _eval_loss(model_fn, loss_fn, params, data_test):
    """Evaluate loss on data_test with fixed params (no gradient computation).

    Args:
        model_fn: JAX model with signature ``(data_i, params) -> output``
            for a single sample. Vmapped over the leading sample axis internally.
        loss_fn: Loss function with signature ``(output, data) -> scalar``,
            called on the full batch after vmapping the model.
        params: Batched parameter pytree with a leading sample axis.
        data_test: JAX data dict with arrays of shape ``(n_samples, ..., n_trials)``.

    Returns:
        Scalar loss as a Python float.
    """
    output = jax.vmap(model_fn, in_axes=(0, 0))(data_test, params)
    return float(loss_fn(output, data_test))


def _eval_fingerprint(model_fn, params, X_eval):
    """Compute model outputs on evaluation grid using optimal params.

    Args:
        model_fn: JAX model with signature ``(data_i, params) -> output``.
        params: Batched parameter pytree with a leading sample axis.
        X_eval: Evaluation data dict for fingerprinting.

    Returns:
        jnp.ndarray of model outputs on the evaluation grid.
    """
    return jax.vmap(model_fn, in_axes=(0, 0))(X_eval, params)


def score_program(program: Program, data: tuple, loss_fn, config: dict, X_eval=None) -> tuple[float, float, jnp.ndarray | None]:
    """Score a program: fit params on data_train, evaluate on data_test.

    Intended to be called inside a subprocess — the process timeout is the
    caller's responsibility (see ``score_with_timeout``).

    Args:
        program: Candidate program to score. Must have ``n_params`` set before
            calling (e.g. via ``program.count_params()``).
        data: Tuple of ``(data_train, data_test)``, each a JAX data dict with
            arrays of shape ``(n_samples, ..., n_trials)``. data_train is used
            for parameter estimation and optimisation; data_test for evaluation.
        loss_fn: Loss function with signature ``(output, data) -> scalar``,
            where output and data both have a leading sample axis.
        config: The ``scoring`` config subsection. Keys:
            ``param_penalty_weight`` (float) and ``gradient_descent`` (dict).
        X_eval: Optional evaluation data dict for computing the deduplication
            fingerprint. If None, eval_fingerprint is not computed.

    Returns:
        Tuple of (final_loss, initial_loss, eval_fingerprint) where:
        - final_loss: loss on data_test after optimization plus complexity penalty
        - initial_loss: cross-validated loss on data_train before optimization, plus complexity penalty
        - eval_fingerprint: model outputs on X_eval with optimal params, or None
    """
    data_train, data_test = data
    model_fn, param_est_fn = program.compile()

    complexity_penalty = config["param_penalty_weight"] * program.n_params
    params_init = _get_params(param_est_fn, model_fn.DEFAULT_PARAMS, data_train)
    initial_loss = _eval_loss(model_fn, loss_fn, params_init, data_train) + complexity_penalty

    params = _optimize(model_fn, loss_fn, params_init, data_train, config["gradient_descent"])

    final_loss = _eval_loss(model_fn, loss_fn, params, data_test) + complexity_penalty
    fingerprint = _eval_fingerprint(model_fn, params, X_eval) if X_eval is not None else None
    return (final_loss, initial_loss, fingerprint)
