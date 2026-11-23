"""Faster per-cell parameter fitting for upstream's `objective`, chosen by backend.

Upstream fits each program with 1,000 Adam steps driven from a Python loop, syncing with the
host after every step. That is reasonable on a GPU but slow on CPU. This module keeps
upstream's checks, data split and scoring by calling upstream's own `objective` with
`max_iter=0`, and swaps only the fitting loop:

* `upstream`: upstream's loop, untouched.
* `adam`:     the same Adam schedule and best-parameter bookkeeping, compiled into one XLA loop.
* `lm`:       batched Levenberg–Marquardt. Each cell is an independent least-squares problem with a
              handful of parameters, so second-order steps converge in tens of iterations.

Both fitters split the cells across every JAX device. On a CPU, `edgar_study/__init__.py` exposes
one device per core before JAX starts, because XLA otherwise runs this elementwise work on about
one thread.

`auto` picks `adam` when JAX only has a CPU and `upstream` otherwise. `lm` fits each cell to
convergence, which lowers training error but loses upstream's implicit early stopping and scores
held-out trials differently, so it is opt-in.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

FITTERS = ("auto", "upstream", "adam", "lm")
LM_ITERATIONS = 100


def resolve(fitter):
    if fitter == "auto":
        return "adam" if jax.default_backend() == "cpu" else "upstream"
    return fitter


def install(hypothesis_engine, fitter, lm_iterations=LM_ITERATIONS):
    """Replace hypothesis_engine.objective with a version that uses `fitter`. Returns the resolved name."""
    fitter = resolve(fitter)
    if fitter != "upstream":
        hypothesis_engine.objective = make_objective(hypothesis_engine.objective, fitter, lm_iterations)
    return fitter


def make_objective(upstream_objective, fitter, lm_iterations=LM_ITERATIONS):
    @functools.wraps(upstream_objective)
    def objective(neuron_model, param_estimator, loss_func, x, y, param_penalty_weight=0.1,
                  fit_params=True, random_seed=0, FAILED_PROGRAM_COST=jnp.inf, tol=1e-2,
                  max_iter=1_000, use_param_estimator=True):
        # Upstream runs the estimator, all validity checks and the initial scores; no fitting.
        result = upstream_objective(neuron_model, param_estimator, loss_func, x, y,
                                    param_penalty_weight=param_penalty_weight, fit_params=fit_params,
                                    random_seed=random_seed, FAILED_PROGRAM_COST=FAILED_PROGRAM_COST,
                                    tol=tol, max_iter=0, use_param_estimator=use_param_estimator)
        initial_loss, initial_params, unfitted_loss, params = result
        if not fit_params or params.shape[1] == 0 or unfitted_loss == FAILED_PROGRAM_COST:
            return result

        # The same trial split as upstream's objective.
        order = jax.random.permutation(jax.random.PRNGKey(random_seed), jnp.arange(y.shape[1]))
        train, test = order[:y.shape[1] // 2], order[y.shape[1] // 2:]
        x_train, y_train, x_test, y_test = x[:, train], y[:, train], x[:, test], y[:, test]

        if fitter == "lm" and loss_func.__name__ == "quadratic_loss":
            fitted = fit_lm(neuron_model, params, x_train, y_train, lm_iterations)
        else:
            fitted = fit_adam(neuron_model, loss_func, params, x_train, y_train, max_iter)

        per_cell = _per_cell_loss(neuron_model, loss_func)
        final = jnp.nanmean(per_cell(fitted, x_test, y_test)) + param_penalty_weight * fitted.shape[1]
        final = float(jnp.nan_to_num(final, nan=FAILED_PROGRAM_COST, posinf=FAILED_PROGRAM_COST,
                                     neginf=FAILED_PROGRAM_COST))
        return initial_loss, initial_params, final, fitted

    return objective


def _shard(*arrays):
    """Pad the cell axis to a multiple of the device count and split it across devices.
    Returns the sharded arrays and a mask that is 1 for real cells and 0 for padding."""
    devices = jax.devices()
    n_cells = arrays[0].shape[0]
    padded = -(-n_cells // len(devices)) * len(devices)
    index = np.concatenate([np.arange(n_cells), np.zeros(padded - n_cells, dtype=int)])
    sharding = NamedSharding(Mesh(np.array(devices), ("cells",)), PartitionSpec("cells"))
    mask = jax.device_put(jnp.asarray(np.arange(padded) < n_cells, jnp.float32), sharding)
    return [jax.device_put(jnp.asarray(a)[index], sharding) for a in arrays], mask, n_cells


def _per_cell_loss(model, loss_func):
    return jax.vmap(lambda p, xc, yc: jnp.mean(loss_func(model(xc, *p), yc)))


def fit_adam(model, loss_func, params, x, y, max_iter, learning_rate=3e-3):
    """Upstream's Adam loop in one compiled while_loop, with identical bookkeeping:
    stop at the first non-finite loss or parameter, keep the parameters after the best step."""
    import optax

    (params, x, y), mask, n_cells = _shard(params, x, y)
    per_cell = _per_cell_loss(model, loss_func)
    loss = lambda p: jnp.sum(per_cell(p, x, y) * mask) / n_cells
    value_and_grad = jax.value_and_grad(loss)
    opt = optax.adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8)

    def cond(state):
        step, _, _, _, _, stop = state
        return (step < max_iter) & ~stop

    def body(state):
        step, p, opt_state, best_loss, best_p, _ = state
        value, grad = value_and_grad(p)
        updates, opt_state = opt.update(grad, opt_state, p)
        p = optax.apply_updates(p, updates)
        stop = ~jnp.isfinite(value) | ~jnp.all(jnp.isfinite(p))
        better = ~stop & (value < best_loss)
        return (step + 1, p, opt_state, jnp.where(better, value, best_loss), jnp.where(better, p, best_p), stop)

    @jax.jit
    def run(p):
        return jax.lax.while_loop(cond, body, (0, p, opt.init(p), loss(p), p, False))[4]

    return jax.device_get(run(params))[:n_cells]


def fit_lm(model, params, x, y, iterations=LM_ITERATIONS):
    """Batched Levenberg–Marquardt on each cell's squared error. Steps are accepted only when they
    lower that cell's error, so every cell ends at least as well fitted as its starting point."""

    def residual(p, xc, yc):
        return model(xc, *p) - yc

    (params, x, y), _, n_cells = _shard(params, x, y)
    residuals = jax.vmap(residual)
    jacobians = jax.vmap(jax.jacfwd(residual))

    def cost(p):
        r = residuals(p, x, y)
        c = jnp.sum(r * r, axis=1)
        return jnp.where(jnp.isfinite(c), c, jnp.inf)

    def body(_, state):
        p, damping, c = state
        r, J = residuals(p, x, y), jacobians(p, x, y)
        gradient = jnp.einsum("ctk,ct->ck", J, r)
        hessian = jnp.einsum("ctk,ctl->ckl", J, J)
        diagonal = jnp.diagonal(hessian, axis1=1, axis2=2)
        scale = jnp.maximum(diagonal, 1e-9 * jnp.max(diagonal, axis=1, keepdims=True) + 1e-12)
        damped = hessian + jax.vmap(jnp.diag)(damping[:, None] * scale)
        step = jnp.linalg.solve(damped, -gradient[..., None])[..., 0]
        candidate = p + jnp.where(jnp.isfinite(step), step, 0.0)
        new_cost = cost(candidate)
        accept = new_cost < c
        return (jnp.where(accept[:, None], candidate, p),
                jnp.clip(jnp.where(accept, damping / 3, damping * 4), 1e-7, 1e7),
                jnp.where(accept, new_cost, c))

    @jax.jit
    def run(p):
        start = (p, jnp.full(p.shape[0], 1e-3, p.dtype), cost(p))
        return jax.lax.fori_loop(0, iterations, body, start)[0]

    return jax.device_get(run(params))[:n_cells]


def benchmark_report(rows):
    """Format benchmark rows as a Markdown table."""
    lines = ["| Program | Fitter | Seconds | Train loss | Score |", "| --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['program']} | {row['fitter']} | {row['seconds']:.1f} | "
                     f"{row['train_loss']:.4f} | {row['score']:.4f} |")
    return "\n".join(lines)


def train_loss(model, loss_func, params, x, y, random_seed=0):
    order = np.asarray(jax.random.permutation(jax.random.PRNGKey(random_seed), jnp.arange(y.shape[1])))
    train = order[:y.shape[1] // 2]
    return float(jnp.nanmean(_per_cell_loss(model, loss_func)(params, x[:, train], y[:, train])))
