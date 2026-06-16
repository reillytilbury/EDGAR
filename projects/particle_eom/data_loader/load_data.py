from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

# `load_data.py` is exec'd from a source string (see edgar/llm/code_loading.py), so it
# has no `__file__` and can't use a relative import for the sibling `generator.py`
# module. Load it by path instead, relative to the current working directory — `edgar
# run`/`edgar test` are always invoked from the repo root, so this mirrors how
# `io.data_path` itself is resolved.
for _collection in ("projects", "experiments"):
    _generator_path = Path(_collection) / "particle_eom" / "data_loader" / "generator.py"
    if _generator_path.exists():
        break
_spec = importlib.util.spec_from_file_location("particle_eom_generator", _generator_path)
_generator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_generator)
sys.modules.setdefault("particle_eom_generator", _generator)

generate_sessions = _generator.generate_sessions
add_noise = _generator.add_noise
minimum_image_diff = _generator.minimum_image_diff
drop_diagonal = _generator.drop_diagonal


def _to_jax(d):
    return {k: jnp.array(v) if k != "_sample_indices" else v for k, v in d.items()}


def _build_neighbor_dx(positions: np.ndarray, L: float) -> np.ndarray:
    """Per-(session, time) pairwise displacement to every other cell, self dropped.

    Args:
        positions: shape (n_sessions, n_recorded, n_cells).
        L: periodic domain length.

    Returns:
        np.ndarray: neighbor_dx[s, t, i, :] = x_j(t) - x_i(t) for every other cell j
            in session s, shape (n_sessions, n_recorded, n_cells, n_cells - 1).
    """
    diff = minimum_image_diff(positions, L)  # (n_sessions, n_recorded, n_cells, n_cells)
    return drop_diagonal(diff)


def _flatten_cell_time(arr: np.ndarray) -> np.ndarray:
    """Flattens the (n_cells, n_recorded) axes into one trial axis, cell-major.

    Args:
        arr: shape (n_sessions, n_recorded, n_cells, *rest).

    Returns:
        np.ndarray: shape (n_sessions, n_cells * n_recorded, *rest). Trial index
            `cell_idx * n_recorded + time_idx` — documented here so any later
            un-flattening (e.g. for plotting) stays consistent.
    """
    n_sessions, n_recorded, n_cells = arr.shape[:3]
    rest = arr.shape[3:]
    arr = np.moveaxis(arr, 2, 1)  # (n_sessions, n_cells, n_recorded, *rest)
    return arr.reshape((n_sessions, n_cells * n_recorded) + rest)


def load_data(
    data_path: str = "",
    seed: int = 0,
    n_cells: int = 40,
    L: float = 40.0,
    A: float = 1.0,
    B: float = 1.0,
    r_c: float = 1.5,
    dt: float = 1e-3,
    n_steps: int = 2000,
    record_every: int = 5,
    train_frac: float = 0.7,
    noise_std_frac: float = 0.05,
    n_sessions_discover: int = 12,
    n_sessions_validate: int = 4,
    n_eval_sessions: int = 3,
    **kwargs,
):
    """
    Generates the overdamped pairwise-force-law particle data and splits it.

    A "sample" is one recording session: a full independently-simulated population of
    `n_cells` particles over `n_steps` integration steps (see
    journal/2026-06-16_eom_discovery_poc_spec.md, "Design revision" section, for why
    the sample unit is the session rather than the particle or a single snapshot).
    Within a session, every (cell, time) pair is a "trial" — `model()` is called once
    per session and must explain every cell at every recorded timestep with one shared
    `(A, B)` (or `k`, or `c`, for the seeds).

    Ground truth `(A, B, r_c, mu=1)` is not returned separately: it is exactly the
    `project_params` this function was called with, and those are already persisted
    per-run in `task_spec.yaml`/`config.yaml` by the engine, so nothing needs to be
    duplicated here for the evaluation hook to read back.

    Args:
        data_path: Unused — data is generated synthetically from the parameters below.
        seed: Base RNG seed.
        n_cells: Number of particles per session.
        L: Periodic domain length.
        A: Ground-truth repulsion strength.
        B: Ground-truth attraction strength.
        r_c: Ground-truth interaction cutoff radius.
        dt: Integration timestep.
        n_steps: Total integration steps per session.
        record_every: Record positions/velocities every this many steps.
        train_frac: Fraction of recorded time points (from the start) used as the
            train block; the remainder is the held-out test block. Both blocks contain
            the same cells — the split is purely along time.
        noise_std_frac: Gaussian noise std on the velocity target, as a fraction of the
            target's std. 0.0 disables noise.
        n_sessions_discover: Number of sessions in the discover split.
        n_sessions_validate: Number of sessions in the validate split.
        n_eval_sessions: Number of discover sessions used for the fingerprint subset.

    Returns:
        tuple: (X_discover, X_validate, X_eval).

        X_discover = (X_disc_train, X_disc_test), X_validate = (X_val_train, X_val_test).
        Each dict has keys:
            'neighbor_dx': shape (n_sessions_in_split, n_cells * n_time_block, n_cells - 1).
            'velocity':    shape (n_sessions_in_split, n_cells * n_time_block).
        X_eval: same keys as X_disc_train (subset of its sessions), plus
            '_sample_indices' (positions within X_disc_train's session axis).
    """
    n_sessions = n_sessions_discover + n_sessions_validate
    positions, velocities = generate_sessions(
        n_sessions=n_sessions,
        n_cells=n_cells,
        L=L,
        A=A,
        B=B,
        r_c=r_c,
        dt=dt,
        n_steps=n_steps,
        record_every=record_every,
        seed=seed,
    )  # each (n_sessions, n_recorded, n_cells)

    noise_rng = np.random.default_rng(seed + 1000)
    velocities = add_noise(velocities, noise_std_frac, noise_rng)

    neighbor_dx = _build_neighbor_dx(positions, L)  # (n_sessions, n_recorded, n_cells, n_cells-1)

    n_recorded = positions.shape[1]
    n_train_time = int(round(train_frac * n_recorded))
    train_time_idx = np.arange(0, n_train_time)
    test_time_idx = np.arange(n_train_time, n_recorded)

    def _split_by_time(time_idx):
        return {
            "neighbor_dx": _flatten_cell_time(neighbor_dx[:, time_idx]),
            "velocity": _flatten_cell_time(velocities[:, time_idx]),
        }

    train = _split_by_time(train_time_idx)
    test = _split_by_time(test_time_idx)

    split_rng = np.random.default_rng(seed + 1)
    session_perm = split_rng.permutation(n_sessions)
    disc_idx = np.sort(session_perm[:n_sessions_discover])
    val_idx = np.sort(session_perm[n_sessions_discover:])

    X_disc_train = {k: v[disc_idx] for k, v in train.items()}
    X_disc_test = {k: v[disc_idx] for k, v in test.items()}
    X_val_train = {k: v[val_idx] for k, v in train.items()}
    X_val_test = {k: v[val_idx] for k, v in test.items()}

    n_eval = min(n_eval_sessions, len(disc_idx))
    eval_sessions = np.sort(split_rng.choice(disc_idx, n_eval, replace=False))
    eval_pos = np.searchsorted(disc_idx, eval_sessions)
    X_eval = {k: v[eval_pos] for k, v in X_disc_train.items()}
    X_eval["_sample_indices"] = eval_pos

    return (
        (_to_jax(X_disc_train), _to_jax(X_disc_test)),
        (_to_jax(X_val_train), _to_jax(X_val_test)),
        _to_jax(X_eval),
    )


def loss_fn(model_output, data):
    """Mean squared error between predicted and observed dx_i/dt, per session.

    Called on the already-`vmap`'d batch (leading axis = sessions), so the trial axis
    (axis=-1) is reduced here while the session axis is kept — see
    `edgar/scoring/scoring.py`'s `_optimize`/`_eval_loss`, which then mean-reduce this
    over sessions themselves.

    Args:
        model_output: JAX array of predicted dx_i/dt, shape (n_sessions, n_trials).
        data: dict of JAX arrays for this split; data['velocity'] shape
            (n_sessions, n_trials).

    Returns:
        JAX array of per-session losses, shape (n_sessions,).
    """
    return jnp.mean((data["velocity"] - model_output) ** 2, axis=-1)
