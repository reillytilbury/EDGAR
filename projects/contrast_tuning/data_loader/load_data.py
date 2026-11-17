from __future__ import annotations

import os

import numpy as np
import jax.numpy as jnp


def load_data(
    data_path: str,
    sessions: list[str] | None = None,
    n_reps: int | None = None,
    sf_threshold: float = 0.4,
    conc_threshold: float = 0.4,
    filter_mode: str = "pref",
    shuffle_phases: bool = True,
    shuffle_seed: int = 0,
    random_seed: int = 42,
    n_eval_samples: int = 100,
):
    """
    Load and preprocess the direction x contrast tuning data (kaesha_contrasts).

    data_path is either one session file (.npz) or the dataset directory, in which case
    `sessions` lists the session files inside it to pool. Each raw file holds responses of
    shape (cells, directions, contrasts, phases, repeats). Every session is preprocessed on
    its own, following figures/exploratory_contrasts.ipynb (EDGAR-main):
      0. Keep only the first n_reps repeats (e.g. 4, to drop the 5th repeat of exp11-10 so
         every session has the same trial layout).
      1. Optionally shuffle start phases across repeats with an independent random Latin
         rectangle per (direction, contrast), then average over phases. This turns the
         repeats into pseudo-repeats.
      2. At every contrast, compute the signal fraction (Sahani & Linden) and the
         doubled-angle circular concentration. With filter_mode='pref', score each cell
         at its max-SF contrast; with 'min', a cell must pass at every contrast.
      3. Divide each repeat by the L2 norm (over directions) of its own 100%-contrast
         curve, so the relative scale across contrasts is kept. Multiply by
         sqrt(n_dirs) so the 100% curve has RMS 1.
    The surviving cells of all sessions are then pooled.

    One sample is one cell. One trial is one (direction, contrast, repeat) entry.
    Repeats are split even -> train and odd -> test, the same even/odd convention the
    notebook uses.

    Returns
    -------
    X_discover, X_validate, X_eval
        Each split is a dict with keys
            "stimulus": direction in radians, shape (n_cells, n_trials)
            "contrast": Michelson contrast as a fraction (0.25, 0.5, 1.0), same shape
            "response": normalised response, same shape
        Cells are split 50/50 into discover / validate at random within each session, so
        both splits hold the same mix of sessions.
        X_eval is a fingerprint subset of the discover train trials.
    """
    if sessions:
        paths = [os.path.join(data_path, f) for f in sessions]
    else:
        paths = [data_path]

    Rs, session_id, theta, contrasts = [], [], None, None
    for k, path in enumerate(paths):
        R_k, theta_k, con_k = _load_session(
            path, n_reps, sf_threshold, conc_threshold, filter_mode,
            shuffle_phases, np.random.default_rng([shuffle_seed, k]),
        )
        if theta is None:
            theta, contrasts = theta_k, con_k
        elif not (np.allclose(theta, theta_k) and np.allclose(contrasts, con_k)):
            raise ValueError(f"{path}: directions/contrasts differ from {paths[0]}")
        Rs.append(R_k)
        session_id.append(np.full(R_k.shape[0], k))
    n_rep_set = {R_k.shape[-1] for R_k in Rs}
    if len(n_rep_set) > 1:
        raise ValueError(f"sessions have different repeat counts {n_rep_set}; set n_reps")
    R = np.concatenate(Rs, axis=0)  # cells x dirs x contrasts x reps
    session_id = np.concatenate(session_id)
    _, n_dirs, n_con, n_reps = R.shape

    # ── flatten (direction, contrast, repeat) into trials; even reps train, odd reps test ──
    stim_grid, con_grid = np.meshgrid(theta, contrasts / 100.0, indexing="ij")

    def _flatten(reps):
        n_r = len(reps)
        resp = R[..., reps].reshape(R.shape[0], -1)
        stim = np.broadcast_to(stim_grid[:, :, None], (n_dirs, n_con, n_r)).reshape(-1)
        con = np.broadcast_to(con_grid[:, :, None], (n_dirs, n_con, n_r)).reshape(-1)
        n = R.shape[0]
        return {
            "stimulus": np.broadcast_to(stim, (n, stim.size)).copy(),
            "contrast": np.broadcast_to(con, (n, con.size)).copy(),
            "response": resp.astype(np.float64),
        }

    train_all = _flatten(np.arange(0, n_reps, 2))
    test_all = _flatten(np.arange(1, n_reps, 2))

    # ── split cells 50/50 into discover / validate ──
    # (stratified by session, so both splits hold the same mix of sessions)
    rng = np.random.default_rng(random_seed)
    disc_idx, val_idx = [], []
    for k in np.unique(session_id):
        cells_k = rng.permutation(np.flatnonzero(session_id == k))
        disc_idx.append(cells_k[: len(cells_k) // 2])
        val_idx.append(cells_k[len(cells_k) // 2 :])
    disc_idx = np.sort(np.concatenate(disc_idx))
    val_idx = np.sort(np.concatenate(val_idx))

    def _take(split, idx):
        return {k: v[idx] for k, v in split.items()}

    X_disc_train, X_disc_test = _take(train_all, disc_idx), _take(test_all, disc_idx)
    X_val_train, X_val_test = _take(train_all, val_idx), _take(test_all, val_idx)

    # ── X_eval: small subset of discover train for fingerprinting ──
    eval_pos = np.sort(
        rng.choice(len(disc_idx), min(n_eval_samples, len(disc_idx)), replace=False)
    )
    X_eval = {k: v[eval_pos] for k, v in X_disc_train.items()}
    X_eval["_sample_indices"] = eval_pos

    return (X_disc_train, X_disc_test), (X_val_train, X_val_test), X_eval


def loss_fn(model_output, data):
    """Scaled squared error loss."""
    return 10 * jnp.mean((data["response"] - model_output) ** 2, axis=-1)


# ── internal helpers ──


def _load_session(path, n_reps, sf_threshold, conc_threshold, filter_mode, shuffle_phases, rng):
    """One session: phase-shuffle, phase-average, filter cells, normalise.
    Returns R (cells x dirs x contrasts x reps), theta (radians), contrasts (percent)."""
    d = np.load(path)
    R = d["responses"].astype(np.float32)  # cells x dirs x contrasts x phases x reps
    if n_reps is not None:
        if R.shape[-1] < n_reps:
            raise ValueError(f"{path} has {R.shape[-1]} repeats, fewer than n_reps={n_reps}")
        R = R[..., :n_reps]
    directions = np.asarray(d["directions"], dtype=np.float64)
    contrasts = np.asarray(d["contrasts"], dtype=np.float64)

    if shuffle_phases:
        R = _shuffle_phases(R, rng)
    R = R.mean(axis=3)  # cells x dirs x contrasts x reps

    theta = np.deg2rad(directions)
    n_cells_raw, n_dirs, n_con, _ = R.shape

    # ── cell filter: SF and concentration at every contrast ──
    sf_c = np.stack(
        [_unbiased_signal_fraction(R[:, :, j]) for j in range(n_con)], axis=1
    )
    mu = R.mean(-1)  # cells x dirs x contrasts
    conc_c = np.abs(
        (np.exp(2j * theta)[None, :, None] * mu).sum(1) / mu.sum(1)
    )  # cells x contrasts
    if filter_mode == "pref":
        best_c = sf_c.argmax(1)
        rows = np.arange(n_cells_raw)
        sf, conc = sf_c[rows, best_c], conc_c[rows, best_c]
    elif filter_mode == "min":
        sf, conc = sf_c.min(1), conc_c.min(1)
    else:
        raise ValueError(f"filter_mode must be 'pref' or 'min', got {filter_mode!r}")
    keep = (sf > sf_threshold) & (conc > conc_threshold)
    R = R[keep]

    # ── normalise each repeat by its own 100%-contrast L2 norm (RMS 1 at max contrast) ──
    i_max = int(np.argmax(contrasts))
    norm = np.linalg.norm(R[:, :, i_max], axis=1, keepdims=True)  # cells x 1 x reps
    R = np.sqrt(n_dirs) * R / np.maximum(norm, 1e-12)[:, :, None, :]
    return R, theta, contrasts


def _random_latin_rectangle(n_rows, n_cols, rng):
    """(n_rows, n_cols) array: each column a permutation of range(n_rows), no repeats in a row."""
    assert n_rows >= n_cols, "need at least as many repeats as phases"
    M = np.empty((n_rows, n_cols), dtype=int)
    for k in range(n_cols):
        while True:
            col = rng.permutation(n_rows)
            if not (col[:, None] == M[:, :k]).any():
                M[:, k] = col
                break
    return M


def _shuffle_phases(R, rng):
    """R: cells x dirs x contrasts x phases x reps. Reassign phases to pseudo-repeats with an
    independent random Latin rectangle per (direction, contrast)."""
    _, n_dirs, n_con, n_ph, n_rep = R.shape
    idx = np.empty((n_dirs, n_con, n_ph, n_rep), dtype=int)  # idx[d, c, k, r] = true repeat
    for di in range(n_dirs):
        for ci in range(n_con):
            idx[di, ci] = _random_latin_rectangle(n_rep, n_ph, rng).T
    return np.take_along_axis(R, idx[None], axis=-1)


def _unbiased_signal_fraction(R):
    """Sahani & Linden (2003) signal fraction. R: cells x dirs x reps -> (cells,)."""
    n_dirs, n_reps = R.shape[1], R.shape[2]
    mu = R.mean(-1)
    var = R.var(-1, ddof=1)
    term1 = ((mu - mu.mean(1, keepdims=True)) ** 2).mean(1)
    term2 = ((n_dirs - 1) / n_dirs**2) * np.sum(var / n_reps, axis=1)
    S2 = term1 - term2
    V2 = np.sum(var / n_reps, axis=1) / n_dirs
    return np.clip(S2 / (S2 + V2), 0, 1)
