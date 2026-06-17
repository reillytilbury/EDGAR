import numpy as np


def model(data, params):
    """
    Overdamped pairwise force law: power-law repulsion competing with linear
    attraction, restricted to neighbors within a cutoff radius. This is the
    ground-truth structural class the evolutionary loop is meant to discover —
    kept here as a reference, not a seed, for sanity-checking the data/loss
    plumbing directly (see checklist item 12 in
    journal/2026-06-16_eom_discovery_poc_spec.md).

    data['neighbor_dx']: shape (n_trials, n_neighbors), neighbor_dx[t, j] =
        x_j(t) - x_i(t) for every other cell j, at the (cell, time) trial t.

    params:
        A: Repulsion strength.
        B: Attraction strength.
        r_c: Interaction cutoff radius.

    Returns:
        np.ndarray: Predicted dx_i/dt, shape (n_trials,).
    """
    A = params["A"]
    B = params["B"]
    r_c = np.clip(params["r_c"], 1e-6, None)
    dx = data["neighbor_dx"]  # x_j - x_i
    r = np.clip(np.abs(dx), 1e-6, None)
    within_cutoff = r < r_c
    repulsion = -A * np.sign(dx) / r**2  # sign(x_i - x_j) = -sign(dx)
    attraction = B * dx
    contribution = np.where(within_cutoff, repulsion + attraction, 0.0)
    return np.sum(contribution, axis=-1)


model.DEFAULT_PARAMS = {
    "A": 1.0,
    "B": 1.0,
    "r_c": 1.5,
}
