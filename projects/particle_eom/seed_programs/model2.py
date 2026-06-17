import numpy as np


def model(data, params):
    """
    Power-law repulsion from every other cell in the session, with no locality
    cutoff (volume-exclusion / "particles can't overlap" term).

    data['neighbor_dx']: shape (n_trials, n_neighbors), neighbor_dx[t, j] =
        x_j(t) - x_i(t) for every other cell j, at the (cell, time) trial t.

    params:
        c: Repulsion strength.

    Returns:
        np.ndarray: Predicted dx_i/dt, shape (n_trials,).
    """
    c = params["c"]
    dx = data["neighbor_dx"]
    r = np.clip(np.abs(dx), 1e-6, None)
    return c * np.sum(-np.sign(dx) / r**2, axis=-1)


model.DEFAULT_PARAMS = {
    "c": 0.1,
}
