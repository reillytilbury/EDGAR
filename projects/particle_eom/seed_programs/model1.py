import numpy as np


def model(data, params):
    """
    Linear spring pulling this cell toward every other cell in the session, with no
    locality cutoff (graph-Laplacian / consensus term).

    data['neighbor_dx']: shape (n_trials, n_neighbors), neighbor_dx[t, j] =
        x_j(t) - x_i(t) for every other cell j, at the (cell, time) trial t.

    params:
        k: Spring constant.

    Returns:
        np.ndarray: Predicted dx_i/dt, shape (n_trials,).
    """
    k = params["k"]
    return k * np.sum(data["neighbor_dx"], axis=-1)


model.DEFAULT_PARAMS = {
    "k": 0.1,
}
