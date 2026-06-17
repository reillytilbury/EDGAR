import numpy as np


def parameter_estimator(data):
    """
    Closed-form OLS estimate of the repulsion strength c, pooled across every cell
    and every training-window timestep within this session.

    Args:
        data (dict): data['neighbor_dx'] shape (n_trials, n_neighbors),
                     data['velocity'] shape (n_trials,) (the noisy dx_i/dt target).

    Returns:
        dict: {"c": float}
    dx = data["neighbor_dx"]
    r = np.clip(np.abs(dx), 1e-6, None)
    feature = np.sum(-np.sign(dx) / r**2, axis=-1)
    target = data["velocity"]
    denom = np.sum(feature**2)
    c = float(np.sum(feature * target) / denom) if denom > 1e-12 else 0.0
    return {"c": c}
