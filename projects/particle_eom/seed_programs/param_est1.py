import numpy as np


def parameter_estimator(data):
    """
    Closed-form OLS estimate of the spring constant k, pooled across every cell and
    every training-window timestep within this session.

    Args:
        data (dict): data['neighbor_dx'] shape (n_trials, n_neighbors),
                     data['velocity'] shape (n_trials,) (the noisy dx_i/dt target).

    Returns:
        dict: {"k": float}
    feature = np.sum(data["neighbor_dx"], axis=-1)
    target = data["velocity"]
    denom = np.sum(feature**2)
    k = float(np.sum(feature * target) / denom) if denom > 1e-12 else 0.0
    return {"k": k}
