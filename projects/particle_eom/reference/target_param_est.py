import numpy as np


def parameter_estimator(data):
    """
    Heuristic initial guess for (A, B, r_c), pooled across every cell and every
    training-window timestep within this session.

    r_c is estimated heuristically from the typical nearest-neighbor spacing (a rough
    guess only — gradient descent refines it). Given that r_c guess, A and B enter the
    model linearly within the cutoff mask, so they are solved with a 2-feature linear
    least-squares fit (normal equations), not a nonlinear optimizer.

    Args:
        data (dict): data['neighbor_dx'] shape (n_trials, n_neighbors),
                     data['velocity'] shape (n_trials,) (the noisy dx_i/dt target).

    Returns:
        dict: {"A", "B", "r_c"}
    """
    dx = data["neighbor_dx"]
    target = data["velocity"]
    r = np.abs(dx)

    nearest_gap = np.min(np.where(r > 0, r, np.inf), axis=-1)
    finite_gaps = nearest_gap[np.isfinite(nearest_gap)]
    r_c = float(2.0 * np.median(finite_gaps)) if finite_gaps.size else 2.0

    mask = r < r_c
    r_safe = np.clip(r, 1e-6, None)
    feature_A = np.sum(np.where(mask, -np.sign(dx) / r_safe**2, 0.0), axis=-1)
    feature_B = np.sum(np.where(mask, dx, 0.0), axis=-1)

    X = np.stack([feature_A, feature_B], axis=-1)  # (n_trials, 2)
    gram = X.T @ X
    rhs = X.T @ target
    if np.linalg.det(gram) > 1e-12:
        A, B = np.linalg.solve(gram, rhs)
    else:
        A, B = 0.0, 0.0

    return {"A": float(A), "B": float(B), "r_c": r_c}
