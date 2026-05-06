import numpy as np


def parameter_estimator(data):
    """Least squares parameter estimator for the linear model."""
    x = data['x']
    y = np.asarray(data['y'])

    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return {"a": float(a), "b": float(b)}
