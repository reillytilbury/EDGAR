import numpy as np


def parameter_estimator(data):
    """
    Estimate parameters for the linear model using least squares.

    Args:
        data (dict): Data dict for one sample with keys 'x' and 'y'.

    Returns:
        dict: Estimated parameters with keys {"a", "b"}.
    """
    x = data['x']
    y = np.asarray(data['y'])

    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return {"a": float(a), "b": float(b)}
