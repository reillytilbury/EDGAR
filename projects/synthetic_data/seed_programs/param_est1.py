import numpy as np


def parameter_estimator(data):
    """
    Estimate parameters for the ReLU model using a simple grid search.

    Args:
        data (dict): Data dict for one sample with keys 'x' and 'y'.

    Returns:
        dict: Estimated parameters with keys {"a", "b"}.
    """
    x = data['x']
    y = np.asarray(data['y'])

    a_values = np.linspace(0.1, 5.0, 20)
    b_values = np.linspace(-1.0, 1.0, 20)

    best_loss = float("inf")
    best_params = (1.0, 0.0)

    for a in a_values:
        for b in b_values:
            y_pred = a * np.maximum(0, x - b)
            loss = np.mean((y - y_pred) ** 2)
            if loss < best_loss:
                best_loss = loss
                best_params = (a, b)
    a = best_params[0]
    b = best_params[1]

    return {"a": a.astype(float), "b": b.astype(float)}
