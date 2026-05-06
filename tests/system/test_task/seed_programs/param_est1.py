import numpy as np


def parameter_estimator(data):
    """Grid search parameter estimator for the ReLU model."""
    x = data['x']
    y = np.asarray(data['y'])

    best_loss, best_params = float("inf"), (1.0, 0.0)
    for a in np.linspace(0.1, 5.0, 20):
        for b in np.linspace(-1.0, 1.0, 20):
            loss = np.mean((y - a * np.maximum(0, x - b)) ** 2)
            if loss < best_loss:
                best_loss, best_params = loss, (a, b)

    return {"a": float(best_params[0]), "b": float(best_params[1])}
