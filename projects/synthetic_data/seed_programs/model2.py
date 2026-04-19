import numpy as np


def model(data, params):
    """
    A simple linear model: y = a*x + b

    Data keys used:
        data['x']  # scalar input, shape (n_trials,)

    Args:
        data (dict): Data dict for one sample with key 'x', shape (n_trials,).
        params (dict): Parameter dictionary with keys:
            - a: Linear slope.
            - b: Linear intercept.

    Returns:
        np.ndarray: Predicted output, shape (n_trials,).
    """
    x = data['x']
    a = params["a"]
    b = params["b"]
    return a * x + b


model.DEFAULT_PARAMS = {"a": 1.0, "b": 0.0}
