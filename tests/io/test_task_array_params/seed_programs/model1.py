import numpy as np


def model(data, params):
    """y = a * x + b"""
    x = data["x"]
    a = params["a"]
    b = params["b"]
    return a * x + b


model.DEFAULT_PARAMS = lambda data: {
    "a": np.ones(data["x"].shape[0]),
    "b": np.zeros(1),
}
