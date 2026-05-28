import numpy as np


def model(data, params):
    """y = a * relu(x - b)"""
    x = data["x"]
    a = params["a"]
    b = params["b"]
    return a * np.maximum(0, x - b)


model.DEFAULT_PARAMS = {"a": float(1.0), "b": float(0.0)}
