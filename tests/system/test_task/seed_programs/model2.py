import numpy as np


def model(data, params):
    """y = a * x + b"""
    x = data['x']
    a = params["a"]
    b = params["b"]
    return a * x + b


model.DEFAULT_PARAMS = {"a": float(1.0), "b": float(0.0)}
