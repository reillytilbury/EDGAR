import numpy as np


def model(data, params):
    """y = a * x + b"""
    x = data['x']
    a = params["a"]
    b = params["b"]
    return a * x + b


model.DEFAULT_PARAMS = {"a": 1.0, "b": 0.0}
