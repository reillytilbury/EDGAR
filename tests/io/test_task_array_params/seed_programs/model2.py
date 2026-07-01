import numpy as np


def model(data, params):
    """y = a * x"""
    x = data["x"]
    a = params["a"]
    return a * x


# unbatched data is passed (first sample)
model.DEFAULT_PARAMS = lambda data: {"a": np.ones(data["x"].shape[0])}
