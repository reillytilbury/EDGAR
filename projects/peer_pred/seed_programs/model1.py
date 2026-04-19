import numpy as np


def model(data, params):
    """
    Linear peer-prediction model.

    Y[c, t] = sum_s A[c, s] * source[s, t]

    data keys: 'source'  # shape (n_source_cells, n_time)
    params: A  # weight matrix, shape (n_target_cells, n_source_cells)

    Returns:
        np.ndarray: Predicted target activity, shape (n_target_cells, n_time).
    """
    return params["A"] @ data["source"]


model.DEFAULT_PARAMS = {"A": np.zeros((1, 1))}
