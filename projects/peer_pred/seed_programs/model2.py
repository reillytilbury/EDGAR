import numpy as np


def model(data, params):
    """
    Linear peer-prediction with cell-specific quadratic nonlinearity.

    Y[c, t] = a0_c + a1_c * (A[c, :] @ source[:, t]) + a2_c * (A[c, :] @ source[:, t])^2

    data keys: 'source'  # shape (n_source_cells, n_time)
    params:
        A: weight matrix, shape (n_target_cells, n_source_cells)
        quadratic: coefficients, shape (n_target_cells, 3)

    Returns:
        np.ndarray: Predicted target activity, shape (n_target_cells, n_time).
    """
    A = params["A"]
    quadratic_coeffs = params["quadratic"]

    intercept = quadratic_coeffs[:, 0:1]
    linear_coef = quadratic_coeffs[:, 1:2]
    quadratic_coef = quadratic_coeffs[:, 2:3]

    Y_linear = A @ data["source"]
    return intercept + linear_coef * Y_linear + quadratic_coef * (Y_linear ** 2)


model.DEFAULT_PARAMS = {"A": np.zeros((1, 1)), "quadratic": np.zeros((1, 3))}
