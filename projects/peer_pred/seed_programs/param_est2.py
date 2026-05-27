import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge


def parameter_estimator(data):
    """
    PLS-Ridge + quadratic estimator for the nonlinear peer-prediction model.

    Stage 1: fit linear weights A via PLS-Ridge.
    Stage 2: fit per-cell quadratic coefficients via least squares.

    data keys: 'source' (n_source, T), 'target' (n_target, T)

    Returns:
        dict: {"A": (n_target, n_source), "quadratic": (n_target, 3)}
    """
    X = np.asarray(data["source"])
    Y = np.asarray(data["target"])

    N_COMPONENTS = 16
    ALPHA = 100.0
    MAX_ITER = 500

    pls = PLSRegression(n_components=N_COMPONENTS, scale=False, max_iter=MAX_ITER, tol=1e-5)
    Z = pls.fit_transform(X.T, Y.T)[0]

    ridge = Ridge(alpha=ALPHA, fit_intercept=False)
    ridge.fit(Z, Y.T)
    A = ridge.coef_ @ pls.x_rotations_.T

    Y_pred_linear = A @ X
    n_target = Y.shape[0]
    quadratic_coeffs = np.zeros((n_target, 3))
    for c in range(n_target):
        X_c = np.stack([np.ones_like(Y_pred_linear[c]), Y_pred_linear[c], Y_pred_linear[c] ** 2], axis=1)
        coeffs, _, _, _ = np.linalg.lstsq(X_c, Y[c], rcond=None)
        quadratic_coeffs[c] = coeffs

    return {"A": A, "quadratic": quadratic_coeffs}
