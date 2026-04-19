import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge


def parameter_estimator(data):
    """
    PLS-Ridge estimator for the linear peer-prediction model.

    Solves A = Y X^T (X X^T + lambda I)^(-1) via PLS dimensionality reduction
    followed by Ridge regression.

    data keys: 'source' (n_source, T), 'target' (n_target, T)

    Returns:
        dict: {"A": weight matrix, shape (n_target, n_source)}
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
    return {"A": A}
