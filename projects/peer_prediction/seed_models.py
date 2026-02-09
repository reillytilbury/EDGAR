import numpy as np


def population_model_pls(stimuli, A):
    """
    Linear mapping from source population activity to a target population.
    Assumes stimuli shape is (time, n_source) and A shape is (n_source, n_target).
    """
    X = np.asarray(stimuli)
    return X @ np.asarray(A)


def parameter_estimator_pls(stimuli, spike_counts, n_components: int = 16):
    """
    Estimate linear weights using PLS regression for a target population.
    Assumes stimuli shape is (time, n_source) and spike_counts shape is (time, n_target).
    Returns A with shape (n_source, n_target).
    """
    from sklearn.cross_decomposition import PLSRegression

    X = np.asarray(stimuli)
    y = np.asarray(spike_counts)
    n_components = int(max(1, min(n_components, X.shape[0] - 1, X.shape[1])))
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X, y)
    A = pls.coef_.T
    return A.reshape(-1)


def population_model_pls_quadratic(stimuli, A, quad_a, quad_b, quad_c):
    """
    Linear PLS map followed by a target-specific quadratic nonlinearity.
    Assumes stimuli shape is (time, n_source).
    A shape: (n_source, n_target); quad_a/b/c are length n_target.
    """
    X = np.asarray(stimuli)
    lin = X @ np.asarray(A)
    quad_a = np.asarray(quad_a)
    quad_b = np.asarray(quad_b)
    quad_c = np.asarray(quad_c)
    return quad_a * lin ** 2 + quad_b * lin + quad_c


def parameter_estimator_pls_quadratic(stimuli, spike_counts, n_components: int = 16):
    """
    Estimate PLS weights and target-specific quadratic mappings.
    Assumes stimuli shape is (time, n_source) and spike_counts shape is (time, n_target).
    Returns (A, quad_a, quad_b, quad_c) with A shape (n_source, n_target).
    """
    from sklearn.cross_decomposition import PLSRegression

    X = np.asarray(stimuli)
    y = np.asarray(spike_counts)
    n_components = int(max(1, min(n_components, X.shape[0] - 1, X.shape[1])))
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X, y)
    A = pls.coef_.T
    lin = X @ A
    quad_coeffs = [np.polyfit(lin[:, i], y[:, i], deg=2) for i in range(y.shape[1])]
    quad = np.asarray(quad_coeffs)  # (n_target, 3)
    return A, quad[:, 0], quad[:, 1], quad[:, 2]
