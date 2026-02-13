import numpy as np

def model_v1(x, a, b):
    """
    model version 1 of the form
    f(x) = a * x + b
    Args:
        x (n_trials,): input data
        a (float): parameter a
        b (float): parameter b
    Returns:
        y (n_trials,): output of the model
    """
    return a * x + b

def parameter_estimator_v1(x, y):
    """
    parameter estimator for model version 1 using least squares regression
    Args:
        x (n_trials,): input data
        y (n_trials,): output data
    Returns:
        a (float): estimated parameter a
        b (float): estimated parameter b
    """
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b

def model_v2(x, a, b, lam):
    """
    model version 2 of the form

    """
    return (a * x + b) * np.exp(lam * x)

def parameter_estimator_v2(x, y):
    """
    parameter estimator for model version 2 using least squares regression with regularization
    Args:
        x (n_trials,): input data
        y (n_trials,): output data
    Returns:
        a (float): estimated parameter a
        b (float): estimated parameter b
        lam (float): estimated parameter lam
    """
    # Near x=0, approximate exponential term as 1 + lam * x, so model can be approximated as
    # f(x) = (a * x + b) * (1 + lam * x) = a * x + b + (a * lam * x^2 + b * lam * x) = (a * lam) * x^2 + (a + b * lam) * x + b
    A = np.vstack([x, x**2, np.ones(len(x))]).T
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    b = coeffs[2]
    a_plus_b_lam = coeffs[0]
    a_lam = coeffs[1]
    # we can solve for a and lam using the relationships above
    a = a_lam / (a_plus_b_lam - b * a_lam / a_lam)
    lam = a_lam / a
    return a, b, lam