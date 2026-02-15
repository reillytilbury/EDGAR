import numpy as np

def model_v1(x, a, b):
    """
    model version 1 of the form
    f(x) = a * x + b
    Args:
        x (1, n_trials) or (n_trials,): input data
        a (float): parameter a
        b (float): parameter b
    Returns:
        y (n_trials,): output of the model
    """
    x_arr = np.asarray(x)
    x_1d = x_arr[0] if x_arr.ndim > 1 else x_arr
    return a * x_1d + b

def parameter_estimator_v1(x, y):
    """
    parameter estimator for model version 1 using least squares regression
    Args:
        x (1, n_trials): input data. x[0] is the sole input feature for the 1 dimensional feature problem
        y (n_trials,): output data
    Returns:
        a (float): estimated parameter a
        b (float): estimated parameter b
    """
    x = x[0]  # extract the 1D input feature from the input array
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return np.array([a, b])

def model_v2(x, a, b, lam):
    """
    model version 2 of the form

    """
    x_arr = np.asarray(x)
    x_1d = x_arr[0] if x_arr.ndim > 1 else x_arr
    exponential_term = np.clip(np.exp(lam * x_1d), None, 1e4)  # clip to prevent overflow
    y = (a * x_1d + b) * exponential_term
    return y

def parameter_estimator_v2(x, y):
    """
    parameter estimator for model version 2 using least squares regression with regularization
    Args:
        x (1, n_trials): input data. x[0] is the sole input feature for the 1 dimensional feature problem
        y (n_trials,): output data
    Returns:
        a (float): estimated parameter a
        b (float): estimated parameter b
        lam (float): estimated parameter lam
    """
    x = np.asarray(x)
    x = x[0] if x.ndim > 1 else x  # extract the 1D input feature from the input array
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
    return np.array([a, b, lam])
