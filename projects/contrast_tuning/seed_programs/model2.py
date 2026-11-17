import numpy as np


def model(data, params):
    """
    Double-peaked Gaussian direction tuning with Naka-Rushton contrast gain.
    Peaks at theta_pref and (theta_pref + pi) % 2pi; both peaks share the same contrast gain.

    data['stimulus'] = theta     # stimulus direction (radians), shape (n_trials,)
    data['contrast'] = c         # stimulus contrast (fraction, 0-1), shape (n_trials,)

    params:
        theta_pref: Preferred direction in radians.
        baseline: Baseline response.
        amplitude_1: Saturated amplitude of the first peak.
        amplitude_2: Saturated amplitude of the second peak.
        tuning_width: Width of both peaks.
        c50: Semi-saturation contrast.
        n: Naka-Rushton exponent.

    Returns:
        np.ndarray: Response, shape (n_trials,).
    """
    theta = data['stimulus']
    c = data['contrast']
    theta_pref = np.clip(params["theta_pref"], 0, 2 * np.pi)
    baseline = np.clip(params["baseline"], 0, None)
    amplitude_1 = np.clip(params["amplitude_1"], 0, None)
    amplitude_2 = np.clip(params["amplitude_2"], 0, None)
    tuning_width = np.clip(params["tuning_width"], 0.01, None)
    c50 = np.clip(params["c50"], 0.01, 2.0)
    n = np.clip(params["n"], 0.5, 6.0)

    gain = c ** n / (c ** n + c50 ** n)
    circ_dist = lambda t1, t2: np.abs(np.arctan2(np.sin(t1 - t2), np.cos(t1 - t2)))
    dist_1 = circ_dist(theta, theta_pref)
    dist_2 = circ_dist(theta, (theta_pref + np.pi) % (2 * np.pi))
    return baseline + gain * (
        amplitude_1 * np.exp(-0.5 * (dist_1 / tuning_width) ** 2)
        + amplitude_2 * np.exp(-0.5 * (dist_2 / tuning_width) ** 2)
    )


model.DEFAULT_PARAMS = {
    "theta_pref": 0.0,
    "baseline": 0.0,
    "amplitude_1": 1.0,
    "amplitude_2": 0.0,
    "tuning_width": 1.0,
    "c50": 0.3,
    "n": 2.0,
}
