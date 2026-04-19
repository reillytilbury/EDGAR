import numpy as np


def model(data, params):
    """
    Double-peaked Gaussian orientation tuning curve.
    Peaks at theta_pref and (theta_pref + pi) % 2pi.

    data['stimulus'] = theta  # stimulus angle (radians), shape (n_trials,)

    params:
        theta_pref: Preferred angle in radians.
        baseline: Baseline firing rate.
        amplitude_1: Amplitude of the first peak.
        amplitude_2: Amplitude of the second peak.
        tuning_width: Width of both peaks.

    Returns:
        np.ndarray: Response, shape (n_trials,).
    """
    theta = data['stimulus']
    theta_pref = np.clip(params["theta_pref"], 0, 2 * np.pi)
    baseline = np.clip(params["baseline"], 0, None)
    amplitude_1 = np.clip(params["amplitude_1"], 0, None)
    amplitude_2 = np.clip(params["amplitude_2"], 0, None)
    tuning_width = np.clip(params["tuning_width"], 0.01, None)

    circ_dist = lambda t1, t2: np.abs(np.arctan2(np.sin(t1 - t2), np.cos(t1 - t2)))
    dist_1 = circ_dist(theta, theta_pref)
    dist_2 = circ_dist(theta, (theta_pref + np.pi) % (2 * np.pi))
    return (baseline
            + amplitude_1 * np.exp(-0.5 * (dist_1 / tuning_width) ** 2)
            + amplitude_2 * np.exp(-0.5 * (dist_2 / tuning_width) ** 2))


model.DEFAULT_PARAMS = {
    "theta_pref": 0.0,
    "baseline": 0.0,
    "amplitude_1": 1.0,
    "amplitude_2": 0.0,
    "tuning_width": 1.0,
}
