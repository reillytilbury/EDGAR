import numpy as np


def model(data, params):
    """
    Gaussian direction tuning with Naka-Rushton contrast gain on the amplitude.

    data['stimulus'] = theta     # stimulus direction (radians), shape (n_trials,)
    data['contrast'] = c         # stimulus contrast (fraction, 0-1), shape (n_trials,)

    params:
        theta_pref: Preferred direction.
        baseline: Baseline response.
        amplitude: Saturated response above baseline (Rmax).
        tuning_width: Width of the tuning curve.
        c50: Semi-saturation contrast.
        n: Naka-Rushton exponent.

    Returns:
        np.ndarray: Response, shape (n_trials,).
    """
    theta = data['stimulus']
    c = data['contrast']
    theta_pref = np.clip(params["theta_pref"], 0, 2 * np.pi)
    baseline = np.clip(params["baseline"], 0, None)
    amplitude = np.clip(params["amplitude"], 0, None)
    tuning_width = np.clip(params["tuning_width"], 0.01, None)
    c50 = np.clip(params["c50"], 0.01, 2.0)
    n = np.clip(params["n"], 0.5, 6.0)

    gain = c ** n / (c ** n + c50 ** n)
    dist = np.abs(np.arctan2(np.sin(theta - theta_pref), np.cos(theta - theta_pref)))
    return baseline + amplitude * gain * np.exp(-0.5 * (dist / tuning_width) ** 2)


model.DEFAULT_PARAMS = {
    "theta_pref": 0.0,
    "baseline": 0.0,
    "amplitude": 1.0,
    "tuning_width": 1.0,
    "c50": 0.3,
    "n": 2.0,
}
