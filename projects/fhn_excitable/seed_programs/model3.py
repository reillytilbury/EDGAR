import numpy as np


def model(state, y_prev, params):
    """Damped harmonic oscillator with Kalman-style observation correction.

    Latent state (x, v) evolves under a fixed-frequency damped oscillator; the
    innovation y_prev - x nudges x toward its most-likely value. This can track
    oscillatory rhythm but has no mechanism to represent the after-spike
    refractory period of an excitable system — the linear ODE has no threshold,
    no reset, no slow gating variable.
    """
    x = state["x"]
    v = state["v"]

    omega = params["omega"]
    damping = params["damping"]
    dt = params["dt"]
    k_gain = params["k_gain"]

    innovation = y_prev - x
    x_corr = x + k_gain * innovation
    v_corr = v

    acc = -2.0 * damping * omega * v_corr - omega * omega * x_corr
    v_new = v_corr + dt * acc
    x_new = x_corr + dt * v_new

    new_state = {"x": x_new, "v": v_new}
    mean = x_new
    return new_state, mean


model.DEFAULT_PARAMS = {
    "omega": 1.0,
    "damping": 0.1,
    "dt": 0.05,
    "k_gain": 0.3,
    "log_sigma_obs": -1.0,
    "s0_x": 0.0,
    "s0_v": 0.0,
}
