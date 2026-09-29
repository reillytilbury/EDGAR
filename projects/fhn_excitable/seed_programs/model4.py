import numpy as np


def model(state, y_prev, params):
    """Damped SHO with observation-driven frequency tracking.

    Extends seed 3 with a running frequency estimate updated from each new
    observation. Still a purely linear oscillator — no slow gating variable,
    no threshold, no way to represent refractoriness. This is the strongest
    "seed floor" that any FHN-like evolved program must clearly beat before
    we can claim excitability was discovered.
    """
    x = state["x"]
    v = state["v"]
    freq_est = state["freq_est"]

    damping = params["damping"]
    dt = params["dt"]
    k_x = params["k_x"]
    k_v = params["k_v"]
    k_freq = params["k_freq"]

    innovation = y_prev - x
    x_corr = x + k_x * innovation
    v_corr = v + k_v * innovation
    freq_corr = freq_est + k_freq * innovation * v_corr

    omega = freq_corr
    acc = -2.0 * damping * omega * v_corr - omega * omega * x_corr
    v_new = v_corr + dt * acc
    x_new = x_corr + dt * v_new

    new_state = {"x": x_new, "v": v_new, "freq_est": freq_corr}
    mean = x_new
    return new_state, mean


model.DEFAULT_PARAMS = {
    "damping": 0.1,
    "dt": 0.05,
    "k_x": 0.3,
    "k_v": 0.1,
    "k_freq": 0.02,
    "log_sigma_obs": -1.0,
    "s0_x": 0.0,
    "s0_v": 0.0,
    "s0_freq_est": 1.0,
}
