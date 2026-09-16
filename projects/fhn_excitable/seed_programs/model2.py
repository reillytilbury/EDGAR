import numpy as np


def model(state, y_prev, params):
    """AR(1) + slow moving average.

    Blends the previous observation with a slow low-pass estimate of the
    signal's local mean. Slightly better than persistence for signals with a
    slowly varying baseline, but still cannot represent the fast-slow spike
    structure of an excitable system.
    """
    alpha = params["alpha"]
    beta = params["beta"]                                # 0 < beta << 1 → slow MA

    y_last = y_prev
    ma_new = (1.0 - beta) * state["ma"] + beta * y_prev

    new_state = {"y_last": y_last, "ma": ma_new}
    mean = alpha * y_last + (1.0 - alpha) * ma_new
    return new_state, mean


model.DEFAULT_PARAMS = {
    "alpha": 0.9,
    "beta": 0.05,
    "log_sigma_obs": 0.0,
    "s0_y_last": 0.0,
    "s0_ma": 0.0,
}
