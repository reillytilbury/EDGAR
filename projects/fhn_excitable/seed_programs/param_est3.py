import numpy as np


def parameter_estimator(data):
    """Damped SHO: seed omega from dominant FFT peak, sigma from residuals.

    ω = 2π * (peak_bin / T) / dt (rad per unit time).
    """
    y = np.asarray(data["y"], dtype=np.float64)
    y = y - y.mean()
    T = y.shape[0]
    dt = 0.05                                        # must match model3's dt
    if T < 8:
        omega_est = 1.0
    else:
        spectrum = np.abs(np.fft.rfft(y))
        spectrum[0] = 0.0
        k_peak = int(np.argmax(spectrum))
        omega_est = max(2.0 * np.pi * k_peak / (T * dt), 0.05)
    residual_std = max(float(np.diff(y).std()), 1e-3)
    return {
        "omega": float(omega_est),
        "damping": 0.1,
        "dt": 0.05,
        "k_gain": 0.3,
        "log_sigma_obs": float(np.log(residual_std)),
        "s0_x": float(y[0]),
        "s0_v": 0.0,
    }
