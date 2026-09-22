import numpy as np


def parameter_estimator(data):
    """Baseline/amplitude heuristic + small positive coupling weights.

    Same resting-drive and stimulus-gain estimates as the independent model;
    the recurrent weights start small and positive and are refined by gradient
    descent (their sign convention is fixed inside the model).
    """
    E = np.asarray(data["E"], dtype=np.float64)   # (n_stim, T)
    I = np.asarray(data["I"], dtype=np.float64)
    b = max(1, E.shape[1] // 10)                  # pre-stimulus window

    E_base = float(E[:, :b].mean())
    I_base = float(I[:, :b].mean())
    E_amp = max(float(E.max() - E_base), 0.1)
    I_amp = max(float(I.max() - I_base), 0.1)

    return {
        "tau_E": 60.0,
        "tau_I": 120.0,
        "W_EE": 0.02,
        "W_EI": 0.01,
        "W_IE": 0.02,
        "W_II": 0.01,
        "C_E": E_base,
        "C_I": I_base,
        "XE": E_amp,
        "XI": I_amp,
        # Obs-noise coef init log(0.01) (matches the generative var = 0.01*mean); GD refines.
        "log_noise_coef": float(np.log(0.01)),
    }
