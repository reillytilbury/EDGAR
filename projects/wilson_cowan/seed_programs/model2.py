import numpy as np

H = 1.0  # integration step (matches the data-generating grid; time is unitless, dt=1)


def model(state, y_prev, params):
    """Linear coupled E/I circuit (recurrent push-pull).

    Adds the recurrent weight matrix to the leaky integrators: E excites both
    populations, I inhibits both. This linear coupling can produce damped
    oscillation and rebound, so we expect it to capture the shape of the evoked response.

    Equation :
        tau_E * dE/dt = -E + W_EE * E - W_EI * I + C_E + XE * stim_E
        tau_I * dI/dt = -I + W_IE * E - W_II * I + C_I + XI * stim_I
    where
        C_E and C_I : constant baseline drive to the excitatory and inhibitory populations, respectively
        stim_E and stim_I : binary variable that is 1 when the stimulus is present and 0 otherwise
        XE and XI : transient input strengths to the excitatory and inhibitory populations, respectively
        W_EE, W_EI, W_IE, W_II : recurrent weights

        This model is fully Markovian with no latent variables, so the state is empty.
    """
    E_prev = y_prev["E_prev"]
    I_prev = y_prev["I_prev"]
    stim_E_prev = y_prev["stim_E_prev"] # 0 if no stimulus, 1 if stimulus present
    stim_I_prev = y_prev["stim_I_prev"] # 0 if no stimulus, 1 if stimulus present

    E_transient_input_strength = params["XE"] * stim_E_prev
    I_transient_input_strength = params["XI"] * stim_I_prev

    drive_E = (params["W_EE"] * E_prev - params["W_EI"] * I_prev
               + params["C_E"] + E_transient_input_strength)
    drive_I = (params["W_IE"] * E_prev - params["W_II"] * I_prev
               + params["C_I"] + I_transient_input_strength)

    E_dot = (-E_prev + drive_E) / params["tau_E"]
    I_dot = (-I_prev + drive_I) / params["tau_I"]

    E = E_prev + H * E_dot
    I = I_prev + H * I_dot

    return {}, (E, I)


model.DEFAULT_PARAMS = {
    "tau_E": 60.0,
    "tau_I": 120.0,
    "W_EE": 0.02,
    "W_EI": 0.01,
    "W_IE": 0.02,
    "W_II": 0.01,
    "C_E": 1.0,
    "C_I": 1.0,
    "XE": 3.0,
    "XI": 1.0,
    "log_noise_coef": -4.6052,  # log(0.01): fitted obs-noise coef, var = exp(·)·max(mean, EPS_MEAN)
}
