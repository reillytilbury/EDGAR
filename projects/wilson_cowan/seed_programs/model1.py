import numpy as np

H = 1.0  # integration step (matches the data-generating grid; time is unitless, dt=1)


def model(state, y_prev, params):
    """Leaky E/I integrators with a latent inhibition variable. 

    Each population relaxes toward a constant baseline drive with its own time
    constant.

    Equation : 
        tau_E * dE/dt = -E + C_E + XE * stim_E - W_ES * S 
        tau_I * dI/dt = -I + C_I + XI * stim_I - W_IS * S
    where 
        C_E and C_I : constant baseline drive to the excitatory and inhibitory populations, respectively,
        stim_E and stim_I : binary variable that is 1 when the stimulus is present and 0 otherwise,
        XE and XI : transient input strengths to the excitatory and inhibitory populations, respectively,
        W_ES and W_IS : weights of the slow inhibition S onto the excitatory and inhibitory populations, respectively.

        state contains one latent variable S, which is a leaky integrator for the difference between the excitatory and inhibitory populations,
        and follows the dynamics: 
           tau_S * dS/dt = -S + (E - I)
    """
    E_prev = y_prev["E_prev"]
    I_prev = y_prev["I_prev"]
    stim_E_prev = y_prev["stim_E_prev"] # 0 if no stimulus, 1 if stimulus present
    stim_I_prev = y_prev["stim_I_prev"] # 0 if no stimulus, 1 if stimulus present

    E_transient_input_strength = params["XE"] * stim_E_prev
    I_transient_input_strength = params["XI"] * stim_I_prev

    S_prev = state["S"]  # latent variable
    S_dot = (-S_prev + (E_prev - I_prev)) / params["tau_S"]
    S = S_prev + H * S_dot

    drive_E = params["C_E"] + E_transient_input_strength - params["W_ES"] * S_prev
    drive_I = params["C_I"] + I_transient_input_strength - params["W_IS"] * S_prev

    E_dot = (-E_prev + drive_E) / params["tau_E"]
    I_dot = (-I_prev + drive_I) / params["tau_I"]

    E = E_prev + H * E_dot
    I = I_prev + H * I_dot

    return {"S": S}, (E, I)

model.INITIAL_STATE = {'S': 1.0}

model.DEFAULT_PARAMS = {
    "tau_E": 60.0,
    "tau_I": 120.0,
    "C_E": 1.0,
    "C_I": 1.0,
    "XE": 3.0,
    "XI": 1.0,
    "tau_S": 300.0,
    "W_ES": 0.5,
    "W_IS": 0.5,
    "log_noise_coef": -4.6052,  # log(0.01): fitted obs-noise coef, var = exp(·)·max(mean, EPS_MEAN)
}
