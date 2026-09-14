import numpy as np
import jax.numpy as jnp

H = 1.0  # integration step; time is unitless (dt=1), so tau values are in time bins


# Wilson-Cowan model WITH slow inhibition (WCS).
#
# Adds a hidden slow-inhibition variable S (from TRN activity) that integrates the
# inhibitory activity and feeds back onto both populations. Under teacher forcing
# I_prev is the observed I, so S is recovered by carrying it through the scan;
# apply_model seeds the carry from `INITIAL_STATE` below. E and I use the PREVIOUS
# S (S_prev), matching the generator in data_loader/simulate_data.py.
def model(state_prev, y_prev, params):
    """Wilson-Cowan-with-slow-inhibition dynamics update.

    Args
    ----
    state_prev : dict with the previous hidden state {'S': S_prev}.
    y_prev : dict {'E_prev','I_prev','stim_E_prev','stim_I_prev'} — the previous
        observation bundled with the previous stimulus.
    params : dict of model parameters (adds tau_S, W_ES, W_IS over the base WC set).
    """
    E_max = params['E_max']
    I_max = params['I_max']
    W_EE = params['W_EE']
    W_EI = params['W_EI']
    W_IE = params['W_IE']
    W_II = params['W_II']
    W_ES = params['W_ES']  # slow inhibition onto excitatory population
    W_IS = params['W_IS']  # slow inhibition onto inhibitory population

    tau_E = params['tau_E']
    tau_I = params['tau_I']
    tau_S = params['tau_S']

    C_E = params['C_E']  # constant input to the excitatory population
    C_I = params['C_I']  # constant input to the inhibitory population

    XE = params['XE']  # transient input to the excitatory population
    XI = params['XI']  # transient input to the inhibitory population

    E_prev = y_prev['E_prev']
    I_prev = y_prev['I_prev']
    stim_E_prev = y_prev['stim_E_prev']
    stim_I_prev = y_prev['stim_I_prev']

    S_prev = state_prev['S']

    S_dot = (-S_prev + I_prev) / tau_S
    S = S_prev + H * S_dot

    E_dot = -E_prev + (E_max - E_prev) * np.maximum((W_EE * E_prev - W_EI * I_prev - W_ES * S_prev + C_E + XE * stim_E_prev), 0)
    E_dot /= tau_E
    I_dot = -I_prev + (I_max - I_prev) * np.maximum((W_IE * E_prev - W_II * I_prev - W_IS * S_prev + C_I + XI * stim_I_prev), 0)
    I_dot /= tau_I
    E = E_prev + H * E_dot
    I = I_prev + H * I_dot

    new_state = {'S': S}
    return new_state, (E, I)


model.DEFAULT_PARAMS = {
    'tau_E': 300.,  # time constant for excitatory population
    'tau_I': 200.,  # time constant for inhibitory population
    'W_EE': 0.01,   # weight of excitatory to excitatory connections
    'W_IE': 0.01,   # weight of inhibitory to excitatory connections
    'W_EI': 0.01,   # weight of excitatory to inhibitory connections
    'W_II': 0.01,   # weight of inhibitory to inhibitory connections
    'E_max': 20.0,
    'I_max': 20.0,
    'C_E': 0.001,
    'C_I': 0.001,
    'tau_S': 300,
    'W_ES': 0.001,
    'W_IS': 0.001,
    'XE': 1.0,
    'XI': 1.0,
    'log_noise_coef': -4.6052,  # log(0.01): fitted obs-noise coef, var = exp(·)·max(mean, EPS_MEAN)
}

# Initial hidden state seeded into the scan carry by apply_model.
model.INITIAL_STATE = {'S': 1.0}


def model_jax(state_prev, y_prev, params):
    '''JAX version of the WCS dynamics update (see `model`).'''
    E_max = params['E_max']
    I_max = params['I_max']
    W_EE = params['W_EE']
    W_EI = params['W_EI']
    W_IE = params['W_IE']
    W_II = params['W_II']
    W_ES = params['W_ES']
    W_IS = params['W_IS']

    tau_E = params['tau_E']
    tau_I = params['tau_I']
    tau_S = params['tau_S']

    C_E = params['C_E']
    C_I = params['C_I']

    XE = params['XE']
    XI = params['XI']

    E_prev = y_prev['E_prev']
    I_prev = y_prev['I_prev']
    stim_E_prev = y_prev['stim_E_prev']
    stim_I_prev = y_prev['stim_I_prev']

    S_prev = state_prev['S']

    S_dot = (-S_prev + I_prev) / tau_S
    S = S_prev + H * S_dot

    E_dot = -E_prev + (E_max - E_prev) * jnp.maximum((W_EE * E_prev - W_EI * I_prev - W_ES * S_prev + C_E + XE * stim_E_prev), 0)
    E_dot /= tau_E
    I_dot = -I_prev + (I_max - I_prev) * jnp.maximum((W_IE * E_prev - W_II * I_prev - W_IS * S_prev + C_I + XI * stim_I_prev), 0)
    I_dot /= tau_I
    E = E_prev + H * E_dot
    I = I_prev + H * I_dot

    new_state = {'S': S}
    return new_state, (E, I)


model_jax.DEFAULT_PARAMS = model.DEFAULT_PARAMS
model_jax.INITIAL_STATE = model.INITIAL_STATE
