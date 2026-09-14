import numpy as np
import jax.numpy as jnp

H = 1/30000


# Wilson-Cowan slow_inhibition model 
def model(state_prev, y_prev, params):
    ''' Wilson-Cowan model dynamics update function. 
    Args
    ----
    state: a dictionary representing the previous state of the system
    y_prev: tuple of ((E_prev, I_prev), (stim_E_prev, stim_I_prev)) representing the previous state and previous stimuli
    params: dictionary of model parameters

    '''
    E_max = params['E_max']
    I_max = params['I_max']
    W_EE = params['W_EE']
    W_EI = params['W_EI']
    W_IE = params['W_IE']
    W_II = params['W_II']
    W_ES = params['W_ES'] # slow inhibition from TRN activity
    W_IS = params['W_IS'] # slow inhibition from TRN activity

    tau_E = params['tau_E']
    tau_I = params['tau_I']
    tau_S = params['tau_S']

    C_E = params['C_E'] # constant input to the excitatory population
    C_I = params['C_I'] # constant input to the inhibitory population    

    XE = params['XE'] # transient input to the excitatary population 
    XI = params['XI'] # transient input to the inhibitory population
    
    ## p27 : inhibitory pulse is considered slower and is applied with an alpha function in the paper. For now use a boxcar function for both
    # tau_alpha = params['tau_alpha'] # time constant for alpha function for inhibitory input
    # XI_drive = XI * _alpha_drive(stimuli[:, 1], tau_alpha, h)    

    E_prev = y_prev['E_prev']
    I_prev = y_prev['I_prev']
    stim_E_prev = y_prev['stim_E_prev']
    stim_I_prev = y_prev['stim_I_prev']

    S_prev = state_prev['S']

    E_dot = -E_prev + (E_max - E_prev)* np.maximum((W_EE * E_prev - W_EI * I_prev - W_ES * S_prev + C_E + XE * stim_E_prev), 0)
    E_dot /= tau_E
    I_dot = -I_prev + (I_max - I_prev)* np.maximum((W_IE * E_prev - W_II * I_prev - W_IS * S_prev + C_I + XI * stim_I_prev), 0)
    I_dot /= tau_I
    E = E_prev + H * E_dot
    I = I_prev + H * I_dot

    S_dot = -S_prev + I_prev
    S_dot /= tau_S
    S = S_prev + h * S_dot
    new_state = {'S' : S}

    return new_state, (E, I)

model.DEFAULT_PARAMS = {
    'tau_E' : 0.001, # time constant for excitatory population
    'tau_I' : 0.001, # time constant for inhibitory population
    'W_EE' : 0.01,  # weight of excitatory to excitatory connections
    'W_IE' : 0.01,  # weight of inhibitory to excitatory connections
    'W_EI' : 0.01,  # weight of excitatory to inhibitory connections
    'W_II' : 0.01,  # weight of inhibitory to inhibitory connections
    'E_max' : 20.0,
    'I_max' : 20.0,
    'C_E' : 0.001,
    'C_I' : 0.001,
    'XE' : 1.0,
    'XI' : 1.0,
    'C_I' : 0.001,
    'tau_S' : 0.01,
    'W_ES' : 0.001,
    'W_IS' : 0.001,

}


# TODO write model_jax 

model_jax.DEFAULT_PARAMS = model.DEFAULT_PARAMS
