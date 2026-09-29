N_SAMPLES = 8          # cells / animals; parameters are fit per sample
N_REPEATS = 12         # noisy repeats per (sample, stim condition)

# Let DT = 1, but in reality this was set up with the reasonable expectation that the 
# measuring frequency is 30kHz. So DT = 1/30 in ms. 
# For a recording of length 350ms, this requires 350*30 = 10500 time points.
# 50ms stim onset equates to 50*30 = 1500 time points.
# Pulse duration is one of (2, 3, 4, 5) ms, which equates to (60, 90, 120, 150) time points.

# DT = 1
# N_TIMES = 10500
# STIM_ONSET = 1000
# STIM_DUR = 60 
# H = 1.0       # integration step

# Make it shorter for testing 
DT = 1
N_TIMES = 800
STIM_ONSET = 200
STIM_DUR = 30
H = 1.0       # integration step

# Per-sample parameters are drawn uniformly on [median - MAD, median + MAD].
PARAM_MEDIAN = {
    'tau_E': 33.0,
    'tau_I': 195.0,
    'W_EE': 0.0396,
    'W_IE': 0.0277,
    'W_EI': 0.0074,
    'W_II': 0.0014, 
    'E_max': 29.5,
    'I_max': 39.9,
    'C_E': 0.0018,
    'C_I': 0.0134,
    'XE': 3.51,
    'XI': 1.26,
}

PARAM_MAD = {
    'tau_E': 6.0,
    'tau_I': 57.0,
    'W_EE': 0.0162,
    'W_IE': 0.005, # make this small
    'W_EI': 0.0033,
    'W_II': 0.0115,
    'E_max': 2.0, # make this small
    'I_max': 2.0, # make this small
    'C_E': 0.0016,
    'C_I': 0.0115,
    'XE': 2.10,
    'XI': 0.86,
}

# Initial state shared across samples (E and I fully observed, no hidden variable).
INITIAL_STATE = {
    'E': 1.0,
    'I': 1.0,
}
