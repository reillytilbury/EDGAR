''' Aim of this file is to simulate population activity data according  to dynamical system model.

Model 1: Wilson-Cowan model (WC)
Model 2 : Lin model (see PDF paper)

Each model specifies the dynamics of the excitatory and inhibitory populations.

Signature of each model should be : initial_state, parameters -> time_series of population activity
Both initial state and parameters are dictionaries of float values.
'''
import os
from typing import Optional

import numpy as np

def _alpha_drive(boxcar, tau_alpha, dt):
    """Convert a 0/1 light-on boxcar into an alpha-function drive time course.

    Paper S1.9: because C1V1_T/T has slower kinetics than ChR2, the inhibitory
    external input X_I is modeled as an alpha function t*exp(-t/tau_alpha) rather
    than a boxcar. The alpha kernel is triggered at each rising edge of the
    boxcar (so paired pulses sum), and peak-normalized to 1 at t=tau_alpha so the
    XI amplitude parameter cleanly means "peak inhibitory drive".

    Parameters
    ----------
    boxcar : (n_times,) array of 0/1 marking when the inhibitory light is on.
    tau_alpha : float, alpha-function time constant (same units as dt*index).
    dt : float, timestep.
    """
    boxcar = np.asarray(boxcar)
    onsets = np.flatnonzero((boxcar > 0) & (np.r_[0, boxcar[:-1]] == 0))
    idx = np.arange(len(boxcar))
    drive = np.zeros(len(boxcar))
    for o in onsets:
        s = (idx - o) * dt
        on = s >= 0
        drive[on] += (s[on] / tau_alpha) * np.exp(1.0 - s[on] / tau_alpha)
    return drive

def wilson_cowan_model(initial_state, parameters, stimuli, time_steps, h=0.01):
    """
    Simulate the Wilson-Cowan model of excitatory and inhibitory populations.

    Parameters
    ----------
    initial_state : dict
        Initial state of the populations, e.g., {'E': 0.1, 'I': 0.1}
    parameters : dict
        Model parameters, e.g., {'E_max': 1.0, 'I_max': 1.0, 'W_EE': 1.0, 'W_EI': 1.0, 'W_IE': 1.0, 'W_II': 1.0, 'C_E': 0.1, 'C_I': 0.1}
    stimuli : (n_time_steps, 2) np.ndarray 
        [:, 0] is the transient input to the excitatory population and [:, 1] is the transient input to the inhibitory population. 
        Each value is either 0 or 1, indicating whether the stimulus is off or on at that time step.
    time_steps : int
        Number of time steps to simulate.

    Returns
    -------
    np.ndarray
        Time series of population activity with shape (time_steps, 2) for E and I.
    """
    E = np.zeros(time_steps)
    I = np.zeros(time_steps)
    
    E[0] = initial_state['E']
    I[0] = initial_state['I']

    E_max = parameters['E_max']
    I_max = parameters['I_max']
    W_EE = parameters['W_EE']
    W_EI = parameters['W_EI']
    W_IE = parameters['W_IE']
    W_II = parameters['W_II']
    tau_E = parameters['tau_E']
    tau_I = parameters['tau_I']

    C_E = parameters['C_E'] # constant input to the excitatory population
    C_I = parameters['C_I'] # constant input to the inhibitory population    

    XE = parameters['XE'] # transient input to the excitatary population 
    XI = parameters['XI'] # transient input to the inhibitory population
    
    ## p27 : inhibitory pulse is considered slower and is applied with an alpha function in the paper. For now use a boxcar function for both
    # tau_alpha = parameters['tau_alpha'] # time constant for alpha function for inhibitory input
    # XI_drive = XI * _alpha_drive(stimuli[:, 1], tau_alpha, h)

    for t in range(1, time_steps):
        stim_E_on = stimuli[t][0]
        stim_I_on = stimuli[t][1]
        E_dot = -E[t-1] + (E_max - E[t-1])* np.maximum((W_EE * E[t-1] - W_EI * I[t-1] + C_E + XE * stim_E_on), 0)
        E_dot /= tau_E
        I_dot = -I[t-1] + (I_max - I[t-1])* np.maximum((W_IE * E[t-1] - W_II * I[t-1] + C_I + XI * stim_I_on), 0)
        I_dot /= tau_I
        E[t] = E[t-1] + h * E_dot
        I[t] = I[t-1] + h * I_dot
            
    return np.column_stack((E, I))

def wilson_cowan_slow_inhibition_model(initial_state, parameters, stimuli, time_steps, h=0.01):
    """
    Simulate the Wilson-Cowan model with slow inhibition - as proposed in Lin's paper

    Parameters
    ----------
    initial_state : dict
        Initial state of the populations, e.g., {'E': 0.1, 'I': 0.1}
    parameters : dict
        Model parameters, e.g., {'E_max': 1.0, 'I_max': 1.0, 'W_EE': 1.0, 'W_EI': 1.0, 'W_IE': 1.0, 'W_II': 1.0, 'C_E': 0.1, 'C_I': 0.1}
    stimuli : (n_time_steps, 2) np.ndarray 
        [:, 0] is the transient input to the excitatory population and [:, 1] is the transient input to the inhibitory population. 
        Each value is either 0 or 1, indicating whether the stimulus is off or on at that time step.
    time_steps : int
        Number of time steps to simulate.

    Returns
    -------
    np.ndarray
        Time series of population activity with shape (time_steps, 2) for E and I.
    """
    E = np.zeros(time_steps)
    I = np.zeros(time_steps)
    S = np.zeros(time_steps) # slow inhibition from TRN activity
    
    E[0] = initial_state['E']
    I[0] = initial_state['I']
    S[0] = initial_state['S']

    E_max = parameters['E_max']
    I_max = parameters['I_max']
    W_EE = parameters['W_EE']
    W_EI = parameters['W_EI']
    W_IE = parameters['W_IE']
    W_II = parameters['W_II']
    W_ES = parameters['W_ES'] # slow inhibition from TRN activity
    W_IS = parameters['W_IS'] # slow inhibition from TRN activity

    tau_E = parameters['tau_E']
    tau_I = parameters['tau_I']
    tau_S = parameters['tau_S']

    C_E = parameters['C_E'] # constant input to the excitatory population
    C_I = parameters['C_I'] # constant input to the inhibitory population    

    XE = parameters['XE'] # transient input to the excitatary population 
    XI = parameters['XI'] # transient input to the inhibitory population

    for t in range(1, time_steps):
        stim_E_on = stimuli[t][0]
        stim_I_on = stimuli[t][1]

        S_dot = -S[t-1] + I[t-1]
        S_dot /= tau_S
        S[t] = S[t-1] + h * S_dot
        
        E_dot = -E[t-1] + (E_max - E[t-1])* np.maximum((W_EE * E[t-1] - W_EI * I[t-1] - W_ES * S[t-1] + C_E + XE * stim_E_on), 0)
        E_dot /= tau_E
        I_dot = -I[t-1] + (I_max - I[t-1])* np.maximum((W_IE * E[t-1] - W_II * I[t-1] - W_IS * S[t-1] + C_I + XI * stim_I_on), 0)
        I_dot /= tau_I
        E[t] = E[t-1] + h * E_dot
        I[t] = I[t-1] + h * I_dot
            
    return np.column_stack((E, I, S))

def lin_model(initial_state, parameters, stimuli, time_steps, h=0.01):
    """
    Simulate Lin's model of excitatory and inhibitory populations as described in paper.

    Parameters
    ----------
    initial_state : dict
        Initial state of the populations + latent variables e.g., {'E': 0.1, 'I': 0.1, 'S': 0.1, 'L': 0.1, 'B': 0.1, 'T': 0.1}
    parameters : dict
        Model parameters, e.g., {'W_EE': 1.0, 'W_EI': 1.0, 'W_IE': 1.0, 'I_EE': 1.0, 'C_E': 0.1, 'C_I': 0.1}
    stimuli : (n_time_steps, 2) np.ndarray 
        [:, 0] is the transient input to the excitatory population and [:, 1] is the transient input to the inhibitory population. 
        Each value is either 0 or 1, indicating whether the stimulus is off or on at that time step.
    time_steps : int
        Number of time steps to simulate.

    Returns
    -------
    np.ndarray
        Time series of population activity with shape (time_steps, 2) for E and I.
    """
    E = np.zeros(time_steps)
    I = np.zeros(time_steps)
    # These 5 values are the latent variables in the model, which will form the "state"
    S = np.zeros(time_steps)
    L = np.zeros(time_steps)
    R = np.zeros(time_steps)
    B = np.zeros(time_steps)
    T = np.zeros(time_steps)
    
    E[0] = initial_state['E']
    I[0] = initial_state['I']
    S[0] = initial_state['S']
    L[0] = initial_state['L'] # LGN neurons
    R[0] = initial_state['R'] # TRN neurons
    B[0] = initial_state['B']
    T[0] = initial_state['T'] # slow inhibition from TRN activity 

    E_max = parameters['E_max']
    I_max = parameters['I_max']
    L_max = parameters['L_max']

    W_EE = parameters['W_EE']
    W_EI = parameters['W_EI']
    W_IE = parameters['W_IE']
    W_II = parameters['W_II']
    W_ES = parameters['W_ES']
    W_IS = parameters['W_IS']
    W_EL = parameters['W_EL']
    W_IL = parameters['W_IL']
    W_LB = parameters['W_LB']
    W_LT = parameters['W_LT']
    W_LE = parameters['W_LE']
    W_RE = parameters['W_RE']
    W_RL = parameters['W_RL']
    
    tau_E = parameters['tau_E']
    tau_I = parameters['tau_I']
    tau_S = parameters['tau_S']
    tau_R = parameters['tau_R']
    tau_B = parameters['tau_B']

    C_E = parameters['C_E'] # constant input to the excitatory population
    C_I = parameters['C_I'] # constant input to the inhibitory population    
    C_L = parameters['C_L']
    C_R = parameters['C_R']

    XE = parameters['XE'] # transient input to the excitatory population 
    XI = parameters['XI'] # transient input to the inhibitory population

    for t in range(1, time_steps):
        stim_E_on = stimuli[t][0]
        stim_I_on = stimuli[t][1]

        S_dot = -S[t-1] + I[t-1]
        S_dot /= tau_S
        S[t] = S[t-1] + h * S_dot

        L_dot = -L[t-1] + (L_max - L[t-1])* np.maximum((W_LE * E[t-1] + C_L + W_LB * L[t-1] * B[t-1] - W_LT * T[t-1]), 0)
        L_dot /= tau_E # use tau_E - same time constant 
        L[t] = L[t-1] + h * L_dot

        T_dot = -T[t-1] + R[t-1]
        T_dot = T_dot / tau_S
        T[t] = T[t-1] + h * T_dot

        B_dot = -B[t-1] + (1 - B[t-1]) * np.maximum(1 - L[t-1], 0)
        B_dot /= tau_B
        B[t] = B[t-1] + h * B_dot

        R_dot = -R[t-1] + (1 - R[t-1]) * np.maximum(W_RE * E[t-1] + W_RL*L[t-1] - C_R, 0)
        R_dot /= tau_R
        R[t] = R[t-1] + h * R_dot        

        E_dot = -E[t-1] + (E_max - E[t-1])* np.maximum(W_EE * E[t-1] - W_EI * I[t-1] + C_E + XE * stim_E_on - W_ES * S[t-1] + W_EL * L[t-1], 0)
        E_dot /= tau_E
        I_dot = -I[t-1] + (I_max - I[t-1])* np.maximum(W_IE * E[t-1] - W_II * I[t-1] + C_I + XI * stim_I_on - W_IS * S[t-1] + W_IL * L[t-1], 0)
        I_dot /= tau_I

        E[t] = E[t-1] + h * E_dot
        I[t] = I[t-1] + h * I_dot

        # state = {
        #     'S' : S[t],
        #     'L' : L[t],
        #     'B' : B[t],
        #     'R' : R[t],
        #     'T' : T[t]
        # }

    return np.column_stack((E, I, S, L, B, R, T))


# ────────────────────────────────────────────────────────────────────────────
# Data synthesis + k-fold split generation
#
# A single model registry (MODELS) lets us synthesise data from either the base
# Wilson-Cowan model or the slow-inhibition variant. Each entry pairs a model
# function with a defaults module that supplies the simulation grid, per-sample
# parameter medians/MADs, and the initial state. Only the observed E, I channels
# are saved (any hidden state, e.g. the WCS slow variable S, is dropped), so the
# data contract stays (n_samples, 2, n_repeats, n_times, 2) for every model.
# ────────────────────────────────────────────────────────────────────────────
from . import wilson_cowan_defaults as wc_defaults
from . import wilson_cowan_slow_defaults as wcs_defaults

MODELS = {
    "wilson_cowan": {
        "model_fn": wilson_cowan_model,
        "defaults": wc_defaults,
        "fold_prefix": "wc",
    },
    "wilson_cowan_slow": {
        "model_fn": wilson_cowan_slow_inhibition_model,
        "defaults": wcs_defaults,
        "fold_prefix": "wcs",
    },
}

DATA_ROOT = "/home/dabin/data/wc_simulations"


def _raw_path(model: str, noiseless: bool = False) -> str:
    """Default output path for a model's raw dataset (noiseless twin in a subdir)."""
    sub = "noiseless" if noiseless else ""
    return os.path.join(DATA_ROOT, sub, f"{model}.npz")


def _generate_parameters(param_medians, param_mad, n_samples, random_seed=0):
    """(n_samples, n_params) array; each param ~ Normal(median, 0.2*MAD), clipped >=0."""
    rng = np.random.default_rng(random_seed)
    parameters = np.zeros((n_samples, len(param_medians)))
    for sample_idx in range(n_samples):
        for param_idx, (param_name, median) in enumerate(param_medians.items()):
            mad = param_mad[param_name]
            parameters[sample_idx, param_idx] = rng.normal(median, 0.2 * mad)
    # ensure that no parameter is negative
    parameters = np.maximum(parameters, 0)
    return parameters


def _add_noise(mean, random_seed=0):
    """Add signal-dependent observation noise: std = 0.1 * sqrt(mean), per time bin.

    Parameters
    ----------
    mean : (n_times, 2) clean E/I response.

    Returns
    -------
    (n_times, 2) noisy signal.
    """
    rng = np.random.default_rng(random_seed)
    noise_std = 0.1 * np.sqrt(mean)
    return mean + rng.normal(0, noise_std, size=mean.shape)

def _cv_gaussian_smooth(repeats):
    """Smooth the repeats with a Gaussian kernel.
    Choose the smoothing bandwidth dynamically by performing a grid search. 

    The input repeats will contain m number of repeats. 
    Iterate through the repeats, and for each repeat smooth the repeat with bandwidth b and compute the MSE between the smoothed repeat and the mean of the remaining repeats. 
    Perform this for a range of bandwidths and choose the bandwidth that minimizes the MSE.

    Parameters
    ----------
    repeats : (n_repeats, n_times, 2) array of noisy repeats.

    Returns
    -------
    repeats : (n_repeats, n_times, 2) array of smoothed repeats.
    bandwidth : float, the optimal bandwidth used for smoothing.
    """
    from scipy.ndimage import gaussian_filter1d

    bandwidth_range = np.geomspace(1, 15, 8) # range of bandwidths to search over
    mse_list = []
    for b in bandwidth_range:
        errs = []
        for i in range(repeats.shape[0]):
            # smoothen the current repeat
            smoothed_repeat = gaussian_filter1d(repeats[i], sigma=b, axis=0)
            # compute the mean of the remaining repeats
            remaining_mean = np.mean(np.delete(repeats, i, axis=0), axis=0)
            # compute the MSE between the smoothed repeat and the mean of the remaining repeats
            mse = np.mean((smoothed_repeat - remaining_mean) ** 2)
            errs.append(mse)
        mse_list.append(np.mean(errs))
    optimal_bandwidth = bandwidth_range[np.argmin(mse_list)]
    smoothed_repeats = np.zeros_like(repeats)
    for i in range(repeats.shape[0]):
        smoothed_repeats[i] = gaussian_filter1d(repeats[i], sigma=optimal_bandwidth, axis=0)
    return smoothed_repeats, optimal_bandwidth

def generate_data(
    model: str = "wilson_cowan",
    noiseless: bool = False,
    warmup_period: int = 0,
    params: Optional[np.ndarray] = None,
    stimuli : Optional[np.ndarray] = None,
    random_seed: int = 0,    
):
    """Simulate a raw dataset for the chosen model.

    Returns ``data`` (n_samples, 2, n_repeats, n_times, 2), ``stimuli``
    (2, n_times, 2) and ``params`` (n_samples, n_params). The two stim conditions
    are an excitatory pulse (drive to E) and an inhibitory pulse (drive to I). Only
    the observed E, I channels are kept; any hidden state (e.g. the WCS slow
    variable S) is dropped so the last axis is always 2.

    ``model`` selects an entry in ``MODELS``; the simulation grid, per-sample
    parameter medians/MADs and initial state all come from that model's defaults
    module. Pass ``params`` (n_samples, n_params) to override sampling.

    If ``noiseless`` is True the observation noise is skipped, so every repeat is
    the identical clean response. Averaging the repeats into k-fold train/test
    then yields train == test — the setup for the GD parameter-recovery sanity
    check (no cross-validation, just "can GD hit the true params on clean data").

    If there is a warmup period, prepend ``warmup_period`` time steps of zeros to the stimuli and 
    the same number of time bins to n_times. And then discard the warmup period from the output data. 
    """
    cfg = MODELS[model]
    defaults = cfg["defaults"]
    model_fn = cfg["model_fn"]

    n_repeats = defaults.N_REPEATS
    n_times = defaults.N_TIMES

    if params is None:
        params = _generate_parameters(
            defaults.PARAM_MEDIAN, defaults.PARAM_MAD, defaults.N_SAMPLES,
            random_seed=random_seed,
        )

    if stimuli is None:
        exc_stimuli = np.zeros((n_times, 2))
        exc_stimuli[defaults.STIM_ONSET:defaults.STIM_ONSET + defaults.STIM_DUR, 0] = 1  # pulse to E
        inh_stimuli = np.zeros((n_times, 2))
        inh_stimuli[defaults.STIM_ONSET:defaults.STIM_ONSET + defaults.STIM_DUR, 1] = 1  # pulse to I
        stimuli = np.stack([exc_stimuli, inh_stimuli], axis=0)

    if warmup_period > 0:
        n_times += warmup_period
        stimuli = np.pad(stimuli, ((0, 0), (warmup_period, 0), (0, 0)), mode='constant', constant_values=0)

    initial_state = defaults.INITIAL_STATE  # hard-coded, shared across samples
    param_names = list(defaults.PARAM_MEDIAN.keys())

    n_samples = len(params)
    data = np.zeros((n_samples, 2, n_repeats, n_times, 2))
    for i, p in enumerate(params):
        p = dict(zip(param_names, p))
        for j, stim in enumerate(stimuli):
            activity = model_fn(initial_state, p, stim, n_times, h=defaults.H)
            activity = activity[:, :2]  # keep observed E, I; drop any hidden state
            for k in range(n_repeats):
                if noiseless:
                    data[i, j, k] = activity
                else:
                    data[i, j, k] = _add_noise(activity, random_seed=i * n_repeats + k)

    if warmup_period > 0:
        data = data[:, :, :, warmup_period:, :]  # discard warmup period
        stimuli = stimuli[:, warmup_period:, :]
        
    return data, stimuli, params

def save_data(
    model: str = "wilson_cowan",
    output_path: Optional[str] = None,
    noiseless: bool = False,
    params: Optional[np.ndarray] = None,
):
    """Simulate the raw dataset for ``model`` and save it as an npz.

    ``output_path`` defaults to ``DATA_ROOT/{model}.npz`` (noiseless twin under a
    ``noiseless/`` subdir).
    """
    output_path = output_path or _raw_path(model, noiseless=noiseless)
    data, stimuli, params = generate_data(model=model, noiseless=noiseless, params=params)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, data=data, params=params, stimuli=stimuli)
    return output_path

def save_kfold_splits(
    raw_path: str,
    out_dir: str | None = None,
    fold_prefix: str = "wc",
    n_folds: int = 3,
    seed: int = 0,
):
    """Derive ``n_folds`` repeat-averaged train/test files from the raw dataset.

    The ``n_repeats`` noisy repeats are randomly partitioned into ``n_folds``
    equal, disjoint folds. For each fold ``f``: the held-out fold is averaged into
    ``test_data`` and the remaining repeats are averaged into ``train_data`` — both
    of shape (n_samples, 2, n_times, 2). Parameters are cross-validated by fitting
    on ``train_data`` and evaluating on the held-out ``test_data``.

    Writes ``{fold_prefix}_fold{f}.npz`` (train_data, test_data, stimuli, params,
    fold, test_repeats, train_repeats) and returns the list of paths.
    """
    raw = np.load(raw_path)
    data = raw["data"]        # (n_samples, 2, n_repeats, n_times, 2)
    stimuli = raw["stimuli"]  # (2, n_times, 2)
    params = raw["params"]    # (n_samples, n_params)

    n_repeats = data.shape[2]
    if n_repeats % n_folds != 0:
        raise ValueError(
            f"n_repeats ({n_repeats}) is not divisible by n_folds ({n_folds}); "
            "folds would be unequal."
        )

    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n_repeats), n_folds)  # disjoint, equal

    out_dir = out_dir or os.path.dirname(raw_path)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for f, test_idx in enumerate(folds):
        test_idx = np.sort(test_idx)
        train_idx = np.sort(np.setdiff1d(np.arange(n_repeats), test_idx))
        test_data = data[:, :, test_idx].mean(axis=2)    # (n_samples, 2, n_times, 2)
        train_data = data[:, :, train_idx].mean(axis=2)
        out_path = os.path.join(out_dir, f"{fold_prefix}_fold{f}.npz")
        np.savez(
            out_path,
            train_data=train_data,
            test_data=test_data,
            stimuli=stimuli,
            params=params,
            fold=f,
            test_repeats=test_idx,
            train_repeats=train_idx,
        )
        paths.append(out_path)
    return paths


def build_dataset(model: str, noiseless: bool = False):
    """(Re)generate a model's raw dataset if missing, then (re)build its k-folds."""
    fold_prefix = MODELS[model]["fold_prefix"]
    raw_path = _raw_path(model, noiseless=noiseless)
    tag = "noiseless " if noiseless else ""
    if not os.path.exists(raw_path):
        print(f"[simulate_data] {tag}raw dataset missing -> generating {raw_path}")
        save_data(model=model, noiseless=noiseless)
    for p in save_kfold_splits(raw_path=raw_path, fold_prefix=fold_prefix):
        print(f"[simulate_data] wrote {p}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Simulate WC / WCS datasets and k-fold splits.")
    ap.add_argument("--model", default="wilson_cowan", choices=list(MODELS),
                    help="which dynamical-system model to simulate")
    args = ap.parse_args()

    # Noisy dataset + folds.
    build_dataset(args.model, noiseless=False)
    # Noiseless twin (train == test) for the GD parameter-recovery sanity check.
    build_dataset(args.model, noiseless=True)
