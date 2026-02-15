import numpy as np
from typing import Tuple
from src.data_structures import Inputs

def target_function(x, a, b, c, k, phi_0):
    """
    target function of the form
    f(x) = (a * x^2 + b * x + c)*sin(k * x + phi_0)

    Args:
        x (array-like): input data
        a (float): parameter a
        b (float): parameter b
        c (float): parameter c
        k (float): parameter k
        phi_0 (float): parameter phi_0

    Returns:
        array-like: output of the target function
    """
    return (a * x**2 + b * x + c) * np.sin(k * x + phi_0)

def load_and_process_data(SEED=42, n_samples=1_000, n_trials=2_000, noise_std=0.1, **kwargs):
    """
    simulate synthetic data for testing of the form
    f(x) = (a * x^2 + b * x + c)*sin(k * x + phi_0) + noise
    where a, b, c, k, phi_0 are randomly generated parameters and noise is Gaussian noise with mean 0 and standard deviation noise_std

    Args:
        SEED (int): random seed for reproducibility
        n_samples (int): number of samples to generate (corresponds to different parameters)
        n_trials (int): number of trials to run
        noise_std (float): standard deviation of the Gaussian noise        
    Returns:
        dict: a dictionary containing the inputs, outputs, response (for now), and parameters
            intputs: x values of shape (n_samples, n_trials)
            outputs: y values of shape (n_samples, n_trials)
            response: same as outputs for now - will be deleted in the future
            parameters: a dictionary containing the parameters a, b, c, k, phi_0 used to generate the data, each of shape (n_samples,)

    """
    rng = np.random.default_rng(SEED)
    # Generate random parameters for each trial
    a = rng.uniform(-1, 1, n_samples)
    b = rng.uniform(-1, 1, n_samples)
    c = rng.uniform(-1, 1, n_samples)
    k = rng.uniform(1.0, 5.0, n_samples)
    phi_0 = rng.uniform(0, 2 * np.pi, n_samples)
    
    # x values will be randomly sampled from -1 to 1 and the same across samples
    x = rng.uniform(-1, 1, n_trials)
    noise = rng.normal(0, noise_std, (n_samples, n_trials))
    y = np.array([target_function(x, a[i], b[i], c[i], k[i], phi_0[i]) for i in range(n_samples)]) + noise 

    # extend x to be the same across samples
    x_tiled = np.tile(x, (n_samples, 1)) 

    return {"inputs": Inputs.from_array(x_tiled, names=['x']), 
            "outputs": y, 
            "response" : y, # fix this once we've resolved the issue with naming things responses ambiguously
            "parameters": {"a": a, "b": b, "c": c, "k": k, "phi_0": phi_0}}

def create_train_test_sample_split(n_samples, training_sample_ratio=0.5, random_seed=0):
    key = jax.random.PRNGKey(random_seed)
    training_size = int(n_samples * training_sample_ratio)
    shuffled_indices = jax.random.permutation(key, jnp.arange(n_samples))
    training_samples = shuffled_indices[:training_size]
    test_samples = shuffled_indices[training_size:]
    return training_samples, test_samples

def create_train_test_trial_split(n_trials: int, random_seed : int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a random split of trial indices into training and test sets.
    """    
    rng = np.random.default_rng(random_seed)
    training_size = n_trials // 2
    shuffled_indices = rng.permutation(n_trials)
    training_trials_idx = shuffled_indices[:training_size]
    test_trials_idx = shuffled_indices[training_size:]
    return training_trials_idx, test_trials_idx