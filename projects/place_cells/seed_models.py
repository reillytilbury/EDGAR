import numpy as np
import jax.numpy as jnp

def place_cell_model_2d(stimuli, center_x=0.0, center_y=0.0, baseline=0.0, amplitude=1.0, sigma=1.0):
    """
    A simple 2D place cell model with a Gaussian receptive field.
    Args:
        stimuli (np.ndarray): Position samples with shape (n_trials, 2).
        center_x (float): Place field center x-coordinate.
        center_y (float): Place field center y-coordinate.
        baseline (float): Baseline firing rate.
        amplitude (float): Peak firing rate above baseline.
        sigma (float): Spatial width (std dev) of the place field.
    Returns:
        np.ndarray: The firing rate for each position sample.
    """
    pos = np.asarray(stimuli)
    dx = pos[:, 0] - center_x
    dy = pos[:, 1] - center_y
    sigma = np.clip(sigma, 1e-6, None)
    r2 = dx ** 2 + dy ** 2
    return baseline + amplitude * np.exp(-0.5 * r2 / (sigma ** 2))


def place_cell_model_2d_jax(stimuli, center_x=0.0, center_y=0.0, baseline=0.0, amplitude=1.0, sigma=1.0):
    """
    JAX version of a 2D place cell model with a Gaussian receptive field.
    Args:
        stimuli (jnp.ndarray): Position samples with shape (n_trials, 2).
        center_x (float): Place field center x-coordinate.
        center_y (float): Place field center y-coordinate.
        baseline (float): Baseline firing rate.
        amplitude (float): Peak firing rate above baseline.
        sigma (float): Spatial width (std dev) of the place field.
    Returns:
        jnp.ndarray: The firing rate for each position sample.
    """
    pos = jnp.asarray(stimuli)
    dx = pos[:, 0] - center_x
    dy = pos[:, 1] - center_y
    sigma = jnp.clip(sigma, 1e-6, None)
    r2 = dx ** 2 + dy ** 2
    return baseline + amplitude * jnp.exp(-0.5 * r2 / (sigma ** 2))


def parameter_estimator_place_cell_2d(stimuli, spike_counts):
    """
    Parameter estimator for the 2D place cell model using weighted stats.
    Args:
        stimuli (np.ndarray): Position samples with shape (n_trials, 2).
        spike_counts (np.ndarray): Spike counts for each position sample.
    Returns:
        np.ndarray: Estimated parameters [center_x, center_y, baseline, amplitude, sigma].
    """
    pos = np.asarray(stimuli)
    weights = np.clip(np.asarray(spike_counts), 0, None)
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        center = np.sum(pos * weights[:, None], axis=0) / weight_sum
    else:
        center = np.mean(pos, axis=0)
    baseline = np.min(spike_counts)
    amplitude = np.max(spike_counts) - baseline
    diffs = pos - center
    r2 = np.sum(diffs ** 2, axis=1)
    sigma = np.sqrt(np.sum(r2 * weights) / (weight_sum + 1e-8)) + 1e-6
    return np.array([center[0], center[1], baseline, amplitude, sigma])


def place_cell_model_2d_aniso(stimuli, center_x=0.0, center_y=0.0, baseline=0.0, amplitude=1.0, sigma_x=1.0, sigma_y=1.0):
    """
    A 2D place cell model with an anisotropic Gaussian receptive field.
    Args:
        stimuli (np.ndarray): Position samples with shape (n_trials, 2).
        center_x (float): Place field center x-coordinate.
        center_y (float): Place field center y-coordinate.
        baseline (float): Baseline firing rate.
        amplitude (float): Peak firing rate above baseline.
        sigma_x (float): Spatial width along x.
        sigma_y (float): Spatial width along y.
    Returns:
        np.ndarray: The firing rate for each position sample.
    """
    pos = np.asarray(stimuli)
    dx = pos[:, 0] - center_x
    dy = pos[:, 1] - center_y
    sigma_x = np.clip(sigma_x, 1e-6, None)
    sigma_y = np.clip(sigma_y, 1e-6, None)
    r2 = (dx ** 2) / (sigma_x ** 2) + (dy ** 2) / (sigma_y ** 2)
    return baseline + amplitude * np.exp(-0.5 * r2)


def place_cell_model_2d_aniso_jax(stimuli, center_x=0.0, center_y=0.0, baseline=0.0, amplitude=1.0, sigma_x=1.0, sigma_y=1.0):
    """
    JAX version of a 2D place cell model with an anisotropic Gaussian receptive field.
    Args:
        stimuli (jnp.ndarray): Position samples with shape (n_trials, 2).
        center_x (float): Place field center x-coordinate.
        center_y (float): Place field center y-coordinate.
        baseline (float): Baseline firing rate.
        amplitude (float): Peak firing rate above baseline.
        sigma_x (float): Spatial width along x.
        sigma_y (float): Spatial width along y.
    Returns:
        jnp.ndarray: The firing rate for each position sample.
    """
    pos = jnp.asarray(stimuli)
    dx = pos[:, 0] - center_x
    dy = pos[:, 1] - center_y
    sigma_x = jnp.clip(sigma_x, 1e-6, None)
    sigma_y = jnp.clip(sigma_y, 1e-6, None)
    r2 = (dx ** 2) / (sigma_x ** 2) + (dy ** 2) / (sigma_y ** 2)
    return baseline + amplitude * jnp.exp(-0.5 * r2)


def parameter_estimator_place_cell_2d_aniso(stimuli, spike_counts):
    """
    Parameter estimator for the anisotropic 2D place cell model using weighted stats.
    Args:
        stimuli (np.ndarray): Position samples with shape (n_trials, 2).
        spike_counts (np.ndarray): Spike counts for each position sample.
    Returns:
        np.ndarray: Estimated parameters [center_x, center_y, baseline, amplitude, sigma_x, sigma_y].
    """
    pos = np.asarray(stimuli)
    weights = np.clip(np.asarray(spike_counts), 0, None)
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        center = np.sum(pos * weights[:, None], axis=0) / weight_sum
    else:
        center = np.mean(pos, axis=0)
    baseline = np.min(spike_counts)
    amplitude = np.max(spike_counts) - baseline
    diffs = pos - center
    var_x = np.sum((diffs[:, 0] ** 2) * weights) / (weight_sum + 1e-8)
    var_y = np.sum((diffs[:, 1] ** 2) * weights) / (weight_sum + 1e-8)
    sigma_x = np.sqrt(var_x) + 1e-6
    sigma_y = np.sqrt(var_y) + 1e-6
    return np.array([center[0], center[1], baseline, amplitude, sigma_x, sigma_y])
