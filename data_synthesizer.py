import numpy as np
import utils

# ----------------------------------------------------------------------------
# Neuron tuning curve – piecewise linear double‑peak
# ----------------------------------------------------------------------------

def neuron_model_true(theta: np.ndarray,
                      peak_angle_rad: float,
                      peak_amplitude: float,
                      secondary_amplitude_ratio: float,
                      width_rad: float) -> np.ndarray:
    """NumPy implementation of the double‑peak piece‑wise‑linear tuning curve."""
    half_width = width_rad / 2.0

    # Primary peak
    dist1 = utils.circular_distance_rad(theta, peak_angle_rad)
    spikes1 = np.where(
        dist1 <= half_width,
        peak_amplitude * (1.0 - dist1 / half_width),
        0.0,
    )

    # Secondary peak (180° apart)
    peak2_loc = (peak_angle_rad + np.pi) % (2 * np.pi)
    amp2 = peak_amplitude * secondary_amplitude_ratio
    dist2 = utils.circular_distance_rad(theta, peak2_loc)
    spikes2 = np.where(
        dist2 <= half_width,
        amp2 * (1.0 - dist2 / half_width),
        0.0,
    )

    return spikes1 + spikes2

# ----------------------------------------------------------------------------
# Single‑cell spike simulation (Poisson)
# ----------------------------------------------------------------------------

def simulate_spikes(rng: np.random.Generator,
                    tuning_curve: np.ndarray,
                    presented_angles_idx: np.ndarray,
                    bias: float = 0.0) -> np.ndarray:
    if not 0.0 <= bias <= 1.0:
        raise ValueError("Bias must be 0–1.")

    mean_spike_counts = tuning_curve[presented_angles_idx] + bias
    return rng.poisson(lam=mean_spike_counts)

# ----------------------------------------------------------------------------
# Population spike simulation
# ----------------------------------------------------------------------------

def simulate_population_spikes(rng: np.random.Generator,
                               n_cells: int,
                               n_trials: int,
                               n_angular_bins: int,
                               shot_noise: float = 0.1):
    if n_cells <= 0 or n_trials <= 0 or n_angular_bins <= 0:
        raise ValueError("n_cells, n_trials, n_angular_bins must be > 0")

    angle_bins = np.linspace(0, 2 * np.pi, n_angular_bins, endpoint=False)

    # Random parameters per neuron
    pref_angle_rad = rng.uniform(0, 2 * np.pi, size=n_cells)
    max_rates = rng.uniform(10, 40, size=n_cells)
    secondary_ratios = rng.uniform(0.0, 1.0, size=n_cells)
    peak_widths_rad = rng.uniform(np.pi / 8, np.pi / 2, size=n_cells)

    # Compute tuning curves vectorised across neurons
    half_widths = peak_widths_rad / 2.0

    # Primary peaks
    dist1 = utils.circular_distance_rad(angle_bins[np.newaxis, :], pref_angle_rad[:, np.newaxis])
    spikes1 = np.where(
        dist1 <= half_widths[:, None],
        max_rates[:, None] * (1.0 - dist1 / half_widths[:, None]),
        0.0,
    )

    # Secondary peaks
    peak2_loc = (pref_angle_rad + np.pi) % (2 * np.pi)
    amp2 = max_rates * secondary_ratios
    dist2 = utils.circular_distance_rad(angle_bins[np.newaxis, :], peak2_loc[:, np.newaxis])
    spikes2 = np.where(
        dist2 <= half_widths[:, None],
        amp2[:, None] * (1.0 - dist2 / half_widths[:, None]),
        0.0,
    )

    tuning_curves = spikes1 + spikes2

    # Trials
    angle_indices = rng.integers(0, n_angular_bins, size=n_trials)
    angles = angle_indices * (2 * np.pi / n_angular_bins)

    mean_rates = tuning_curves[:, angle_indices] + shot_noise
    spike_matrix = rng.poisson(lam=mean_rates)

    return spike_matrix, tuning_curves + shot_noise, angles, angle_indices

# ----------------------------------------------------------------------------
# Calcium‑transient simulation (zero‑inflated exponential)
# ----------------------------------------------------------------------------

def simulate_calcium_transients(rng: np.random.Generator,
                                tuning_curve: np.ndarray,
                                presented_angles_idx: np.ndarray,
                                bias: float = 0.0,
                                p_zero: float = 0.0) -> np.ndarray:
    if not 0.0 <= bias <= 1.0:
        raise ValueError("Bias must be 0–1.")
    if not 0.0 <= p_zero <= 1.0:
        raise ValueError("p_zero must be 0–1.")

    noise_free = tuning_curve[presented_angles_idx] + bias
    keep_mask = rng.random(size=len(noise_free)) > p_zero
    return noise_free * rng.exponential(scale=1.0, size=len(noise_free)) * keep_mask

# ----------------------------------------------------------------------------
# Population calcium simulation
# ----------------------------------------------------------------------------

def simulate_population_calcium_transients(rng: np.random.Generator,
                                           n_cells: int,
                                           n_trials: int,
                                           n_angular_bins: int,
                                           shot_noise: float = 0.0,
                                           p_zero: float = 0.0):
    if n_cells <= 0 or n_trials <= 0 or n_angular_bins <= 0:
        raise ValueError("n_cells, n_trials, n_angular_bins must be > 0")

    angle_bins = np.linspace(0, 2 * np.pi, n_angular_bins, endpoint=False)

    # Random parameters
    pref_angle_rad = rng.uniform(0, 2 * np.pi, size=n_cells)
    max_rates = rng.uniform(10, 40, size=n_cells)
    secondary_ratios = rng.uniform(0.0, 1.0, size=n_cells)
    peak_widths_rad = rng.uniform(np.pi / 8, np.pi / 2, size=n_cells)

    # Compute tuning curves across neurons
    tuning_curves = np.zeros((n_cells, n_angular_bins))
    for c in range(n_cells):
        tuning_curves[c] = neuron_model_true(angle_bins, pref_angle_rad[c], max_rates[c], secondary_ratios[c], peak_widths_rad[c])

    # simulate spikes for each cell
    angle_indices = rng.integers(0, n_angular_bins, size=n_trials)
    angles = angle_indices * (2 * np.pi / n_angular_bins)

    # neuron noise free calcium transients
    noise_free = tuning_curves[:, angle_indices] + shot_noise
    # add neuron noise
    exp_noise = rng.exponential(scale=1.0, size=(n_cells, n_trials))
    keep_mask = rng.random(size=(n_cells, n_trials)) > p_zero
    activity_matrix = noise_free * exp_noise * keep_mask

    return activity_matrix, tuning_curves + shot_noise, angles, angle_indices
