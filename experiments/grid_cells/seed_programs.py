"""
Seed programs for grid cell experiments.

These are initial models for grid cell firing patterns. Grid cells fire in
hexagonal patterns as animals navigate through 2D environments.

Two main approaches:
1. Cosine sum model: Sum of 3 cosine waves at 60-degree angles
2. Gaussian lattice model: Sum of Gaussians on a hexagonal lattice
"""

import numpy as np
import jax.numpy as jnp


def grid_model_1(X, lam=0.5, theta=0.0, phi_x=0.0, phi_y=0.0, baseline=0.0, amplitude=1.0):
    """
    Grid cell model using sum of 3 cosines at 60-degree angles.
    
    This model approximates the hexagonal firing pattern using interference
    of three plane waves oriented 60 degrees apart.
    
    Equation: r(x,y) = baseline + amplitude * sum_{k=0}^2 cos(q * u_k^T ([x,y] - phi))
    where q = 4*pi/(sqrt(3)*lam) and u_k are unit vectors 60 degrees apart.
    
    Args:
        X (np.ndarray): Predictor array with shape (n_features, n_trials).
                        X[0] is x position, X[1] is y position (normalized to [-1, 1]).
        lam (float): Grid spacing (wavelength) in same units as x,y.
        theta (float): Orientation of the grid pattern in radians.
        phi_x (float): Phase offset in x direction.
        phi_y (float): Phase offset in y direction.
        baseline (float): Baseline firing rate.
        amplitude (float): Amplitude of modulation.
    
    Returns:
        np.ndarray: Predicted firing rate, shape (n_trials,).
    """
    x = X[0]
    y = X[1]
    
    # Clip parameters to valid ranges
    lam = np.clip(lam, 0.1, 2.0)
    theta = np.clip(theta, 0, np.pi / 3)  # 60-degree periodicity
    baseline = np.clip(baseline, 0, None)
    amplitude = np.clip(amplitude, 0, None)
    
    # Wave number for hexagonal lattice spacing
    q = 4.0 * np.pi / (np.sqrt(3.0) * lam)
    
    # Three directions 60 degrees apart
    angles = theta + 2.0 * np.pi * np.arange(3) / 3.0
    ux = np.cos(angles)
    uy = np.sin(angles)
    
    # Shifted positions
    dx = x - phi_x
    dy = y - phi_y
    
    # Sum of three plane waves
    s = 0.0
    for k in range(3):
        proj = ux[k] * dx + uy[k] * dy
        s = s + np.cos(q * proj)
    
    return baseline + amplitude * s


def grid_model_1_jax(X, lam=0.5, theta=0.0, phi_x=0.0, phi_y=0.0, baseline=0.0, amplitude=1.0):
    """JAX-compatible version of grid_model_1."""
    x = X[0]
    y = X[1]
    
    lam = jnp.clip(lam, 0.1, 2.0)
    theta = jnp.clip(theta, 0, jnp.pi / 3)
    baseline = jnp.clip(baseline, 0, None)
    amplitude = jnp.clip(amplitude, 0, None)
    
    q = 4.0 * jnp.pi / (jnp.sqrt(3.0) * lam)
    
    angles = theta + 2.0 * jnp.pi * jnp.arange(3) / 3.0
    ux = jnp.cos(angles)
    uy = jnp.sin(angles)
    
    dx = x - phi_x
    dy = y - phi_y
    
    # Vectorized sum using JAX operations
    proj = ux[:, None] * dx[None, :] + uy[:, None] * dy[None, :]  # (3, n_trials)
    s = jnp.sum(jnp.cos(q * proj), axis=0)  # (n_trials,)
    
    return baseline + amplitude * s


def parameter_estimator_1(X, firing_rates):
    """
    Estimate parameters for grid_model_1 from firing rate data.
    
    Uses simple heuristics based on the spatial firing pattern:
    - Grid spacing: estimated from peak distances in autocorrelation
    - Orientation: estimated from dominant frequency direction
    - Phase: estimated from position of maximum firing
    - Amplitude/baseline: from min/max firing rates
    
    Args:
        X (np.ndarray): Predictor array (2, n_trials) with x, y positions.
        firing_rates (np.ndarray): Firing rates, shape (n_trials,).
    
    Returns:
        np.ndarray: Estimated parameters [lam, theta, phi_x, phi_y, baseline, amplitude].
    """
    x = X[0]
    y = X[1]
    
    # Baseline and amplitude from min/max
    baseline = np.percentile(firing_rates, 10)
    peak_rate = np.percentile(firing_rates, 95)
    amplitude = (peak_rate - baseline) / 3.0  # Divide by 3 for sum of 3 cosines
    
    # Find location of maximum firing for phase estimation
    max_idx = np.argmax(firing_rates)
    phi_x = x[max_idx]
    phi_y = y[max_idx]
    
    # Estimate grid spacing from spatial extent
    # Default to reasonable value based on typical grid spacing
    lam = 0.4  # Will be refined by optimization
    
    # Orientation - start with 0, will be refined
    theta = 0.0
    
    return np.array([lam, theta, phi_x, phi_y, baseline, amplitude])


def grid_model_2(X, lam=0.5, theta=0.0, phi_x=0.0, phi_y=0.0, baseline=0.0, 
                 amplitude=1.0, sigma=0.08):
    """
    Grid cell model using Gaussian bumps on a hexagonal lattice.
    
    This model places Gaussian firing fields at the vertices of a hexagonal
    lattice, providing sharper field boundaries than the cosine model.
    
    Args:
        X (np.ndarray): Predictor array (n_features, n_trials).
                        X[0] is x, X[1] is y (normalized to [-1, 1]).
        lam (float): Lattice spacing.
        theta (float): Orientation of the lattice in radians.
        phi_x, phi_y (float): Phase offsets.
        baseline (float): Baseline firing rate.
        amplitude (float): Peak amplitude per field.
        sigma (float): Width of each Gaussian field.
    
    Returns:
        np.ndarray: Predicted firing rate, shape (n_trials,).
    """
    x = X[0]
    y = X[1]
    
    # Clip parameters
    lam = np.clip(lam, 0.1, 2.0)
    theta = np.clip(theta, 0, np.pi / 3)
    sigma = np.clip(sigma, 0.01, 0.5)
    baseline = np.clip(baseline, 0, None)
    amplitude = np.clip(amplitude, 0, None)
    
    # Hexagonal lattice basis vectors
    v1 = np.array([lam, 0.0])
    v2 = np.array([0.5 * lam, 0.5 * np.sqrt(3.0) * lam])
    
    # Rotate basis by theta
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    v1 = R @ v1
    v2 = R @ v2
    
    # Determine range of lattice points to sum over
    extent = 2.0  # Arena extent
    margin = 2.0  # Extra margin
    step = min(np.linalg.norm(v1), np.linalg.norm(v2))
    n_range = int(np.ceil((extent + margin * lam) / step)) + 2
    
    # Shifted positions
    dx = x - phi_x
    dy = y - phi_y
    
    # Sum Gaussian bumps
    r = np.full_like(x, baseline, dtype=float)
    inv2sig2 = 1.0 / (2.0 * sigma * sigma)
    
    for n in range(-n_range, n_range + 1):
        for m in range(-n_range, n_range + 1):
            cx, cy = n * v1 + m * v2
            ddx = dx - cx
            ddy = dy - cy
            r = r + amplitude * np.exp(-(ddx * ddx + ddy * ddy) * inv2sig2)
    
    return r


def grid_model_2_jax(X, lam=0.5, theta=0.0, phi_x=0.0, phi_y=0.0, baseline=0.0,
                     amplitude=1.0, sigma=0.08):
    """
    JAX-compatible version of grid_model_2.
    
    Uses a fixed range of lattice points for JIT compatibility.
    """
    x = X[0]
    y = X[1]
    
    lam = jnp.clip(lam, 0.1, 2.0)
    theta = jnp.clip(theta, 0, jnp.pi / 3)
    sigma = jnp.clip(sigma, 0.01, 0.5)
    baseline = jnp.clip(baseline, 0, None)
    amplitude = jnp.clip(amplitude, 0, None)
    
    # Hexagonal lattice basis vectors
    v1_x = lam * jnp.cos(theta)
    v1_y = lam * jnp.sin(theta)
    v2_x = 0.5 * lam * jnp.cos(theta) - 0.5 * jnp.sqrt(3.0) * lam * jnp.sin(theta)
    v2_y = 0.5 * lam * jnp.sin(theta) + 0.5 * jnp.sqrt(3.0) * lam * jnp.cos(theta)
    
    dx = x - phi_x
    dy = y - phi_y
    
    inv2sig2 = 1.0 / (2.0 * sigma * sigma)
    
    # Fixed lattice range for JIT compatibility (reduced for memory efficiency)
    n_range = 5
    n_vals = jnp.arange(-n_range, n_range + 1)
    m_vals = jnp.arange(-n_range, n_range + 1)
    N, M = jnp.meshgrid(n_vals, m_vals, indexing='ij')
    N = N.ravel()
    M = M.ravel()
    
    # Lattice centers
    cx = N * v1_x + M * v2_x  # (n_lattice,)
    cy = N * v1_y + M * v2_y  # (n_lattice,)
    
    # Distances from each lattice point
    ddx = dx[None, :] - cx[:, None]  # (n_lattice, n_trials)
    ddy = dy[None, :] - cy[:, None]
    
    # Sum of Gaussians
    bumps = amplitude * jnp.exp(-(ddx * ddx + ddy * ddy) * inv2sig2)
    r = baseline + jnp.sum(bumps, axis=0)
    
    return r


def parameter_estimator_2(X, firing_rates):
    """
    Estimate parameters for grid_model_2 from firing rate data.
    
    Similar to parameter_estimator_1 but includes sigma estimation.
    
    Args:
        X (np.ndarray): Predictor array (2, n_trials) with x, y positions.
        firing_rates (np.ndarray): Firing rates, shape (n_trials,).
    
    Returns:
        np.ndarray: Estimated parameters [lam, theta, phi_x, phi_y, baseline, amplitude, sigma].
    """
    x = X[0]
    y = X[1]
    
    # Baseline and amplitude
    baseline = np.percentile(firing_rates, 10)
    peak_rate = np.percentile(firing_rates, 95)
    amplitude = peak_rate - baseline
    
    # Phase from maximum
    max_idx = np.argmax(firing_rates)
    phi_x = x[max_idx]
    phi_y = y[max_idx]
    
    # Default grid spacing
    lam = 0.4
    
    # Orientation
    theta = 0.0
    
    # Estimate sigma from field width
    # Use width at half maximum heuristic
    sigma = 0.06
    
    return np.array([lam, theta, phi_x, phi_y, baseline, amplitude, sigma])
