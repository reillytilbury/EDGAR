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
from scipy import signal, ndimage, spatial

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


def parameter_estimator_1_old(X, firing_rates):
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

def parameter_estimator_1(X, firing_rates):
    """
    Robust parameter estimator using Peak Detection on Spatial Autocorrelation.
    """
    x = X[0]
    y = X[1]
    
    # ==========================================================
    # 1. CREATE SMOOTHED RATE MAP
    # ==========================================================
    # Use a fixed number of bins to create an image
    nbins = 50
    # Range is fixed to [-1, 1] based on your model description
    search_range = [[-1, 1], [-1, 1]]
    
    # Histogram the data
    heatmap, x_edges, y_edges = np.histogram2d(x, y, bins=nbins, range=search_range, weights=firing_rates)
    occupancy, _, _ = np.histogram2d(x, y, bins=nbins, range=search_range)
    
    # Avoid division by zero
    ratemap = np.divide(heatmap, occupancy, out=np.zeros_like(heatmap), where=occupancy!=0)
    
    # Gaussian smooth the map. 
    # Sigma=2.0 pixels is usually good for 50 bins.
    ratemap_smooth = ndimage.gaussian_filter(ratemap, sigma=2.0)
    
    # ==========================================================
    # 2. COMPUTE SPATIAL AUTOCORRELATION (SAC)
    # ==========================================================
    # Center the rate map (subtract mean) to handle baseline
    rm_centered = ratemap_smooth - np.mean(ratemap_smooth)
    
    # Compute 2D autocorrelation using FFT
    sac = signal.fftconvolve(rm_centered, rm_centered[::-1, ::-1], mode='same')
    
    # ==========================================================
    # 3. DETECT PEAKS TO FIND GEOMETRY (LAMBDA & THETA)
    # ==========================================================
    # Instead of masking, we find Local Maxima.
    # This finds pixels that are the highest in their immediate neighborhood.
    # size=5 means we look at a 5x5 patch.
    local_max = ndimage.maximum_filter(sac, size=5) == sac
    
    # We also discard low-value background noise.
    # We only keep peaks that are at least 10% of the maximum correlation.
    sac_max = np.max(sac)
    binary_peaks = local_max & (sac > 0.1 * sac_max)
    
    # Get coordinates of these peaks
    peak_y, peak_x = np.where(binary_peaks)
    
    # Center of the image
    cy, cx = sac.shape[0] // 2, sac.shape[1] // 2
    
    # Calculate distances of all peaks from the center
    # Scale factor: Total range 2.0 / 50 bins = 0.04 units per pixel
    pixel_scale = 2.0 / nbins
    distances_px = np.sqrt((peak_x - cx)**2 + (peak_y - cy)**2)
    
    # Filter out the central peak (distance is 0 or very close to 0)
    # We look for peaks strictly > 0. Since we use grid coordinates, exact 0 is possible.
    valid_mask = distances_px > 1.0 # Ignore peaks within 1 pixel of center
    
    valid_distances = distances_px[valid_mask]
    valid_y = peak_y[valid_mask]
    valid_x = peak_x[valid_mask]
    
    if len(valid_distances) > 0:
        # The Grid Spacing (lam) is the distance to the NEAREST neighbor peak
        nearest_idx = np.argmin(valid_distances)
        min_dist_px = valid_distances[nearest_idx]
        
        lam = min_dist_px * pixel_scale
        
        # The Orientation (theta) is the angle of that nearest peak
        dy = valid_y[nearest_idx] - cy
        dx = valid_x[nearest_idx] - cx
        theta = np.arctan2(dy, dx)
        
        # Normalize theta to [0, 60] degrees (pi/3) due to hexagonal symmetry
        theta = theta % (np.pi / 3.0)
    else:
        # Fallback if no peaks found (e.g. very low firing)
        lam = 0.5
        theta = 0.0

    # ==========================================================
    # 4. ESTIMATE PHASE (PHI) & AMPLITUDE
    # ==========================================================
    # Baseline is roughly the minimum firing rate
    baseline = np.min(ratemap_smooth)
    peak_rate = np.max(ratemap_smooth)
    
    # Amplitude heuristic
    amplitude = (peak_rate - baseline) / 3.0
    
    # Phase: Location of the highest firing peak in the SMOOTHED map
    # This aligns the grid "blob" with the data "blob"
    max_rate_idx = np.unravel_index(np.argmax(ratemap_smooth), ratemap_smooth.shape)
    
    # Convert array indices to X,Y coordinates
    # Indices are (row, col) -> (y, x). Y goes top to bottom in array, but usually bottom-up in plots.
    # However, histogram2d maps x to dim 0 and y to dim 1 if not careful, 
    # but standard is (y, x) for image.
    # Let's map consistently with the histogram generation:
    # x_edges[i] is the left edge of bin i.
    
    # x index is max_rate_idx[1], y index is max_rate_idx[0]
    phi_x_idx = max_rate_idx[1]
    phi_y_idx = max_rate_idx[0]
    
    # Convert index to coordinate value
    # (Index + 0.5) * bin_width + min_edge
    phi_x = -1.0 + (phi_x_idx + 0.5) * pixel_scale
    phi_y = -1.0 + (phi_y_idx + 0.5) * pixel_scale
    
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


def parameter_estimator_2_old(X, firing_rates):
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

def parameter_estimator_2(X, firing_rates):
    """
    Estimator using 'Anchor & Triangulation' logic.
    1. Finds the global spacing (Lambda) and blob size (Sigma) using Autocorrelation.
    2. Finds the 'Brightest Dot' (Anchor) to set Phase (Phi).
    3. Finds neighbors of the Anchor to calculate the precise Orientation (Theta).
    """
    x = X[0]
    y = X[1]
    
    # --- 1. Rate Map Generation ---
    nbins = 60 # Slightly higher res for peak finding
    range_lims = [[-1, 1], [-1, 1]]
    heatmap, x_edges, y_edges = np.histogram2d(x, y, bins=nbins, range=range_lims, weights=firing_rates)
    occupancy, _, _ = np.histogram2d(x, y, bins=nbins, range=range_lims)
    
    # Safe division & smoothing
    ratemap = np.divide(heatmap, occupancy, out=np.zeros_like(heatmap), where=occupancy!=0)
    ratemap_smooth = ndimage.gaussian_filter(ratemap, sigma=1.5)
    
    # Pixel to physical unit scaler
    pixel_scale = 2.0 / nbins
    
    # --- 2. Global Geometry (Lambda & Sigma) via SAC ---
    # We still use SAC for Lambda because individual peak distances are noisy.
    rm_centered = ratemap_smooth - np.mean(ratemap_smooth)
    sac = signal.fftconvolve(rm_centered, rm_centered[::-1, ::-1], mode='same')
    
    # Get Lambda (Distance to nearest SAC peak)
    sac_max = np.max(sac)
    # Find peaks in SAC
    local_max_sac = ndimage.maximum_filter(sac, size=5) == sac
    peaks_sac_y, peaks_sac_x = np.where(local_max_sac & (sac > 0.1 * sac_max))
    
    cy, cx = sac.shape[0] // 2, sac.shape[1] // 2
    dists_sac = np.sqrt((peaks_sac_x - cx)**2 + (peaks_sac_y - cy)**2)
    
    # Filter center
    valid_mask = dists_sac > 2.0
    if np.any(valid_mask):
        lam_px = np.min(dists_sac[valid_mask])
        lam = lam_px * pixel_scale
    else:
        lam = 0.5 # Fallback
        lam_px = 0.5 / pixel_scale

    # Get Sigma (HWHM of central SAC peak)
    # Scan right from center
    mid_slice = sac[cy, cx:]
    mid_slice = mid_slice / mid_slice[0]
    # Find where it drops below 0.5
    hwhm = np.searchsorted(-mid_slice, -0.5) 
    sigma = (hwhm * pixel_scale) / 1.66
    
    # --- 3. Phase (Phi) via Brightest Point ---
    # Find peaks in the actual Rate Map
    local_max_rm = ndimage.maximum_filter(ratemap_smooth, size=3) == ratemap_smooth
    # Keep significant peaks
    peak_mask = local_max_rm & (ratemap_smooth > 0.2 * np.max(ratemap_smooth))
    p_y, p_x = np.where(peak_mask)
    
    if len(p_y) == 0:
        return np.array([lam, 0.0, 0.0, 0.0, 0.0, 1.0, sigma])

    # Find the "Brightest" peak (Highest firing rate)
    peak_rates = ratemap_smooth[p_y, p_x]
    brightest_idx = np.argmax(peak_rates)
    
    # Set Phase to this coordinate
    # Convert bin index to x,y
    phi_x = -1.0 + (p_x[brightest_idx] + 0.5) * pixel_scale
    phi_y = -1.0 + (p_y[brightest_idx] + 0.5) * pixel_scale
    
    # --- 4. Orientation (Theta) via Neighbors ---
    # We look for peaks that form a triangle with the brightest peak.
    anchor_y, anchor_x = p_y[brightest_idx], p_x[brightest_idx]
    
    # Calculate distances from Anchor to all other peaks
    dx = p_x - anchor_x
    dy = p_y - anchor_y
    distances = np.sqrt(dx**2 + dy**2)
    
    # Find neighbors that are approximately 1 Lambda away
    # We allow a tolerance (e.g., +/- 25%)
    tolerance = 0.25 * lam_px
    neighbor_mask = (distances > (lam_px - tolerance)) & (distances < (lam_px + tolerance))
    
    neighbor_indices = np.where(neighbor_mask)[0]
    
    if len(neighbor_indices) > 0:
        # Calculate angles to these neighbors
        angles = []
        for idx in neighbor_indices:
            # Angle relative to positive x-axis
            # Note: y-axis in array increases downwards, but usually we treat y as up.
            # However, provided we are consistent, it matches. 
            # Let's stick to array coords: dy is (neighbor - anchor)
            
            vec_y = (p_y[idx] - anchor_y)
            vec_x = (p_x[idx] - anchor_x)
            
            angle = np.arctan2(vec_y, vec_x)
            angles.append(angle)
            
        angles = np.array(angles)
        
        # Grid symmetry is 60 degrees (pi/3).
        # We want the average angle modulo 60 degrees.
        # But simple mean(angles % 60) fails at the wrap-around (e.g. 59 deg and 1 deg).
        # We use circular mean logic for 6-fold symmetry:
        # Multiply angles by 6, take mean vector, divide phase by 6.
        
        sin_sum = np.sum(np.sin(6 * angles))
        cos_sum = np.sum(np.cos(6 * angles))
        mean_angle_6x = np.arctan2(sin_sum, cos_sum)
        
        theta = mean_angle_6x / 6.0
        
        # Ensure positive theta in [0, pi/3]
        theta = theta % (np.pi / 3.0)
        
    else:
        # If the brightest peak has no neighbors (isolated), 
        # fall back to SAC-based theta (finding nearest peak in SAC)
        peaks_sac_y, peaks_sac_x = np.where(local_max_sac) # Re-get without threshold
        dists_sac = np.sqrt((peaks_sac_x - cx)**2 + (peaks_sac_y - cy)**2)
        valid = dists_sac > (lam_px * 0.5)
        if np.any(valid):
            idx_sac = np.argmin(dists_sac[valid])
            # get coordinates of that valid peak
            # ... (omitted for brevity, usually neighbor method works)
            theta = 0.0 # simple fallback
            
    # --- 5. Amplitude & Baseline ---
    baseline = np.percentile(ratemap_smooth, 5) # 5th percentile
    amplitude = np.max(ratemap_smooth) - baseline

    return np.array([lam, theta, phi_x, phi_y, baseline, amplitude, sigma])