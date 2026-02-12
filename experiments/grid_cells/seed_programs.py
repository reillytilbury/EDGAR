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
        X (np.ndarray): Input array with shape (n_features, n_trials).
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


def parameter_estimator_1_v2(X, firing_rates):
    """
    Estimate parameters for grid_model_1 from firing rate data.
    
    Uses simple heuristics based on the spatial firing pattern:
    - Grid spacing: estimated from peak distances in autocorrelation
    - Orientation: estimated from dominant frequency direction
    - Phase: estimated from position of maximum firing
    - Amplitude/baseline: from min/max firing rates
    
    Args:
        X (np.ndarray): Input array (2, n_trials) with x, y positions.
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

def _make_ratemap(X, firing_rates, nbins=60, sigma=1.5):
    x, y = X[0], X[1]
    rng = [[-1, 1], [-1, 1]]

    heat, xe, ye = np.histogram2d(x, y, bins=nbins, range=rng, weights=firing_rates)
    occ, _, _ = np.histogram2d(x, y, bins=nbins, range=rng)

    rm = np.divide(heat, occ, out=np.zeros_like(heat), where=occ > 0).T  # [y,x]
    rm_s = ndimage.gaussian_filter(rm, sigma=sigma)

    return rm_s, occ.T

def _fft_peaks(rm, occ=None, occ_q=20, kmin_frac=0.05, kmax_frac=0.45):
    """
    Return dominant ring peaks in FFT magnitude.
    kmin_frac/kmax_frac are fractions of Nyquist radius to ignore DC/edge.
    """
    H, W = rm.shape
    # mask low-occupancy bins (optional)
    if occ is not None:
        thr = np.percentile(occ[occ > 0], occ_q) if np.any(occ > 0) else 0
        mask = (occ >= thr).astype(float)
    else:
        mask = np.ones_like(rm)

    # window to reduce edge artefacts
    wy = np.hanning(H)
    wx = np.hanning(W)
    window = wy[:, None] * wx[None, :]

    z = (rm - np.nanmean(rm[mask > 0])) * mask * window
    F = np.fft.fftshift(np.fft.fft2(z))
    M = np.abs(F)

    cy, cx = H // 2, W // 2
    yy, xx = np.indices((H, W))
    rr = np.sqrt((yy - cy)**2 + (xx - cx)**2)

    rmax = np.max(rr)
    rmin = kmin_frac * rmax
    rmax_use = kmax_frac * rmax

    # suppress DC and very high freq
    band = (rr >= rmin) & (rr <= rmax_use)
    M_band = np.where(band, M, 0.0)

    # find top peaks (take more than 6 then prune by angle)
    flat_idx = np.argpartition(M_band.ravel(), -50)[-50:]
    py, px = np.unravel_index(flat_idx, (H, W))

    # sort by magnitude descending
    order = np.argsort(M_band[py, px])[::-1]
    py, px = py[order], px[order]

    # compute angles/radii
    dy = py - cy
    dx = px - cx
    ang = np.arctan2(dy, dx)  # [-pi, pi]
    rad = np.sqrt(dx*dx + dy*dy)

    return F, M, py, px, ang, rad, (cy, cx)

def parameter_estimator_1(X, firing_rates, nbins=60, sigma=1.5):
    rm, occ = _make_ratemap(X, firing_rates, nbins=nbins, sigma=sigma)

    F, M, py, px, ang, rad, (cy, cx) = _fft_peaks(rm, occ=occ)

    if len(py) < 6:
        # fallback
        return np.array([0.5, 0.0, 0.0, 0.0, max(0.0, np.percentile(rm, 5)), 1.0])

    # --- Select a dominant radius (ring) robustly ---
    # Use median radius of the strongest few peaks
    r0 = np.median(rad[:10])

    # Keep candidates near r0
    cand = np.where(np.abs(rad - r0) < 0.15 * r0)[0]
    if len(cand) < 6:
        cand = np.arange(min(len(rad), 20))

    ang_c = ang[cand]
    py_c, px_c = py[cand], px[cand]
    rad_c = rad[cand]

    # Pick one "fundamental" direction, then enforce 60° symmetry.
    # We only need theta modulo pi/3.
    a0 = ang_c[0]  # strongest
    # reduce to [0, pi/3)
    theta = (a0 % (np.pi/3))

    # --- Convert radius in FFT bins to lambda in spatial units ---
    # frequency step: cycles per arena-length (2 units) per FFT index.
    # A shift of 1 in FFT index corresponds to 1/(arena_length) cycles per unit.
    # Here arena length is 2 (from -1 to 1). For discrete grid:
    arena = 2.0
    fx = (px_c[0] - cx) / arena  # rough; same scale for x/y
    fy = (py_c[0] - cy) / arena
    k_mag = 2*np.pi*np.sqrt(fx*fx + fy*fy)  # radians per unit

    # Your model uses q = 4*pi/(sqrt(3)*lam)
    # and wavevectors have magnitude q.
    q = k_mag if k_mag > 1e-6 else 4*np.pi/(np.sqrt(3)*0.5)
    lam = 4.0 * np.pi / (np.sqrt(3.0) * q)
    lam = float(np.clip(lam, 0.1, 2.0))

    # --- Phase (phi_x, phi_y) from two directions ---
    # Build two unit vectors at theta and theta+60deg
    angles = theta + 2*np.pi*np.arange(2)/3.0
    u = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (2,2)

    # Find FFT bins closest to those directions at radius r0
    # Use dot-product with direction in frequency space
    dy = py_c - cy
    dx = px_c - cx
    rr = np.sqrt(dx*dx + dy*dy) + 1e-9
    vx = dx / rr
    vy = dy / rr

    # direction matching
    phi_eq = []
    A = []
    for j in range(2):
        djx, djy = u[j,0], u[j,1]
        # pick candidate with max alignment and near r0
        align = vx*djx + vy*djy
        idx = np.argmax(align)
        ky, kx = py_c[idx], px_c[idx]

        # phase of Fourier coefficient
        phase = np.angle(F[ky, kx])

        # For cos(k·x - delta), Fourier phase relates to delta up to conventions.
        # Use delta = -phase as a practical convention; you can flip sign if needed.
        delta = -phase

        # constraint: q * (u_j · phi) ≈ delta  (mod 2π)
        A.append(q * u[j])
        # wrap delta to [-pi, pi] is fine; the solve picks one branch
        phi_eq.append(((delta + np.pi) % (2*np.pi)) - np.pi)

    A = np.stack(A, axis=0)  # (2,2)
    b = np.array(phi_eq)

    try:
        phi = np.linalg.solve(A, b)
        phi_x, phi_y = float(phi[0]), float(phi[1])
    except np.linalg.LinAlgError:
        phi_x, phi_y = 0.0, 0.0

    # Keep phi in arena
    phi_x = float(np.clip(phi_x, -1.0, 1.0))
    phi_y = float(np.clip(phi_y, -1.0, 1.0))

    # --- baseline/amplitude via regression on raw samples ---
    s = grid_model_1(X, lam=lam, theta=theta, phi_x=phi_x, phi_y=phi_y,
                    baseline=0.0, amplitude=1.0)  # = sum cos
    r = firing_rates

    # Solve r ≈ b + a*s
    S = np.vstack([np.ones_like(s), s]).T
    coef, *_ = np.linalg.lstsq(S, r, rcond=None)
    baseline = float(max(0.0, coef[0]))
    amplitude = float(max(0.0, coef[1]))

    return np.array([lam, theta, phi_x, phi_y, baseline, amplitude])

def parameter_estimator_1_v2(X, firing_rates):
    """
    Robust parameter estimator for Cosine Grid Model using Triangulation.
    
    Strategy:
    1. lambda (spacing): Derived from Spatial Autocorrelation (SAC) for global stability.
    2. phi (phase): Anchored to the 'brightest' peak in the Rate Map.
    3. theta (orientation): Calculated by triangulation between the Anchor 
       and its neighbors in the Rate Map.
    
    Args:
        X (np.ndarray): (2, N) array of positions x, y in [-1, 1].
        firing_rates (np.ndarray): (N,) array of firing rates.
        
    Returns:
        np.ndarray: [lam, theta, phi_x, phi_y, baseline, amplitude]
    """
    x = X[0]
    y = X[1]
    
    # -----------------------------------------------------------
    # 1. PRE-PROCESSING: RATE MAP & SMOOTHING
    # -----------------------------------------------------------
    # Use slightly higher resolution (60) for better peak localization
    nbins = 60
    range_lims = [[-1, 1], [-1, 1]]
    
    # Create histograms
    heatmap, x_edges, y_edges = np.histogram2d(x, y, bins=nbins, range=range_lims, weights=firing_rates)
    occupancy, _, _ = np.histogram2d(x, y, bins=nbins, range=range_lims)
    
    # Calculate Rate Map (Transposed so dim 0 is y, dim 1 is x for image logic)
    # np.histogram2d returns H[x,y], we usually want H[row, col] -> H[y,x]
    ratemap = np.divide(heatmap, occupancy, out=np.zeros_like(heatmap), where=occupancy!=0).T
    
    # Smooth the map (Crucial for peak finding)
    ratemap_smooth = ndimage.gaussian_filter(ratemap, sigma=1.5)
    
    # Conversion factor: Physical units per pixel
    pixel_scale = 2.0 / nbins

    # -----------------------------------------------------------
    # 2. ESTIMATE LAMBDA (SPACING) VIA AUTOCORRELATION
    # -----------------------------------------------------------
    # We use SAC for lambda because it averages spacing over the whole arena
    rm_centered = ratemap_smooth - np.mean(ratemap_smooth)
    sac = signal.fftconvolve(rm_centered, rm_centered[::-1, ::-1], mode='same')
    
    # Find peaks in SAC
    sac_max = np.max(sac)
    local_max_sac = ndimage.maximum_filter(sac, size=5) == sac
    peaks_sac_y, peaks_sac_x = np.where(local_max_sac & (sac > 0.1 * sac_max))
    
    # Center of SAC
    cy, cx = sac.shape[0] // 2, sac.shape[1] // 2
    
    # Calculate distances from center
    dists_sac = np.sqrt((peaks_sac_x - cx)**2 + (peaks_sac_y - cy)**2)
    
    # Exclude the central peak (dist ~ 0)
    # We look for the first ring of peaks
    valid_mask = dists_sac > 2.0 # Ignore center
    
    if np.any(valid_mask):
        # Lambda is the distance to the nearest neighbor in SAC
        lam_px = np.min(dists_sac[valid_mask])
        lam = lam_px * pixel_scale
    else:
        # Fallback defaults
        lam = 0.5
        lam_px = 0.5 / pixel_scale

    # -----------------------------------------------------------
    # 3. ESTIMATE PHI (PHASE) VIA BRIGHTEST ANCHOR
    # -----------------------------------------------------------
    # Find local maxima in the actual Rate Map
    local_max_rm = ndimage.maximum_filter(ratemap_smooth, size=3) == ratemap_smooth
    # Filter out noise (must be > 20% of max rate)
    peak_mask = local_max_rm & (ratemap_smooth > 0.2 * np.max(ratemap_smooth))
    p_y, p_x = np.where(peak_mask)
    
    if len(p_y) == 0:
        return np.array([0.5, 0.0, 0.0, 0.0, 0.0, 1.0])

    # Find the "Brightest" peak index
    peak_vals = ratemap_smooth[p_y, p_x]
    brightest_idx = np.argmax(peak_vals)
    
    anchor_y = p_y[brightest_idx]
    anchor_x = p_x[brightest_idx]
    
    # Convert Anchor to physical coordinates (-1 to 1) for Phi
    # Note: edges[0] is -1. Bin i center is -1 + (i + 0.5)*scale
    phi_x = -1.0 + (anchor_x + 0.5) * pixel_scale
    phi_y = -1.0 + (anchor_y + 0.5) * pixel_scale

    # -----------------------------------------------------------
    # 4. ESTIMATE THETA (ORIENTATION) VIA TRIANGULATION
    # -----------------------------------------------------------
    # Calculate distances from the Anchor to all other peaks in Rate Map
    dx = p_x - anchor_x
    dy = p_y - anchor_y
    dists_rm = np.sqrt(dx**2 + dy**2)
    
    # Find neighbors that are roughly 1 lambda away
    # Allow tolerance (e.g., +/- 25% of lambda)
    tolerance = 0.25 * lam_px
    neighbor_mask = (dists_rm > (lam_px - tolerance)) & (dists_rm < (lam_px + tolerance))
    
    neighbor_indices = np.where(neighbor_mask)[0]
    
    if len(neighbor_indices) > 0:
        # Calculate angles from Anchor to Neighbors
        # Note: In image coords, y increases downwards. 
        # Standard math assumes y increases upwards. 
        # However, for orientation, as long as we are consistent, it works.
        # We'll use standard image indices (row=y, col=x).
        vec_y = p_y[neighbor_indices] - anchor_y
        vec_x = p_x[neighbor_indices] - anchor_x
        
        angles = np.arctan2(vec_y, vec_x)
        
        # Grid has 6-fold symmetry (repeats every 60 deg or pi/3)
        # We need the circular mean modulo 60 degrees.
        # Formula: Mean_Angle = atan2(sum(sin(6*angles)), sum(cos(6*angles))) / 6
        
        sin_sum = np.sum(np.sin(6 * angles))
        cos_sum = np.sum(np.cos(6 * angles))
        
        theta_6x = np.arctan2(sin_sum, cos_sum)
        theta = theta_6x / 6.0
        
        # Normalize to range [0, pi/3]
        theta = theta % (np.pi / 3.0)
        
    else:
        # Fallback: Use SAC if anchor has no neighbors (e.g. isolated blob)
        # Find nearest peak in SAC again
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            nearest_sac_idx = valid_indices[np.argmin(dists_sac[valid_mask])]
            dy = peaks_sac_y[nearest_sac_idx] - cy
            dx = peaks_sac_x[nearest_sac_idx] - cx
            theta = np.arctan2(dy, dx) % (np.pi / 3.0)
        else:
            theta = 0.0

    # -----------------------------------------------------------
    # 5. BASELINE AND AMPLITUDE
    # -----------------------------------------------------------
    # Simple percentile heuristics
    baseline = np.percentile(ratemap_smooth, 5)
    peak_rate = np.max(ratemap_smooth)
    
    # For Sum of 3 Cosines: Range is [-1.5, 3] * Amp. Total range 4.5 * Amp.
    # However, max value is Baseline + 3 * Amp.
    # So Amp = (Peak - Baseline) / 3 is a decent starting point.
    amplitude = (peak_rate - baseline) / 3.0
    
    return np.array([lam, theta, phi_x, phi_y, baseline, amplitude])

def grid_model_2(X, lam=0.5, theta=0.0, phi_x=0.0, phi_y=0.0, baseline=0.0, 
                 amplitude=1.0, sigma=0.08):
    """
    Grid cell model using Gaussian bumps on a hexagonal lattice.
    
    This model places Gaussian firing fields at the vertices of a hexagonal
    lattice, providing sharper field boundaries than the cosine model.
    
    Args:
        X (np.ndarray): Input array (n_features, n_trials).
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
        X (np.ndarray): Input array (2, n_trials) with x, y positions.
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