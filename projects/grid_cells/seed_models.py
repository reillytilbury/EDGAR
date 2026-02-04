import numpy as np
import jax.numpy as jnp

def grid_cell_model(stimuli, lattice_spacing=1.0, orientation=0.0, phase_x=0.0, phase_y=0.0,
                    baseline=0.0, amplitude=1.0, sigma=0.25):
    """
    Grid cell model: sum of Gaussian blobs on a hexagonal lattice.
    Args:
        stimuli (np.ndarray): Position samples with shape (n_trials, 2).
        lattice_spacing (float): Distance between neighboring grid fields.
        orientation (float): Lattice rotation (radians).
        phase_x (float): Lattice phase shift along x.
        phase_y (float): Lattice phase shift along y.
        baseline (float): Baseline firing rate.
        amplitude (float): Peak firing rate above baseline.
        sigma (float): Blob width (std dev).
    Returns:
        np.ndarray: Firing rate for each position sample.
    """
    def _hex_lattice_centers(lattice_spacing, orientation, phase_x, phase_y, extent):
        a1 = lattice_spacing * np.array([np.cos(orientation), np.sin(orientation)])
        a2 = lattice_spacing * np.array([np.cos(orientation + np.pi / 3), np.sin(orientation + np.pi / 3)])
        n_max = int(np.ceil((extent / lattice_spacing) * 2)) + 1
        centers = []
        for i in range(-n_max, n_max + 1):
            for j in range(-n_max, n_max + 1):
                c = i * a1 + j * a2 + np.array([phase_x, phase_y])
                if np.linalg.norm(c) <= extent * 1.2:
                    centers.append(c)
        return np.asarray(centers)

    pos = np.asarray(stimuli)
    extent = float(np.max(np.linalg.norm(pos, axis=1)) + 1e-6)
    lattice_spacing = np.clip(lattice_spacing, 1e-6, None)
    sigma = np.clip(sigma, 1e-6, None)
    centers = _hex_lattice_centers(lattice_spacing, orientation, phase_x, phase_y, extent)
    diffs = pos[:, None, :] - centers[None, :, :]
    r2 = np.sum(diffs ** 2, axis=-1)
    blobs = np.exp(-0.5 * r2 / (sigma ** 2))
    return baseline + amplitude * np.max(blobs, axis=1)

def grid_cell_model_jax(stimuli, lattice_spacing=1.0, orientation=0.0, phase_x=0.0, phase_y=0.0,
                        baseline=0.0, amplitude=1.0, sigma=0.25):
    """
    JAX version of grid_cell_model.
    """
    def _hex_lattice_centers_jax(lattice_spacing, orientation, phase_x, phase_y, extent):
        a1 = lattice_spacing * jnp.array([jnp.cos(orientation), jnp.sin(orientation)])
        a2 = lattice_spacing * jnp.array([jnp.cos(orientation + jnp.pi / 3), jnp.sin(orientation + jnp.pi / 3)])
        n_max = int(np.ceil((extent / float(lattice_spacing)) * 2)) + 1
        centers = []
        for i in range(-n_max, n_max + 1):
            for j in range(-n_max, n_max + 1):
                c = i * a1 + j * a2 + jnp.array([phase_x, phase_y])
                if float(jnp.linalg.norm(c)) <= extent * 1.2:
                    centers.append(c)
        return jnp.stack(centers, axis=0)

    pos = jnp.asarray(stimuli)
    extent = jnp.max(jnp.linalg.norm(pos, axis=1)) + 1e-6
    lattice_spacing = jnp.clip(lattice_spacing, 1e-6, None)
    sigma = jnp.clip(sigma, 1e-6, None)
    centers = _hex_lattice_centers_jax(lattice_spacing, orientation, phase_x, phase_y, float(extent))
    diffs = pos[:, None, :] - centers[None, :, :]
    r2 = jnp.sum(diffs ** 2, axis=-1)
    blobs = jnp.exp(-0.5 * r2 / (sigma ** 2))
    return baseline + amplitude * jnp.max(blobs, axis=1)

def parameter_estimator_grid_cell(stimuli, spike_counts, grid_size=64):
    """
    Estimate grid parameters from the autocorrelation of a binned rate map.
    Returns [lattice_spacing, orientation, phase_x, phase_y, baseline, amplitude, sigma].
    """
    def _autocorr2d(Z):
        Z = Z - np.mean(Z)
        F = np.fft.fft2(Z)
        ac = np.fft.ifft2(F * np.conj(F)).real
        ac = np.fft.fftshift(ac)
        ac /= (np.max(ac) + 1e-8)
        return ac

    def _estimate_grid_from_autocorr(ac, spacing, top_k=6):
        h, w = ac.shape
        cy, cx = h // 2, w // 2
        Y, X = np.indices(ac.shape)
        R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = R > 2
        flat_idx = np.argpartition(ac[mask], -top_k)[-top_k:]
        peak_coords = np.column_stack(np.nonzero(mask))[flat_idx]
        vecs = peak_coords - np.array([cy, cx])
        dists = np.sqrt(np.sum(vecs ** 2, axis=1))
        if len(dists) == 0:
            return 1.0, 0.0
        spacing_pix = np.median(dists)
        angs = np.arctan2(vecs[:, 0], vecs[:, 1])
        orientation = np.median(angs) % (np.pi / 3)
        lattice_spacing = spacing_pix * spacing
        return lattice_spacing, orientation

    def _estimate_blob_sigma_from_autocorr(ac, spacing):
        h, w = ac.shape
        cy, cx = h // 2, w // 2
        center_line = ac[cy, :]
        half_max = (np.max(center_line) + np.min(center_line)) / 2
        idx = np.where(center_line >= half_max)[0]
        if len(idx) < 2:
            return 0.25
        fwhm = (idx[-1] - idx[0]) * spacing
        sigma = fwhm / (2.0 * np.sqrt(2 * np.log(2)))
        return max(sigma, 1e-3)

    pos = np.asarray(stimuli)
    rates = np.asarray(spike_counts)
    baseline = np.min(rates)
    amplitude = np.max(rates) - baseline
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    ix = np.clip(((pos[:, 0] - mins[0]) / span[0] * (grid_size - 1)).astype(int), 0, grid_size - 1)
    iy = np.clip(((pos[:, 1] - mins[1]) / span[1] * (grid_size - 1)).astype(int), 0, grid_size - 1)
    rate_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    counts = np.zeros((grid_size, grid_size), dtype=np.float32)
    np.add.at(rate_map, (iy, ix), rates)
    np.add.at(counts, (iy, ix), 1.0)
    rate_map = rate_map / (counts + 1e-6)
    ac = _autocorr2d(rate_map)
    spacing = float(np.mean(span) / grid_size)
    lattice_spacing, orientation = _estimate_grid_from_autocorr(ac, spacing)
    sigma = _estimate_blob_sigma_from_autocorr(ac, spacing)
    phase_x, phase_y = 0.0, 0.0
    return np.array([lattice_spacing, orientation, phase_x, phase_y, baseline, amplitude, sigma])

def grid_cell_model_v2(stimuli, lattice_spacing=1.0, orientation=0.0, phase_x=0.0, phase_y=0.0,
                       baseline=0.0, amplitude=1.0, sigma_major=0.25, sigma_minor=0.25,
                       blob_orientation=0.0):
    """
    Grid cell model v2: anisotropic Gaussian blobs on a hexagonal lattice.
    """
    def _hex_lattice_centers(lattice_spacing, orientation, phase_x, phase_y, extent):
        a1 = lattice_spacing * np.array([np.cos(orientation), np.sin(orientation)])
        a2 = lattice_spacing * np.array([np.cos(orientation + np.pi / 3), np.sin(orientation + np.pi / 3)])
        n_max = int(np.ceil((extent / lattice_spacing) * 2)) + 1
        centers = []
        for i in range(-n_max, n_max + 1):
            for j in range(-n_max, n_max + 1):
                c = i * a1 + j * a2 + np.array([phase_x, phase_y])
                if np.linalg.norm(c) <= extent * 1.2:
                    centers.append(c)
        return np.asarray(centers)

    pos = np.asarray(stimuli)
    extent = float(np.max(np.linalg.norm(pos, axis=1)) + 1e-6)
    lattice_spacing = np.clip(lattice_spacing, 1e-6, None)
    sigma_major = np.clip(sigma_major, 1e-6, None)
    sigma_minor = np.clip(sigma_minor, 1e-6, None)
    centers = _hex_lattice_centers(lattice_spacing, orientation, phase_x, phase_y, extent)
    diffs = pos[:, None, :] - centers[None, :, :]
    c, s = np.cos(blob_orientation), np.sin(blob_orientation)
    rot = np.array([[c, -s], [s, c]])
    diffs = diffs @ rot.T
    r2 = (diffs[..., 0] ** 2) / (sigma_major ** 2) + (diffs[..., 1] ** 2) / (sigma_minor ** 2)
    blobs = np.exp(-0.5 * r2)
    return baseline + amplitude * np.max(blobs, axis=1)

def grid_cell_model_v2_jax(stimuli, lattice_spacing=1.0, orientation=0.0, phase_x=0.0, phase_y=0.0,
                           baseline=0.0, amplitude=1.0, sigma_major=0.25, sigma_minor=0.25,
                           blob_orientation=0.0):
    def _hex_lattice_centers_jax(lattice_spacing, orientation, phase_x, phase_y, extent):
        a1 = lattice_spacing * jnp.array([jnp.cos(orientation), jnp.sin(orientation)])
        a2 = lattice_spacing * jnp.array([jnp.cos(orientation + jnp.pi / 3), jnp.sin(orientation + jnp.pi / 3)])
        n_max = int(np.ceil((extent / float(lattice_spacing)) * 2)) + 1
        centers = []
        for i in range(-n_max, n_max + 1):
            for j in range(-n_max, n_max + 1):
                c = i * a1 + j * a2 + jnp.array([phase_x, phase_y])
                if float(jnp.linalg.norm(c)) <= extent * 1.2:
                    centers.append(c)
        return jnp.stack(centers, axis=0)

    pos = jnp.asarray(stimuli)
    extent = jnp.max(jnp.linalg.norm(pos, axis=1)) + 1e-6
    lattice_spacing = jnp.clip(lattice_spacing, 1e-6, None)
    sigma_major = jnp.clip(sigma_major, 1e-6, None)
    sigma_minor = jnp.clip(sigma_minor, 1e-6, None)
    centers = _hex_lattice_centers_jax(lattice_spacing, orientation, phase_x, phase_y, float(extent))
    diffs = pos[:, None, :] - centers[None, :, :]
    c, s = jnp.cos(blob_orientation), jnp.sin(blob_orientation)
    rot = jnp.array([[c, -s], [s, c]])
    diffs = diffs @ rot.T
    r2 = (diffs[..., 0] ** 2) / (sigma_major ** 2) + (diffs[..., 1] ** 2) / (sigma_minor ** 2)
    blobs = jnp.exp(-0.5 * r2)
    return baseline + amplitude * jnp.max(blobs, axis=1)

def parameter_estimator_grid_cell_v2(stimuli, spike_counts, grid_size=64):
    """
    Estimate grid + anisotropic blob parameters from autocorrelation.
    Returns [lattice_spacing, orientation, phase_x, phase_y, baseline, amplitude, sigma_major, sigma_minor, blob_orientation].
    """
    def _autocorr2d(Z):
        Z = Z - np.mean(Z)
        F = np.fft.fft2(Z)
        ac = np.fft.ifft2(F * np.conj(F)).real
        ac = np.fft.fftshift(ac)
        ac /= (np.max(ac) + 1e-8)
        return ac

    def _estimate_grid_from_autocorr(ac, spacing, top_k=6):
        h, w = ac.shape
        cy, cx = h // 2, w // 2
        Y, X = np.indices(ac.shape)
        R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = R > 2
        flat_idx = np.argpartition(ac[mask], -top_k)[-top_k:]
        peak_coords = np.column_stack(np.nonzero(mask))[flat_idx]
        vecs = peak_coords - np.array([cy, cx])
        dists = np.sqrt(np.sum(vecs ** 2, axis=1))
        if len(dists) == 0:
            return 1.0, 0.0
        spacing_pix = np.median(dists)
        angs = np.arctan2(vecs[:, 0], vecs[:, 1])
        orientation = np.median(angs) % (np.pi / 3)
        lattice_spacing = spacing_pix * spacing
        return lattice_spacing, orientation

    def _estimate_blob_sigma_from_autocorr(ac, spacing):
        h, w = ac.shape
        cy, cx = h // 2, w // 2
        center_line = ac[cy, :]
        half_max = (np.max(center_line) + np.min(center_line)) / 2
        idx = np.where(center_line >= half_max)[0]
        if len(idx) < 2:
            return 0.25
        fwhm = (idx[-1] - idx[0]) * spacing
        sigma = fwhm / (2.0 * np.sqrt(2 * np.log(2)))
        return max(sigma, 1e-3)

    pos = np.asarray(stimuli)
    rates = np.asarray(spike_counts)
    baseline = np.min(rates)
    amplitude = np.max(rates) - baseline
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    ix = np.clip(((pos[:, 0] - mins[0]) / span[0] * (grid_size - 1)).astype(int), 0, grid_size - 1)
    iy = np.clip(((pos[:, 1] - mins[1]) / span[1] * (grid_size - 1)).astype(int), 0, grid_size - 1)
    rate_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    counts = np.zeros((grid_size, grid_size), dtype=np.float32)
    np.add.at(rate_map, (iy, ix), rates)
    np.add.at(counts, (iy, ix), 1.0)
    rate_map = rate_map / (counts + 1e-6)
    ac = _autocorr2d(rate_map)
    spacing = float(np.mean(span) / grid_size)
    lattice_spacing, orientation = _estimate_grid_from_autocorr(ac, spacing)
    sigma = _estimate_blob_sigma_from_autocorr(ac, spacing)
    sigma_major = sigma
    sigma_minor = sigma
    blob_orientation = orientation
    phase_x, phase_y = 0.0, 0.0
    return np.array([lattice_spacing, orientation, phase_x, phase_y, baseline, amplitude, sigma_major, sigma_minor, blob_orientation])
