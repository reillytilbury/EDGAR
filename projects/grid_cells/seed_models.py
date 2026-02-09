import numpy as np


def grid_cell_model_gaussian_lattice(
    stimuli,
    lattice_spacing=0.5,
    orientation=0.0,
    phase_x=0.0,
    phase_y=0.0,
    baseline=0.0,
    amplitude=1.0,
    sigma=0.12,
    n_max=8,
):
    """
    Grid cell model: sum of isotropic Gaussian bumps on a hexagonal lattice.
    """
    pos = np.asarray(stimuli)
    if pos.ndim != 2:
        raise ValueError(f"stimuli must be 2D. Got shape {pos.shape}.")
    if pos.shape[0] == 2 and pos.shape[1] != 2:
        pos = pos.T
    if pos.shape[1] != 2:
        raise ValueError(f"stimuli must have 2 columns (x,y). Got shape {pos.shape}.")

    lattice_spacing = np.clip(lattice_spacing, 1e-3, None)
    sigma = np.clip(sigma, 1e-3, None)
    amplitude = np.clip(amplitude, 0.0, None)
    baseline = np.clip(baseline, 0.0, None)

    a1 = lattice_spacing * np.array([np.cos(orientation), np.sin(orientation)])
    a2 = lattice_spacing * np.array([np.cos(orientation + np.pi / 3), np.sin(orientation + np.pi / 3)])

    n = np.arange(-n_max, n_max + 1)
    N, M = np.meshgrid(n, n, indexing="ij")
    centers = N[..., None] * a1 + M[..., None] * a2 + np.array([phase_x, phase_y])
    centers = centers.reshape(-1, 2)

    diffs = pos[:, None, :] - centers[None, :, :]
    r2 = np.sum(diffs ** 2, axis=-1)
    blobs = np.exp(-0.5 * r2 / (sigma ** 2))
    return baseline + amplitude * np.sum(blobs, axis=1)




def grid_cell_model_gaussian_lattice_aniso(
    stimuli,
    lattice_spacing=0.5,
    orientation=0.0,
    phase_x=0.0,
    phase_y=0.0,
    baseline=0.0,
    amplitude=1.0,
    sigma_major=0.14,
    sigma_minor=0.08,
    n_max=8,
):
    """
    Grid cell model: anisotropic Gaussian bumps on a hexagonal lattice.
    The anisotropy is aligned with the grid orientation.
    """
    pos = np.asarray(stimuli)
    if pos.ndim != 2:
        raise ValueError(f"stimuli must be 2D. Got shape {pos.shape}.")
    if pos.shape[0] == 2 and pos.shape[1] != 2:
        pos = pos.T
    if pos.shape[1] != 2:
        raise ValueError(f"stimuli must have 2 columns (x,y). Got shape {pos.shape}.")

    lattice_spacing = np.clip(lattice_spacing, 1e-3, None)
    sigma_major = np.clip(sigma_major, 1e-3, None)
    sigma_minor = np.clip(sigma_minor, 1e-3, None)
    amplitude = np.clip(amplitude, 0.0, None)
    baseline = np.clip(baseline, 0.0, None)

    a1 = lattice_spacing * np.array([np.cos(orientation), np.sin(orientation)])
    a2 = lattice_spacing * np.array([np.cos(orientation + np.pi / 3), np.sin(orientation + np.pi / 3)])

    n = np.arange(-n_max, n_max + 1)
    N, M = np.meshgrid(n, n, indexing="ij")
    centers = N[..., None] * a1 + M[..., None] * a2 + np.array([phase_x, phase_y])
    centers = centers.reshape(-1, 2)

    diffs = pos[:, None, :] - centers[None, :, :]
    c, s = np.cos(orientation), np.sin(orientation)
    rot = np.array([[c, -s], [s, c]])
    diffs = diffs @ rot.T
    r2 = (diffs[..., 0] ** 2) / (sigma_major ** 2) + (diffs[..., 1] ** 2) / (sigma_minor ** 2)
    blobs = np.exp(-0.5 * r2)
    return baseline + amplitude * np.sum(blobs, axis=1)




def parameter_estimator_grid_gaussian(stimuli, spike_counts, grid_size=64):
    """
    Estimate [lattice_spacing, orientation, phase_x, phase_y, baseline, amplitude, sigma]
    using autocorrelation of a binned rate map.
    """
    pos = np.asarray(stimuli)
    if pos.ndim != 2:
        raise ValueError(f"stimuli must be 2D. Got shape {pos.shape}.")
    if pos.shape[0] == 2 and pos.shape[1] != 2:
        pos = pos.T
    if pos.shape[1] != 2:
        raise ValueError(f"stimuli must have 2 columns (x,y). Got shape {pos.shape}.")

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
        if np.sum(mask) == 0:
            return 1.0, 0.0
        vals = ac[mask]
        coords = np.column_stack(np.nonzero(mask))
        if vals.size == 0:
            return 1.0, 0.0
        top_idx = np.argpartition(vals, -min(top_k, vals.size))[-min(top_k, vals.size):]
        peak_coords = coords[top_idx]
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

    rates = np.asarray(spike_counts)
    baseline = float(np.min(rates))
    amplitude = float(np.max(rates) - baseline)
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


def parameter_estimator_grid_gaussian_aniso(stimuli, spike_counts, grid_size=64):
    """
    Estimate [lattice_spacing, orientation, phase_x, phase_y, baseline, amplitude, sigma_major, sigma_minor]
    using autocorrelation of a binned rate map.
    """
    pos = np.asarray(stimuli)
    if pos.ndim != 2:
        raise ValueError(f"stimuli must be 2D. Got shape {pos.shape}.")
    if pos.shape[0] == 2 and pos.shape[1] != 2:
        pos = pos.T
    if pos.shape[1] != 2:
        raise ValueError(f"stimuli must have 2 columns (x,y). Got shape {pos.shape}.")

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
        if np.sum(mask) == 0:
            return 1.0, 0.0
        vals = ac[mask]
        coords = np.column_stack(np.nonzero(mask))
        if vals.size == 0:
            return 1.0, 0.0
        top_idx = np.argpartition(vals, -min(top_k, vals.size))[-min(top_k, vals.size):]
        peak_coords = coords[top_idx]
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

    def _estimate_anisotropy_from_autocorr(ac, spacing):
        h, w = ac.shape
        cy, cx = h // 2, w // 2
        Y, X = np.indices(ac.shape)
        dx = (X - cx) * spacing
        dy = (Y - cy) * spacing
        R = np.sqrt(dx ** 2 + dy ** 2)
        mask = R <= (4.0 * spacing)
        if np.sum(mask) < 10:
            return 1.0, 1.0
        weights = np.clip(ac[mask], 0.0, None)
        wsum = float(np.sum(weights))
        if wsum <= 0:
            return 1.0, 1.0
        dxm = dx[mask]
        dym = dy[mask]
        cov_xx = float(np.sum(weights * dxm * dxm) / wsum)
        cov_yy = float(np.sum(weights * dym * dym) / wsum)
        cov_xy = float(np.sum(weights * dxm * dym) / wsum)
        cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.clip(eigvals, 1e-6, None)
        mean_eig = float(np.mean(eigvals))
        if mean_eig <= 0:
            return 1.0, 1.0
        ratio_major = np.sqrt(float(eigvals[1] / mean_eig))
        ratio_minor = np.sqrt(float(eigvals[0] / mean_eig))
        return ratio_major, ratio_minor

    rates = np.asarray(spike_counts)
    baseline = float(np.min(rates))
    amplitude = float(np.max(rates) - baseline)
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
    ratio_major, ratio_minor = _estimate_anisotropy_from_autocorr(ac, spacing)
    sigma_major = max(sigma * ratio_major, 1e-3)
    sigma_minor = max(sigma * ratio_minor, 1e-3)
    phase_x, phase_y = 0.0, 0.0
    return np.array([lattice_spacing, orientation, phase_x, phase_y, baseline, amplitude, sigma_major, sigma_minor])
