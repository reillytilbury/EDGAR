import numpy as np
from scipy.ndimage import gaussian_filter


def parameter_estimator(data):
    """
    Autocorrelation-based initializer for an isotropic hex-grid Gaussian-bump model.

    Estimates lam (lattice spacing) and theta (orientation) from the 2D autocorrelation
    of the rate map, then estimates (phi_x, phi_y) via FFT cross-correlation with a
    template lattice.

    Args:
        data (dict): Keys 'pos_x', 'pos_y', 'response' — shape (n_trials,).

    Returns:
        dict: {"lam", "theta", "phi_x", "phi_y", "baseline", "amplitude", "sigma"}
    """
    x = np.asarray(data['pos_x'], float)
    y = np.asarray(data['pos_y'], float)
    yobs = np.asarray(data['response'], float)

    nbins = 48
    smooth_sigma = 2.5
    occ_prct = 20
    topk = 120

    baseline = float(np.percentile(yobs, 10))
    amplitude = float(max(1e-6, np.percentile(yobs, 95) - baseline))

    edges = np.linspace(-1.0, 1.0, nbins + 1)
    occ, _, _ = np.histogram2d(x, y, bins=[edges, edges])
    heat, _, _ = np.histogram2d(x, y, bins=[edges, edges], weights=yobs)
    Rs = gaussian_filter(heat / (occ + 1e-8), smooth_sigma)
    Os = gaussian_filter(occ, smooth_sigma)
    Rm = Rs / (Os + 1e-8)

    if np.any(occ > 0):
        thr = np.percentile(occ[occ > 0], occ_prct)
        mask = (occ >= thr).astype(float)
    else:
        mask = np.ones_like(Rm)

    wy = np.hanning(nbins)
    win = wy[:, None] * wy[None, :]
    w = mask * win
    mu = np.sum(Rm * w) / (np.sum(w) + 1e-12)
    Z = (Rm - mu) * w

    F = np.fft.fft2(Z)
    A = np.fft.fftshift(np.fft.ifft2(np.abs(F) ** 2).real)
    A = A / (np.max(A) + 1e-12)

    cy, cx = nbins // 2, nbins // 2
    yy, xx = np.indices((nbins, nbins))
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    band = (rr >= max(0.08 * np.max(rr), 0.15 * np.max(rr))) & (rr <= 0.6 * np.max(rr))
    Ab = np.where(band, A, -np.inf)
    flat = Ab.ravel()
    k = min(topk, flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    py, px = np.unravel_index(idx, (nbins, nbins))
    vals = Ab[py, px]
    order = np.argsort(vals)[::-1]
    py, px = py[order], px[order]

    if len(py) == 0 or not np.isfinite(vals).any():
        lam, theta = 0.6, 0.0
    else:
        py0, px0 = py[0], px[0]
        r0 = float(rr[py0, px0])
        ang0 = float(np.arctan2(py0 - cy, px0 - cx))
        binw = 2.0 / nbins
        lam = float(np.clip(r0 * binw, 1.0, 1.5))
        theta = float(ang0 % (np.pi / 3))

    sigma = float(np.clip(0.20 * lam, 0.01, 0.6))

    # Phase via FFT cross-correlation with template
    xs = (np.arange(nbins) + 0.5) * (2.0 / nbins) - 1.0
    Xg, Yg = np.meshgrid(xs, xs, indexing="ij")
    c, s = np.cos(theta), np.sin(theta)
    v1 = np.array([lam * c, lam * s])
    v2 = np.array([0.5 * lam * c - 0.5 * np.sqrt(3.0) * lam * s,
                   0.5 * lam * s + 0.5 * np.sqrt(3.0) * lam * c])
    K = int(np.clip(np.ceil(2.0 / max(lam, 1e-3)), 2, 5))
    ns = np.arange(-K, K + 1)
    nn, mm = np.meshgrid(ns, ns, indexing="ij")
    centers = (nn[..., None] * v1 + mm[..., None] * v2).reshape(-1, 2)
    inv2 = 1.0 / (2.0 * sigma * sigma + 1e-12)
    dx0 = Xg[None, :, :] - centers[:, 0][:, None, None]
    dy0 = Yg[None, :, :] - centers[:, 1][:, None, None]
    T = np.exp(-(dx0 * dx0 + dy0 * dy0) * inv2).sum(axis=0)

    Rz = (Rm - Rm.mean()) / (Rm.std() + 1e-12)
    Tz = (T - T.mean()) / (T.std() + 1e-12)
    C = np.fft.ifft2(np.fft.fft2(Rz) * np.conj(np.fft.fft2(Tz))).real
    iy, ix = np.unravel_index(np.argmax(C), C.shape)

    binw = 2.0 / nbins
    sy = iy if iy <= nbins // 2 else iy - nbins
    sx = ix if ix <= nbins // 2 else ix - nbins
    phi_x = float(np.clip(-sx * binw, -1.0, 1.0))
    phi_y = float(np.clip(-sy * binw, -1.0, 1.0))

    return {
        "lam": float(lam),
        "theta": float(theta),
        "phi_x": phi_x,
        "phi_y": phi_y,
        "baseline": float(baseline),
        "amplitude": float(amplitude),
        "sigma": sigma,
    }
