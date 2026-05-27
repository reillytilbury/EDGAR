import numpy as np
from scipy.ndimage import gaussian_filter


def parameter_estimator(data):
    """
    Autocorrelation-based initializer for an anisotropic hex-grid model.

    Same as param_est1 for lam/theta/phi, then estimates sigma_par/sigma_perp
    from weighted second moments of the rate map in the lattice frame.

    Args:
        data (dict): Keys 'pos_x', 'pos_y', 'response' — shape (n_trials,).

    Returns:
        dict: {"lam", "theta", "phi_x", "phi_y", "baseline", "amplitude",
               "sigma_par", "sigma_perp"}
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

    cy_idx, cx_idx = nbins // 2, nbins // 2
    yy, xx = np.indices((nbins, nbins))
    rr = np.sqrt((xx - cx_idx) ** 2 + (yy - cy_idx) ** 2)

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
        ang0 = float(np.arctan2(py0 - cy_idx, px0 - cx_idx))
        binw = 2.0 / nbins
        lam = float(np.clip(r0 * binw, 1.0, 1.5))
        theta = float(ang0 % (np.pi / 3))

    sigma0 = float(np.clip(0.20 * lam, 0.01, 0.6))
    binw = 2.0 / nbins

    # Phase via cross-correlation with template
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
    inv2 = 1.0 / (2.0 * sigma0 * sigma0 + 1e-12)
    dx0 = Xg[None, :, :] - centers[:, 0][:, None, None]
    dy0 = Yg[None, :, :] - centers[:, 1][:, None, None]
    T = np.exp(-(dx0 * dx0 + dy0 * dy0) * inv2).sum(axis=0)

    Rz = (Rm - Rm.mean()) / (Rm.std() + 1e-12)
    Tz = (T - T.mean()) / (T.std() + 1e-12)
    Ccorr = np.fft.ifft2(np.fft.fft2(Rz) * np.conj(np.fft.fft2(Tz))).real
    iy, ix = np.unravel_index(np.argmax(Ccorr), Ccorr.shape)

    sy = iy if iy <= nbins // 2 else iy - nbins
    sx_v = ix if ix <= nbins // 2 else ix - nbins
    phi_x = float(np.clip(-sx_v * binw, -1.0, 1.0))
    phi_y = float(np.clip(-sy * binw, -1.0, 1.0))

    # Anisotropy from local second moments around the peak of the rate map
    iy0, ix0 = np.unravel_index(np.argmax(Rm), Rm.shape)
    lam_bins = max(1.0, lam / binw)
    rad = int(np.clip(np.round(0.6 * lam_bins), 3, 12))

    y0_s, y1_s = max(0, iy0 - rad), min(nbins, iy0 + rad + 1)
    x0_s, x1_s = max(0, ix0 - rad), min(nbins, ix0 + rad + 1)
    patch = Rm[y0_s:y1_s, x0_s:x1_s].copy()

    floor = np.percentile(patch, 30)
    wgt = np.maximum(patch - floor, 0.0) + 1e-12
    wgt /= np.sum(wgt)

    ys_loc = (np.arange(y0_s, y1_s) - float(iy0)) * binw
    xs_loc = (np.arange(x0_s, x1_s) - float(ix0)) * binw
    YY, XX = np.meshgrid(ys_loc, xs_loc, indexing="ij")

    mx = np.sum(wgt * XX)
    my = np.sum(wgt * YY)
    DX = XX - mx
    DY = YY - my
    Cxx = np.sum(wgt * DX * DX)
    Cyy = np.sum(wgt * DY * DY)
    Cxy = np.sum(wgt * DX * DY)

    Rmat = np.array([[c, s], [-s, c]])
    Cmat = np.array([[Cxx, Cxy], [Cxy, Cyy]])
    Cuv = Rmat @ Cmat @ Rmat.T

    sigma_par = float(np.clip(np.sqrt(max(Cuv[0, 0], 1e-8)), 0.01, 0.8))
    sigma_perp = float(np.clip(np.sqrt(max(Cuv[1, 1], 1e-8)), 0.01, 0.8))

    ratio = sigma_par / max(sigma_perp, 1e-8)
    if ratio > 2.5 or ratio < 0.4:
        sigma_par = 0.5 * sigma_par + 0.5 * sigma0
        sigma_perp = 0.5 * sigma_perp + 0.5 * sigma0
    else:
        sigma_par = 0.7 * sigma_par + 0.3 * sigma0
        sigma_perp = 0.7 * sigma_perp + 0.3 * sigma0

    return {
        "lam": float(lam),
        "theta": float(theta),
        "phi_x": phi_x,
        "phi_y": phi_y,
        "baseline": float(baseline),
        "amplitude": float(amplitude),
        "sigma_par": float(sigma_par),
        "sigma_perp": float(sigma_perp),
    }
