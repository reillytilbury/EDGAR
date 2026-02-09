import numpy as np
from scipy.ndimage import gaussian_filter

N_BINS = 65
SMOOTHING_SIGMA = 1.0
HIST_RANGE = ((-1.0, 1.0), (-1.0, 1.0))
TIME_BIN = 0.01
X_KEY = "x"
Y_KEY = "y"
T_KEY = "t"
SPIKES_KEY = "spikes_mod1"
EPS = 1e-6


def load_data(
    data_path: str | None = None,
    min_spikes: int = 20,
    spatial_reliability_min: float | None = 0.2,
    grid_score_min: float | None = -0.5,
):
    """
    Load grid cell dataset and return (X, Y).

    Supports .npz or .npy (dict-like) with keys for x, y, t, and spikes_mod1.

    X: (2, n_timepoints) with x,y positions
    Y: (n_cells, n_timepoints) spike counts per time bin

    This mirrors the grid-cells scratchpad preprocessing:
    - fixed keys: x, y, t, spikes_mod1
    - fixed time bin: 0.01s
    - histogram range: [-1, 1] for both x and y
    - rate maps are smoothed occupancy-normalized (no min_occ masking)
    - spatial reliability and grid score computed from these smoothed maps
    """
    if data_path is None:
        raise ValueError("Grid cell data_path is not set. Provide data_path or set DATA_PATH.")

    data = np.load(data_path, allow_pickle=True)

    def _get(key: str):
        if isinstance(data, np.lib.npyio.NpzFile):
            if key not in data.files:
                raise KeyError(f"Missing key '{key}' in grid cell data.")
            return data[key]
        if data.shape == () and hasattr(data, "item"):
            obj = data.item()
        else:
            obj = data
        if not isinstance(obj, dict):
            raise ValueError("Unsupported data format for grid cells.")
        if key not in obj:
            raise KeyError(f"Missing key '{key}' in grid cell data.")
        return obj[key]

    x_all = np.asarray(_get(X_KEY)).reshape(-1)
    y_all = np.asarray(_get(Y_KEY)).reshape(-1)
    t_all = np.asarray(_get(T_KEY)).reshape(-1)

    if x_all.shape != y_all.shape or x_all.shape != t_all.shape:
        raise ValueError(
            f"x, y, t must share shape. Got x {x_all.shape}, y {y_all.shape}, t {t_all.shape}."
        )

    time_mask = np.ones_like(t_all, dtype=bool)

    x0 = x_all[time_mask]
    y0 = y_all[time_mask]
    t0 = t_all[time_mask]
    if t0.size == 0:
        return np.zeros((2, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    valid = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(t0)
    valid_idx = np.where(valid)[0]
    idx_map = -np.ones(len(t0), dtype=int)
    idx_map[valid_idx] = np.arange(valid_idx.size)

    x = x0[valid]
    y = y0[valid]
    n_valid = len(x)
    if n_valid == 0:
        return np.zeros((2, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    dt = TIME_BIN
    t_start_val = float(t0[0])
    t_min = float(t0[0])
    t_max = float(t0[-1])

    def _rate_map(pos_xy, counts, bins=N_BINS, smoothing_sigma=SMOOTHING_SIGMA, eps=EPS):
        occ, xedges, yedges = np.histogram2d(
            pos_xy[:, 0],
            pos_xy[:, 1],
            bins=bins,
            range=HIST_RANGE,
        )
        spk, _, _ = np.histogram2d(
            pos_xy[:, 0],
            pos_xy[:, 1],
            bins=[xedges, yedges],
            weights=counts,
        )
        occ_s = gaussian_filter(occ, sigma=smoothing_sigma)
        spk_s = gaussian_filter(spk, sigma=smoothing_sigma)
        return spk_s / (occ_s + eps)

    def _spatial_reliability(pos_xy, counts):
        n = len(counts)
        if n < 10:
            return 0.0
        split = n // 2
        rate1 = _rate_map(pos_xy[:split], counts[:split])
        rate2 = _rate_map(pos_xy[split:], counts[split:])
        v1 = rate1.ravel()
        v2 = rate2.ravel()
        if np.std(v1) == 0 or np.std(v2) == 0:
            return 0.0
        return float(np.corrcoef(v1, v2)[0, 1])

    def _autocorr2d(Z):
        Z = Z - np.mean(Z)
        F = np.fft.fft2(Z)
        ac = np.fft.ifft2(F * np.conj(F)).real
        ac = np.fft.fftshift(ac)
        ac /= (np.max(ac) + 1e-8)
        return ac

    def _estimate_grid_from_autocorr(ac, top_k=6):
        h, w = ac.shape
        cy, cx = h // 2, w // 2
        Y, X = np.indices(ac.shape)
        R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = R > 2
        if np.sum(mask) == 0:
            return 0.0
        vals = ac[mask]
        coords = np.column_stack(np.nonzero(mask))
        if vals.size == 0:
            return 0.0
        top_n = min(top_k, vals.size)
        top_idx = np.argpartition(vals, -top_n)[-top_n:]
        peak_coords = coords[top_idx]
        vecs = peak_coords - np.array([cy, cx])
        dists = np.sqrt(np.sum(vecs ** 2, axis=1))
        if len(dists) == 0:
            return 0.0
        return float(np.median(dists))

    def _rotate_grid(ac, angle_deg):
        angle = np.deg2rad(angle_deg)
        c, s = np.cos(angle), np.sin(angle)
        h, w = ac.shape
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        Y, X = np.indices(ac.shape)
        x0 = X - cx
        y0 = Y - cy
        xr = c * x0 + s * y0 + cx
        yr = -s * x0 + c * y0 + cy

        x0f = np.floor(xr).astype(int)
        y0f = np.floor(yr).astype(int)
        x1f = x0f + 1
        y1f = y0f + 1

        valid = (x0f >= 0) & (x1f < w) & (y0f >= 0) & (y1f < h)
        out = np.zeros_like(ac)
        if not np.any(valid):
            return out

        x0v = x0f[valid]
        y0v = y0f[valid]
        x1v = x1f[valid]
        y1v = y1f[valid]
        dx = xr[valid] - x0v
        dy = yr[valid] - y0v

        v00 = ac[y0v, x0v]
        v10 = ac[y0v, x1v]
        v01 = ac[y1v, x0v]
        v11 = ac[y1v, x1v]

        out[valid] = (
            v00 * (1 - dx) * (1 - dy)
            + v10 * dx * (1 - dy)
            + v01 * (1 - dx) * dy
            + v11 * dx * dy
        )
        return out

    def _grid_score(pos_xy, counts):
        rate = _rate_map(pos_xy, counts)
        ac = _autocorr2d(rate)
        spacing_pix = _estimate_grid_from_autocorr(ac)
        if spacing_pix <= 0:
            return -np.inf
        h, w = ac.shape
        cy, cx = h // 2, w // 2
        Y, X = np.indices(ac.shape)
        R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        inner = 0.5 * spacing_pix
        outer = 1.5 * spacing_pix
        ring = (R >= inner) & (R <= outer)
        if np.sum(ring) < 10:
            return -np.inf

        def _corr(a, b, mask):
            a = a[mask].ravel()
            b = b[mask].ravel()
            if a.size < 10:
                return 0.0
            a = a - np.mean(a)
            b = b - np.mean(b)
            denom = np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-8
            return float(np.sum(a * b) / denom)

        ac30 = _rotate_grid(ac, 30.0)
        ac60 = _rotate_grid(ac, 60.0)
        ac90 = _rotate_grid(ac, 90.0)
        ac120 = _rotate_grid(ac, 120.0)
        ac150 = _rotate_grid(ac, 150.0)

        c30 = _corr(ac, ac30, ring)
        c60 = _corr(ac, ac60, ring)
        c90 = _corr(ac, ac90, ring)
        c120 = _corr(ac, ac120, ring)
        c150 = _corr(ac, ac150, ring)

        return min(c60, c120) - max(c30, c90, c150)

    spikes = _get(SPIKES_KEY)
    if isinstance(spikes, np.ndarray) and spikes.shape == () and hasattr(spikes, "item"):
        spikes = spikes.item()

    if isinstance(spikes, dict):
        spike_list = list(spikes.values())
    else:
        spike_list = list(spikes)

    pos_xy = np.stack([x, y], axis=1).astype(np.float32)

    unit_counts = []
    for times in spike_list:
        if times is None:
            continue
        times = np.asarray(times, dtype=np.float32).reshape(-1)
        times = times[(times >= t_min) & (times <= t_max)]
        if times.size == 0:
            continue
        idx_full = np.rint((times - t_start_val) / dt).astype(int)
        idx_full = idx_full[(idx_full >= 0) & (idx_full < len(t0))]
        if idx_full.size == 0:
            continue
        idx_valid = idx_map[idx_full]
        idx_valid = idx_valid[idx_valid >= 0]
        if idx_valid.size == 0:
            continue
        counts = np.bincount(idx_valid, minlength=n_valid).astype(np.float32)
        total = float(np.sum(counts))
        if total < min_spikes:
            continue
        if spatial_reliability_min is not None:
            rel = _spatial_reliability(pos_xy, counts)
            if not np.isfinite(rel) or rel < spatial_reliability_min:
                continue
        if grid_score_min is not None:
            score = _grid_score(pos_xy, counts)
            if not np.isfinite(score) or score < grid_score_min:
                continue
        unit_counts.append(counts)

    if len(unit_counts) == 0:
        Y = np.zeros((0, n_valid), dtype=np.float32)
    else:
        Y = np.stack(unit_counts, axis=0).astype(np.float32)
    X = np.stack([x, y], axis=0).astype(np.float32)
    return X, Y

# # ---------------------------------------------------------------------------
# # Example visualization (commented out): plot a few rate maps with metrics.
# # ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     data_path = "/home/reilly/Desktop/Toroidal_topology_grid_cell_data/rat_q_grid_modules_1_2.npz"
#     X, Y = load_data(
#         data_path=data_path,
#         spatial_reliability_min=None,
#         grid_score_min=None,
#     )
#     if Y.shape[0] == 0:
#         print("No cells to plot.")
#     else:
#         pos_xy = X.T

#         def _rate_map(pos_xy, counts, bins=N_BINS, smoothing_sigma=SMOOTHING_SIGMA, eps=EPS):
#             occ, xedges, yedges = np.histogram2d(
#                 pos_xy[:, 0],
#                 pos_xy[:, 1],
#                 bins=bins,
#                 range=HIST_RANGE,
#             )
#             spk, _, _ = np.histogram2d(
#                 pos_xy[:, 0],
#                 pos_xy[:, 1],
#                 bins=[xedges, yedges],
#                 weights=counts,
#             )
#             occ_s = gaussian_filter(occ, sigma=smoothing_sigma)
#             spk_s = gaussian_filter(spk, sigma=smoothing_sigma)
#             return spk_s / (occ_s + eps)

#         def _spatial_reliability(pos_xy, counts):
#             n = len(counts)
#             if n < 10:
#                 return 0.0
#             split = n // 2
#             rate1 = _rate_map(pos_xy[:split], counts[:split])
#             rate2 = _rate_map(pos_xy[split:], counts[split:])
#             v1 = rate1.ravel()
#             v2 = rate2.ravel()
#             if np.std(v1) == 0 or np.std(v2) == 0:
#                 return 0.0
#             return float(np.corrcoef(v1, v2)[0, 1])

#         def _autocorr2d(Z):
#             Z = Z - np.mean(Z)
#             F = np.fft.fft2(Z)
#             ac = np.fft.ifft2(F * np.conj(F)).real
#             ac = np.fft.fftshift(ac)
#             ac /= (np.max(ac) + 1e-8)
#             return ac

#         def _estimate_grid_from_autocorr(ac, top_k=6):
#             h, w = ac.shape
#             cy, cx = h // 2, w // 2
#             Y, X = np.indices(ac.shape)
#             R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
#             mask = R > 2
#             if np.sum(mask) == 0:
#                 return 0.0
#             vals = ac[mask]
#             coords = np.column_stack(np.nonzero(mask))
#             if vals.size == 0:
#                 return 0.0
#             top_n = min(top_k, vals.size)
#             top_idx = np.argpartition(vals, -top_n)[-top_n:]
#             peak_coords = coords[top_idx]
#             vecs = peak_coords - np.array([cy, cx])
#             dists = np.sqrt(np.sum(vecs ** 2, axis=1))
#             return float(np.median(dists)) if len(dists) else 0.0

#         def _rotate_grid(ac, angle_deg):
#             angle = np.deg2rad(angle_deg)
#             c, s = np.cos(angle), np.sin(angle)
#             h, w = ac.shape
#             cy = (h - 1) / 2.0
#             cx = (w - 1) / 2.0
#             Y, X = np.indices(ac.shape)
#             x0 = X - cx
#             y0 = Y - cy
#             xr = c * x0 + s * y0 + cx
#             yr = -s * x0 + c * y0 + cy

#             x0f = np.floor(xr).astype(int)
#             y0f = np.floor(yr).astype(int)
#             x1f = x0f + 1
#             y1f = y0f + 1

#             valid = (x0f >= 0) & (x1f < w) & (y0f >= 0) & (y1f < h)
#             out = np.zeros_like(ac)
#             if not np.any(valid):
#                 return out

#             x0v = x0f[valid]
#             y0v = y0f[valid]
#             x1v = x1f[valid]
#             y1v = y1f[valid]
#             dx = xr[valid] - x0v
#             dy = yr[valid] - y0v

#             v00 = ac[y0v, x0v]
#             v10 = ac[y0v, x1v]
#             v01 = ac[y1v, x0v]
#             v11 = ac[y1v, x1v]

#             out[valid] = (
#                 v00 * (1 - dx) * (1 - dy)
#                 + v10 * dx * (1 - dy)
#                 + v01 * (1 - dx) * dy
#                 + v11 * dx * dy
#             )
#             return out

#         def _grid_score(pos_xy, counts):
#             rate = _rate_map(pos_xy, counts)
#             ac = _autocorr2d(rate)
#             spacing_pix = _estimate_grid_from_autocorr(ac)
#             if spacing_pix <= 0:
#                 return -np.inf
#             h, w = ac.shape
#             cy, cx = h // 2, w // 2
#             Y, X = np.indices(ac.shape)
#             R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
#             inner = 0.5 * spacing_pix
#             outer = 1.5 * spacing_pix
#             ring = (R >= inner) & (R <= outer)
#             if np.sum(ring) < 10:
#                 return -np.inf

#             def _corr(a, b, mask):
#                 a = a[mask].ravel()
#                 b = b[mask].ravel()
#                 if a.size < 10:
#                     return 0.0
#                 a = a - np.mean(a)
#                 b = b - np.mean(b)
#                 denom = np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-8
#                 return float(np.sum(a * b) / denom)

#             ac30 = _rotate_grid(ac, 30.0)
#             ac60 = _rotate_grid(ac, 60.0)
#             ac90 = _rotate_grid(ac, 90.0)
#             ac120 = _rotate_grid(ac, 120.0)
#             ac150 = _rotate_grid(ac, 150.0)

#             c30 = _corr(ac, ac30, ring)
#             c60 = _corr(ac, ac60, ring)
#             c90 = _corr(ac, ac90, ring)
#             c120 = _corr(ac, ac120, ring)
#             c150 = _corr(ac, ac150, ring)

#             return min(c60, c120) - max(c30, c90, c150)

#         n_show = min(6, Y.shape[0])
#         idx = np.random.choice(Y.shape[0], size=n_show, replace=False)
#         n_cols = min(3, n_show)
#         n_rows = (n_show + n_cols - 1) // n_cols
#         fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
#         for ax, i in zip(axes.ravel(), idx):
#             rm = _rate_map(pos_xy, Y[i])
#             rel = _spatial_reliability(pos_xy, Y[i])
#             score = _grid_score(pos_xy, Y[i])
#             ax.imshow(rm, origin="lower", cmap="viridis")
#             ax.set_title(f"cell {i} | rel {rel:.2f} | grid {score:.2f}")
#             ax.set_xticks([])
#             ax.set_yticks([])
#         for ax in axes.ravel()[len(idx):]:
#             ax.axis("off")
#         plt.tight_layout()
#         plt.show()
