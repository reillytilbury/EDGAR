import numpy as np
from pathlib import Path
import glob


def load_data(data_path: str | None = None,
              min_spikes: int = 20,
              max_spikes: int | None = 4000,
              spatial_reliability_min: float | None = 0.4,
              reliability_bins: int = 40,
              reliability_min_occ: int = 3,
              reliability_smoothing_sigma: float = 1.2,
              fs_spk: float = 20000.0,
              fs_pos: float = 39.06):
    """
    Load place cell dataset and return merged (X, Y) across sessions.

    Supports:
    - a directory containing session subfolders (e.g., /home/reilly/Desktop/hc2)
    - a single session folder with .whl/.res/.clu files

    X: (n_stim_dim, n_trials) with n_stim_dim=4 (x, y, head_direction, speed)
    Y: (n_cells, n_trials) spike counts (zero-padded across sessions)
    """
    if data_path is None:
        raise ValueError("Place cell data_path is not set. Provide data_path or set DATA_PATH.")

    base = Path(data_path)
    if isinstance(data_path, (list, tuple)):
        sessions = [Path(p) for p in data_path]
    elif base.is_dir():
        whl_files = list(base.glob("*.whl"))
        if whl_files:
            sessions = [base]
        else:
            sessions = sorted([p for p in base.iterdir() if p.is_dir()])
    else:
        raise ValueError(f"Invalid data_path: {data_path}")

    def _load_session(session_dir: Path):
        def _rate_map(pos_xy, counts, bins, min_occ, smoothing_sigma, eps=1e-6):
            def _gaussian_kernel(sigma):
                if sigma <= 0:
                    return None
                size = int(max(3, np.ceil(6 * sigma)))
                if size % 2 == 0:
                    size += 1
                ax = np.arange(size) - size // 2
                xx, yy = np.meshgrid(ax, ax)
                kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
                kernel /= np.sum(kernel)
                return kernel

            def _fft_convolve2d(a, b):
                out_shape = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)
                fa = np.fft.rfftn(a, s=out_shape)
                fb = np.fft.rfftn(b, s=out_shape)
                out = np.fft.irfftn(fa * fb, s=out_shape)
                start0 = (b.shape[0] - 1) // 2
                start1 = (b.shape[1] - 1) // 2
                return out[start0:start0 + a.shape[0], start1:start1 + a.shape[1]]

            occ, xedges, yedges = np.histogram2d(pos_xy[:, 0], pos_xy[:, 1], bins=bins)
            spk, _, _ = np.histogram2d(pos_xy[:, 0], pos_xy[:, 1], bins=[xedges, yedges], weights=counts)
            rate = spk / (occ + eps)
            rate = np.where(occ >= min_occ, rate, np.nan)
            kernel = _gaussian_kernel(smoothing_sigma)
            if kernel is not None:
                rate_filled = np.nan_to_num(rate, nan=0.0)
                occ_mask = np.isfinite(rate).astype(float)
                rate_s = _fft_convolve2d(rate_filled, kernel)
                norm = _fft_convolve2d(occ_mask, kernel)
                rate = rate_s / (norm + eps)
                rate = np.where(norm > 0, rate, np.nan)
            return rate

        def _spatial_reliability(pos_xy, counts):
            n = len(counts)
            if n < 10:
                return 0.0
            split = n // 2
            rate1 = _rate_map(pos_xy[:split], counts[:split], reliability_bins, reliability_min_occ, reliability_smoothing_sigma)
            rate2 = _rate_map(pos_xy[split:], counts[split:], reliability_bins, reliability_min_occ, reliability_smoothing_sigma)
            mask = np.isfinite(rate1) & np.isfinite(rate2)
            if np.sum(mask) < 10:
                return 0.0
            v1 = rate1[mask].ravel()
            v2 = rate2[mask].ravel()
            if np.std(v1) == 0 or np.std(v2) == 0:
                return 0.0
            return float(np.corrcoef(v1, v2)[0, 1])

        prefix = session_dir.name[-10:]
        whl_path = session_dir / f"{prefix}.whl"
        if not whl_path.exists():
            raise FileNotFoundError(f"Missing .whl file in {session_dir}")
        whl = np.loadtxt(whl_path)
        x0, y0 = whl[:, 0], whl[:, 1]
        valid = (x0 >= 0) & (y0 >= 0)
        valid_idx = np.where(valid)[0]
        x = x0[valid]
        y = y0[valid]
        n_valid = len(x)
        dt = 1.0 / fs_pos

        idx_map = -np.ones(len(x0), dtype=int)
        idx_map[valid_idx] = np.arange(n_valid)

        def spikes_to_pos_idx(res_samples):
            spk_t = res_samples / fs_spk
            spk_idx = (spk_t / dt).astype(int)
            ok = (spk_idx >= 0) & (spk_idx < len(x0)) & valid[spk_idx]
            return spk_idx[ok]

        clu_files = sorted(glob.glob(str(session_dir / f"{prefix}.clu.*")))
        shanks = sorted({int(Path(f).suffix.split('.')[-1]) for f in clu_files})

        unit_counts = []
        for shank in shanks:
            res = np.loadtxt(session_dir / f"{prefix}.res.{shank}", dtype=int)
            clu = np.loadtxt(session_dir / f"{prefix}.clu.{shank}", dtype=int)
            n_units = int(clu[0])
            clu = clu[1:]
            for u in range(n_units):
                res_u = res[clu == u]
                if len(res_u) < min_spikes:
                    continue
                if max_spikes is not None and len(res_u) > max_spikes:
                    continue
                spk_idx_full = spikes_to_pos_idx(res_u)
                spk_idx_valid = idx_map[spk_idx_full]
                spk_idx_valid = spk_idx_valid[spk_idx_valid >= 0]
                counts = np.bincount(spk_idx_valid, minlength=n_valid).astype(np.float32)
                if spatial_reliability_min is not None:
                    rel = _spatial_reliability(np.stack([x, y], axis=1), counts)
                    if not np.isfinite(rel) or rel < spatial_reliability_min:
                        continue
                unit_counts.append(counts)

        if len(unit_counts) == 0:
            Y = np.zeros((0, n_valid), dtype=np.float32)
        else:
            Y = np.stack(unit_counts, axis=0)
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        speed = np.sqrt(dx ** 2 + dy ** 2) / dt
        head_dir = np.arctan2(dy, dx)
        X = np.stack([x, y, head_dir, speed], axis=0).astype(np.float32)
        return X, Y

    session_data = []
    total_trials = 0
    for s in sessions:
        X_s, Y_s = _load_session(s)
        session_data.append((X_s, Y_s))
        total_trials += X_s.shape[1]

    if total_trials == 0:
        return np.zeros((2, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    X_total = np.concatenate([X_s for X_s, _ in session_data], axis=1)
    Y_blocks = []
    offset = 0
    for X_s, Y_s in session_data:
        n_trials = X_s.shape[1]
        pad_left = offset
        pad_right = total_trials - offset - n_trials
        if pad_left > 0 or pad_right > 0:
            Y_pad = np.pad(Y_s, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0)
        else:
            Y_pad = Y_s
        Y_blocks.append(Y_pad)
        offset += n_trials
    Y_total = np.concatenate(Y_blocks, axis=0) if Y_blocks else np.zeros((0, total_trials), dtype=np.float32)
    return X_total, Y_total
