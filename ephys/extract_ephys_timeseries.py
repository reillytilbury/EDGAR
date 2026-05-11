"""
Extract binned spike count time series from Hayley's curated npz files.

Usage
-----
python extract_ephys_timeseries.py                       # 50 ms bins, full stim window
python extract_ephys_timeseries.py --bin_size 0.1        # 100 ms bins
python extract_ephys_timeseries.py --t_start -0.2 --t_stop 1.2  # include pre/post stim

Output shape: (n_cells, n_orientations, n_repeats, n_time_bins) = (564, 360, 6, n_t), float32
  axis 0: global cell index 0–563  (0–229: 2026-03-11, 230–368: 2026-03-12, 369–563: 2026-03-13)
  axis 1: orientation in degrees   (index == degrees, 0–359)
  axis 2: repeat                   (0–5; NaN where session 1 has only 5 repeats)
  axis 3: time bin                 (bin i covers [t_start + i*bin_size, t_start + (i+1)*bin_size) s)

Divide by bin_size to convert to Hz.
Metadata saved alongside: bin_edges (n_t+1,) in seconds relative to stim onset.
"""

import argparse
import numpy as np
from pathlib import Path

DATA_ROOT = Path('/home/reilly/datasets/hayley_data')
OUT_DIR   = DATA_ROOT

SESSIONS = [
    ('HB005_2026-03-11_g0', 'HB005', '2026-03-11'),
    ('HB005_2026-03-12_g0', 'HB005', '2026-03-12'),
    ('HB005_2026-03-13_g0', 'HB005', '2026-03-13'),
]

N_ORI = 360
N_REP = 6


def compute_spike_timeseries(t_start=0.0, t_stop=1.0, bin_size=0.05):
    bin_edges = np.arange(t_start, t_stop + bin_size / 2, bin_size)
    bin_edges[-1] = t_stop   # snap last edge to exact t_stop
    n_t = len(bin_edges) - 1
    all_counts = []

    for session_id, _, _ in SESSIONS:
        npz = DATA_ROOT / session_id / f'{session_id}_curated_results.npz'
        raw = np.load(npz, allow_pickle=True)

        spike_times    = raw['spike_times']
        spike_clusters = raw['spike_clusters']
        sua_units      = raw['sua_units'].astype(int)
        trial_starts   = raw['trial_starts']
        trial_ends     = raw['trial_ends']
        stimulus_type  = raw['stimulus_type']
        stimulus_id    = raw['stimulus_id']

        grating_mask  = stimulus_type == 'fs_grating_static'
        g_starts      = trial_starts[grating_mask]
        g_ids         = stimulus_id[grating_mask].astype(int)
        stim_duration = float(np.median(trial_ends[grating_mask] - g_starts))

        sua_mask = np.isin(spike_clusters, sua_units)
        st_sua   = spike_times[sua_mask]
        sc_sua   = spike_clusters[sua_mask]

        assert np.all(np.diff(st_sua) >= 0), f'{session_id}: spike times not sorted'

        n_cells = len(sua_units)
        counts  = np.full((n_cells, N_ORI, N_REP, n_t), np.nan, dtype=np.float32)

        for ci, cell_id in enumerate(sua_units):
            cell_st = st_sua[sc_sua == cell_id]

            for ori in range(N_ORI):
                for r, t_idx in enumerate(np.where(g_ids == ori)[0]):
                    t0 = g_starts[t_idx]
                    # clip bin edges to stim window so we don't count spikes past trial end
                    edges = np.clip(t0 + bin_edges, t0 + t_start, t0 + stim_duration)
                    counts[ci, ori, r] = np.diff(np.searchsorted(cell_st, edges))

            if (ci + 1) % 50 == 0 or ci == n_cells - 1:
                print(f'  {session_id}: {ci + 1}/{n_cells} cells')

        all_counts.append(counts)

    return np.concatenate(all_counts, axis=0), bin_edges


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--t_start',  type=float, default=0.0,
                        help='Window start in s relative to stim onset (default 0)')
    parser.add_argument('--t_stop',   type=float, default=1.0,
                        help='Window stop in s relative to stim onset (default 1)')
    parser.add_argument('--bin_size', type=float, default=0.02,
                        help='Bin width in seconds (default 0.02 = 20 ms)')
    args = parser.parse_args()

    print(f'Extracting time series: [{args.t_start}, {args.t_stop}) s, bin_size={args.bin_size*1000:.0f} ms ...')
    counts, bin_edges = compute_spike_timeseries(args.t_start, args.t_stop, args.bin_size)

    tag     = f'bs{int(args.bin_size*1000)}ms_{args.t_start}-{args.t_stop}'.replace('.', 'p')
    outpath = OUT_DIR / f'spike_timeseries_{tag}.npy'
    np.save(outpath, counts)
    np.save(OUT_DIR / f'spike_timeseries_{tag}_bin_edges.npy', bin_edges)

    print(f'Saved {outpath}')
    print(f'Shape: {counts.shape}  dtype: {counts.dtype}')
    print(f'Bin edges: {bin_edges[0]:.3f} – {bin_edges[-1]:.3f} s  ({len(bin_edges)-1} bins)')
    print(f'NaN entries: {np.isnan(counts).sum():,}')
    print(f'Value range: {np.nanmin(counts):.0f} – {np.nanmax(counts):.0f} spikes/bin')
