"""
Compute spike count arrays directly from Hayley's curated npz files.

Usage
-----
python extract_ephys.py                             # full stim window [0, 1.019 s]
python extract_ephys.py --t_start 0 --t_stop 0.25  # first 250 ms sub-window

Output shape: (n_cells, n_orientations, n_repeats) = (564, 360, 6), float32
  axis 0: global cell index 0–563  (0–229: 2026-03-11, 230–368: 2026-03-12, 369–563: 2026-03-13)
  axis 1: orientation in degrees   (index == degrees, 0–359)
  axis 2: repeat                   (0–5; NaN where session 1 has only 5 repeats)

Divide by (t_stop - t_start) to convert to Hz.
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

OPTO_TYPES = {'-1', 'opto_stim_grating'}
N_ORI      = 360
N_REP      = 6


def compute_spike_counts(t_start=0.0, t_stop=1.0):
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

        grating_mask = stimulus_type == 'fs_grating_static'
        g_starts     = trial_starts[grating_mask]
        g_ids        = stimulus_id[grating_mask].astype(int)
        orientations = np.arange(N_ORI)

        stim_duration = float(np.median(trial_ends[grating_mask] - g_starts))

        # resolve t_stop relative to stim window
        effective_stop = min(t_stop, stim_duration)

        sua_mask = np.isin(spike_clusters, sua_units)
        st_sua   = spike_times[sua_mask]
        sc_sua   = spike_clusters[sua_mask]

        assert np.all(np.diff(st_sua) >= 0), f'{session_id}: spike times not sorted'

        n_cells = len(sua_units)
        counts  = np.full((n_cells, N_ORI, N_REP), np.nan, dtype=np.float32)

        for ci, cell_id in enumerate(sua_units):
            cell_st = st_sua[sc_sua == cell_id]

            for ori in orientations:
                ori_trial_idxs = np.where(g_ids == ori)[0]
                for r, t_idx in enumerate(ori_trial_idxs):
                    t0 = g_starts[t_idx]
                    lo = np.searchsorted(cell_st, t0 + t_start, side='left')
                    hi = np.searchsorted(cell_st, t0 + effective_stop, side='right')
                    counts[ci, ori, r] = hi - lo

        all_counts.append(counts)
        print(f'  {session_id}: {n_cells} cells done')

    return np.concatenate(all_counts, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--t_start', type=float, default=0.0,
                        help='Window start in s relative to stim onset (default 0)')
    parser.add_argument('--t_stop',  type=float, default=1.0,
                        help='Window stop in s relative to stim onset (default 1)')
    args = parser.parse_args()

    print(f'Computing spike counts in [{args.t_start}, {args.t_stop}) s ...')
    counts = compute_spike_counts(args.t_start, args.t_stop)

    tag     = f'{args.t_start}-{args.t_stop}'.replace('.', 'p')
    outpath = OUT_DIR / f'spike_counts_{tag}.npy'
    np.save(outpath, counts)

    print(f'Saved {outpath}')
    print(f'Shape: {counts.shape}  dtype: {counts.dtype}')
    print(f'NaN entries: {np.isnan(counts).sum():,}')
    print(f'Value range: {np.nanmin(counts):.0f} – {np.nanmax(counts):.0f} spikes')
