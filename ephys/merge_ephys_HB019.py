"""
Merge spike count arrays from two HB019 recording sessions into a single dataset.

Sessions merged
---------------
Session 0 — HB019_2026-05-28  (131 cells after contamination filtering)
Session 1 — HB019_2026-05-29  (186 cells after contamination filtering)

The two sessions used an identical stimulus grid (180 directions × 4 phases × 4
repeats), so their spike count arrays can be concatenated directly along the cell
axis.  Cluster IDs are local to each session's Kilosort output and can overlap,
so the merged metadata carries a session index to let you trace any cell back to
its source file.

Output
------
HB019_merged_tw2p0.npy
    float32, shape (317, 180, 4, 4)
    axis 0 — cell index across both sessions (0–130: session 0; 131–316: session 1)
    axis 1 — orientation index (ori_values[i] gives degrees)
    axis 2 — phase index (0→0°, 1→90°, 2→180°, 3→270°)
    axis 3 — repeat index (0–3, chronological within each session)

HB019_merged_tw2p0_meta.npz
    ori_values              : int array (180,)  — grating direction in degrees
    phase_values            : int array (4,)    — spatial phase in degrees
    session                 : int array (317,)  — 0 = May-28, 1 = May-29
    sua_unit_ids            : int array (317,)  — Kilosort cluster ID within that session
    sua_isi_violations_ratio: float array (317,) — contamination metric (primary filter)
    sua_frac_rpv            : float array (317,) — raw refractory period violation rate

Usage
-----
python merge_ephys_HB019.py
"""

import numpy as np
from pathlib import Path

SESSIONS = [
    Path('/home/reilly/datasets/hayley_data/HB019_2026-05-28'),
    Path('/home/reilly/datasets/hayley_data/HB019_2026-05-29'),
]
TAG    = 'tw1p5'
OUT_DIR = Path('/home/reilly/datasets/hayley_data')


def load_session(session_dir, tag):
    counts = np.load(session_dir / f'spike_counts_HB019_{tag}.npy').astype(np.float32)
    meta   = np.load(session_dir / f'spike_counts_HB019_{tag}_meta.npz')
    return counts, meta


def main():
    all_counts = []
    all_session = []
    all_unit_ids = []
    all_isi = []
    all_rpv = []
    ref_ori = ref_phase = None

    for si, session_dir in enumerate(SESSIONS):
        counts, meta = load_session(session_dir, TAG)
        ori    = meta['ori_values']
        phase  = meta['phase_values']

        # Verify stimulus grids match across sessions
        if ref_ori is None:
            ref_ori, ref_phase = ori, phase
        else:
            assert np.array_equal(ori, ref_ori),   f'ori_values mismatch in {session_dir}'
            assert np.array_equal(phase, ref_phase), f'phase_values mismatch in {session_dir}'

        n_cells = counts.shape[0]
        print(f'Session {si} ({session_dir.name}): {n_cells} cells, shape {counts.shape}')

        all_counts.append(counts)
        all_session.append(np.full(n_cells, si, dtype=np.int32))
        all_unit_ids.append(meta['sua_unit_ids'])
        all_isi.append(meta['sua_isi_violations_ratio'])
        all_rpv.append(meta['sua_frac_rpv'])

    merged_counts  = np.concatenate(all_counts,   axis=0)
    merged_session = np.concatenate(all_session,   axis=0)
    merged_ids     = np.concatenate(all_unit_ids,  axis=0)
    merged_isi     = np.concatenate(all_isi,       axis=0)
    merged_rpv     = np.concatenate(all_rpv,       axis=0)

    out_counts = OUT_DIR / f'HB019_merged_{TAG}.npy'
    out_meta   = OUT_DIR / f'HB019_merged_{TAG}_meta.npz'

    np.save(out_counts, merged_counts)
    np.savez(out_meta,
             ori_values=ref_ori,
             phase_values=ref_phase,
             session=merged_session,
             sua_unit_ids=merged_ids,
             sua_isi_violations_ratio=merged_isi,
             sua_frac_rpv=merged_rpv)

    print(f'\nMerged counts: {merged_counts.shape}  dtype: {merged_counts.dtype}')
    print(f'  Session 0 cells: {(merged_session == 0).sum()}')
    print(f'  Session 1 cells: {(merged_session == 1).sum()}')
    print(f'Saved {out_counts}')
    print(f'Saved {out_meta}')


if __name__ == '__main__':
    main()
