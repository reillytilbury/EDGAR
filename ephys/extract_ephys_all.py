"""
Extract spike count arrays for every session in hayley_data_all/, auto-detecting whether
each session used static gratings (HB005/HB006-style: `stimulus_id`, single presentation
per trial) or drifting gratings (HB008/HB019/HB020-style: `stimulus_ori` + `stimulus_phase`).
Then merges same-animal sessions that share an identical stimulus grid into one file per
animal (mirrors what merge_ephys_HB019.py did by hand for the two original HB019 sessions).

Per-session outputs (saved inside that session's own folder, same convention as
extract_ephys_HB019.py):
    spike_counts_<session>_<tag>.npy
    spike_counts_<session>_<tag>_meta.npz

Per-animal merged outputs (only when >=2 sessions share the same kind + stimulus grid;
saved at the hayley_data_all/ root):
    <ANIMAL>_merged_<tag>.npy
    <ANIMAL>_merged_<tag>_meta.npz

Drifting sessions  -> counts shape (n_cells, 180 ori, 4 phase, n_rep), meta has
                       ori_values, phase_values, sua_unit_ids, sua_isi_violations_ratio,
                       sua_frac_rpv, session, session_names (merged only).
Static sessions    -> counts shape (n_cells, n_unique_stimulus_ids, n_rep), meta has
                       id_values, grating_label, sua_unit_ids, sua_isi_violations_ratio,
                       sua_frac_rpv, session, session_names (merged only).

Contamination filtering (MAX_ISI_VIOLATIONS_RATIO) is applied only when the session's npz
carries `all_unit_isi_violations_ratio` (HB019, HB020). HB005/HB006/HB008 don't have this
metric, so all `sua_units` are kept for those and sua_isi/sua_frac_rpv are saved as NaN.

Usage
-----
python extract_ephys_all.py
"""

import re
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_ROOT = Path('/home/reilly/datasets/hayley_data_all')

T_WINDOW_DRIFT = 1.5   # s, spike-counting window ending at trial_end (drifting gratings)
T_START_STATIC = 0.0   # s, relative to trial start (static gratings)
T_STOP_STATIC  = 1.0

MAX_ISI_VIOLATIONS_RATIO = 0.05   # primary contamination filter, when available; None = no filter
MAX_FRAC_RPV             = None


def animal_from_session(session_id):
    return re.match(r'([A-Za-z]+\d+)', session_id).group(1)


def apply_contamination_filter(raw, sua_units):
    """Filter by ISI-violation ratio / frac RPV when the npz carries those metrics;
    otherwise keep every SUA unit and report NaN for the (unavailable) metrics."""
    if 'all_unit_isi_violations_ratio' not in raw.files:
        return sua_units, np.full(len(sua_units), np.nan), np.full(len(sua_units), np.nan)

    isi_ratio = raw['all_unit_isi_violations_ratio']
    frac_rpv  = raw['all_unit_frac_rpv']
    mask = np.ones(len(sua_units), dtype=bool)
    if MAX_ISI_VIOLATIONS_RATIO is not None:
        mask &= isi_ratio[sua_units] <= MAX_ISI_VIOLATIONS_RATIO
    if MAX_FRAC_RPV is not None:
        mask &= frac_rpv[sua_units] <= MAX_FRAC_RPV
    n_dropped = (~mask).sum()
    if n_dropped:
        print(f'    Contamination filter: dropped {n_dropped}/{len(sua_units)} SUA units '
              f'(ISI ratio <= {MAX_ISI_VIOLATIONS_RATIO}, frac RPV <= {MAX_FRAC_RPV})')
    sua_units = sua_units[mask]
    return sua_units, isi_ratio[sua_units], frac_rpv[sua_units]


# -----------------------------------------------------------------------------
# Drifting gratings (HB008 / HB019 / HB020-style): stimulus_ori + stimulus_phase
# -----------------------------------------------------------------------------

def extract_drifting(npz_path, t_window=T_WINDOW_DRIFT):
    raw = np.load(npz_path, allow_pickle=True)
    spike_times    = raw['spike_times']
    spike_clusters = raw['spike_clusters']
    sua_units      = raw['sua_units'].astype(int)
    trial_ends     = raw['trial_ends']      # only reliable time reference (see extract_ephys_HB019.py)
    stimulus_ori   = raw['stimulus_ori']
    stimulus_phase = raw['stimulus_phase']
    assert np.all(np.diff(spike_times) >= 0), f'{npz_path}: spike_times not sorted'

    sua_units, sua_isi, sua_rpv = apply_contamination_filter(raw, sua_units)

    ori_values   = np.arange(0, 360, 2, dtype=int)
    phase_values = np.array([0, 90, 180, 270], dtype=int)
    ori_index    = {o: i for i, o in enumerate(ori_values)}
    phase_index  = {p: i for i, p in enumerate(phase_values)}

    grating_mask = ~np.isnan(stimulus_ori)
    trial_ends     = trial_ends[grating_mask]
    stimulus_ori   = stimulus_ori[grating_mask]
    stimulus_phase = stimulus_phase[grating_mask]

    order       = np.argsort(trial_ends)   # chronological
    t_ends_sort = trial_ends[order]
    ori_sort    = stimulus_ori[order]
    phase_sort  = stimulus_phase[order]

    oi_arr = np.array([ori_index[int(round(o))]   for o in ori_sort])
    pi_arr = np.array([phase_index[int(round(p))] for p in phase_sort])

    n_ori, n_phase = len(ori_values), len(phase_values)
    rep_counter = np.zeros((n_ori, n_phase), dtype=int)
    ri_arr = np.empty(len(order), dtype=int)
    for k in range(len(order)):
        oi, pi = oi_arr[k], pi_arr[k]
        ri_arr[k] = rep_counter[oi, pi]
        rep_counter[oi, pi] += 1
    n_rep = int(rep_counter.max())

    t_starts = t_ends_sort - t_window
    n_cells  = len(sua_units)
    counts   = np.zeros((n_cells, n_ori, n_phase, n_rep), dtype=np.float32)

    sua_mask = np.isin(spike_clusters, sua_units)
    st_sua   = spike_times[sua_mask]
    sc_sua   = spike_clusters[sua_mask]

    for ci, cell_id in enumerate(sua_units):
        cell_st = st_sua[sc_sua == cell_id]
        lo = np.searchsorted(cell_st, t_starts,   side='left')
        hi = np.searchsorted(cell_st, t_ends_sort, side='right')
        counts[ci, oi_arr, pi_arr, ri_arr] = (hi - lo).astype(np.float32)

    meta = dict(ori_values=ori_values, phase_values=phase_values,
                sua_unit_ids=sua_units, sua_isi_violations_ratio=sua_isi, sua_frac_rpv=sua_rpv)
    tag = f'tw{t_window}'.replace('.', 'p')
    return counts, meta, tag


# -----------------------------------------------------------------------------
# Static gratings (HB005 / HB006-style): stimulus_id, single presentation per trial
# -----------------------------------------------------------------------------

def extract_static(npz_path, t_start=T_START_STATIC, t_stop=T_STOP_STATIC):
    raw = np.load(npz_path, allow_pickle=True)
    spike_times    = raw['spike_times']
    spike_clusters = raw['spike_clusters']
    sua_units      = raw['sua_units'].astype(int)
    trial_starts   = raw['trial_starts']
    trial_ends     = raw['trial_ends']
    stimulus_type  = raw['stimulus_type'].astype(str)
    stimulus_id    = raw['stimulus_id']
    assert np.all(np.diff(spike_times) >= 0), f'{npz_path}: spike_times not sorted'

    sua_units, sua_isi, sua_rpv = apply_contamination_filter(raw, sua_units)

    grating_labels = [s for s in np.unique(stimulus_type) if 'grating_static' in s]
    assert len(grating_labels) == 1, \
        f'{npz_path}: expected exactly 1 static-grating stimulus_type label, got {grating_labels}'
    grating_label = grating_labels[0]

    grating_mask = stimulus_type == grating_label
    g_starts = trial_starts[grating_mask]
    g_ends   = trial_ends[grating_mask]
    g_ids    = stimulus_id[grating_mask].astype(int)

    id_values = np.unique(g_ids)
    id_index  = {v: i for i, v in enumerate(id_values)}
    n_ids     = len(id_values)

    stim_duration  = float(np.median(g_ends - g_starts))
    effective_stop = min(t_stop, stim_duration)

    order       = np.argsort(g_starts)   # chronological
    starts_sort = g_starts[order]
    ids_sort    = g_ids[order]
    ii_arr      = np.array([id_index[i] for i in ids_sort])

    rep_counter = np.zeros(n_ids, dtype=int)
    ri_arr = np.empty(len(order), dtype=int)
    for k in range(len(order)):
        ri_arr[k] = rep_counter[ii_arr[k]]
        rep_counter[ii_arr[k]] += 1
    n_rep = int(rep_counter.max())

    sua_mask = np.isin(spike_clusters, sua_units)
    st_sua   = spike_times[sua_mask]
    sc_sua   = spike_clusters[sua_mask]

    n_cells = len(sua_units)
    counts  = np.zeros((n_cells, n_ids, n_rep), dtype=np.float32)

    for ci, cell_id in enumerate(sua_units):
        cell_st = st_sua[sc_sua == cell_id]
        lo = np.searchsorted(cell_st, starts_sort + t_start,        side='left')
        hi = np.searchsorted(cell_st, starts_sort + effective_stop, side='right')
        counts[ci, ii_arr, ri_arr] = (hi - lo).astype(np.float32)

    meta = dict(id_values=id_values, grating_label=grating_label,
                sua_unit_ids=sua_units, sua_isi_violations_ratio=sua_isi, sua_frac_rpv=sua_rpv)
    tag = f't{t_start}-{t_stop}'.replace('.', 'p')
    return counts, meta, tag


# -----------------------------------------------------------------------------
# Main: discover sessions, extract each, then merge same-animal same-grid sessions
# -----------------------------------------------------------------------------

def merge_drifting(animal, sessions, tag):
    ref_ori, ref_phase = sessions[0][2]['ori_values'], sessions[0][2]['phase_values']
    if not all(np.array_equal(m['ori_values'], ref_ori) and np.array_equal(m['phase_values'], ref_phase)
               for _, _, m, _, _ in sessions):
        print(f'{animal}: stimulus grid mismatch across sessions, skipping merge')
        return

    all_counts, all_session, all_ids, all_isi, all_rpv, names = [], [], [], [], [], []
    for si, (sid, counts, meta, _, _) in enumerate(sessions):
        all_counts.append(counts)
        all_session.append(np.full(counts.shape[0], si, dtype=np.int32))
        all_ids.append(meta['sua_unit_ids'])
        all_isi.append(meta['sua_isi_violations_ratio'])
        all_rpv.append(meta['sua_frac_rpv'])
        names.append(sid)

    merged = np.concatenate(all_counts, axis=0)
    out_counts = DATA_ROOT / f'{animal}_merged_{tag}.npy'
    out_meta   = DATA_ROOT / f'{animal}_merged_{tag}_meta.npz'
    np.save(out_counts, merged)
    np.savez(out_meta, ori_values=ref_ori, phase_values=ref_phase,
             session=np.concatenate(all_session), sua_unit_ids=np.concatenate(all_ids),
             sua_isi_violations_ratio=np.concatenate(all_isi), sua_frac_rpv=np.concatenate(all_rpv),
             session_names=np.array(names))
    print(f'{animal}: merged {len(sessions)} sessions -> {merged.shape}  ({out_counts.name})')


def merge_static(animal, sessions, tag):
    ref_ids = sessions[0][2]['id_values']
    if not all(np.array_equal(m['id_values'], ref_ids) for _, _, m, _, _ in sessions):
        print(f'{animal}: stimulus-id grid mismatch across sessions, skipping merge')
        return

    max_rep = max(c.shape[2] for _, c, *_ in sessions)
    all_counts, all_session, all_ids, all_isi, all_rpv, names = [], [], [], [], [], []
    for si, (sid, counts, meta, _, _) in enumerate(sessions):
        if counts.shape[2] < max_rep:
            pad = np.zeros((counts.shape[0], counts.shape[1], max_rep - counts.shape[2]), dtype=np.float32)
            counts = np.concatenate([counts, pad], axis=2)
        all_counts.append(counts)
        all_session.append(np.full(counts.shape[0], si, dtype=np.int32))
        all_ids.append(meta['sua_unit_ids'])
        all_isi.append(meta['sua_isi_violations_ratio'])
        all_rpv.append(meta['sua_frac_rpv'])
        names.append(sid)

    merged = np.concatenate(all_counts, axis=0)
    out_counts = DATA_ROOT / f'{animal}_merged_{tag}.npy'
    out_meta   = DATA_ROOT / f'{animal}_merged_{tag}_meta.npz'
    np.save(out_counts, merged)
    np.savez(out_meta, id_values=ref_ids,
             session=np.concatenate(all_session), sua_unit_ids=np.concatenate(all_ids),
             sua_isi_violations_ratio=np.concatenate(all_isi), sua_frac_rpv=np.concatenate(all_rpv),
             session_names=np.array(names))
    print(f'{animal}: merged {len(sessions)} sessions -> {merged.shape}  ({out_counts.name})')


def main():
    npz_paths = sorted(DATA_ROOT.glob('*/*_curated_results.npz'))
    print(f'Found {len(npz_paths)} sessions in {DATA_ROOT}')

    extracted = defaultdict(list)   # animal -> [(session_id, counts, meta, tag, kind), ...]

    for npz_path in npz_paths:
        session_dir = npz_path.parent
        session_id  = session_dir.name
        animal      = animal_from_session(session_id)
        raw_keys    = np.load(npz_path, allow_pickle=True).files

        print(f'\n{session_id}:')
        if 'stimulus_ori' in raw_keys:
            counts, meta, tag = extract_drifting(npz_path)
            kind = 'drifting'
        else:
            counts, meta, tag = extract_static(npz_path)
            kind = 'static'
        print(f'  [{kind}] shape={counts.shape}  cells={counts.shape[0]}')

        out_counts = session_dir / f'spike_counts_{session_id}_{tag}.npy'
        out_meta   = session_dir / f'spike_counts_{session_id}_{tag}_meta.npz'
        np.save(out_counts, counts)
        np.savez(out_meta, **meta)
        print(f'  Saved {out_counts.name}, {out_meta.name}')

        extracted[animal].append((session_id, counts, meta, tag, kind))

    print('\n--- Merging same-animal sessions ---')
    for animal, sessions in extracted.items():
        if len(sessions) < 2:
            continue
        kinds = {k for *_, k in sessions}
        tags  = {t for *_, t, _ in sessions}
        if len(kinds) > 1:
            print(f'{animal}: mixed session kinds {kinds}, skipping merge')
            continue
        if len(tags) > 1:
            print(f'{animal}: sessions used different extraction tags {tags}, skipping merge')
            continue
        kind, tag = kinds.pop(), tags.pop()
        (merge_drifting if kind == 'drifting' else merge_static)(animal, sessions, tag)


if __name__ == '__main__':
    main()
