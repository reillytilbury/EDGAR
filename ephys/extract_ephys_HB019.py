"""
Extract spike count arrays from HB019_2026-05-28_curated_results.npz (drifting gratings).

=============================================================================
DATASET CONTENTS  (HB019_2026-05-28_curated_results.npz)
=============================================================================

Raw spike data
--------------
spike_times     : float array, shape (n_spikes,)
    Timestamp in seconds of every detected spike in the recording, sorted in
    ascending order.  All values share the same time reference as trial_ends
    (see the NOTE below about trial_starts).

spike_clusters  : int array, shape (n_spikes,)
    For each spike, the cluster (unit) ID it was assigned to by the spike
    sorter.  Cluster IDs are integers 0–969 and index directly into the
    all_unit_* quality-metric arrays described below.

Unit classification
-------------------
sua_units       : int array, shape (174,)
    Cluster IDs that the curator classified as Single-Unit Activity — i.e.
    isolated recordings from one neuron.  These are the cells used for
    analysis.

mua_units       : int array, shape (400,)
    Cluster IDs classified as Multi-Unit Activity — spikes from multiple
    neurons that could not be cleanly separated.  Not used for analysis but
    kept for reference.

    NOTE: sua + mua = 574 labelled units.  The remaining ~396 cluster IDs
    (out of 970 total) are noise or unlabelled clusters.

Contamination / quality metrics  (one value per cluster, indexed by cluster ID)
-------------------------------------------------------------------------------
These arrays have length 970 (one entry per cluster ID).  To get the metric
for a particular unit, index directly with its cluster ID — e.g.
all_unit_frac_rpv[cluster_id].

all_unit_frac_rpv           : float array, shape (970,)
    Fraction of spike pairs that fall within the refractory period (~2 ms).
    A real neuron cannot fire twice that fast, so any pair within that window
    is biologically impossible.  High values (e.g. >10 %) indicate that two
    neurons were merged into one cluster.  Typical threshold for clean single
    units: <5 %.

    CAUTION: fast-firing cells can show elevated RPV even when clean, because
    the raw fraction is not corrected for firing rate.  Use
    all_unit_isi_violations_ratio as the primary contamination filter.

all_unit_isi_violations_ratio : float array, shape (970,)
    A firing-rate-corrected estimate of the fraction of spikes that came from
    a contaminating (wrong) neuron.  Unlike frac_rpv, this metric accounts
    for how often a clean unit would be expected to fire within the refractory
    window by chance, so it is more reliable for high-rate cells.  Values
    above 0.10 mean >10 % of the unit's spikes probably belong to another
    neuron.  Typical threshold: <0.10 for analysis; <0.05 for strict use.

    For the 174 SUA units in this file: max = 0.11, so all pass a 0.10
    threshold and all pass a 0.15 threshold comfortably.

Spatial coordinates  (one value per cluster, indexed by cluster ID)
--------------------------------------------------------------------
all_unit_dist_from_tip  : float array, shape (970,)
    Estimated distance of the unit from the probe tip along the shank, in
    micrometres.  Larger values = further from the tip = deeper in tissue.

all_unit_x_loc          : float array, shape (970,)
    Lateral position of the unit across the probe face, in micrometres.
    Together with dist_from_tip, this locates each neuron on the probe.

Trial / stimulus structure
--------------------------
trial_starts    : float array, shape (n_trials,)
    *** BROKEN TIME REFERENCE *** — these timestamps use a different clock
    origin than spike_times and trial_ends.  Do not use to align spikes.

trial_ends      : float array, shape (n_trials,)
    Time in seconds when each stimulus trial ended, in the same reference
    frame as spike_times.  Used as the stimulus anchor: spikes are counted
    in [trial_end - t_window, trial_end].

stimulus_ori    : float array, shape (n_trials,)
    Drifting grating direction shown on each trial, in degrees (0, 2, 4, …,
    358).  180 unique directions × 4 phases × 4 repeats = 2880 trials total.

stimulus_phase  : float array, shape (n_trials,)
    Spatial phase of the grating on each trial, in degrees (0, 90, 180, 270).

stimulus_type   : array, shape (n_trials,)
    Categorical label for the trial type (e.g. drifting grating vs. blank).

stimulus_epoch  : array, shape (n_trials,)
    Index of the experimental epoch / block the trial belongs to.

=============================================================================
HOW EXTRACTION WORKS
=============================================================================

1.  Load & sanity-check
    Read the .npz and confirm spike_times is sorted (a requirement for the
    fast binary-search spike counting below).

2.  Build a trial table  (build_trial_table)
    Sort trials chronologically by trial_end.  Map each trial's orientation
    and phase values to integer indices into a fixed grid
    (ori: 0–179, phase: 0–3).  Assign a repeat index (0–3) per condition by
    counting how many times that (ori, phase) pair has already appeared,
    in chronological order.

3.  Count spikes per cell per trial  (compute_spike_counts)
    Filter the global spike list down to SUA spikes only.  For each of the
    174 SUA cells, extract its spike timestamps and use np.searchsorted to
    count how many fall in [trial_end - t_window, trial_end] for every
    trial simultaneously.  Store the result in the output array at position
    [cell, ori_idx, phase_idx, rep_idx].

    The result is a float32 array of shape (174, 180, 4, 4):
        axis 0 — cell index (maps to sua_units[i])
        axis 1 — orientation index (ori_values[j] gives the degree value)
        axis 2 — phase index (0→0°, 1→90°, 2→180°, 3→270°)
        axis 3 — repeat index (0–3, chronological)

4.  Optionally extract time series  (compute_spike_timeseries)
    Instead of a single count per trial, bin the spikes within the window
    into fixed-width bins (e.g. 50 ms).  Output shape is
    (174, 180, 4, 4, n_bins).

=============================================================================
NOTE ON trial_starts / trial_ends TIME REFERENCE
=============================================================================
trial_starts uses a different clock origin than spike_times and trial_ends.
If you align spikes to trial_starts you will get systematically wrong counts.
Always use trial_ends as the stimulus anchor.

=============================================================================
OUTPUT FILES
=============================================================================
spike_counts_HB019_{tag}.npy      : float32, shape (174, 180, 4, 4)
spike_counts_HB019_{tag}_meta.npz : ori_values (shape 180), phase_values (shape 4)

Usage
-----
python extract_ephys_HB019.py                      # default 2.0 s window
python extract_ephys_HB019.py --t_window 1.5       # custom window
python extract_ephys_HB019.py --bin_size 0.05      # time series with 50 ms bins
"""

import argparse
import numpy as np
from pathlib import Path

# -----------------------------------------------------------------------------
# DATASET PATH — swap between sessions by commenting/uncommenting
# -----------------------------------------------------------------------------
# NPZ_PATH = Path('/home/reilly/datasets/hayley_data/HB019_2026-05-28/HB019_2026-05-28_curated_results.npz')
NPZ_PATH = Path('/home/reilly/datasets/hayley_data/HB019_2026-05-29/HB019_2026-05-29_curated_results.npz')

OUT_DIR = NPZ_PATH.parent

# -----------------------------------------------------------------------------
# QUALITY-FILTER THRESHOLDS
# Set to None to disable a filter entirely.
#
# MAX_ISI_VIOLATIONS_RATIO : primary contamination filter (firing-rate corrected).
#   Fraction of spikes estimated to come from a contaminating neuron.
#   <0.05 = strict  |  <0.10 = standard  |  None = no filter
#
# MAX_FRAC_RPV : secondary filter (raw refractory-period violation rate).
#   Less reliable than ISI ratio for fast-firing cells — use as a sanity check
#   rather than a primary gate.
#   <0.05 = strict  |  <0.10 = lenient  |  None = no filter
# -----------------------------------------------------------------------------
MAX_ISI_VIOLATIONS_RATIO = 0.05
MAX_FRAC_RPV             = None

N_ORI   = 180   # 0, 2, 4, …, 358 degrees
N_PHASE = 4     # 0, 90, 180, 270 degrees
N_REP   = 4     # repeats per (direction, phase) condition


def load_data():
    raw = np.load(NPZ_PATH, allow_pickle=True)
    spike_times    = raw['spike_times']
    spike_clusters = raw['spike_clusters']
    sua_units      = raw['sua_units'].astype(int)
    trial_ends     = raw['trial_ends']      # seconds, same reference as spike_times
    stimulus_ori   = raw['stimulus_ori']    # degrees: 0, 2, 4, …, 358
    stimulus_phase = raw['stimulus_phase']  # degrees: 0, 90, 180, 270
    assert np.all(np.diff(spike_times) >= 0), 'spike_times not sorted'

    # Apply contamination filters (thresholds set at top of file)
    isi_ratio = raw['all_unit_isi_violations_ratio']   # indexed by cluster ID
    frac_rpv  = raw['all_unit_frac_rpv']               # indexed by cluster ID
    mask = np.ones(len(sua_units), dtype=bool)
    if MAX_ISI_VIOLATIONS_RATIO is not None:
        mask &= isi_ratio[sua_units] <= MAX_ISI_VIOLATIONS_RATIO
    if MAX_FRAC_RPV is not None:
        mask &= frac_rpv[sua_units] <= MAX_FRAC_RPV
    n_dropped = (~mask).sum()
    if n_dropped:
        print(f'Contamination filter: dropped {n_dropped}/{len(sua_units)} SUA units '
              f'(ISI ratio <= {MAX_ISI_VIOLATIONS_RATIO}, frac RPV <= {MAX_FRAC_RPV})')
    sua_units = sua_units[mask]
    sua_isi   = isi_ratio[sua_units]
    sua_rpv   = frac_rpv[sua_units]

    return spike_times, spike_clusters, sua_units, sua_isi, sua_rpv, trial_ends, stimulus_ori, stimulus_phase


def build_trial_table(trial_ends, stimulus_ori, stimulus_phase):
    """Return sorted trial_ends and (ori_idx, phase_idx, rep_idx) per trial."""
    ori_values   = np.arange(0, 360, 2, dtype=int)   # 0, 2, …, 358
    phase_values = np.array([0, 90, 180, 270], dtype=int)
    ori_index    = {o: i for i, o in enumerate(ori_values)}
    phase_index  = {p: i for i, p in enumerate(phase_values)}

    # Drop blank trials (ori/phase are NaN for non-grating epochs)
    grating_mask = ~np.isnan(stimulus_ori)
    n_blank = (~grating_mask).sum()
    if n_blank:
        print(f'Skipping {n_blank} blank trials (NaN stimulus_ori)')
    trial_ends    = trial_ends[grating_mask]
    stimulus_ori   = stimulus_ori[grating_mask]
    stimulus_phase = stimulus_phase[grating_mask]

    order       = np.argsort(trial_ends)   # process chronologically
    t_ends_sort = trial_ends[order]
    ori_sort    = stimulus_ori[order]
    phase_sort  = stimulus_phase[order]

    oi_arr = np.array([ori_index[int(o)]   for o in ori_sort])
    pi_arr = np.array([phase_index[int(p)] for p in phase_sort])

    # Assign repeat index per (ori, phase) condition chronologically
    rep_counter = np.zeros((N_ORI, N_PHASE), dtype=int)
    ri_arr = np.empty(len(order), dtype=int)
    for k in range(len(order)):
        oi, pi = oi_arr[k], pi_arr[k]
        ri_arr[k] = rep_counter[oi, pi]
        rep_counter[oi, pi] += 1

    return t_ends_sort, oi_arr, pi_arr, ri_arr, ori_values, phase_values


def compute_spike_counts(t_window=2.0):
    spike_times, spike_clusters, sua_units, sua_isi, sua_rpv, trial_ends, stimulus_ori, stimulus_phase = load_data()

    t_ends, oi_arr, pi_arr, ri_arr, ori_values, phase_values = build_trial_table(
        trial_ends, stimulus_ori, stimulus_phase)

    t_starts = t_ends - t_window    # estimated stimulus onset
    n_trials = len(t_ends)
    n_cells  = len(sua_units)
    counts   = np.zeros((n_cells, N_ORI, N_PHASE, N_REP), dtype=np.float32)

    sua_mask = np.isin(spike_clusters, sua_units)
    st_sua   = spike_times[sua_mask]
    sc_sua   = spike_clusters[sua_mask]

    for ci, cell_id in enumerate(sua_units):
        cell_st = st_sua[sc_sua == cell_id]
        lo = np.searchsorted(cell_st, t_starts, side='left')
        hi = np.searchsorted(cell_st, t_ends,   side='right')
        cell_counts = (hi - lo).astype(np.float32)
        counts[ci, oi_arr, pi_arr, ri_arr] = cell_counts

        if (ci + 1) % 50 == 0 or ci == n_cells - 1:
            print(f'  {ci + 1}/{n_cells} cells')

    return counts, sua_units, sua_isi, sua_rpv, ori_values, phase_values


def compute_spike_timeseries(t_window=2.0, bin_size=0.05):
    spike_times, spike_clusters, sua_units, sua_isi, sua_rpv, trial_ends, stimulus_ori, stimulus_phase = load_data()

    t_ends, oi_arr, pi_arr, ri_arr, ori_values, phase_values = build_trial_table(
        trial_ends, stimulus_ori, stimulus_phase)

    bin_edges = np.arange(0, t_window + bin_size / 2, bin_size)
    bin_edges[-1] = t_window
    n_t = len(bin_edges) - 1

    n_cells    = len(sua_units)
    timeseries = np.zeros((n_cells, N_ORI, N_PHASE, N_REP, n_t), dtype=np.float32)

    sua_mask = np.isin(spike_clusters, sua_units)
    st_sua   = spike_times[sua_mask]
    sc_sua   = spike_clusters[sua_mask]

    for ci, cell_id in enumerate(sua_units):
        cell_st = st_sua[sc_sua == cell_id]

        for k in range(len(t_ends)):
            t0    = t_ends[k] - t_window
            edges = t0 + bin_edges
            timeseries[ci, oi_arr[k], pi_arr[k], ri_arr[k]] = np.diff(
                np.searchsorted(cell_st, edges))

        if (ci + 1) % 25 == 0 or ci == n_cells - 1:
            print(f'  {ci + 1}/{n_cells} cells')

    return timeseries, sua_units, sua_isi, sua_rpv, bin_edges, ori_values, phase_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--t_window', type=float, default=2.0,
                        help='Spike counting window in s ending at trial_end (default 2.0)')
    parser.add_argument('--bin_size', type=float, default=None,
                        help='If set, extract time series with this bin size in s (e.g. 0.05)')
    args = parser.parse_args()

    if args.bin_size is None:
        print(f'Computing spike counts in [trial_end - {args.t_window}, trial_end] s ...')
        counts, sua_units, sua_isi, sua_rpv, ori_values, phase_values = compute_spike_counts(args.t_window)

        tag     = f'tw{args.t_window}'.replace('.', 'p')
        outpath = OUT_DIR / f'spike_counts_HB019_{tag}.npy'
        np.save(outpath, counts)
        np.savez(OUT_DIR / f'spike_counts_HB019_{tag}_meta.npz',
                 ori_values=ori_values, phase_values=phase_values,
                 sua_unit_ids=sua_units,
                 sua_isi_violations_ratio=sua_isi,
                 sua_frac_rpv=sua_rpv)

        print(f'Saved {outpath}')
        print(f'Shape: {counts.shape}  dtype: {counts.dtype}')
        print(f'Value range: {counts.min():.0f} – {counts.max():.0f} spikes')
        print(f'Mean firing rate: {counts.mean() / args.t_window:.2f} Hz')

    else:
        print(f'Extracting time series: window={args.t_window} s, '
              f'bin_size={args.bin_size*1000:.0f} ms ...')
        ts, sua_units, sua_isi, sua_rpv, bin_edges, ori_values, phase_values = compute_spike_timeseries(
            args.t_window, args.bin_size)

        tag     = f'tw{args.t_window}_bs{int(args.bin_size*1000)}ms'.replace('.', 'p')
        outpath = OUT_DIR / f'spike_timeseries_HB019_{tag}.npy'
        np.save(outpath, ts)
        np.savez(OUT_DIR / f'spike_timeseries_HB019_{tag}_meta.npz',
                 bin_edges=bin_edges, ori_values=ori_values, phase_values=phase_values,
                 sua_unit_ids=sua_units,
                 sua_isi_violations_ratio=sua_isi,
                 sua_frac_rpv=sua_rpv)

        print(f'Saved {outpath}')
        print(f'Shape: {ts.shape}  dtype: {ts.dtype}')
        print(f'Bin edges: {bin_edges[0]:.3f} – {bin_edges[-1]:.3f} s  '
              f'({len(bin_edges)-1} bins)')
        print(f'Value range: {ts.min():.0f} – {ts.max():.0f} spikes/bin')
