"""
Run configurations for EDGAR. Each entry is passed as kwargs to main() in hypothesis_engine.py.

To run locally:  python3 hypothesis_engine.py         (runs all configs sequentially)
To run remotely: ./launch_gcp.sh                      (one VM per entry in RUN_CONFIGS)
"""

# ---- DATA PATHS ----
data_path_bz15 = [
    [
        '/home/reilly/datasets/jacob data/bz15/data2.npy',
        '/home/reilly/datasets/jacob data/bz15/data3.npy',
        '/home/reilly/datasets/jacob data/bz15/data5.npy',
    ],
    [
        '/home/reilly/datasets/jacob data/bz15/meta2.mat',
        '/home/reilly/datasets/jacob data/bz15/meta3.mat',
        '/home/reilly/datasets/jacob data/bz15/meta5.mat',
    ],
]

data_path_bz16 = [
    ['/home/reilly/datasets/jacob data/bz16/BZ016_2025-06-24_1_dspikes.npy'],
    ['/home/reilly/datasets/jacob data/bz16/2025-06-24_1_BZ016_Block.mat'],
]

data_path_gt1 = '/home/reilly/datasets/stringer_2021/gratings_drifting_GT1_2019_04_12_1.npy'
data_path_gt2 = '/home/reilly/datasets/stringer_2021/gratings_drifting_GT2_2019_04_05_1.npy'
data_path_gt3 = '/home/reilly/datasets/stringer_2021/gratings_drifting_GT3_2019_04_05_1.npy'

data_path_ali = [
    '/home/reilly/datasets/ali data/stim_sequence.npy',
    '/home/reilly/datasets/ali data/stim_resps.npy',
]

data_path_hayley = '/home/reilly/datasets/hayley_data/spike_counts.npy'

data_path_hd = '/home/reilly/datasets/hd_cells/hd_with_repeats.npz'

# ---- REMOTE CONFIGS (one VM per entry) ----
# CONFIGS_BZ16 = [
#     dict(run_name='bz16',
#          n_iterations=12, time_limit=60,
#          data_path=data_path_bz16, data_type='jacob',
#          use_image_feedback=True, use_large_every=3,
#          param_penalty_weight=0.01 / 3,
#          data_scale_factor=150,
#          activity_thresh=0.2, signal_fraction_thresh=0.0, conc_thresh=0.4,
#          n_bins=256, min_repeats=6,
#          diag_kwargs={'point_alpha': 0.1, 'n_mean': 50},
#          exploration_topology=[1, 2, 3, 4, 5, 6, 7, 0], exploit_point=0.5)
# ] * 4

# CONFIGS_GT1 = [
#     dict(run_name='gt1',
#          n_iterations=12, time_limit=60,
#          data_path=data_path_gt1, data_type='stringer',
#          use_image_feedback=True, use_large_every=3,
#          data_scale_factor=100,
#          nonzero_filter=True,
#          activity_thresh=0.4, signal_fraction_thresh=0.0, conc_thresh=0.55,
#          diag_kwargs={'point_alpha': 0.1, 'n_mean': 50},
#          n_bins=256, min_repeats=6,
#          exploration_topology=[1, 2, 3, 4, 5, 6, 7, 0], exploit_point=0.5)
# ] * 4

# CONFIGS_GT2 = [
#     dict(run_name='gt2',
#          n_iterations=12, time_limit=60,
#          data_path=data_path_gt2,
#          data_type='stringer',
#          use_image_feedback=True, use_large_every=3,
#          param_penalty_weight=0.01 * 3,
#          data_scale_factor=170,
#          activity_thresh=0.4, signal_fraction_thresh=0.0, conc_thresh=0.55,
#          diag_kwargs={'point_alpha': 0.1, 'n_mean': 50},
#          n_bins=256, min_repeats=6,
#          exploration_topology=[1, 2, 3, 4, 5, 6, 7, 0], exploit_point=0.5)
# ] * 4

CONFIGS_HD = [
    dict(run_name='hd',
         n_iterations=8, time_limit=60,
         data_path=data_path_hd, data_type='hd',
         use_image_feedback=True, use_large_every=3,
         param_penalty_weight=0.01,
         data_scale_factor=70,
         activity_thresh=0.0, signal_fraction_thresh=0.0, conc_thresh=0.0,
         n_bins=180, min_repeats=None,
         diag_kwargs={'point_alpha': 0.15, 'n_mean': 50},
         exploration_topology=[1, 2, 3, 4, 5, 6, 7, 0], exploit_point=0.5)
] * 6

CONFIGS_BZ15 = [
    dict(run_name='bz15',
         n_iterations=12, time_limit=30,
         data_path=data_path_bz15, data_type='jacob',
         use_image_feedback=True, use_large_every=3,
         param_penalty_weight=0.01,
         data_scale_factor=100,
         activity_thresh=0.0, signal_fraction_thresh=0.85, conc_thresh=0.4,
         n_bins=256, min_repeats=12,
         diag_kwargs={'point_alpha': 0.1, 'n_mean': 60},
         exploration_topology=[1, 2, 3, 4, 5, 6, 7, 0], exploit_point=0.5)
] * 6

RUN_CONFIGS = CONFIGS_BZ15