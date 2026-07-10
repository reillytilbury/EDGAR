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

data_path_hayley_hb019 = '/home/reilly/datasets/hayley_data_all/HB019_merged_tw1p5.npy'   # 3 sessions -> 362 cells
data_path_hayley_hb020 = '/home/reilly/datasets/hayley_data_all/HB020_merged_tw1p5.npy'   # 3 sessions -> 216 cells

data_path_hd = '/home/reilly/datasets/hd_cells/hd_with_repeats.npz'

data_path_bz15_dff = [
    '/home/reilly/datasets/jacob data/bz15_dff/BZ015_2025-07-03_2_dff.npy',
    '/home/reilly/datasets/jacob data/bz15_dff/BZ015_2025-07-03_3_dff.npy',
    '/home/reilly/datasets/jacob data/bz15_dff/BZ015_2025-07-03_5_dff.npy',
]

# CONFIGS_BZ16 = [
#     dict(run_name='bz16',
#          n_iterations=12, time_limit=60,
#          data_path=data_path_bz16, data_type='jacob',
#          use_image_feedback=True, use_large_every=3,
#          param_penalty_weight=0.01 / 3,
#          data_scale_factor=140,
#          activity_thresh=0.0, signal_fraction_thresh=0.6, conc_thresh=0.3,
#          n_bins=256, min_repeats=6,
#          diag_kwargs={'point_alpha': 0.1, 'n_mean': 50},
#          exploration_topology=[1, 2, 3, 4, 5, 6, 7, 0], exploit_point=0.5)
# ] * 4


# CONFIGS_GT1 = [
#     dict(run_name=f'gt1-cp-10',
#          n_iterations=12, time_limit=30,
#          data_path=data_path_gt1, data_type='stringer',
#          use_image_feedback=True, use_large_every=3,
#          data_scale_factor=100,
#          nonzero_filter=True,
#          param_penalty_weight=0.1,   
#          activity_thresh=0.4, signal_fraction_thresh=0.0, conc_thresh=0.55,
#          diag_kwargs={'point_alpha': 0.1, 'n_mean': 50},
#          n_bins=256, min_repeats=6,
#          exploration_topology=[1, 2, 3, 4, 5, 6, 7, 0], exploit_point=0.5)
# ] 

# # add 2 config of GT1 with standard conc_thresh of 0.55 but no image feedback
# CONFIGS_GT1 += [
#     dict(run_name='gt1-blind',
#          n_iterations=12, time_limit=60,
#          data_path=data_path_gt1, data_type='stringer',
#          use_image_feedback=False, use_large_every=3,
#          data_scale_factor=100,
#          nonzero_filter=True,
#          activity_thresh=0.4, signal_fraction_thresh=0.0, conc_thresh=0.55,
#          diag_kwargs={'point_alpha': 0.1, 'n_mean': 50},
#          n_bins=256, min_repeats=6,
#          exploration_topology=[1, 2, 3, 4, 5, 6, 7, 0], exploit_point=0.5)
# ] * 2

# import seed_programs

# CONFIGS_GT1 = [
#     dict(run_name=f'gt1',
#          n_iterations=12, time_limit=60,
#          data_path=data_path_gt1, data_type='stringer',
#          use_image_feedback=True, use_large_every=3,
#          data_scale_factor=100,
#          nonzero_filter=True,
#          max_cells=None,
#          seed_numpy_programs=[seed_programs.neuron_model_3, seed_programs.neuron_model_4],
#          seed_jax_programs=[seed_programs.neuron_model_3_jax, seed_programs.neuron_model_4_jax],
#          seed_param_estimators=[seed_programs.parameter_estimator_3, seed_programs.parameter_estimator_4],
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

# CONFIGS_GT3 = [
#     dict(run_name='gt3',
#          n_iterations=12, time_limit=30,
#          data_path=data_path_gt3,
#          data_type='stringer',
#          use_image_feedback=True, use_large_every=3,
#          param_penalty_weight=0.01,
#          nonzero_filter=True,
#          data_scale_factor=100,
#          activity_thresh=0.4, signal_fraction_thresh=0.0, conc_thresh=0.5,
#          diag_kwargs={'point_alpha': 0.1, 'n_mean': 50},
#          n_bins=256, min_repeats=6,
#          exploration_topology=[1, 2, 3, 4, 5, 6, 7, 0], exploit_point=0.5)
# ] * 4

# CONFIGS_HD = [
#     dict(run_name='hd',
#          n_iterations=12, time_limit=45,
#          data_path=data_path_hd, data_type='hd',
#          use_image_feedback=True, use_large_every=3,
#          param_penalty_weight=0.05,
#          data_scale_factor=100,
#          activity_thresh=0.0, signal_fraction_thresh=0.0, conc_thresh=0.0,
#          n_bins=180, min_repeats=None,
#          diag_kwargs={'point_alpha': 0.15, 'n_mean': 50},
#          exploration_topology=[1, 2, 3, 4, 5, 6, 7, 0], exploit_point=0.5)
# ] * 4

# CONFIGS_BZ15 = [
#     dict(run_name='bz15',
#          n_iterations=12, time_limit=30,
#          data_path=data_path_bz15, data_type='jacob',
#          use_image_feedback=True, use_large_every=3,
#          param_penalty_weight=0.01,
#          data_scale_factor=100,
#          activity_thresh=0.0, signal_fraction_thresh=0.85, conc_thresh=0.4,
#          n_bins=256, min_repeats=12,
#          diag_kwargs={'point_alpha': 0.1, 'n_mean': 60},
#          exploration_topology=[1, 2, 3, 4, 5, 6, 7, 0], exploit_point=0.5)
# ] * 6

# CONFIGS_HAYLEY_HB019 = [
#     dict(run_name='hayley_hb019',
#          n_iterations=8, time_limit=45,
#          data_path=data_path_hayley_hb019, data_type='hayley',
#          use_image_feedback=True, use_large_every=3,
#          param_penalty_weight=0.01,
#          data_scale_factor=40,
#          activity_thresh=0.0, signal_fraction_thresh=0.5, conc_thresh=0.15,
#          n_bins=180, min_repeats=None,
#          sort_by_sf=True,
#          diag_kwargs={'point_alpha': 0.2, 'n_mean': 50},
#          exploration_topology=[1, 2, 3, 4, 5, 6, 7, 0], exploit_point=0.5)
# ] * 4

CONFIGS_HAYLEY_HB020 = [
    dict(run_name='hayley_hb020',
         n_iterations=8, time_limit=20,
         data_path=data_path_hayley_hb020, data_type='hayley',
         use_image_feedback=True, use_large_every=3,
         param_penalty_weight=0.01,
         data_scale_factor=40,
         activity_thresh=0.0, signal_fraction_thresh=0.5, conc_thresh=0.125,
         n_bins=180, min_repeats=None,
         sort_by_sf=False,
         diag_kwargs={'point_alpha': 0.2, 'n_mean': 40},
         exploration_topology=[1, 2, 3, 4, 5, 6, 7, 0], exploit_point=0.5)
] * 8

# CONFIGS_BZ15_DFF = [
#     dict(run_name='bz15_dff',
#          n_iterations=8, time_limit=45,
#          data_path=data_path_bz15_dff, data_type='jacob_dff',
#          use_image_feedback=True, use_large_every=3,
#          param_penalty_weight=0.01, 
#          data_scale_factor=170,
#          activity_thresh=0.6, signal_fraction_thresh=0.6, conc_thresh=0.2,
#          sort_by_sf=True, n_bins=256, min_repeats=3,
#          diag_kwargs={'point_alpha': 0.05, 'n_mean': 42},
#          exploration_topology=[1, 2, 3, 4, 5, 6, 7, 0], exploit_point=0.5)
# ] * 4

RUN_CONFIGS = CONFIGS_HAYLEY_HB020