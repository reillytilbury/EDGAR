from pathlib import Path

import numpy as np
import pytest

from projects.minimodel_discovery.seed_programs import model1 as _model1_mod, model2 as _model2_mod
from projects.minimodel_discovery.seed_programs import param_est1 as _param_est1_mod, param_est2 as _param_est2_mod
from projects.minimodel_discovery.data_loader.load_data import (
    load_and_process_data,
    train_test_split,
    loss_fn,
    build_evaluation_points,
)
from projects.minimodel_discovery.image_feedback.plot import plot_model_fits
from src import utils
from src.hypothesis_engine import objective

model_v1 = _model1_mod.model
model_v2 = _model2_mod.model
param_est_v1 = _param_est1_mod.parameter_estimator
param_est_v2 = _param_est2_mod.parameter_estimator


DATA_PATH = "/home/reilly/datasets/image_responses/FX8_nat60k_2023_05_16.npz"
IMAGE_PATH = "/home/reilly/datasets/image_responses/nat60k_text16.mat"
MINIMODEL_REPO_PATH = "/home/reilly/Documents/code/minimodel"
TEACHER_CHECKPOINT_PATH = (
    "/home/reilly/Documents/code/minimodel/notebooks/checkpoints/"
    "FX8_051623_2layer_16_320_clamp_norm_depthsep_pool.pt"
)


def _load_small_dataset(*, use_teacher: bool, max_cells: int = 6, max_train_images: int = 48):
    return load_and_process_data(
        data_path=DATA_PATH,
        image_path=IMAGE_PATH,
        mouse_id=4,
        min_fev=0.15,
        max_cells=max_cells,
        max_train_images=max_train_images,
        anchor_cell_count=min(4, max_cells),
        minimodel_repo_path=MINIMODEL_REPO_PATH,
        teacher_checkpoint_path=TEACHER_CHECKPOINT_PATH,
        use_teacher_diagnostics=use_teacher,
        teacher_device="cpu",
    )


def _build_seed_programs_list(data):
    n_samples = utils.data_n_samples(data)
    params_v1 = [param_est_v1(utils.get_data_sample(data, idx)) for idx in range(n_samples)]
    params_v2 = [param_est_v2(utils.get_data_sample(data, idx)) for idx in range(n_samples)]
    return [
        {
            "model": model_v1,
            "params": utils.stack_params(params_v1),
            "losses": np.full((n_samples,), 0.0, dtype=np.float32),
        },
        {
            "model": model_v2,
            "params": utils.stack_params(params_v2),
            "losses": np.full((n_samples,), 0.0, dtype=np.float32),
        },
    ]


def test_loader_returns_fx8_image_response_contract():
    data = _load_small_dataset(use_teacher=False, max_cells=8, max_train_images=64)

    assert set(data) == {"image", "response", "response_repeats", "stimulus_id", "cell_index"}
    assert data["image"].shape[0] == data["response"].shape[0]
    assert data["image"].shape[1:3] == (66, 130)
    assert data["response"].shape[1] == data["image"].shape[-1]
    assert data["response_repeats"].shape[2] == data["image"].shape[-1]
    assert data["response_repeats"].shape[1] == 10
    assert data["stimulus_id"].shape == data["response"].shape
    assert data["cell_index"].shape == data["response"].shape

    np.testing.assert_allclose(data["image"][0], data["image"][1])
    assert np.all(np.isnan(data["response_repeats"][:, :, :64]))
    assert np.all(np.isfinite(data["response_repeats"][:, :, 64:]))

    train_samples, train_trials = train_test_split(data, random_seed=7)
    assert train_samples.ndim == 1
    assert np.array_equal(train_trials, np.arange(64))


def test_seeds_and_estimators_are_finite_and_improve_tiny_objective():
    data = _load_small_dataset(use_teacher=False, max_cells=4, max_train_images=24)
    train_samples, train_trials = train_test_split(data, random_seed=3)
    train = utils.slice_data_trials(utils.slice_data_samples(data, train_samples), train_trials)

    for model_fn, param_est_fn in ((model_v1, param_est_v1), (model_v2, param_est_v2)):
        sample = utils.get_data_sample(train, 0)
        params = param_est_fn(sample)
        pred = np.asarray(utils.call_model(model_fn, sample, params))
        assert pred.shape == sample["response"].shape
        assert np.isfinite(pred).all()

        initial_loss, _, final_loss, _ = objective(
            model=model_fn,
            param_estimator=param_est_fn,
            data=[train, train],
            loss_fn=loss_fn,
            param_penalty_weight=0.0,
            fit_params=True,
            max_iter=12,
            learning_rate=0.05,
            use_param_estimator=True,
            trial_batch_size=16,
        )
        assert np.isfinite(initial_loss)
        assert np.isfinite(final_loss)
        assert final_loss <= initial_loss + 1e-4


def test_custom_build_evaluation_points_uses_real_train_images():
    data = _load_small_dataset(use_teacher=False, max_cells=4, max_train_images=32)
    train_samples, train_trials = train_test_split(data, random_seed=2)
    train = utils.slice_data_trials(utils.slice_data_samples(data, train_samples), train_trials)

    eval_points = build_evaluation_points(train, random_seed=5, n_eval_images=12)
    assert eval_points["image"].shape[-1] == 12
    assert eval_points["response"].shape[-1] == 12
    assert np.all(np.isin(eval_points["stimulus_id"][0], train["stimulus_id"][0]))

    params = param_est_v1(utils.get_data_sample(train, 0))
    eval_matrix = utils.compute_evaluation_matrix(model_v1, utils.stack_params([params] * utils.data_n_samples(train)), eval_points)
    assert np.asarray(eval_matrix).shape == (utils.data_n_samples(train), 12)


def test_plot_model_fits_smoke_without_teacher(tmp_path: Path):
    data = _load_small_dataset(use_teacher=False, max_cells=4, max_train_images=24)
    train_samples, train_trials = train_test_split(data, random_seed=1)
    train = utils.slice_data_trials(utils.slice_data_samples(data, train_samples), train_trials)
    eval_points = build_evaluation_points(train, random_seed=1, n_eval_images=12)
    programs_list = _build_seed_programs_list(train)

    out_path = tmp_path / "minimodel_diag_no_teacher.png"
    plot_model_fits(
        data=train,
        programs_list=programs_list,
        X_eval=eval_points,
        save_path=str(out_path),
        labels=["seed_v1", "seed_v2"],
        title_prefix="smoke",
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_model_fits_smoke_with_teacher(tmp_path: Path):
    pytest.importorskip("torch")
    if not Path(TEACHER_CHECKPOINT_PATH).exists():
        pytest.skip("teacher checkpoint unavailable")

    data = _load_small_dataset(use_teacher=True, max_cells=2, max_train_images=12)
    train_samples, train_trials = train_test_split(data, random_seed=0)
    train = utils.slice_data_trials(utils.slice_data_samples(data, train_samples), train_trials)
    eval_points = build_evaluation_points(train, random_seed=0, n_eval_images=8)
    programs_list = _build_seed_programs_list(train)

    out_path = tmp_path / "minimodel_diag_teacher.png"
    plot_model_fits(
        data=train,
        programs_list=programs_list,
        X_eval=eval_points,
        save_path=str(out_path),
        labels=["seed_v1", "seed_v2"],
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 0
