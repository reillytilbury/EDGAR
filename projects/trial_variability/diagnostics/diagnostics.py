"""
Diagnostics for the trial_variability project.
Computes quantitative model fit metrics (R^2 overall, R^2 signal, R^2 noise, Fano slopes)
and provides visualizations for model evaluation and LLM multimodal feedback.
"""

from __future__ import annotations

from typing import Any
import warnings
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from edgar.projects.diagnostics import BaseDiagnostics
from edgar.data.neural.cells import preferred_orientation
from edgar.data.neural.utils import bin_x


def fold_over_repeats(
    sorted_angles: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Folds angles and arrays over repeats of unique stimulus angles."""
    _, repeat_counts = np.unique(sorted_angles, return_counts=True)
    n_repeats = int(max(repeat_counts))
    n_angles = len(repeat_counts)

    def fold_array(arr: np.ndarray) -> np.ndarray:
        folded = np.full((n_repeats, n_angles) + arr.shape[1:], np.nan)
        i2 = 0
        for i, rc in enumerate(repeat_counts):
            i1, i2 = i2, i2 + rc
            folded[:rc, i] = arr[i1:i2]
        return folded

    return (fold_array(sorted_angles),) + tuple(fold_array(arr) for arr in arrays)


def fold_trials_and_cells(
    repeat_counts: np.ndarray,
    cotune_count: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...] | np.ndarray:
    """Folds 2D arrays of shape (n_trials, n_cells) based on repeat and cotuning counts."""
    n_repeats = int(max(repeat_counts))
    n_angles = len(repeat_counts)
    n_orientations = len(cotune_count)
    n_cotuned = int(max(cotune_count))

    folded_list = [
        np.full((n_repeats, n_angles, n_orientations, n_cotuned), np.nan)
        for _ in arrays
    ]

    trial_idx_end = 0
    for a_idx, rc in enumerate(repeat_counts):
        trial_idx_start = trial_idx_end
        trial_idx_end = trial_idx_start + rc

        cell_idx_end = 0
        for o_idx, cc in enumerate(cotune_count):
            cell_idx_start = cell_idx_end
            cell_idx_end = cell_idx_start + cc

            for folded, arr in zip(folded_list, arrays):
                folded[:rc, a_idx, o_idx, :cc] = arr[
                    trial_idx_start:trial_idx_end, cell_idx_start:cell_idx_end
                ]

    return tuple(folded_list) if len(folded_list) > 1 else folded_list[0]


def find_top_disagreement_cell(pred1: np.ndarray, pred2: np.ndarray) -> int:
    """Finds the cell index that maximizes mean-squared disagreement between two predictions."""
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        disagreement = np.nanmean((pred1 - pred2) ** 2, axis=tuple(range(pred1.ndim - 1)))
    valid_disagreement = np.nan_to_num(disagreement, nan=-1.0)
    if np.all(valid_disagreement < 0):
        return 0
    return int(np.argmax(valid_disagreement))


def find_top_disagreement_repeats(
    pred1: np.ndarray,
    pred2: np.ndarray,
    folded_angles: np.ndarray,
) -> tuple[int, int, int]:
    """Finds the angle with the highest average disagreement across cells and repeats,
    and returns that angle index along with its two most disagreeing repeat indices.
    """
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cell_disagreement = np.nanmean((pred1 - pred2) ** 2, axis=-1)  # (n_repeats, n_angles)
        mean_disagreement_per_angle = np.nanmean(cell_disagreement, axis=0)  # (n_angles,)

    # Require at least 2 valid repeats for repeat comparisons
    valid_repeat_counts = np.sum(~np.isnan(cell_disagreement), axis=0)
    mean_disagreement_per_angle[valid_repeat_counts < 2] = -np.inf

    if np.all(np.isneginf(mean_disagreement_per_angle)) or np.all(np.isnan(mean_disagreement_per_angle)):
        top_angle_idx = 0
    else:
        top_angle_idx = int(np.nanargmax(mean_disagreement_per_angle))

    disagreement_at_angle = np.nan_to_num(cell_disagreement[:, top_angle_idx], nan=-np.inf)
    valid_repeats = np.where(~np.isneginf(disagreement_at_angle))[0]
    if len(valid_repeats) >= 2:
        top_in_valid = np.argsort(disagreement_at_angle[valid_repeats])[-2:]
        r1, r2 = int(valid_repeats[top_in_valid[0]]), int(valid_repeats[top_in_valid[1]])
    elif len(valid_repeats) == 1:
        r1, r2 = int(valid_repeats[0]), int(valid_repeats[0])
    else:
        r1, r2 = 0, min(1, pred1.shape[0] - 1)
    return top_angle_idx, int(r1), int(r2)


def find_top_disagreement_cotuned_cells(
    pred1_res: np.ndarray,
    pred2_res: np.ndarray,
    cotune_count: np.ndarray,
) -> tuple[int, int, int]:
    """Finds the orientation index and two cotuned cell indices with maximum disagreement between two residual predictions."""
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        disagreement = np.nanmean((pred1_res - pred2_res) ** 2, axis=(0, 1, 3))  # (n_orientations,)
    # Set disagreement to -1 for any orientation bins with fewer than 2 cells
    for i, cc in enumerate(cotune_count):
        if cc < 2:
            disagreement[i] = -1.0

    valid_disagreement = np.nan_to_num(disagreement, nan=-1.0)
    if np.all(valid_disagreement < 0):
        best_ori = int(np.argmax(cotune_count))
        max_c = cotune_count[best_ori]
        return best_ori, 0, min(1, max_c - 1) if max_c > 1 else 0

    orientation_idx = int(np.argmax(valid_disagreement))
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        disagreement_per_orientation = np.nanmean(
            (pred1_res[:, :, orientation_idx, :] - pred2_res[:, :, orientation_idx, :]) ** 2,
            axis=(0, 1),
        )  # (n_cotuned,)
    valid_dis_cotuned = np.nan_to_num(disagreement_per_orientation, nan=-1.0)
    max_c = cotune_count[orientation_idx]
    if max_c >= 2:
        c1, c2 = np.argsort(valid_dis_cotuned[:max_c])[-2:]
    else:
        c1, c2 = 0, 0
    return int(orientation_idx), int(c1), int(c2)


def reduced_rank_regression(
    X: np.ndarray, Y: np.ndarray, rank: int, alpha: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduced Rank Regression (RRR)."""
    X_mean = np.nanmean(X, axis=0)
    Y_mean = np.nanmean(Y, axis=0)
    X_c = np.nan_to_num(X - X_mean, nan=0.0)
    Y_c = np.nan_to_num(Y - Y_mean, nan=0.0)

    n_features = X_c.shape[1]
    A = X_c.T @ X_c + alpha * np.eye(n_features)
    B = X_c.T @ Y_c
    W_ols = np.linalg.solve(A, B)

    Y_hat = X_c @ W_ols
    _, _, Vh = np.linalg.svd(Y_hat, full_matrices=False)

    rank = min(rank, Vh.shape[0])
    V_r = Vh[:rank, :].T
    W_rrr = W_ols @ V_r @ V_r.T

    return W_rrr, X_mean, Y_mean


def compute_rrr_prediction(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    rank: int = 120,
    alpha: float = 5.0,
) -> np.ndarray:
    """Computes RRR predictions on test data."""
    W, xm, ym = reduced_rank_regression(X_train, Y_train, rank=rank, alpha=alpha)
    X_test_c = np.nan_to_num(X_test - xm, nan=0.0)
    Y_pred = X_test_c @ W + ym
    return np.maximum(Y_pred, 0.0)


class Diagnostics(BaseDiagnostics):
    """Project diagnostics for trial-to-trial neural response variability modeling."""

    def compute_metrics(
        self,
        data: dict[str, np.ndarray],
        y_pred: np.ndarray,
        params: dict[str, Any] | None = None,
        program: Any | None = None,
    ) -> dict[str, float]:
        """Computes R^2 overall, R^2 signal, R^2 noise, and Fano slopes on held-out test data.

        Args:
            data: NumPy dictionary of test data containing 'stimulus' and 'response'.
            y_pred: Model predictions evaluated on `data`.
            params: Dictionary of parameters (unused).
            program: Program instance (unused).

        Returns:
            dict containing:
            - r2_overall: Fraction of variance explained on test trials/cells.
            - r2_signal: Fraction of variance explained for stimulus tuning curves.
            - r2_noise: Variance explained of trial-to-trial residual fluctuations.
            - fano_slope_data: Variance-to-mean scaling slope in real neural data.
            - fano_slope_pred: Variance-to-mean scaling slope in model predictions.
        """
        try:
            # Handle sample dimension if present
            sample_idx = 0
            angles = np.asarray(data["stimulus"])
            if angles.ndim > 1:
                angles = angles[sample_idx]
            resp = np.asarray(data["response"])
            if resp.ndim > 2:
                resp = resp[sample_idx]
            pred = np.asarray(y_pred)
            if pred.ndim > 2:
                pred = pred[sample_idx]

            # Restrict to test split (bottom right: n_trials//2:, n_cells//2:)
            n_trials, n_cells = resp.shape
            trial_mid, cell_mid = n_trials // 2, n_cells // 2

            test_resp = resp[trial_mid:, cell_mid:]
            test_pred = pred[trial_mid:, cell_mid:]
            test_angles = angles[trial_mid:]

            if np.all(np.isnan(test_resp)):
                test_resp = resp
                test_pred = pred
                test_angles = angles

            # Overall R^2
            diff_sq = (test_resp - test_pred) ** 2
            mse = float(np.nanmean(diff_sq))
            var_data = float(np.nanvar(test_resp))
            r2_overall = float(1.0 - mse / var_data) if var_data > 1e-12 else 0.0

            # Sort trials by angle for folding over repeats
            sort_idx = np.argsort(test_angles)
            sorted_angles = test_angles[sort_idx]
            sorted_resp = test_resp[sort_idx]
            sorted_pred = test_pred[sort_idx]

            _, folded_resp, folded_pred = fold_over_repeats(
                sorted_angles, sorted_resp, sorted_pred
            )

            # Signal R^2
            r_averaged = np.nanmean(folded_pred, axis=0)
            signal_mse = float(np.mean(np.nanvar(folded_resp, axis=0, mean=r_averaged)))
            r2_signal = float(1.0 - signal_mse / var_data) if var_data > 1e-12 else 0.0

            # Noise R^2
            true_mean = np.nanmean(folded_resp, axis=0, keepdims=True)
            pred_mean = np.nanmean(folded_pred, axis=0, keepdims=True)
            delta_y = folded_resp - true_mean
            delta_y_hat = folded_pred - pred_mean
            noise_mse = float(np.nanmean((delta_y - delta_y_hat) ** 2))
            noise_var = float(np.nanmean(delta_y ** 2))
            r2_noise = float(1.0 - noise_mse / noise_var) if noise_var > 1e-12 else 0.0

            # Fano Slopes
            mean_resp_cond = np.nanmean(folded_resp, axis=0).flatten()
            mean_pred_cond = np.nanmean(folded_pred, axis=0).flatten()
            var_resp_cond = np.nanvar(folded_resp, axis=0).flatten()
            var_pred_cond = np.nanvar(folded_pred, axis=0).flatten()

            mask_resp = (
                ~np.isnan(mean_resp_cond)
                & ~np.isnan(var_resp_cond)
                & (mean_resp_cond > 1e-5)
            )
            mask_pred = (
                ~np.isnan(mean_pred_cond)
                & ~np.isnan(var_pred_cond)
                & (mean_pred_cond > 1e-5)
            )

            slope_resp = (
                float(np.polyfit(mean_resp_cond[mask_resp], var_resp_cond[mask_resp], 1)[0])
                if np.sum(mask_resp) > 1
                else 0.0
            )
            slope_pred = (
                float(np.polyfit(mean_pred_cond[mask_pred], var_pred_cond[mask_pred], 1)[0])
                if np.sum(mask_pred) > 1
                else 0.0
            )

            return {
                "r2_overall": round(r2_overall, 4),
                "r2_signal": round(r2_signal, 4),
                "r2_noise": round(r2_noise, 4),
                "fano_slope_data": round(slope_resp, 4),
                "fano_slope_pred": round(slope_pred, 4),
            }
        except Exception as e:
            return {"error": str(e)}

    def plot_model_fits(
        self,
        data: dict[str, np.ndarray],
        programs: list[Any],
        rng: np.random.Generator,
        save_path: str = "",
        losses: list[float] | None = None,
        sample_losses: list[np.ndarray] | None = None,
        program_names: list[str] | None = None,
        params: list[dict] | None = None,
        title_prefix: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Plot combined diagnostic figures containing tuning curves, repeat consistency, and cotuned noise correlations."""
        if save_path == "":
            raise ValueError("plot_model_fits requires a non-empty save_path")

        if program_names is None:
            program_names = [getattr(p, "name", f"Model {i+1}") for i, p in enumerate(programs)]
        if params is None:
            params = [getattr(p, "params", {}) for p in programs]

        # 1. Extract data for single sample
        sample_idx = 0
        raw_stim = np.asarray(data["stimulus"])
        stims = np.asarray(raw_stim[sample_idx] if raw_stim.ndim > 1 else raw_stim).reshape(-1)
        raw_resp = np.asarray(data["response"])
        resp = np.asarray(raw_resp[sample_idx] if raw_resp.ndim > 2 else raw_resp)

        n_trials, n_cells = resp.shape
        trial_mid, cell_mid = n_trials // 2, n_cells // 2

        # Partition into train predictors/targets and test predictors/targets
        X_train = resp[:trial_mid, :cell_mid]
        Y_train = resp[:trial_mid, cell_mid:]
        X_test = resp[trial_mid:, :cell_mid]
        test_responses = resp[trial_mid:, cell_mid:]
        test_angles = stims[trial_mid:]

        if np.all(np.isnan(test_responses)):
            test_responses = resp
            test_angles = stims
            X_train = resp[:trial_mid, :cell_mid]
            Y_train = resp[:trial_mid, cell_mid:]
            X_test = resp[trial_mid:, :cell_mid]

        # 2. Compute predictions for all models
        predictions_test = []
        for i, program in enumerate(programs):
            model_fn = (
                program.compile_model()
                if hasattr(program, "compile_model")
                else (program["model"] if isinstance(program, dict) and "model" in program else program)
            )
            p_dict = params[i] if i < len(params) else {}

            def _slice_leaf(x):
                if (
                    isinstance(x, (np.ndarray, jnp.ndarray))
                    and x.ndim > 0
                    and x.shape[0] > sample_idx
                ):
                    return x[sample_idx]
                return x

            plot_params = jax.tree_util.tree_map(_slice_leaf, p_dict) if p_dict is not None else {}

            single_sample_data = {
                k: v[sample_idx]
                if hasattr(v, "__getitem__") and len(v) > sample_idx and getattr(v, "ndim", 0) > (1 if k == "stimulus" else 2)
                else v
                for k, v in data.items()
            }

            y_pred = np.asarray(model_fn(single_sample_data, plot_params))
            if y_pred.ndim > 2:
                y_pred = y_pred[sample_idx]

            if y_pred.shape == resp.shape and not np.all(np.isnan(resp[trial_mid:, cell_mid:])):
                y_pred_test = y_pred[trial_mid:, cell_mid:]
            else:
                y_pred_test = y_pred

            predictions_test.append(y_pred_test)

        if len(predictions_test) == 0:
            return

        if len(predictions_test) == 1:
            y1 = predictions_test[0]
            y2 = predictions_test[0]
            name1 = program_names[0] if program_names else "Model 1"
            name2 = f"{name1} (Copy)"
        else:
            y1 = predictions_test[0]
            y2 = predictions_test[1]
            name1 = program_names[0] if program_names else "Model 1"
            name2 = program_names[1] if len(program_names) > 1 else "Model 2"

        # 3. Compute Reduced Rank Regression (RRR) baseline
        rank = min(120, max(1, min(X_train.shape[1], Y_train.shape[1], X_train.shape[0])))
        rrr = compute_rrr_prediction(X_train, Y_train, X_test, rank=rank, alpha=5.0)
        if rrr.shape != test_responses.shape:
            rrr = np.zeros_like(test_responses)

        # 4. Fold data over repeats for Tuning Curves and Repeat Scatter
        sort_idx = np.argsort(test_angles)
        sorted_angles = test_angles[sort_idx]
        sorted_responses, sorted_y1, sorted_y2, sorted_rrr = [
            arr[sort_idx] for arr in (test_responses, y1, y2, rrr)
        ]

        folded_angles, folded_responses, folded_y1, folded_y2, folded_rrr = fold_over_repeats(
            sorted_angles, sorted_responses, sorted_y1, sorted_y2, sorted_rrr
        )

        # Find max disagreement cells for Section 1
        cell_12 = find_top_disagreement_cell(folded_y1, folded_y2)
        cell_1rrr = find_top_disagreement_cell(folded_y1, folded_rrr)
        cell_2rrr = find_top_disagreement_cell(folded_y2, folded_rrr)

        # Find max disagreement repeats for Section 2
        angle_12, r1_12, r2_12 = find_top_disagreement_repeats(folded_y1, folded_y2, folded_angles)
        angle_1rrr, r1_1rrr, r2_1rrr = find_top_disagreement_repeats(folded_y1, folded_rrr, folded_angles)
        angle_2rrr, r1_2rrr, r2_2rrr = find_top_disagreement_repeats(folded_y2, folded_rrr, folded_angles)

        # 5. Compute cotuned sorting and residuals for Section 3
        cell_preferred_orientations = preferred_orientation(test_responses, test_angles)
        n_orientations = 128
        binned_cells = bin_x(
            cell_preferred_orientations,
            n_bins=n_orientations,
            min_val=0,
            max_val=2 * np.pi,
        )
        cell_sort_idx = np.argsort(binned_cells)
        sorted_orientations = binned_cells[cell_sort_idx]

        trial_sort_idx = np.argsort(test_angles)
        sorted_angles_trial = test_angles[trial_sort_idx]
        sorted_both_responses, sorted_both_y1, sorted_both_y2, sorted_both_rrr = [
            arr[trial_sort_idx][:, cell_sort_idx] for arr in (test_responses, y1, y2, rrr)
        ]

        _, repeat_counts = np.unique(sorted_angles_trial, return_counts=True)
        _, cotune_count = np.unique(sorted_orientations, return_counts=True)

        folded_both_responses, folded_both_y1, folded_both_y2, folded_both_rrr = fold_trials_and_cells(
            repeat_counts,
            cotune_count,
            sorted_both_responses,
            sorted_both_y1,
            sorted_both_y2,
            sorted_both_rrr,
        )

        # Subtract signal component (average over repeats) to obtain residuals
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            res_responses, res_y1, res_y2, res_rrr = [
                arr - np.nanmean(arr, axis=0, keepdims=True)
                for arr in (folded_both_responses, folded_both_y1, folded_both_y2, folded_both_rrr)
            ]

        idx_12 = find_top_disagreement_cotuned_cells(res_y1, res_y2, cotune_count)
        idx_1rrr = find_top_disagreement_cotuned_cells(res_y1, res_rrr, cotune_count)
        idx_2rrr = find_top_disagreement_cotuned_cells(res_y2, res_rrr, cotune_count)

        # 6. Create Large Combined Figure
        fig = plt.figure(figsize=(15, 27), constrained_layout=True)
        outer_subfigs = fig.subfigures(3, 1, height_ratios=[1.0, 1.0, 1.0])

        # --- SECTION 1: TUNING CURVES & TRIAL-TO-TRIAL ENVELOPES (3 rows x 2 cols) ---
        outer_subfigs[0].suptitle(
            "1. Single-Cell Tuning Curves & Trial-to-Trial Response Envelopes",
            fontsize=13,
            fontweight="bold",
        )
        axes1 = outer_subfigs[0].subplots(3, 2)
        rows_info_tc = [
            (cell_12, f"{name1} vs {name2}"),
            (cell_1rrr, f"{name1} vs RRR"),
            (cell_2rrr, f"{name2} vs RRR"),
        ]

        angles_x = folded_angles[0, :]
        for row_idx, (cell_idx, comp_name) in enumerate(rows_info_tc):
            resp_cell = folded_responses[:, :, cell_idx]
            rrr_cell = folded_rrr[:, :, cell_idx]

            mean_resp = np.nanmean(resp_cell, axis=0)
            min_rrr, max_rrr = np.nanmin(rrr_cell, axis=0), np.nanmax(rrr_cell, axis=0)
            is_first_row = row_idx == 0

            for col_idx, (model_cell, model_name, col_color) in enumerate(
                [
                    (folded_y1[:, :, cell_idx], name1, "C0"),
                    (folded_y2[:, :, cell_idx], name2, "C1"),
                ]
            ):
                ax = axes1[row_idx, col_idx]
                lbl = (lambda name: name) if is_first_row else (lambda name: None)

                # Signals
                ax.plot(
                    angles_x,
                    mean_resp,
                    "-",
                    color="black",
                    linewidth=2.0,
                    label=lbl("Data (Signal)"),
                    zorder=5,
                )
                ax.plot(
                    angles_x,
                    np.nanmean(model_cell, axis=0),
                    "-",
                    color=col_color,
                    linewidth=2.0,
                    label=lbl(f"{model_name} (Signal)"),
                    zorder=6,
                )

                # Envelopes
                ax.fill_between(
                    angles_x,
                    np.nanmin(model_cell, axis=0),
                    np.nanmax(model_cell, axis=0),
                    color=col_color,
                    alpha=0.20,
                    label=lbl(f"{model_name} Envelope"),
                    zorder=3,
                )
                ax.fill_between(
                    angles_x,
                    min_rrr,
                    max_rrr,
                    color="C4",
                    alpha=0.12,
                    label=lbl("RRR Envelope"),
                    zorder=2,
                )

                ax.set_title(
                    f"Max Disagreement: {comp_name} (Cell {cell_idx})\n{model_name} vs RRR",
                    fontsize=10,
                )
                ax.set_xlabel("Stimulus Angle (rad)", fontsize=9)
                ax.set_ylabel("Normalized Response", fontsize=9)
                ax.tick_params(labelsize=8)

            if is_first_row:
                axes1[0, 0].legend(loc="upper right", fontsize=8)
                axes1[0, 1].legend(loc="upper right", fontsize=8)

        # --- SECTION 2: REPEAT SCATTER COMPARISON (3 rows x 3 cols) ---
        outer_subfigs[1].suptitle(
            "2. Single-Trial Repeat Consistency Across Cells",
            fontsize=13,
            fontweight="bold",
        )
        row_subfigs2 = outer_subfigs[1].subfigures(3, 1)
        rows_info_rep = [
            (angle_12, r1_12, r2_12, f"{name1} vs {name2}"),
            (angle_1rrr, r1_1rrr, r2_1rrr, f"{name1} vs RRR"),
            (angle_2rrr, r1_2rrr, r2_2rrr, f"{name2} vs RRR"),
        ]

        for row_idx, (top_angle_idx, r1, r2, comp_name) in enumerate(rows_info_rep):
            subfig = row_subfigs2[row_idx]
            angle_val = folded_angles[0, top_angle_idx]
            subfig.suptitle(
                f"Trial maximizing {comp_name} disagreement (Stimulus: {angle_val / np.pi:.2f}π rad)",
                fontsize=10,
                fontweight="bold",
            )
            ax_row = subfig.subplots(1, 3)

            cols_config = [
                (folded_rrr, "C4", "RRR", f"Response repeat {r2}"),
                (folded_y1, "C0", name1, None),
                (folded_y2, "C1", name2, None),
            ]

            all_vals = []
            for arr, _, _, _ in cols_config:
                all_vals.extend([arr[r1, top_angle_idx, :], arr[r2, top_angle_idx, :]])
            flat_vals = np.concatenate([v.flatten() for v in all_vals if v is not None])
            min_val = (
                float(np.nanmin(flat_vals) - 0.05)
                if len(flat_vals) > 0 and not np.all(np.isnan(flat_vals))
                else 0.0
            )
            max_val = (
                float(np.nanmax(flat_vals) + 0.05)
                if len(flat_vals) > 0 and not np.all(np.isnan(flat_vals))
                else 1.0
            )

            for col_idx, (arr, color, col_title, ylabel) in enumerate(cols_config):
                ax = ax_row[col_idx]
                x_data = arr[r1, top_angle_idx, :]
                y_data = arr[r2, top_angle_idx, :]

                ax.scatter(x_data, y_data, color=color, alpha=0.5, s=15, zorder=2)
                ax.plot(
                    [min_val, max_val],
                    [min_val, max_val],
                    "gray",
                    linestyle="--",
                    alpha=0.5,
                    zorder=1,
                )

                valid_mask = ~np.isnan(x_data) & ~np.isnan(y_data)
                if np.sum(valid_mask) > 1:
                    try:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            r_coef = np.corrcoef(x_data[valid_mask], y_data[valid_mask])[0, 1]
                        if not np.isnan(r_coef):
                            ax.text(
                                0.05,
                                0.95,
                                f"r = {r_coef:.3f}",
                                transform=ax.transAxes,
                                verticalalignment="top",
                                fontsize=8,
                                fontweight="bold",
                                bbox=dict(
                                    facecolor="white",
                                    alpha=0.8,
                                    boxstyle="round,pad=0.2",
                                    edgecolor="gray",
                                ),
                            )
                    except Exception:
                        pass

                ax.set_xlim(min_val, max_val)
                ax.set_ylim(min_val, max_val)
                ax.set_title(col_title, fontsize=10)
                ax.set_xlabel(f"Response repeat {r1}", fontsize=9)
                if ylabel:
                    ax.set_ylabel(ylabel, fontsize=9)
                ax.tick_params(labelsize=8)

        # --- SECTION 3: RESIDUAL SCATTER IN COTUNED CELLS (3 rows x 3 cols) ---
        outer_subfigs[2].suptitle(
            "3. Cotuned Noise Correlations (Residual Scatter Across Trials)",
            fontsize=13,
            fontweight="bold",
        )
        row_subfigs3 = outer_subfigs[2].subfigures(3, 1)
        rows_info_res = [
            (idx_12, f"{name1} vs {name2}"),
            (idx_1rrr, f"{name1} vs RRR"),
            (idx_2rrr, f"{name2} vs RRR"),
        ]

        for row_idx, ((orientation_idx, c1, c2), comp_name) in enumerate(rows_info_res):
            subfig = row_subfigs3[row_idx]
            subfig.suptitle(
                f"Cotuned cells with maximum noise disagreement ({comp_name})",
                fontsize=10,
                fontweight="bold",
            )
            ax_row = subfig.subplots(1, 3)

            get_flat = lambda arr: (
                arr[:, :, orientation_idx, c1].flatten(),
                arr[:, :, orientation_idx, c2].flatten(),
            )
            x_rrr, y_rrr = get_flat(res_rrr)
            x_y1, y_y1 = get_flat(res_y1)
            x_y2, y_y2 = get_flat(res_y2)

            all_res = [x_rrr, y_rrr, x_y1, y_y1, x_y2, y_y2]
            flat_res = np.concatenate([v.flatten() for v in all_res if v is not None])
            max_val = (
                float(np.nanmax(np.abs(flat_res)) * 1.1)
                if len(flat_res) > 0 and not np.all(np.isnan(flat_res))
                else 1.0
            )
            if max_val < 1e-4 or np.isnan(max_val):
                max_val = 1.0

            cols_config = [
                (x_rrr, y_rrr, "C4", "RRR", "Cell j Noise"),
                (x_y1, y_y1, "C0", name1, None),
                (x_y2, y_y2, "C1", name2, None),
            ]

            for col_idx, (x, y, color, col_title, ylabel) in enumerate(cols_config):
                ax = ax_row[col_idx]
                ax.scatter(x, y, color=color, alpha=0.5, s=15, zorder=2)
                ax.set_xlim(-max_val, max_val)
                ax.set_ylim(-max_val, max_val)
                ax.axhline(0, color="gray", linestyle="--", alpha=0.3, zorder=1)
                ax.axvline(0, color="gray", linestyle="--", alpha=0.3, zorder=1)

                valid_mask = ~np.isnan(x) & ~np.isnan(y)
                if np.sum(valid_mask) > 1 and np.var(x[valid_mask]) > 1e-12:
                    try:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            r_noise = np.corrcoef(x[valid_mask], y[valid_mask])[0, 1]
                            slope, intercept = np.polyfit(x[valid_mask], y[valid_mask], 1)

                        fit_x = np.array([-max_val, max_val])
                        fit_y = slope * fit_x + intercept
                        ax.plot(
                            fit_x,
                            fit_y,
                            color="gray",
                            linestyle=":",
                            linewidth=1.5,
                            alpha=0.7,
                            zorder=1,
                        )

                        if not np.isnan(r_noise):
                            ax.text(
                                0.05,
                                0.95,
                                f"r_noise = {r_noise:.3f}",
                                transform=ax.transAxes,
                                verticalalignment="top",
                                fontsize=8,
                                fontweight="bold",
                                bbox=dict(
                                    facecolor="white",
                                    alpha=0.8,
                                    boxstyle="round,pad=0.2",
                                    edgecolor="gray",
                                ),
                            )
                    except Exception:
                        pass

                ax.set_title(col_title, fontsize=10)
                ax.set_xlabel("Cell i Noise", fontsize=9)
                if ylabel:
                    ax.set_ylabel(ylabel, fontsize=9)
                ax.tick_params(labelsize=8)

        plt.savefig(save_path, bbox_inches="tight", dpi=100)
        plt.close(fig)


def plot_model_fits(*args, **kwargs):
    """Module-level compatibility wrapper for Diagnostics.plot_model_fits."""
    return Diagnostics().plot_model_fits(*args, **kwargs)
