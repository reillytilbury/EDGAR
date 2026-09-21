"""
Diagnostics for the trial_variability project.
Computes quantitative model fit metrics (R^2 overall, R^2 signal, R^2 noise, Fano slopes)
and provides visualizations for model evaluation and LLM multimodal feedback.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import gaussian_kde
from typing import Any

from edgar.projects.diagnostics import BaseDiagnostics


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
            stim = np.asarray(data["stimulus"])
            if stim.ndim > 1:
                stim = stim[sample_idx]
            resp = np.asarray(data["response"])
            if resp.ndim > 2:
                resp = resp[sample_idx]
            pred = np.asarray(y_pred)
            if pred.ndim > 2:
                pred = pred[sample_idx]

            n_trials, n_cells = resp.shape
            trial_mid, cell_mid = n_trials // 2, n_cells // 2

            # Evaluate on held-out test region (bottom-right: test trials & test cells)
            test_angles = stim[trial_mid:].reshape(-1)
            test_resp = resp[trial_mid:, cell_mid:]
            test_pred = pred[trial_mid:, cell_mid:]

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
        """Plot observed target responses and overlaid model predictions for population & single cells."""
        model_colors = [
            "tab:orange",
            "tab:blue",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:olive",
        ]
        model_alphas = [0.8, 0.5, 0.3]

        if save_path == "":
            raise ValueError("plot_model_fits requires a non-empty save_path")

        # 1. Resolve arguments
        if program_names is None:
            program_names = [p.name for p in programs]
        if losses is None:
            losses = [
                p.program_losses.discover.final
                if hasattr(p, "program_losses")
                else None
                for p in programs
            ]
        if params is None:
            params = [p.params for p in programs]

        # 2. Handle the sample dimension (take the first sample for plotting)
        sample_idx = 0
        stims = np.asarray(data["stimulus"][sample_idx]).reshape(-1)
        actual_response = np.asarray(data["response"][sample_idx])
        sig = np.asarray(data["signal"][sample_idx])  # (n_trials, n_cells)

        # 3. Calculate Preferred Angles using Vector Concentration
        complex_sum = np.nansum(sig * np.exp(2j * stims)[:, np.newaxis], axis=0)
        pref_angles = (np.angle(complex_sum) / 2.0) % np.pi

        # Restrict to test-set region: bottom-right corner (trials n//2:, cells n//2:)
        n_trials, n_cells = actual_response.shape
        trial_mid, cell_mid = n_trials // 2, n_cells // 2

        valid_cells = np.where(~np.all(np.isnan(actual_response), axis=0))[0]
        valid_cells = valid_cells[valid_cells >= cell_mid]
        if len(valid_cells) == 0:
            valid_cells = np.arange(cell_mid, n_cells)

        # Sort valid cells
        pref_angles_valid = pref_angles[valid_cells]
        sort_idx = np.argsort(pref_angles_valid)

        final_cell_idx = valid_cells[sort_idx]
        sorted_pref_angles = pref_angles[final_cell_idx]
        sorted_actual = actual_response[:, final_cell_idx]

        # 4. Pick random test trials
        valid_trials = np.where(~np.all(np.isnan(actual_response), axis=1))[0]
        valid_trials = valid_trials[valid_trials >= trial_mid]
        if len(valid_trials) == 0:
            valid_trials = np.arange(trial_mid, n_trials)

        n_show = min(3, len(valid_trials))
        random_trials = rng.choice(valid_trials, size=n_show, replace=False)

        # 5. Compute predictions
        predictions_sorted = []
        predictions_raw = []

        for i, program in enumerate(programs):
            model_fn = (
                program.compile_model()
                if hasattr(program, "compile_model")
                else program["model"]
            )
            p_dict = params[i]

            def _slice_leaf(x):
                if (
                    isinstance(x, (np.ndarray, jnp.ndarray))
                    and x.ndim > 0
                    and x.shape[0] > sample_idx
                ):
                    return x[sample_idx]
                return x

            plot_params = jax.tree_util.tree_map(_slice_leaf, p_dict)

            single_sample_data = {
                k: v[sample_idx]
                if hasattr(v, "__getitem__") and len(v) > sample_idx
                else v
                for k, v in data.items()
            }

            y_pred = np.asarray(model_fn(single_sample_data, plot_params))
            predictions_raw.append(y_pred)
            y_pred_sorted = y_pred[:, final_cell_idx]
            predictions_sorted.append(y_pred_sorted)

        chosen_cells = rng.choice(
            valid_cells, size=min(3, len(valid_cells)), replace=False
        )

        fig = plt.figure(figsize=(18, 16))
        outer_gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
        best_model_idx = len(programs) - 1

        # --- ROW 1: POPULATION FITS (3 trials) ---
        for i, trial_idx in enumerate(random_trials):
            ax_slot = outer_gs[0, i]
            inner_gs = ax_slot.subgridspec(2, 1, height_ratios=[4, 1.2], hspace=0.08)

            ax1 = fig.add_subplot(inner_gs[0])
            ax2 = fig.add_subplot(inner_gs[1], sharex=ax1)

            angle = stims[trial_idx]
            sort_idx_cell = np.argsort(predictions_sorted[best_model_idx][trial_idx])
            n_cells_shown = len(final_cell_idx)
            x_coords_cells = np.arange(n_cells_shown)

            for j, y_pred_sorted in enumerate(predictions_sorted):
                label = (
                    program_names[j] if j < len(program_names) else f"Model {j + 1}"
                )
                ax1.plot(
                    x_coords_cells,
                    y_pred_sorted[trial_idx][sort_idx_cell],
                    color=model_colors[j % len(model_colors)],
                    linewidth=1.5,
                    alpha=model_alphas[j % len(model_alphas)],
                    label=label,
                    zorder=5,
                )

                sq_err = (y_pred_sorted[trial_idx] - sorted_actual[trial_idx]) ** 2
                ax2.plot(
                    x_coords_cells,
                    sq_err[sort_idx_cell],
                    color=model_colors[j % len(model_colors)],
                    linewidth=1.5,
                    alpha=model_alphas[j % len(model_alphas)],
                )

            angle_diffs = np.abs(
                np.angle(np.exp(2j * (sorted_pref_angles - angle))) / 2.0
            )

            sc = ax1.scatter(
                x_coords_cells,
                sorted_actual[trial_idx][sort_idx_cell],
                c=angle_diffs[sort_idx_cell],
                cmap="viridis",
                vmin=0,
                vmax=np.pi / 2.0,
                s=15,
                alpha=0.5,
                label="Observed",
                zorder=10,
            )

            cbar = fig.colorbar(sc, ax=[ax1, ax2], pad=0.02, aspect=25)
            cbar.set_label("Orientation difference (stim - pref) (rad)")

            ax1.set_title(
                f"Population Response: Trial {trial_idx}\n(Stimulus: {angle:.2f} rad)",
                fontsize=11,
                fontweight="bold",
            )
            ax1.set_ylabel("Response", fontsize=10)
            ax1.tick_params(labelbottom=False, labelsize=9)

            ax2.set_xlabel("Cells (sorted by best model prediction)", fontsize=10)
            ax2.set_ylabel("Squared Error", fontsize=9)
            ax2.tick_params(labelsize=9)
            if i == 0:
                ax1.legend(fontsize=8, loc="upper right")

        # --- ROW 2: SINGLE-CELL TUNING CURVES (3 cells) ---
        sort_idx = np.argsort(stims)
        for i, cell in enumerate(chosen_cells):
            ax_slot = outer_gs[1, i]
            inner_gs = ax_slot.subgridspec(2, 1, height_ratios=[4, 1.2], hspace=0.08)

            ax1 = fig.add_subplot(inner_gs[0])
            ax2 = fig.add_subplot(inner_gs[1], sharex=ax1)

            for j, y_pred in enumerate(predictions_raw):
                label = (
                    program_names[j] if j < len(program_names) else f"Model {j + 1}"
                )
                ax1.plot(
                    stims[sort_idx],
                    y_pred[sort_idx, cell],
                    color=model_colors[j % len(model_colors)],
                    linewidth=1.5,
                    alpha=model_alphas[j % len(model_alphas)],
                    label=label,
                    zorder=5,
                )

                sq_err = (y_pred[:, cell] - actual_response[:, cell]) ** 2
                ax2.plot(
                    stims[sort_idx],
                    sq_err[sort_idx],
                    color=model_colors[j % len(model_colors)],
                    linewidth=1.5,
                    alpha=model_alphas[j % len(model_alphas)],
                )

            non_nan_mask = ~np.isnan(stims) & ~np.isnan(actual_response[:, cell])
            x_coords_tc = stims[non_nan_mask]
            y_coords_tc = actual_response[non_nan_mask, cell]

            if len(x_coords_tc) > 1:
                xy = np.vstack([x_coords_tc, y_coords_tc])
                density = gaussian_kde(xy)(xy)
            else:
                density = np.ones_like(x_coords_tc)

            sc = ax1.scatter(
                x_coords_tc,
                y_coords_tc,
                c=density,
                cmap="viridis",
                s=15,
                alpha=0.5,
                label="Observed",
                zorder=10,
            )

            cbar = fig.colorbar(sc, ax=[ax1, ax2], pad=0.02, aspect=25)
            cbar.set_label("Density")

            ax1.set_title(
                f"Tuning Curve: Cell {cell}\n(Pref. Orientation: {pref_angles[cell]:.2f} rad)",
                fontsize=11,
                fontweight="bold",
            )
            ax1.set_ylabel("Response", fontsize=10)
            ax1.tick_params(labelbottom=False, labelsize=9)

            ax2.set_xlabel("Stimulus Angle (rad)", fontsize=10)
            ax2.set_ylabel("Squared Error", fontsize=9)
            ax2.tick_params(labelsize=9)
            if i == 0:
                ax1.legend(fontsize=8, loc="upper right")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, bbox_inches="tight", dpi=140)
        plt.close()


def plot_model_fits(*args, **kwargs):
    """Module-level compatibility wrapper for Diagnostics.plot_model_fits."""
    return Diagnostics().plot_model_fits(*args, **kwargs)
