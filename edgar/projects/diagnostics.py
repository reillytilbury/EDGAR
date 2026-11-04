"""
edgar/projects/diagnostics.py

Base class definition for project-level diagnostics, metrics, and visual feedback.
"""

from __future__ import annotations
from typing import Any
import numpy as np


class BaseDiagnostics:
    """Base class for project-specific diagnostics, scalar evaluation, and visualizations."""

    def compute_metrics(
        self,
        data: dict[str, np.ndarray],
        y_pred: np.ndarray,
        params: dict[str, Any] | None = None,
        program: Any | None = None,
    ) -> dict[str, Any]:
        """Computes scalar or summary diagnostic metrics for a single program.

        Executed inside the scoring subprocess worker immediately following parameter
        optimization on the test split.

        Args:
            data: NumPy dictionary of test data for the current split (e.g., test split of X_discover).
            y_pred: NumPy array of model predictions evaluated on `data`.
            params: Dictionary of optimized model parameters.
            program: The Program instance being scored.

        Returns:
            Dictionary of scalar/summary metrics, e.g.
            {"r2_overall": 0.85, "r2_signal": 0.92, "r2_noise": 0.45, "fano_slope": 1.15}
        """
        return {}

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
        """Visualizes model fits against ground truth data (used for post-hoc analysis & Dashboard).

        Args:
            data: NumPy dictionary of data.
            programs: List of Program objects.
            rng: NumPy random generator.
            save_path: Path to save the plot image.
            losses: Scalar losses for each program.
            sample_losses: Per-sample losses for each program.
            program_names: Names for each program.
            params: Parameters for each program.
            title_prefix: Prefix for plot title.
        """
        pass

    def generate_feedback_image(
        self,
        data: dict[str, np.ndarray],
        parents: list[Any],
        rng: np.random.Generator,
        save_path: str = "",
        **kwargs: Any,
    ) -> None:
        """Visualizes parent models for multimodal LLM feedback. Defaults to plot_model_fits.

        Args:
            data: NumPy dictionary of data.
            parents: List of parent Program objects.
            rng: NumPy random generator.
            save_path: Path to save the feedback image.
        """
        self.plot_model_fits(
            data,
            parents,
            rng=rng,
            save_path=save_path,
            **kwargs,
        )
