"""
Diagnostics Manager for EDGAR.

Each experiment should provide a `diagnostics.py` file that defines four
functions used by the engine:

1. `select_evaluation_points(inputs, n_points=100, random_seed=0, **kwargs)`
   - Chooses experiment-specific evaluation inputs used for model evaluation
     summaries (`evaluation_matrix`).
   - Return shape should be:
     - `(n_samples, n_eval)` for single-input experiments, or
     - `(n_samples, n_features, n_eval)` for multi-input experiments.

2. `plot_model_fits(plot_data, ...)`
   - Draws multi-model fit diagnostics for selected samples/cells.
   - Receives precomputed tensors from the engine in `plot_data` and should
     only handle visualization decisions (layout/styling/annotations).
   - Called repeatedly during runs for image feedback and saved diagnostics.

3. `plot_single_model_fit(...)`
   - Draws a detailed fit plot for one model.
   - Used for final reports of top-ranked programs.

4. `plot_train_vs_test_loss(...)`
   - Draws train-vs-test scatter diagnostics across discovered programs.

Process in a typical run:
- `load_diagnostics(experiment_path)` imports `experiments/<task>/diagnostics.py`
  and validates that all required functions exist.
- `hypothesis_engine` calls `select_evaluation_points(...)` once on training
  inputs to get evaluation inputs for the run.
- `hypothesis_engine` computes `evaluation_matrix` values using those inputs.
- `hypothesis_engine` prepares `plot_data` (predictions, observations, losses,
  evaluation-grid outputs, and subplot metadata) and then calls plotting
  functions to save iteration/final figures.

This module is the contract boundary that guarantees every experiment exposes
the same diagnostics entry points while still letting each task control
evaluation-point policy and visualization style.
"""

import importlib.util
import logging
from pathlib import Path
from typing import Protocol, Optional, Callable, TypedDict, runtime_checkable
import numpy as np
import pandas as pd
import jax.numpy as jnp


class ModelFitPlotData(TypedDict):
    """Structured payload consumed by experiment `plot_model_fits(plot_data=...)`."""

    sample_selection: np.ndarray
    stimuli_1d: jnp.ndarray
    spike_matrix: jnp.ndarray
    point_losses: jnp.ndarray
    x_values_mean: jnp.ndarray
    binned_mean: jnp.ndarray
    x_values_eval: jnp.ndarray
    model_outputs: jnp.ndarray
    n_row_cols: int
    n_models: int
    n_cells: int
    n_trials: int
    n_eval: int
    n_mean: int
    input_idx: int


@runtime_checkable
class DiagnosticsProtocol(Protocol):
    """
    Protocol defining the required diagnostic functions.
    
    Any experiment-specific diagnostics module must implement these functions
    with compatible signatures. The Protocol ensures type safety and provides
    IDE autocomplete support.
    """
    
    def select_evaluation_points(self,
                                 inputs: jnp.ndarray,
                                 n_points: int = 100,
                                 random_seed: int = 0) -> jnp.ndarray:
        """Select experiment-specific evaluation points for model diagnostics."""
        ...

    def plot_model_fits(self,
                        plot_data: ModelFitPlotData,
                        colours: list = ...,
                        labels: Optional[list] = None,
                        title: str = '',
                        line_width: float = 4.0,
                        line_alpha: float = 1.0,
                        point_alpha: float = 0.1,
                        point_size: int = 80,
                        legend_fontsize: int = 12,
                        dpi: float = 100.0,
                        save_path: Optional[str] = None) -> None:
        """
        Plot fits of multiple models using precomputed plotting tensors.

        Required `plot_data` keys are produced by
        `hypothesis_engine.prepare_model_fit_plot_data(...)`.
        """
        ...
    
    def plot_single_model_fit(self,
                              model: Callable,
                              loss_function: Callable,
                              x: jnp.ndarray,
                              y: jnp.ndarray,
                              params: jnp.ndarray,
                              n_eval: int = 100,
                              n_mean: int = 50,
                              dpi: float = 100.0,
                              title: str = '',
                              save_path: Optional[str] = None,
                              input_idx: int = 0) -> None:
        """Plot fit of a single model."""
        ...
    
    def plot_train_vs_test_loss(self,
                                programs_df: pd.DataFrame,
                                island_labels: list,
                                save_path: Optional[str] = None) -> None:
        """Plot train vs test loss scatter."""
        ...


# Required function names that must exist in a diagnostics module
REQUIRED_FUNCTIONS = [
    'select_evaluation_points',
    'plot_model_fits',
    'plot_single_model_fit',
    'plot_train_vs_test_loss',
]


def load_diagnostics(experiment_path: Optional[str] = None):
    """
    Load a diagnostics module from an experiment directory.
    
    Args:
        experiment_path: Path to experiment directory containing diagnostics.py.
                        If None or if diagnostics.py doesn't exist, returns None
                        (diagnostics will be disabled).
                        Can be relative (e.g., 'experiments/orientation_tuning')
                        or absolute.
    
    Returns:
        A module implementing the DiagnosticsProtocol functions, or None if
        no diagnostics module is available.
    
    Raises:
        ValueError: If required functions are missing from the module.
    
    Example:
        # Load diagnostics (returns None if not found)
        diag = load_diagnostics('experiments/orientation_tuning')
        if diag:
            diag.plot_model_fits(plot_data=...)
    """
    if experiment_path is None:
        logging.info("No diagnostics path specified, diagnostics disabled.")
        return None
    
    # Resolve the path to diagnostics.py
    exp_path = Path(experiment_path)
    if not exp_path.is_absolute():
        # Assume relative to project root
        exp_path = Path(__file__).parent.parent / exp_path
    
    diagnostics_file = exp_path / 'diagnostics.py'
    
    if not diagnostics_file.exists():
        logging.info(
            f"No diagnostics.py found at {diagnostics_file}, diagnostics disabled."
        )
        return None
    
    # Dynamically import the module
    spec = importlib.util.spec_from_file_location(
        f"diagnostics_{exp_path.name}",
        diagnostics_file
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Validate required functions exist
    missing = [fn for fn in REQUIRED_FUNCTIONS if not hasattr(module, fn)]
    if missing:
        raise ValueError(
            f"Diagnostics module at {diagnostics_file} is missing required functions: "
            f"{missing}. See DiagnosticsProtocol for required signatures."
        )
    
    logging.info(f"Loaded diagnostics from {diagnostics_file}")
    return module


def get_default_diagnostics():
    """
    Convenience function to get the default diagnostics module.
    
    Returns:
        The src/diagnostic module.
    """
    return load_diagnostics(None)
