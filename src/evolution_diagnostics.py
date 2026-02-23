"""
Diagnostics Manager for EDGAR.

Each experiment should provide a `diagnostics.py` file that defines two
functions used by the engine:

1. `select_evaluation_points(inputs, n_points=100, random_seed=0, **kwargs)`
   - Chooses experiment-specific evaluation inputs used for model evaluation
     summaries (`evaluation_matrix`).
   - Return shape should be:
     - `(n_samples, n_eval)` for single-input experiments, or
     - `(n_samples, n_features, n_eval)` for multi-input experiments.

2. `plot_model_fits(X, Y, programs_list, ...)`
   - Draws multi-model fit diagnostics for selected samples/cells.
   - Called repeatedly during runs for image feedback and saved diagnostics.

Process in a typical run:
- `load_diagnostics(experiment_path)` imports `experiments/<task>/diagnostics.py`
  and validates that all required functions exist.
- `hypothesis_engine` calls `select_evaluation_points(...)` once on training
  inputs to get evaluation inputs for the run.
- `hypothesis_engine` computes `evaluation_matrix` values using those inputs.
- `hypothesis_engine` passes raw inputs/outputs plus program parameters
  to task-specific plotting functions to save iteration/final figures.

This module is the contract boundary that guarantees every experiment exposes
the same diagnostics entry points while still letting each task control
evaluation-point policy and visualization style.
"""

import importlib.util
import logging
from pathlib import Path
from typing import Protocol, Optional, runtime_checkable
import numpy as np
import pandas as pd
import jax.numpy as jnp


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
                        X: jnp.ndarray,
                        Y: jnp.ndarray,
                        programs_list: list,
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
        """Plot fits of multiple models using raw inputs/outputs."""
        ...


# Required function names that must exist in a diagnostics module
REQUIRED_FUNCTIONS = [
    'select_evaluation_points',
    'plot_model_fits',
]


def plot_train_vs_test_loss(programs_df: pd.DataFrame,
                            island_labels: Optional[list] = None,
                            save_path: Optional[str] = None,
                            loss_thresh: float = 1000.0,
                            title: str = 'Train vs Test Loss') -> None:
    """
    Shared train-vs-test loss scatter used across experiments.

    This central implementation replaces per-experiment duplicates while keeping
    behavior robust for both island-labeled and unlabeled data.
    """
    if 'train_loss' not in programs_df.columns or 'test_loss' not in programs_df.columns:
        raise ValueError("DataFrame must contain 'train_loss' and 'test_loss' columns.")

    train_loss = np.asarray(programs_df['train_loss'].to_numpy(), dtype=float)
    test_loss = np.asarray(programs_df['test_loss'].to_numpy(), dtype=float)

    train_loss = np.nan_to_num(train_loss, nan=np.inf, posinf=np.inf, neginf=np.inf)
    test_loss = np.nan_to_num(test_loss, nan=np.inf, posinf=np.inf, neginf=np.inf)

    finite_mask = np.isfinite(train_loss) & np.isfinite(test_loss)
    if np.isfinite(loss_thresh):
        finite_mask &= (train_loss < loss_thresh) & (test_loss < loss_thresh)

    train_loss = train_loss[finite_mask]
    test_loss = test_loss[finite_mask]

    birth_island = None
    if 'birth_island' in programs_df.columns:
        birth_island = np.asarray(programs_df['birth_island'].to_numpy())[finite_mask]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if train_loss.size == 0:
        plt.figure(figsize=(10, 10))
        if np.isfinite(loss_thresh):
            msg = f'No valid programs\n(all losses >= {loss_thresh})'
        else:
            msg = 'No valid programs'
        plt.text(0.5, 0.5, msg, ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        plt.xlabel('Train Loss')
        plt.ylabel('Test Loss')
        plt.title(title)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
        return

    plt.figure(figsize=(10, 10))
    cmap = plt.get_cmap('tab10')

    if birth_island is not None and birth_island.size == train_loss.size:
        for island_id in np.unique(birth_island):
            island_mask = (birth_island == island_id)
            island_int = int(island_id)
            if island_labels is not None and 0 <= island_int < len(island_labels):
                label = island_labels[island_int]
            else:
                label = f'Island {island_int}'
            plt.scatter(
                train_loss[island_mask],
                test_loss[island_mask],
                label=label,
                color=cmap(island_int % 10),
                alpha=0.9,
            )
    else:
        plt.scatter(train_loss, test_loss, alpha=0.9, color='tab:blue', edgecolor='none')

    lo = min(float(np.min(train_loss)), float(np.min(test_loss)))
    hi = max(float(np.max(train_loss)), float(np.max(test_loss)))
    if hi <= lo:
        hi = lo + 1e-6
    pad = 0.05 * (hi - lo)
    plt.xlim(lo - pad, hi + pad)
    plt.ylim(lo - pad, hi + pad)
    plt.plot([lo, hi], [lo, hi], color='black', linestyle='--', alpha=0.6)

    plt.xlabel('Train Loss')
    plt.ylabel('Test Loss')
    plt.title(title)
    if birth_island is not None and np.unique(birth_island).size > 1:
        plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


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
        diag.plot_model_fits(X=..., Y=..., programs_list=..., save_path=...)
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
