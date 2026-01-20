"""
Diagnostics Manager for EDGAR.

This module provides a way to load experiment-specific diagnostic functions
while ensuring they implement the required interface. This allows different
experiments (e.g., orientation_tuning, spatial_frequency) to have their own
visualization functions while maintaining a consistent API.

Usage:
    from src.diagnostics_manager import load_diagnostics
    
    # Load experiment-specific diagnostics
    diagnostics = load_diagnostics('experiments/orientation_tuning')
    diagnostics.plot_model_fits(...)
    
    # Or use default (src/diagnostic.py)
    diagnostics = load_diagnostics()
"""

import importlib.util
import logging
from pathlib import Path
from typing import Protocol, Optional, Callable, Sequence, runtime_checkable

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
    
    def plot_model_fits(self,
                        programs_df: pd.DataFrame,
                        loss_function: Callable,
                        x: jnp.ndarray,
                        y: jnp.ndarray,
                        cell_selection: Sequence[int],
                        n_eval: int = 100,
                        n_mean: int = 50,
                        colours: list = ...,
                        labels: Optional[list] = None,
                        title: str = '',
                        line_width: float = 4.0,
                        line_alpha: float = 1.0,
                        point_alpha: float = 0.1,
                        point_size: int = 80,
                        legend_fontsize: int = 12,
                        dpi: float = 100.0,
                        save_path: Optional[str] = None,
                        predictor_idx: int = 0) -> None:
        """Plot fits of multiple models over selected cells."""
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
                              predictor_idx: int = 0) -> None:
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
            diag.plot_model_fits(...)
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
