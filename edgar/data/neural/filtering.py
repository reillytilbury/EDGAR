import numpy as np
from typing import Callable
from . import signal


def activity(response: np.ndarray) -> np.ndarray:
    """
    Calculates firing probabilities for each cell.
    """
    active_elements = (response > 0).astype(np.float32)
    return np.mean(active_elements, axis=1)


def vector_concentration(response: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    Calculates vector concentrations for each cell.
    """
    conc = np.abs(
        np.sum(np.exp(2j * angles)[np.newaxis, :] * response, axis=1)
        / np.sum(response, axis=1)
    )
    return conc


def signal_fraction(response: np.ndarray, min_repeats: int = 2) -> np.ndarray:
    """
    Calculates unbiased signal fractions for each cell.
    """
    signal_fraction, _ = signal._unbiased_fraction(response, min_repeats=min_repeats)
    return signal_fraction


def apply(
    response: np.ndarray,
    filter_quantity: Callable,
    threshold: float,
    cell_axis: int = 0,
    **kwargs,
) -> np.ndarray:
    """
    Filters cells based on a given metric and threshold.
    """
    filter_values = filter_quantity(response, **kwargs)
    good_cells = np.where(filter_values > threshold)[0]
    print(
        f"Selected {len(good_cells)} / {response.shape[cell_axis]} cells with {filter_quantity.__name__} > {threshold}."
    )
    return np.take(response, good_cells, axis=cell_axis)


def by_activity(
    response: np.ndarray, threshold: float, cell_axis: int = 0
) -> np.ndarray:
    return apply(response, activity, threshold, cell_axis=cell_axis)


def by_vector_concentration(
    response: np.ndarray, angles: np.ndarray, threshold: float
) -> np.ndarray:
    return apply(response, vector_concentration, threshold, angles=angles)


def by_signal_fraction(
    response: np.ndarray, threshold: float, min_repeats: int = 2
) -> np.ndarray:
    return apply(
        response,
        signal_fraction,
        threshold,
        cell_axis=1,
        min_repeats=min_repeats,
    )
