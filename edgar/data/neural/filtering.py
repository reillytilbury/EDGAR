"""
This module provides functions for filtering neural responses based on various metrics.

It includes utilities to calculate:
- Firing probabilities (activity).
- Vector concentrations (directionality).
- Unbiased signal fractions (stimulus-related variance).

These metrics can then be used to filter cells from a neural response dataset
based on a specified threshold.
"""

import numpy as np
from typing import Callable
from . import signal


def activity(response: np.ndarray) -> np.ndarray:
    """Calculates firing probabilities for each cell.

    The activity is defined as the proportion of non-zero (active) elements
    across the second dimension (e.g., trials or time points) for each cell.

    Args:
        response: A NumPy array representing neural responses. Expected shape
            is (n_cells, n_observations).

    Returns:
        A NumPy array of shape (n_cells,) containing the firing probability
        for each cell.
    """
    active_elements = (response > 0).astype(np.float32)
    return np.mean(active_elements, axis=1)


def vector_concentration(response: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Calculates vector concentrations for each cell.

    Vector concentration is a measure of the circular variance of responses,
    often used to quantify the strength of directional tuning in neurons.
    A higher concentration indicates stronger tuning.

    The calculation is based on the formula:
    $$ \\text{Concentration} = \\left| \\frac{\\sum_{k} e^{2i \\theta_k} R_k}{\\sum_{k} R_k} \\right| $$
    where $R_k$ is the response at angle $\\theta_k$.

    Args:
        response: A NumPy array representing neural responses. Expected shape
            is (n_cells, n_angles).
        angles: A NumPy array of shape (n_angles,) containing the angles
            (in radians) corresponding to the responses.

    Returns:
        A NumPy array of shape (n_cells,) containing the vector concentration
        for each cell.
    """
    conc = np.abs(
        np.sum(np.exp(2j * angles)[np.newaxis, :] * response, axis=1)
        / np.sum(response, axis=1)
    )
    return conc


def signal_fraction(response: np.ndarray, min_repeats: int = 2) -> np.ndarray:
    """Calculates unbiased signal fractions for each cell.

    This function computes the unbiased fraction of stimulus-related variance
    as described by Sahani & Linden (2003). It quantifies how much of the
    total response variance is attributable to the stimulus, independent of
    noise.

    The underlying calculation is performed by `signal._unbiased_fraction`.

    Args:
        response: A NumPy array of neural responses. Expected shape
            is (n_repeats, n_cells, n_angles).
        min_repeats: The minimum number of repeats required per angle to
            perform the calculation.

    Returns:
        A NumPy array of shape (n_cells,) containing the unbiased signal
        fraction for each cell, clipped to be between 0 and 1.
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
    """Filters cells based on a given metric and threshold.

    This generic function applies a `filter_quantity` callable to the `response`
    data, then selects cells where the calculated filter value exceeds the
    specified `threshold`.

    Args:
        response: A NumPy array containing the neural response data.
        filter_quantity: A callable function that takes the `response` array
            and `**kwargs` as input and returns a NumPy array of filter values
            for each cell (e.g., `activity`, `vector_concentration`).
        threshold: The scalar threshold value. Cells with `filter_values > threshold`
            will be retained.
        cell_axis: The axis along which cells are organized in the `response` array.
            Defaults to 0.
        **kwargs: Additional keyword arguments to pass to the `filter_quantity` function.

    Returns:
        A NumPy array containing the filtered `response` data, with only the
        cells that passed the filter.
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
    """Filters cells based on their firing activity.

    This is a convenience wrapper around the `apply` function, using the
    `activity` function as the filtering metric.

    Args:
        response: A NumPy array representing neural responses. Expected shape
            is (n_cells, n_observations).
        threshold: The minimum firing probability a cell must have to be
            retained.
        cell_axis: The axis along which cells are organized in the `response` array.
            Defaults to 0.

    Returns:
        A NumPy array containing the filtered `response` data, with only the
        cells that have activity greater than the `threshold`.
    """
    return apply(response, activity, threshold, cell_axis=cell_axis)


def by_vector_concentration(
    response: np.ndarray, angles: np.ndarray, threshold: float
) -> np.ndarray:
    """Filters cells based on their vector concentration.

    This is a convenience wrapper around the `apply` function, using the
    `vector_concentration` function as the filtering metric.

    Args:
        response: A NumPy array representing neural responses. Expected shape
            is (n_cells, n_angles).
        angles: A NumPy array of shape (n_angles,) containing the angles
            (in radians) corresponding to the responses.
        threshold: The minimum vector concentration a cell must have to be
            retained.

    Returns:
        A NumPy array containing the filtered `response` data, with only the
        cells that have vector concentration greater than the `threshold`.
    """
    return apply(response, vector_concentration, threshold, angles=angles)


def by_signal_fraction(
    response: np.ndarray, threshold: float, min_repeats: int = 2
) -> np.ndarray:
    """Filters cells based on their unbiased signal fraction.

    This is a convenience wrapper around the `apply` function, using the
    `signal_fraction` function as the filtering metric.

    Args:
        response: A NumPy array of neural responses. Expected shape
            is (n_repeats, n_cells, n_angles).
        threshold: The minimum unbiased signal fraction a cell must have
            to be retained.
        min_repeats: The minimum number of repeats required per angle for
            the signal fraction calculation.

    Returns:
        A NumPy array containing the filtered `response` data, with only the
        cells that have an unbiased signal fraction greater than the `threshold`.
    """
    return apply(
        response,
        signal_fraction,
        threshold,
        cell_axis=1,
        min_repeats=min_repeats,
    )