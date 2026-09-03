"""Utilities for handling neural data trials.

This module provides functions for manipulating experimental trial data,
specifically for shuffling multiple arrays consistently along a specified axis.
"""
import numpy as np


def shuffle(*arrays: np.ndarray, axis: int = -1) -> list[np.ndarray]:
    """Shuffles multiple arrays along the specified axis.

    This function shuffles multiple input arrays consistently along a specified
    axis, meaning that the same permutation is applied to all arrays. This is
    useful for maintaining the correspondence between different data arrays
    that represent related aspects of experimental trials.

    Args:
        *arrays: The arrays to be shuffled. All arrays must have the same size
            along the specified `axis`.
        axis: The axis along which to shuffle the arrays. Defaults to -1 (the
            last axis).

    Returns:
        A list of the shuffled arrays. The order of arrays in the output list
        corresponds to the order of arrays in the input `arrays`.
    """
    if not arrays:
        return []

    n_trials = arrays[0].shape[axis]
    shuff_idx = np.random.permutation(n_trials)

    # Use take to shuffle along the specified axis
    return [
        np.take(arr, shuff_idx, axis=axis) if arr.ndim > 0 else arr for arr in arrays
    ]