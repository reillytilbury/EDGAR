"""
Data structures for flexible multi-predictor support in the hypothesis engine.

This module provides a generic Predictors class that allows models to work with
arbitrary numbers of input predictors while maintaining backward compatibility
with single-predictor (2D array) inputs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union, List, Dict, Optional, Sequence
import numpy as np
import jax.numpy as jnp


ArrayLike = Union[np.ndarray, jnp.ndarray]


@dataclass
class Predictors:
    """
    A container for multiple predictor variables used in model fitting.
    
    Supports both index-based access (X[0], X[1]) and name-based access (X['theta']).
    Automatically handles conversion between 2D (n_cells, n_trials) and 
    3D (n_cells, n_features, n_trials) representations.
    
    Attributes:
        data: Internal storage as a 3D array of shape (n_cells, n_features, n_trials)
        names: List of predictor names in order (e.g., ['theta', 'speed'])
        
    Example:
        # Single predictor (backward compatible)
        >>> angles = np.random.rand(100, 50)  # (n_cells, n_trials)
        >>> predictors = Predictors.from_array(angles, names=['theta'])
        >>> predictors.shape
        (100, 1, 50)
        >>> predictors[0].shape  # Access by index
        (100, 50)
        >>> predictors['theta'].shape  # Access by name
        (100, 50)
        
        # Multiple predictors
        >>> angles = np.random.rand(100, 50)
        >>> speed = np.random.rand(100, 50)
        >>> predictors = Predictors.from_dict({'theta': angles, 'speed': speed})
        >>> predictors.shape
        (100, 2, 50)
        >>> predictors[1].shape  # speed
        (100, 50)
    """
    data: ArrayLike
    names: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate and normalize the data to 3D format."""
        # Convert to array if needed
        if not isinstance(self.data, (np.ndarray, jnp.ndarray)):
            self.data = np.asarray(self.data)
        
        # Auto-expand 2D to 3D: (n_cells, n_trials) -> (n_cells, 1, n_trials)
        if self.data.ndim == 2:
            self.data = self.data[:, np.newaxis, :]
        
        if self.data.ndim != 3:
            raise ValueError(
                f"Predictors data must be 2D (n_cells, n_trials) or "
                f"3D (n_cells, n_features, n_trials), got shape {self.data.shape}"
            )
        
        # Auto-generate names if not provided
        n_features = self.data.shape[1]
        if not self.names:
            self.names = [f"x{i}" for i in range(n_features)]
        elif len(self.names) != n_features:
            raise ValueError(
                f"Number of names ({len(self.names)}) must match "
                f"number of predictors ({n_features})"
            )
    
    @classmethod
    def from_array(
        cls, 
        data: ArrayLike, 
        names: Optional[List[str]] = None
    ) -> Predictors:
        """
        Create Predictors from a numpy/jax array.
        
        Args:
            data: Array of shape (n_cells, n_trials) or (n_cells, n_features, n_trials)
            names: Optional list of predictor names
            
        Returns:
            Predictors instance
        """
        return cls(data=data, names=names or [])
    
    @classmethod
    def from_dict(
        cls, 
        predictors_dict: Dict[str, ArrayLike],
        order: Optional[List[str]] = None
    ) -> Predictors:
        """
        Create Predictors from a dictionary of named arrays.
        
        Args:
            predictors_dict: Dict mapping predictor names to arrays of shape (n_cells, n_trials)
            order: Optional list specifying the order of predictors. 
                   If None, uses dict iteration order.
                   
        Returns:
            Predictors instance
            
        Example:
            >>> predictors = Predictors.from_dict({
            ...     'theta': angles,  # (n_cells, n_trials)
            ...     'speed': speeds,  # (n_cells, n_trials)
            ... })
        """
        if order is None:
            order = list(predictors_dict.keys())
        else:
            # Validate order contains all keys
            if set(order) != set(predictors_dict.keys()):
                raise ValueError(
                    f"Order {order} doesn't match dict keys {list(predictors_dict.keys())}"
                )
        
        # Stack arrays along new axis
        arrays = [predictors_dict[name] for name in order]
        
        # Validate shapes match
        shapes = [arr.shape for arr in arrays]
        if len(set(shapes)) > 1:
            raise ValueError(f"All predictor arrays must have the same shape, got {shapes}")
        
        # Stack: each array is (n_cells, n_trials) -> result is (n_cells, n_features, n_trials)
        stacked = np.stack(arrays, axis=1)
        
        return cls(data=stacked, names=order)
    
    @property
    def shape(self) -> tuple:
        """Return the shape (n_cells, n_features, n_trials)."""
        return self.data.shape
    
    @property
    def n_cells(self) -> int:
        """Number of cells/samples."""
        return self.data.shape[0]
    
    @property
    def n_features(self) -> int:
        """Number of predictor variables."""
        return self.data.shape[1]
    
    @property
    def n_trials(self) -> int:
        """Number of trials/observations per cell."""
        return self.data.shape[2]
    
    def __getitem__(self, key: Union[int, str, slice]) -> ArrayLike:
        """
        Access predictor(s) by index or name.
        
        Args:
            key: Integer index, string name, or slice
            
        Returns:
            For single predictor: array of shape (n_cells, n_trials)
            For slice: array of shape (n_cells, n_selected, n_trials)
        """
        if isinstance(key, str):
            if key not in self.names:
                raise KeyError(f"Predictor '{key}' not found. Available: {self.names}")
            idx = self.names.index(key)
            return self.data[:, idx, :]
        elif isinstance(key, int):
            if key < 0 or key >= self.n_features:
                raise IndexError(
                    f"Predictor index {key} out of range for {self.n_features} predictors"
                )
            return self.data[:, key, :]
        elif isinstance(key, slice):
            return self.data[:, key, :]
        else:
            raise TypeError(f"Invalid key type: {type(key)}. Use int, str, or slice.")
    
    def to_tensor(self) -> ArrayLike:
        """
        Return the full 3D tensor representation.
        
        Returns:
            Array of shape (n_cells, n_features, n_trials)
        """
        return self.data
    
    def to_2d(self, predictor: Union[int, str] = 0) -> ArrayLike:
        """
        Extract a single predictor as a 2D array (backward compatibility).
        
        Args:
            predictor: Index or name of the predictor to extract
            
        Returns:
            Array of shape (n_cells, n_trials)
        """
        return self[predictor]
    
    def to_dict(self) -> Dict[str, ArrayLike]:
        """
        Convert to a dictionary of named arrays.
        
        Returns:
            Dict mapping predictor names to arrays of shape (n_cells, n_trials)
        """
        return {name: self[name] for name in self.names}
    
    def get_cell(self, cell_idx: int) -> ArrayLike:
        """
        Get all predictors for a single cell.
        
        Args:
            cell_idx: Index of the cell
            
        Returns:
            Array of shape (n_features, n_trials)
        """
        return self.data[cell_idx, :, :]
    
    def slice_cells(self, indices: ArrayLike) -> Predictors:
        """
        Create a new Predictors with a subset of cells.
        
        Args:
            indices: Array of cell indices to select
            
        Returns:
            New Predictors instance with selected cells
        """
        return Predictors(data=self.data[indices], names=self.names.copy())
    
    def slice_trials(self, indices: ArrayLike) -> Predictors:
        """
        Create a new Predictors with a subset of trials.
        
        Args:
            indices: Array of trial indices to select
            
        Returns:
            New Predictors instance with selected trials
        """
        return Predictors(data=self.data[:, :, indices], names=self.names.copy())
    
    def as_jax(self) -> Predictors:
        """Convert internal data to JAX array."""
        if isinstance(self.data, jnp.ndarray):
            return self
        return Predictors(data=jnp.asarray(self.data), names=self.names.copy())
    
    def as_numpy(self) -> Predictors:
        """Convert internal data to NumPy array."""
        if isinstance(self.data, np.ndarray):
            return self
        return Predictors(data=np.asarray(self.data), names=self.names.copy())
    
    def __repr__(self) -> str:
        return (
            f"Predictors(shape={self.shape}, names={self.names}, "
            f"dtype={self.data.dtype})"
        )
    
    def __len__(self) -> int:
        """Return number of predictors."""
        return self.n_features


def ensure_predictors(
    x: Union[ArrayLike, Predictors, Dict[str, ArrayLike]],
    names: Optional[List[str]] = None
) -> Predictors:
    """
    Convert various input formats to a Predictors object.
    
    This is the main entry point for backward compatibility. It accepts:
    - Predictors: returned as-is
    - 2D array (n_cells, n_trials): wrapped as single predictor
    - 3D array (n_cells, n_features, n_trials): wrapped directly
    - Dict of arrays: converted via from_dict
    
    Args:
        x: Input data in any supported format
        names: Optional predictor names (used for array inputs)
        
    Returns:
        Predictors instance
        
    Example:
        # All of these work:
        >>> ensure_predictors(angles_2d)  # (n_cells, n_trials)
        >>> ensure_predictors(angles_3d)  # (n_cells, n_features, n_trials)  
        >>> ensure_predictors({'theta': angles, 'speed': speeds})
        >>> ensure_predictors(existing_predictors)
    """
    if isinstance(x, Predictors):
        return x
    elif isinstance(x, dict):
        return Predictors.from_dict(x, order=names)
    else:
        return Predictors.from_array(x, names=names)
