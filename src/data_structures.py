"""
Data structures for flexible multi-input support in the hypothesis engine.

This module provides a generic Inputs class that allows models to work with
arbitrary numbers of input variables while maintaining backward compatibility
with single-input (2D array) inputs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union, List, Dict, Optional, Sequence
import numpy as np
import jax.numpy as jnp


ArrayLike = Union[np.ndarray, jnp.ndarray]


@dataclass
class Inputs:
    """
    A container for multiple input variables used in model fitting.
    
    Supports both index-based access (X[0], X[1]) and name-based access (X['theta']).
    Automatically handles conversion between 2D (n_samples, n_trials) and 
    3D (n_samples, n_features, n_trials) representations.
    
    Attributes:
        data: Internal storage as a 3D array of shape (n_samples, n_features, n_trials)
        names: List of input variable names in order (e.g., ['theta', 'speed'])
        
    Example:
        # Single input (backward compatible)
        >>> angles = np.random.rand(100, 50)  # (n_samples, n_trials)
        >>> inputs = Inputs.from_array(angles, names=['theta'])
        >>> inputs.shape
        (100, 1, 50)
        >>> inputs[0].shape  # Access by index
        (100, 50)
        >>> inputs['theta'].shape  # Access by name
        (100, 50)
        
        # Multiple inputs
        >>> angles = np.random.rand(100, 50)
        >>> speed = np.random.rand(100, 50)
        >>> inputs = Inputs.from_dict({'theta': angles, 'speed': speed})
        >>> inputs.shape
        (100, 2, 50)
        >>> inputs[1].shape  # speed
        (100, 50)
    """
    data: ArrayLike
    names: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate and normalize the data to 3D format."""
        # Convert to array if needed
        if not isinstance(self.data, (np.ndarray, jnp.ndarray)):
            self.data = np.asarray(self.data)
        
        # Auto-expand 2D to 3D: (n_samples, n_trials) -> (n_samples, 1, n_trials)
        if self.data.ndim == 2:
            self.data = self.data[:, np.newaxis, :]
        
        if self.data.ndim != 3:
            raise ValueError(
                f"Inputs data must be 2D (n_samples, n_trials) or "
                f"3D (n_samples, n_features, n_trials), got shape {self.data.shape}"
            )
        
        # Auto-generate names if not provided
        n_features = self.data.shape[1]
        if not self.names:
            self.names = [f"x{i}" for i in range(n_features)]
        elif len(self.names) != n_features:
            raise ValueError(
                f"Number of names ({len(self.names)}) must match "
                f"number of inputs ({n_features})"
            )
    
    @classmethod
    def from_array(
        cls, 
        data: ArrayLike, 
        names: Optional[List[str]] = None
    ) -> Inputs:
        """
        Create Inputs from a numpy/jax array.
        
        Args:
            data: Array of shape (n_samples, n_trials) or (n_samples, n_features, n_trials)
            names: Optional list of input variable names
            
        Returns:
            Inputs instance
        """
        return cls(data=data, names=names or [])
    
    @classmethod
    def from_dict(
        cls, 
        inputs_dict: Dict[str, ArrayLike],
        order: Optional[List[str]] = None
    ) -> Inputs:
        """
        Create Inputs from a dictionary of named arrays.
        
        Args:
            inputs_dict: Dict mapping input variable names to arrays of shape (n_samples, n_trials)
            order: Optional list specifying the order of inputs. 
                   If None, uses dict iteration order.
                   
        Returns:
            Inputs instance
            
        Example:
            >>> inputs = Inputs.from_dict({
            ...     'theta': angles,  # (n_samples, n_trials)
            ...     'speed': speeds,  # (n_samples, n_trials)
            ... })
        """
        if order is None:
            order = list(inputs_dict.keys())
        else:
            # Validate order contains all keys
            if set(order) != set(inputs_dict.keys()):
                raise ValueError(
                    f"Order {order} doesn't match dict keys {list(inputs_dict.keys())}"
                )
        
        # Stack arrays along new axis
        arrays = [inputs_dict[name] for name in order]
        
        # Validate shapes match
        shapes = [arr.shape for arr in arrays]
        if len(set(shapes)) > 1:
            raise ValueError(f"All input arrays must have the same shape, got {shapes}")
        
        # Stack: each array is (n_samples, n_trials) -> result is (n_samples, n_features, n_trials)
        stacked = np.stack(arrays, axis=1)
        
        return cls(data=stacked, names=order)
    
    @property
    def shape(self) -> tuple:
        """Return the shape (n_samples, n_features, n_trials)."""
        return self.data.shape
    
    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self.data.shape[0]
    
    @property
    def n_features(self) -> int:
        """Number of input variables."""
        return self.data.shape[1]
    
    @property
    def n_trials(self) -> int:
        """Number of trials/observations per cell."""
        return self.data.shape[2]
    
    def __getitem__(self, key: Union[int, str, slice]) -> ArrayLike:
        """
        Access input(s) by index or name.
        
        Args:
            key: Integer index, string name, or slice
            
        Returns:
            For single input: array of shape (n_samples, n_trials)
            For slice: array of shape (n_samples, n_selected, n_trials)
        """
        if isinstance(key, str):
            if key not in self.names:
                raise KeyError(f"Input '{key}' not found. Available: {self.names}")
            idx = self.names.index(key)
            return self.data[:, idx, :]
        elif isinstance(key, int):
            if key < 0 or key >= self.n_features:
                raise IndexError(
                    f"Input index {key} out of range for {self.n_features} inputs"
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
            Array of shape (n_samples, n_features, n_trials)
        """
        return self.data
    
    def to_2d(self, input_var: Union[int, str] = 0) -> ArrayLike:
        """
        Extract a single input as a 2D array (backward compatibility).
        
        Args:
            input_var: Index or name of the input to extract
            
        Returns:
            Array of shape (n_samples, n_trials)
        """
        return self[input_var]
    
    def to_dict(self) -> Dict[str, ArrayLike]:
        """
        Convert to a dictionary of named arrays.
        
        Returns:
            Dict mapping input names to arrays of shape (n_samples, n_trials)
        """
        return {name: self[name] for name in self.names}
    
    def get_sample(self, sample_idx: int) -> ArrayLike:
        """
        Get all inputs for a single sample.
        
        Args:
            sample_idx: Index of the sample
            
        Returns:
            Array of shape (n_features, n_trials)
        """
        return self.data[sample_idx, :, :]
    
    def slice_samples(self, indices: ArrayLike) -> Inputs:
        """
        Create a new Inputs with a subset of samples.
        
        Args:
            indices: Array of sample indices to select
            
        Returns:
            New Inputs instance with selected samples
        """
        return Inputs(data=self.data[indices], names=self.names.copy())
    
    def slice_trials(self, indices: ArrayLike) -> Inputs:
        """
        Create a new Inputs with a subset of trials.
        
        Args:
            indices: Array of trial indices to select
            
        Returns:
            New Inputs instance with selected trials
        """
        return Inputs(data=self.data[:, :, indices], names=self.names.copy())
    
    def as_jax(self) -> Inputs:
        """Convert internal data to JAX array."""
        if isinstance(self.data, jnp.ndarray):
            return self
        return Inputs(data=jnp.asarray(self.data), names=self.names.copy())
    
    def as_numpy(self) -> Inputs:
        """Convert internal data to NumPy array."""
        if isinstance(self.data, np.ndarray):
            return self
        return Inputs(data=np.asarray(self.data), names=self.names.copy())
    
    def __repr__(self) -> str:
        return (
            f"Inputs(shape={self.shape}, names={self.names}, "
            f"dtype={self.data.dtype})"
        )
    
    def __len__(self) -> int:
        """Return number of inputs."""
        return self.n_features


def ensure_inputs(
    x: Union[ArrayLike, Inputs, Dict[str, ArrayLike]],
    names: Optional[List[str]] = None
) -> Inputs:
    """
    Convert various input formats to a Inputs object.
    
    This is the main entry point for backward compatibility. It accepts:
    - Inputs: returned as-is
    - 2D array (n_samples, n_trials): wrapped as single input
    - 3D array (n_samples, n_features, n_trials): wrapped directly
    - Dict of arrays: converted via from_dict
    
    Args:
        x: Input data in any supported format
        names: Optional input names (used for array inputs)
        
    Returns:
        Inputs instance
        
    Example:
        # All of these work:
        >>> ensure_inputs(angles_2d)  # (n_samples, n_trials)
        >>> ensure_inputs(angles_3d)  # (n_samples, n_features, n_trials)  
        >>> ensure_inputs({'theta': angles, 'speed': speeds})
        >>> ensure_inputs(existing_inputs)
    """
    if isinstance(x, Inputs):
        return x
    elif isinstance(x, dict):
        return Inputs.from_dict(x, order=names)
    else:
        return Inputs.from_array(x, names=names)

@dataclass
class Outputs:
    """
    A container for multiple output/target variables used in model fitting.
    
    Supports both index-based access (Y[0], Y[1]) and name-based access (Y['cell_1']).
    Automatically handles conversion between 2D (n_samples, n_trials) and 
    3D (n_samples, n_targets, n_trials) representations.
    
    This enables models to predict multiple targets simultaneously (e.g., firing rates
    of multiple neurons) while maintaining backward compatibility with scalar outputs.
    
    Attributes:
        data: Internal storage as a 3D array of shape (n_samples, n_targets, n_trials)
        names: List of output/target names in order (e.g., ['cell_1', 'cell_2'])
        
    Example:
        # Single output (backward compatible with legacy scalar outputs)
        >>> firing_rates = np.random.rand(100, 50)  # (n_samples, n_trials)
        >>> outputs = Outputs.from_array(firing_rates, names=['firing_rate'])
        >>> outputs.shape
        (100, 1, 50)
        >>> outputs[0].shape  # Access by index
        (100, 50)
        >>> outputs['firing_rate'].shape  # Access by name
        (100, 50)
        
        # Multiple outputs (vectorized targets)
        >>> cell_1_rates = np.random.rand(100, 50)
        >>> cell_2_rates = np.random.rand(100, 50)
        >>> outputs = Outputs.from_dict({'cell_1': cell_1_rates, 'cell_2': cell_2_rates})
        >>> outputs.shape
        (100, 2, 50)
        >>> outputs[1].shape  # cell_2
        (100, 50)
    """
    data: ArrayLike
    names: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate and normalize the data to 3D format."""
        # Convert to array if needed
        if not isinstance(self.data, (np.ndarray, jnp.ndarray)):
            self.data = np.asarray(self.data)
        
        # Auto-expand 2D to 3D: (n_samples, n_trials) -> (n_samples, 1, n_trials)
        # This provides backward compatibility with scalar outputs
        if self.data.ndim == 2:
            self.data = self.data[:, np.newaxis, :]
        
        if self.data.ndim != 3:
            raise ValueError(
                f"Outputs data must be 2D (n_samples, n_trials) or "
                f"3D (n_samples, n_targets, n_trials), got shape {self.data.shape}"
            )
        
        # Auto-generate names if not provided
        n_targets = self.data.shape[1]
        if not self.names:
            self.names = [f"y{i}" for i in range(n_targets)]
        elif len(self.names) != n_targets:
            raise ValueError(
                f"Number of names ({len(self.names)}) must match "
                f"number of targets ({n_targets})"
            )
    
    @classmethod
    def from_array(
        cls, 
        data: ArrayLike, 
        names: Optional[List[str]] = None
    ) -> Outputs:
        """
        Create Outputs from a numpy/jax array.
        
        Args:
            data: Array of shape (n_samples, n_trials) or (n_samples, n_targets, n_trials)
            names: Optional list of output/target names
            
        Returns:
            Outputs instance
        """
        return cls(data=data, names=names or [])
    
    @classmethod
    def from_dict(
        cls, 
        outputs_dict: Dict[str, ArrayLike],
        order: Optional[List[str]] = None
    ) -> Outputs:
        """
        Create Outputs from a dictionary of named arrays.
        
        Args:
            outputs_dict: Dict mapping output names to arrays of shape (n_samples, n_trials)
            order: Optional list specifying the order of outputs. 
                   If None, uses dict iteration order.
                   
        Returns:
            Outputs instance
            
        Example:
            >>> outputs = Outputs.from_dict({
            ...     'cell_1': rates_1,  # (n_samples, n_trials)
            ...     'cell_2': rates_2,  # (n_samples, n_trials)
            ... })
        """
        if order is None:
            order = list(outputs_dict.keys())
        else:
            # Validate order contains all keys
            if set(order) != set(outputs_dict.keys()):
                raise ValueError(
                    f"Order {order} doesn't match dict keys {list(outputs_dict.keys())}"
                )
        
        # Stack arrays along new axis
        arrays = [outputs_dict[name] for name in order]
        
        # Validate shapes match
        shapes = [arr.shape for arr in arrays]
        if len(set(shapes)) > 1:
            raise ValueError(f"All output arrays must have the same shape, got {shapes}")
        
        # Stack: each array is (n_samples, n_trials) -> result is (n_samples, n_targets, n_trials)
        stacked = np.stack(arrays, axis=1)
        
        return cls(data=stacked, names=order)
    
    @property
    def shape(self) -> tuple:
        """Return the shape (n_samples, n_targets, n_trials)."""
        return self.data.shape
    
    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self.data.shape[0]
    
    @property
    def n_targets(self) -> int:
        """Number of output/target variables."""
        return self.data.shape[1]
    
    @property
    def n_trials(self) -> int:
        """Number of trials/observations per sample."""
        return self.data.shape[2]
    
    @property
    def is_scalar(self) -> bool:
        """Check if this represents scalar (single-target) outputs."""
        return self.n_targets == 1
    
    def __getitem__(self, key: Union[int, str, slice]) -> ArrayLike:
        """
        Access output(s) by index or name.
        
        Args:
            key: Integer index, string name, or slice
            
        Returns:
            For single output: array of shape (n_samples, n_trials)
            For slice: array of shape (n_samples, n_selected, n_trials)
        """
        if isinstance(key, str):
            if key not in self.names:
                raise KeyError(f"Output '{key}' not found. Available: {self.names}")
            idx = self.names.index(key)
            return self.data[:, idx, :]
        elif isinstance(key, int):
            if key < 0 or key >= self.n_targets:
                raise IndexError(
                    f"Output index {key} out of range for {self.n_targets} targets"
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
            Array of shape (n_samples, n_targets, n_trials)
        """
        return self.data
    
    def to_2d(self, target: Union[int, str] = 0) -> ArrayLike:
        """
        Extract a single target as a 2D array (backward compatibility).
        
        Args:
            target: Index or name of the target to extract
            
        Returns:
            Array of shape (n_samples, n_trials)
        """
        return self[target]
    
    def to_2d_legacy(self) -> ArrayLike:
        """
        Return data in legacy 2D format. Only valid for scalar outputs.
        
        Returns:
            Array of shape (n_samples, n_trials)
            
        Raises:
            ValueError: If n_targets > 1
        """
        if not self.is_scalar:
            raise ValueError(
                f"to_2d_legacy() only valid for scalar outputs, "
                f"but this has {self.n_targets} targets. Use to_2d(target_idx) instead."
            )
        return self.data[:, 0, :]
    
    def to_dict(self) -> Dict[str, ArrayLike]:
        """
        Convert to a dictionary of named arrays.
        
        Returns:
            Dict mapping output names to arrays of shape (n_samples, n_trials)
        """
        return {name: self[name] for name in self.names}
    
    def get_sample(self, sample_idx: int) -> ArrayLike:
        """
        Get all outputs for a single sample.
        
        Args:
            sample_idx: Index of the sample
            
        Returns:
            Array of shape (n_targets, n_trials)
        """
        return self.data[sample_idx, :, :]
    
    def slice_samples(self, indices: ArrayLike) -> Outputs:
        """
        Create a new Outputs with a subset of samples.
        
        Args:
            indices: Array of sample indices to select
            
        Returns:
            New Outputs instance with selected samples
        """
        return Outputs(data=self.data[indices], names=self.names.copy())
    
    def slice_trials(self, indices: ArrayLike) -> Outputs:
        """
        Create a new Outputs with a subset of trials.
        
        Args:
            indices: Array of trial indices to select
            
        Returns:
            New Outputs instance with selected trials
        """
        return Outputs(data=self.data[:, :, indices], names=self.names.copy())
    
    def slice_targets(self, indices: Union[ArrayLike, List[int], List[str]]) -> Outputs:
        """
        Create a new Outputs with a subset of targets.
        
        Args:
            indices: Array of target indices or list of target names to select
            
        Returns:
            New Outputs instance with selected targets
        """
        if isinstance(indices, list) and len(indices) > 0 and isinstance(indices[0], str):
            # Convert names to indices
            int_indices = [self.names.index(name) for name in indices]
            new_names = indices
        else:
            int_indices = list(indices)
            new_names = [self.names[i] for i in int_indices]
        
        return Outputs(data=self.data[:, int_indices, :], names=new_names)
    
    def as_jax(self) -> Outputs:
        """Convert internal data to JAX array."""
        if isinstance(self.data, jnp.ndarray):
            return self
        return Outputs(data=jnp.asarray(self.data), names=self.names.copy())
    
    def as_numpy(self) -> Outputs:
        """Convert internal data to NumPy array."""
        if isinstance(self.data, np.ndarray):
            return self
        return Outputs(data=np.asarray(self.data), names=self.names.copy())
    
    def __repr__(self) -> str:
        return (
            f"Outputs(shape={self.shape}, names={self.names}, "
            f"dtype={self.data.dtype}, is_scalar={self.is_scalar})"
        )
    
    def __len__(self) -> int:
        """Return number of targets."""
        return self.n_targets


def ensure_outputs(
    y: Union[ArrayLike, Outputs, Dict[str, ArrayLike]],
    names: Optional[List[str]] = None
) -> Outputs:
    """
    Convert various output formats to an Outputs object.
    
    This is the main entry point for backward compatibility. It accepts:
    - Outputs: returned as-is
    - 2D array (n_samples, n_trials): wrapped as single target (scalar output)
    - 3D array (n_samples, n_targets, n_trials): wrapped directly (vectorized output)
    - Dict of arrays: converted via from_dict
    
    Args:
        y: Output data in any supported format
        names: Optional target names (used for array inputs)
        
    Returns:
        Outputs instance
        
    Example:
        # All of these work:
        >>> ensure_outputs(firing_rates_2d)  # (n_samples, n_trials) -> scalar
        >>> ensure_outputs(firing_rates_3d)  # (n_samples, n_targets, n_trials) -> vectorized
        >>> ensure_outputs({'cell_1': rates_1, 'cell_2': rates_2})
        >>> ensure_outputs(existing_outputs)
    """
    if isinstance(y, Outputs):
        return y
    elif isinstance(y, dict):
        return Outputs.from_dict(y, order=names)
    else:
        return Outputs.from_array(y, names=names)