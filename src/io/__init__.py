"""
Configuration and persistence for EDGAR experiments.

Modules:
  - task_spec: TaskSpec dataclass with persistence (frozen experiment configuration)
  - config: Load and merge configs, load project code
  - project_output_path: Create timestamped output directories and artifacts
"""

# Re-export data/param helpers so project code can import from src.io
# rather than from src.utils (which is slated for removal).
from ..utils import (  # noqa: F401
    broadcast_params,
    call_model,
    data_as_jax,
    data_as_numpy,
    data_n_samples,
    data_n_trials,
    get_data_sample,
    slice_data_samples,
    slice_data_trials,
    slice_params,
    stack_params,
    tree_to_jax,
)
