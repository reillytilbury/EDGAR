"""
Configuration and record management for EDGAR experiments.

Modules:
  - task_spec: TaskSpec dataclass (frozen experiment configuration)
  - config: Load and merge configs, import project code
  - record: Save/load TaskSpec for reproducibility
  - run_dir: Create timestamped run directories
"""
