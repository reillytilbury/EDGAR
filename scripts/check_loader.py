"""Smoke-test a project's data loader.

Loads the project's config, resolves its load_data() function, calls it on the
configured data path, and prints array shapes for each returned group. Useful
after changing the data file, the loader, or the filter thresholds — confirms
the loader runs and filters left enough data behind.

Usage:
    python scripts/check_loader.py projects/orientation_tuning/config.yaml
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.io.config import Config  # noqa: E402
from src.io.task_spec import TaskSpec  # noqa: E402


def _summarize(obj, prefix: str = "") -> None:
    """Recursively print shape (or len) of every leaf array in a nested dict/tuple."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _summarize(v, f"{prefix}{k}.")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _summarize(v, f"{prefix}[{i}].")
    else:
        shape = getattr(obj, "shape", None)
        if shape is None:
            shape = f"len={len(obj)}" if hasattr(obj, "__len__") else type(obj).__name__
        print(f"  {prefix.rstrip('.'):50s} {shape}")


def main(config_path: str) -> int:
    config = Config.from_yaml(Path(config_path))
    spec = TaskSpec.from_config(config)
    print(f"Loading {spec.io['data_path']} with project_params={spec.project_params}\n")
    result = spec.load_data_fn(spec.io["data_path"], **spec.project_params)
    print(f"load_data returned a {type(result).__name__} of length {len(result)}:")
    group_names = ["discover", "validate", "eval"]
    for i, group in enumerate(result):
        label = group_names[i] if i < len(group_names) else f"group[{i}]"
        print(f"\n[{label}]")
        _summarize(group, prefix="  ")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_loader.py <config.yaml>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
