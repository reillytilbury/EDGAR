"""
Record saving and loading: serialize TaskSpec to disk and reconstruct from disk.

save_record writes a yaml file capturing merged config, seed source code, git state, and timestamp.
load_record reads that file back into a TaskSpec (requires project code to still exist on disk).

File location: <run_dir>/task_spec_record.yaml

Contains:
  - task_name, git_sha, git_dirty, created_at
  - merged io, evolution, llms, scoring, project_params
  - seed program source code
  - prompt schemas (model, param_est, jax)

Example usage:
--------------
    # Save after a run
    spec = TaskSpec.from_config("projects/my_task/config.yaml")
    spec.save_record(run_dir)

    # Load for reproduction
    spec = TaskSpec.from_record("runs/2024-03-15/10-30-45/task_spec_record.yaml")
    # TaskSpec is now identical to original (prompts, seed code, config, etc.)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from ..evolution.program import Program, BirthCertificate, Code
from ..llm.prompt_schema import PromptSchema
from .task_spec import TaskSpec


RECORD_FILENAME = "task_spec_record.yaml"


def save_record(spec: TaskSpec, run_dir: Path) -> Path:
    """
    Save TaskSpec to a yaml record file for reproducibility.

    Args:
        spec: TaskSpec instance to save
        run_dir: Directory to save record into (will create task_spec_record.yaml)

    Returns:
        Path to the saved record file
    """
    record = {
        "task_name": spec.task_name,
        "git_sha": spec.git_sha,
        "git_dirty": spec.git_dirty,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "io": spec.io,
        "evolution": spec.evolution,
        "llms": spec.llms,
        "scoring": spec.scoring,
        "project_params": spec.project_params,
        "seed_programs": [
            {
                "batch_index": p.birth.batch_index,
                "model_code": p.code.model,
                "param_est_code": p.code.param_est,
            }
            for p in spec.seed_programs
        ],
        "prompt_schemas": {
            "model": spec.model_prompt_schema.model_dump(),
            "param_est": spec.param_est_prompt_schema.model_dump(),
            "jax": spec.jax_prompt_schema.model_dump(),
        },
    }

    path = Path(run_dir) / RECORD_FILENAME
    with open(path, "w") as f:
        yaml.dump(record, f, default_flow_style=False, sort_keys=False)
    return path


def load_record(record_path: Path) -> TaskSpec:
    """
    Reconstruct TaskSpec from a saved record file.

    Args:
        record_path: Path to task_spec_record.yaml from a previous run

    Returns:
        TaskSpec with identical settings as the original run

    Raises:
        FileNotFoundError: If record file doesn't exist
        ImportError: If project code (load_data_fn, loss_fn, etc.) can't be imported
        ValueError: If record is malformed
    """
    with open(record_path) as f:
        record = yaml.safe_load(f)

    task_name = record["task_name"]

    seed_programs = [
        Program(
            birth=BirthCertificate(
                generation=-1,
                island=-1,
                batch_index=s["batch_index"],
                mode="seed",
            ),
            code=Code(model=s["model_code"], param_est=s["param_est_code"]),
        )
        for s in record["seed_programs"]
    ]

    schemas = record["prompt_schemas"]

    # Re-import project callables from the task folder.
    # This requires the project code to still be on disk.
    from .config import _import_fn, _import_fn_optional

    return TaskSpec(
        task_name=task_name,
        git_sha=record["git_sha"],
        git_dirty=record["git_dirty"],
        io=record["io"],
        evolution=record["evolution"],
        llms=record["llms"],
        scoring=record["scoring"],
        project_params=record["project_params"],
        model_prompt_schema=PromptSchema(**schemas["model"]),
        param_est_prompt_schema=PromptSchema(**schemas["param_est"]),
        jax_prompt_schema=PromptSchema(**schemas["jax"]),
        load_data_fn=_import_fn(f"projects.{task_name}.data_loader.load_data", "load_and_process_data"),
        loss_fn=_import_fn(f"projects.{task_name}.data_loader.load_data", "loss_fn"),
        plot_fn=_import_fn_optional(f"projects.{task_name}.image_feedback.plot", "plot_model_fits"),
        seed_programs=seed_programs,
    )
