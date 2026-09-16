#!/usr/bin/env python
"""Interactive triage: run one state-space program's scan step-by-step.

Runs the LLM's ``model(state, y_prev, params)`` in a plain Python for-loop
(not ``jax.lax.scan``) for a fixed number of steps, printing the carried state
and the predicted observation at every step. Useful for figuring out WHY a
program produces NaNs or inf losses during regular scoring.

Not integrated into the scoring pipeline — the pipeline is jit-wrapped, and
Python ``print`` inside a traced function fires only once at trace time.

Project-agnostic across state-space projects. The one project-specific piece —
how a data trajectory maps to the per-step ``y_prev`` the model consumes — is
supplied by an optional hook in the project's ``data_loader/load_data.py``:

    def debug_trajectory(data) -> list[y_prev]:
        '''Return one trajectory's per-step y_prev pytrees (what apply_model scans).'''

Everything else is generic: the carry is seeded from ``model.INITIAL_STATE``
(empty dict for stateless models), and the model output is flattened as a pytree
for printing and the non-finite check, so scalar / tuple / dict predictions all work.

Usage:
    EDGAR_SCAN_DEBUG=1 python scripts/debug_program.py \\
        projects/wilson_cowan/config.yaml \\
        projects/wilson_cowan/seed_programs/model1.py
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import jax
import jax.numpy as jnp

from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from edgar.llm.code_loading import load_function_from_source


def _numpy_to_jax_source(src: str) -> str:
    """Trivial numpy → jax rewrite (mirrors score_seeds.py)."""
    out = src.replace("import numpy as np", "import jax.numpy as jnp")
    out = out.replace("np.", "jnp.")
    if "import jax.numpy as jnp" not in out:
        out = "import jax.numpy as jnp\n" + out
    return out


def _extract_default_params(source: str) -> dict:
    """Read ``model.DEFAULT_PARAMS`` out of a program source string."""
    ns = {}
    exec(source, ns)
    model = ns.get("model")
    if model is None or not hasattr(model, "DEFAULT_PARAMS"):
        raise ValueError("program must define model.DEFAULT_PARAMS")
    return dict(model.DEFAULT_PARAMS)


def _floats(tree):
    """Same pytree structure, every leaf rounded to a plain float (for printing)."""
    return jax.tree_util.tree_map(lambda v: round(float(v), 4), tree)


def _all_finite(tree) -> bool:
    return all(bool(jnp.isfinite(v)) for v in jax.tree_util.tree_leaves(tree))


def debug_run(config_path: Path, program_path: Path, max_steps: int = 50) -> int:
    if not os.environ.get("EDGAR_SCAN_DEBUG"):
        print("[debug_program] set EDGAR_SCAN_DEBUG=1 to enable step-by-step prints",
              file=sys.stderr)
        return 2

    config = Config.from_yaml(config_path)
    spec = TaskSpec.from_config(config)
    (X_train, _), _, _ = spec.load_data_fn(spec.io["data_path"], **spec.project_params)

    # Project hook: turn the loaded data into one trajectory's per-step y_prev sequence.
    loader_src = (spec.project_dir / "data_loader" / "load_data.py").read_text()
    debug_trajectory = load_function_from_source(loader_src, "debug_trajectory")
    if debug_trajectory is None:
        print(
            "[debug_program] this project's data_loader/load_data.py does not define "
            "debug_trajectory(data); add it to enable step-by-step debugging "
            "(see projects/wilson_cowan for the canonical example).",
            file=sys.stderr,
        )
        return 2
    y_prev_seq = list(debug_trajectory(X_train))
    if not y_prev_seq:
        print("[debug_program] debug_trajectory returned an empty sequence", file=sys.stderr)
        return 1

    src = program_path.read_text()
    model_fn = load_function_from_source(_numpy_to_jax_source(src), "model")
    if model_fn is None:
        print(f"[debug_program] couldn't load model() from {program_path}", file=sys.stderr)
        return 1

    default_params = _extract_default_params(src)
    params = jax.tree_util.tree_map(jnp.asarray, default_params)

    # Carry seeded from the model's INITIAL_STATE (empty dict for stateless models).
    init_state = getattr(model_fn, "INITIAL_STATE", {})
    state = jax.tree_util.tree_map(jnp.asarray, dict(init_state))

    n_steps = min(max_steps, len(y_prev_seq))
    print(f"=== {program_path.name} on {config_path.parent.name} ===")
    print(f"init state:  {_floats(state)}")
    print(f"params:      {list(default_params)}")
    print(f"---\nstepping {n_steps} iterations (Python for-loop, outside jit):")

    for i in range(n_steps):
        y_prev = jax.tree_util.tree_map(jnp.asarray, y_prev_seq[i])
        new_state, mean = model_fn(state, y_prev, params)
        print(f"  t={i:3d}  y_prev={_floats(y_prev)}  pred={_floats(mean)}  state={_floats(new_state)}")
        if not _all_finite(mean):
            print(f"[debug_program] non-finite prediction at step {i}, stopping.")
            return 1
        state = new_state
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("program", type=Path)
    ap.add_argument("--max-steps", type=int, default=50)
    args = ap.parse_args()
    return debug_run(args.config, args.program, max_steps=args.max_steps)


if __name__ == "__main__":
    sys.exit(main())
