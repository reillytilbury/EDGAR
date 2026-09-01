import argparse
import asyncio
import numpy as np
from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from edgar.scoring.scoring import _get_params, _eval_loss, _optimize
from pathlib import Path
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_command_buffer=" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_enable_command_buffer=").strip()


async def main(project_name: str):
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / "projects" / project_name / "config.yaml"

    print(f"Loading config from: {path}")
    config = Config.from_yaml(path)
    spec = TaskSpec.from_config(config)

    # Load data
    print("Loading data...")
    X_discover, X_validate, X_eval = spec.load_data_fn(
        data_path=spec.io["data_path"], **spec.project_params
    )

    # We will test both Seed Program 1 and Seed Program 2
    for idx, program in enumerate(spec.seed_programs):
        print("\n======================================")
        print(f"Testing Program {idx + 1}: {program.name}")
        print("======================================")

        # Load Python model code and translate manually for quick JAX compilation testing
        model_code = program.code.model

        # Simple JAX translation rules
        model_jax_code = model_code.replace(
            "import numpy as np", "import jax.numpy as jnp"
        )
        model_jax_code = model_jax_code.replace("np.", "jnp.")
        program.code.model_jax = model_jax_code

        try:
            model_fn = program.compile_model()
            print("✓ Compiled JAX model successfully")
        except Exception as e:
            print(f"✗ Failed to compile JAX model: {e}")
            continue

        try:
            param_est_fn = program.compile_param_est()
            print("✓ Compiled param estimator successfully")
        except Exception as e:
            print(f"✗ Failed to compile param estimator: {e}")
            continue

        # Get initial parameters
        try:
            params_init = _get_params(
                param_est_fn, program.default_params, X_discover[0]
            )
            print("✓ Estimated initial parameters successfully")
            for k, v in params_init.items():
                print(f"  - {k}: shape {v.shape}, mean {np.nanmean(v):.4f}")
        except Exception as e:
            print(f"✗ Failed to estimate initial parameters: {e}")
            import traceback

            traceback.print_exc()
            continue

        # Evaluate initial loss
        try:
            initial_loss = _eval_loss(
                model_fn, spec.loss_fn, params_init, X_discover[1]
            )
            print(f"Initial loss on discovery test split: {initial_loss:.4f}")
        except Exception as e:
            print(f"✗ Failed to evaluate initial loss: {e}")
            import traceback

            traceback.print_exc()
            continue

        # Optimize parameters using gradient descent (Adam)
        try:
            print("Optimizing parameters...")
            gd_config = spec.scoring["gradient_descent"].copy()
            # gd_config["max_iter"] = 100
            params_opt, _ = _optimize(
                model_fn, spec.loss_fn, params_init, X_discover[0], gd_config
            )
            params_opt = params_opt[0]
            print("✓ Optimization complete")
        except Exception as e:
            print(f"✗ Failed during optimization: {e}")
            import traceback

            traceback.print_exc()
            continue

        # Evaluate final loss
        try:
            final_loss = _eval_loss(model_fn, spec.loss_fn, params_opt, X_discover[1])
            print(f"Final loss on discovery test split: {final_loss:.4f}")
            if final_loss < initial_loss:
                print("✓ Success: Loss decreased during optimization!")
            else:
                print("✗ Warning: Loss did not decrease!")
        except Exception as e:
            print(f"✗ Failed to evaluate final loss: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify and score seed programs for a given project."
    )
    parser.add_argument(
        "project_name",
        type=str,
        help="Name of the project directory under the projects/ directory.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.project_name))
