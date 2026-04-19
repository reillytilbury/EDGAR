import logging
import os
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

from ..evolution import genetic_helpers
from ..monitoring import create_dynamic_progress_update, create_family_tree
from ..monitoring.diagnostics import _programs_df_to_programs_list
from ..monitoring.log import _update_generation_log_test_losses_and_mark_winner
from .jax_objective import _call_objective, _clear_jax_runtime_cache


def finalize_run(
    *,
    islands,
    X,
    X_eval_test,
    paths,
    n_islands: int,
    fit_params: bool,
    max_iter: int,
    param_penalty_weight: float,
    use_param_estimator: bool,
    trial_batch_size,
    param_estimator_timeout_s,
    objective_timeout_s,
    use_simple_objective: bool,
    loss_fn,
    has_spec_plotter: bool,
    plot_model_fits,
    open_family_tree: bool,
) -> str:
    logging.info("Calculating loss on test set...")
    for island_idx in range(n_islands):
        logging.info("Island %s programs: %s programs to evaluate.", island_idx, len(islands[island_idx]))
        for j in range(len(islands[island_idx])):
            _clear_jax_runtime_cache()
            program = islands[island_idx].iloc[j]
            try:
                _, _, test_loss, optimized_params = _call_objective(
                    use_simple_objective,
                    model=program["program"],
                    param_estimator=program["parameter_estimator"],
                    data=[X[1, 0], X[1, 1]],
                    loss_fn=loss_fn,
                    fit_params=fit_params,
                    max_iter=max_iter,
                    param_penalty_weight=param_penalty_weight,
                    use_param_estimator=use_param_estimator,
                    trial_batch_size=trial_batch_size,
                    timeout_s=param_estimator_timeout_s,
                    objective_timeout_s=objective_timeout_s,
                )
                islands[island_idx].at[j, "test_loss"] = test_loss
                islands[island_idx].at[j, "params"] = optimized_params
                islands[island_idx].at[j, "mean_loss"] = np.mean(test_loss)
                print(f"Test loss: {test_loss:.2f}")
            except Exception as test_eval_error:
                logging.exception(
                    "Test evaluation failed (island=%s, idx=%s): %s",
                    island_idx,
                    j,
                    test_eval_error,
                )
                islands[island_idx].at[j, "test_loss"] = np.inf
                islands[island_idx].at[j, "mean_loss"] = np.inf
                print("Test loss: inf", flush=True)
            _clear_jax_runtime_cache()

    try:
        combined_dir = os.path.join(paths.base_dir, paths.date_stamp, paths.time_stamp, "combined")
        os.makedirs(combined_dir, exist_ok=True)
        combined_programs_dataframe = pd.concat(islands, ignore_index=True)
        combined_programs_dataframe, _ = genetic_helpers.remove_duplicates(
            combined_programs_dataframe,
            mode="complicated",
            loss_tol=0.025,
            cosine_tol=0.99,
            loss_type="test_loss",
            iteration=-1,
        )
        combined_programs_dataframe = combined_programs_dataframe.sort_values(by="mean_loss").reset_index(drop=True)

        _update_generation_log_test_losses_and_mark_winner(paths.generation_log_path, islands)
        combined_programs_dataframe = combined_programs_dataframe[
            [
                "iteration_number",
                "birth_island",
                "batch_index",
                "train_loss",
                "test_loss",
                "program_code_string",
                "parameter_estimator_code_string",
                "program",
                "parameter_estimator",
                "params",
                "parent1_id",
                "parent2_id",
                "llm_name",
            ]
        ]
        combined_programs_dataframe.to_csv(os.path.join(combined_dir, "programs_db.csv"), index=False)

        for island_id, island_df in enumerate(islands):
            island_dir = os.path.join(paths.base_dir, paths.date_stamp, paths.time_stamp, f"island_{island_id}")
            os.makedirs(island_dir, exist_ok=True)
            island_df.to_csv(os.path.join(island_dir, "programs_db.csv"), index=False)
    except Exception as postprocess_error:
        logging.exception("Post-processing failed: %s", postprocess_error)
        print("Warning: post-processing failed; returning partial run outputs.", flush=True)
        return paths.full_dir

    _plot_final_top_models(
        combined_programs_dataframe=combined_programs_dataframe,
        islands=islands,
        paths=paths,
        n_islands=n_islands,
        X=X,
        X_eval_test=X_eval_test,
        loss_fn=loss_fn,
        param_penalty_weight=param_penalty_weight,
        has_spec_plotter=has_spec_plotter,
        plot_model_fits=plot_model_fits,
    )

    create_dynamic_progress_update(paths.generation_log_path, paths.full_dir)
    create_family_tree(paths.generation_log_path, paths.full_dir, n_islands)
    if open_family_tree:
        family_tree_path = os.path.join(paths.full_dir, "genealogy.html")
        try:
            if os.path.isfile(family_tree_path):
                webbrowser.open(Path(family_tree_path).resolve().as_uri())
            else:
                logging.info("Family tree HTML not found at %s; skipping auto-open.", family_tree_path)
        except Exception as exc:
            logging.info("Failed to open family tree HTML: %s", exc)

    return paths.full_dir


def _plot_final_top_models(
    *,
    combined_programs_dataframe,
    islands,
    paths,
    n_islands: int,
    X,
    X_eval_test,
    loss_fn,
    param_penalty_weight: float,
    has_spec_plotter: bool,
    plot_model_fits,
) -> None:
    if not has_spec_plotter:
        return

    df_list = [combined_programs_dataframe] + islands
    df_dirs = [os.path.join(paths.base_dir, paths.date_stamp, paths.time_stamp, "combined")]
    df_dirs += [os.path.join(paths.base_dir, paths.date_stamp, paths.time_stamp, f"island_{i}") for i in range(n_islands)]

    try:
        for i, df in enumerate(df_list):
            df = df.head(3)
            df = df.sort_values(by="test_loss", ascending=False).reset_index(drop=True)
            programs_list = _programs_df_to_programs_list(
                df,
                loss_func=loss_fn,
                data=X[1, 1],
                complexity_penalty=param_penalty_weight,
            )
            plot_model_fits(
                data=X[1, 1],
                programs_list=programs_list,
                X_eval=X_eval_test,
                save_path=os.path.join(df_dirs[i], "top_model_fits.png"),
            )
            for j in range(min(3, len(df))):
                model_df = df.iloc[[j]].copy().reset_index(drop=True)
                plot_model_fits(
                    data=X[1, 1],
                    programs_list=_programs_df_to_programs_list(
                        model_df,
                        loss_func=loss_fn,
                        data=X[1, 1],
                        complexity_penalty=param_penalty_weight,
                    ),
                    X_eval=X_eval_test,
                    save_path=os.path.join(df_dirs[i], f"top_model_fit_{min(3, len(df)) - j}.png"),
                    labels=["model"],
                )
    except Exception as top_plot_error:
        logging.exception("Top-model plotting failed: %s", top_plot_error)
