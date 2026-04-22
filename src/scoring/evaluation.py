"""
evaluation.py

Batch evaluation of candidate models during evolutionary search. Validates,
translates, and scores evolved candidates from LLM generation.

Result containers track outputs at each generation stage:
- ModelGenerationResult: NumPy and JAX code, prompts, and callables for model.
- ParamEstimatorGenerationResult: Code, callable, and metadata for parameter estimator.
- CandidateGenerationResult: Combined model + param estimator results for one candidate.

Main function:
- evaluate_candidate_batch: Evaluate a batch of candidates per iteration. Handles:
  1) Model code parsing (NumPy -> JAX translation)
  2) Parameter estimator parsing and optional refinement
  3) Validation: translation checks, JAX tracing, finite output
  4) Optimization: fit parameters on training data
  5) Evaluation: compute loss on test data
  6) Visualization: plot model fits if enabled
  Returns success_rate and evaluation_log_updates dict.

Example usage:
--------------
    success_rate, log_updates = evaluate_candidate_batch(
        iteration=0,
        islands=[island_0_df, island_1_df],
        island_results=[[cand_0, cand_1], [cand_2, cand_3]],
        parent_ids=[(None, None), (0, 1), (1, 2), (2, 3)],
        X=[train_data, test_data],
        X_eval_train=X_eval,
        loss_fn=loss_fn,
        ...
    )
    print(f"Success rate: {success_rate:.1%}")
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from jax.flatten_util import ravel_pytree

from .. import utils
from ..llm.candidates import _run_translation_check_on_eval
from ..llm.code_loading import load_function_from_source
from ..monitoring.diagnostics import _programs_df_to_programs_list
from .objective import _call_objective, _clear_jax_runtime_cache


# ---------------------------------------------------------------------------
# Result containers for a single generation step
# ---------------------------------------------------------------------------

@dataclass
class ModelGenerationResult:
    numpy_code: str | None
    prompt: str | None
    llm_response: str | None
    jax_code: str | None = None
    jax_callable: Callable | None = None
    jax_prompt: str | None = None
    jax_raw_response: str | None = None


@dataclass
class ParamEstimatorGenerationResult:
    code: str | None
    callable_obj: Callable | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateGenerationResult:
    model: ModelGenerationResult
    param_estimator: ParamEstimatorGenerationResult


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_candidate_batch(
    *,
    iteration: int,
    islands,
    island_results,
    parent_ids,
    model_name: str,
    X,
    X_eval_train,
    loss_fn,
    use_simple_objective: bool,
    param_penalty_weight: float,
    fit_params: bool,
    use_param_estimator: bool,
    max_iter: int,
    trial_batch_size,
    param_estimator_timeout_s,
    objective_timeout_s,
    has_spec_plotter: bool,
    plot_model_fits,
    image_param_est_vs_gd_dir: str,
    image_family_tree_fits_dir: str,
    llm_name: str,
    temperature: float,
    mode: str,
    model_image_dirs,
    log_prompts: bool,
    log_jax_translations: bool,
    t_start: float,
) -> tuple[float, dict]:
    i = iteration
    n_islands = len(islands)
    batch_size = len(island_results[0]) if n_islands else 0
    success_rate = 0.0
    evaluation_log_updates = {}
    for island_idx, j in np.ndindex(n_islands, batch_size):
        _clear_jax_runtime_cache()
        candidate_result = island_results[island_idx][j]
        model_result = candidate_result.model
        param_est_result = candidate_result.param_estimator
        model_code_string = model_result.numpy_code
        prompt = model_result.prompt
        model_llm_response = model_result.llm_response
        model_code_string_jax = model_result.jax_code
        model_new = model_result.jax_callable
        jax_prompt = model_result.jax_prompt
        jax_raw_response = model_result.jax_raw_response
        param_est_code_string = param_est_result.code
        param_est_new = param_est_result.callable_obj
        pe_metadata = param_est_result.metadata
        parent1_id, parent2_id = parent_ids[island_idx * batch_size + j]
        candidate_key = (i, island_idx, j)

        print(
            f"=== iter={i} island={island_idx} batch={j} === "
            f"(mode={mode})",
            flush=True,
        )
        log_lines = [
            f"=== iter={i} island={island_idx} batch={j} ===",
            f"mode={mode}",
            f"parent_ids={parent1_id},{parent2_id}",
        ]
        score_line_idx = len(log_lines)
        log_lines.append("score=<pending>")
        if log_prompts:
            log_lines.append("[Model prompt]")
            log_lines.append(prompt or "<none>")
            log_lines.append("[Model response]")
            log_lines.append(model_llm_response or "<none>")
            if isinstance(pe_metadata, dict):
                log_lines.append("[Param estimator prompt]")
                log_lines.append(pe_metadata.get("initial_prompt") or "<none>")
                log_lines.append("[Param estimator response]")
                log_lines.append(pe_metadata.get("initial_response") or "<none>")
                if pe_metadata.get("refinement_prompts"):
                    for ridx, ref_prompt in enumerate(pe_metadata.get("refinement_prompts", []), start=1):
                        ref_resp = pe_metadata.get("refinement_responses", [None] * ridx)
                        ref_code = pe_metadata.get("refinement_codes", [None] * ridx)
                        log_lines.append(f"[Refinement {ridx} prompt]")
                        log_lines.append(ref_prompt or "<none>")
                        log_lines.append(f"[Refinement {ridx} response]")
                        log_lines.append(ref_resp[ridx - 1] if ridx - 1 < len(ref_resp) else "<none>")
                        log_lines.append(f"[Refinement {ridx} code]")
                        log_lines.append(ref_code[ridx - 1] if ridx - 1 < len(ref_code) else "<none>")

        if log_jax_translations:
            log_lines.append("[JAX translator prompt]")
            log_lines.append(jax_prompt or "<none>")
            log_lines.append("[JAX translator response]")
            log_lines.append(jax_raw_response or "<none>")
            log_lines.append("[Parsed JAX code]")
            log_lines.append(model_code_string_jax or "<none>")

        log_lines.append("[Parsed model code]")
        log_lines.append(model_code_string or "<none>")
        log_lines.append("[Parsed parameter estimator code]")
        log_lines.append(param_est_code_string or "<none>")
        if isinstance(pe_metadata, dict) and pe_metadata.get("status"):
            log_lines.append(f"[Param estimator status] {pe_metadata['status']}")
        status_notes = []
        if model_code_string is None:
            status_notes.append("model code missing")
        if param_est_code_string is None:
            status_notes.append("parameter estimator code missing")
        if model_new is None:
            status_notes.append("model parse failed")
        if param_est_new is None:
            status_notes.append("parameter estimator parse failed")
        if model_code_string_jax is None:
            status_notes.append("JAX translation missing")

        def _flush_log(extra_note: str | None = None):
            if extra_note:
                status_notes.append(extra_note)
            if status_notes:
                log_lines.append("[Status] " + " | ".join(status_notes))
                print("Status:", " | ".join(status_notes), flush=True)
            logging.info("\n".join(log_lines))

        def _set_score(value):
            try:
                scalar = float(value)
                if np.isfinite(scalar):
                    log_lines[score_line_idx] = f"score={scalar:.6f}"
                else:
                    log_lines[score_line_idx] = "score=inf"
            except Exception:
                log_lines[score_line_idx] = "score=<unknown>"

        if model_new is None or param_est_new is None:
            if model_code_string is None or param_est_code_string is None:
                debug_blocks = utils.extract_code_blocks(model_llm_response)
                debug_text = "\n\n".join(debug_blocks).strip() if debug_blocks else (model_llm_response or "").strip()
                debug_imports = []
                for line in debug_text.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("import ") or stripped.startswith("from "):
                        debug_imports.append(stripped)
                def _debug_extract(code: str, name_prefixes: list[str]) -> str | None:
                    for prefix in name_prefixes:
                        safe_prefix = re.escape(prefix)
                        pattern = rf"^\s*def\s+{safe_prefix}\d*\s*\(.*?(?=^\s*def\s+|\Z)"
                        match = re.search(pattern, code, flags=re.MULTILINE | re.DOTALL)
                        if match:
                            return match.group(0).strip()
                    return None

                debug_model = _debug_extract(
                    debug_text,
                    [f"{model_name}_v", model_name, "model_v", "model"],
                )
                debug_param = _debug_extract(debug_text, ["parameter_estimator_v", "parameter_estimator"])
                if log_prompts:
                    log_lines.append("[Extracted code text]")
                    log_lines.append(debug_text or "<none>")
                log_lines.append("[Extracted imports]")
                log_lines.append("\n".join(dict.fromkeys(debug_imports)) or "<none>")
                log_lines.append("[Extracted model block]")
                log_lines.append((debug_model or "<none>") if log_prompts else ("present" if debug_model else "missing"))
                log_lines.append("[Extracted parameter_estimator block]")
                log_lines.append((debug_param or "<none>") if log_prompts else ("present" if debug_param else "missing"))

            if model_code_string is None:
                evaluation_log_updates[candidate_key] = {
                    "status": "model_generation_failed",
                    "failure_stage": "model_generation",
                    "failure_message": "No NumPy model code generated.",
                }
            elif model_new is None:
                evaluation_log_updates[candidate_key] = {
                    "status": "jax_translation_failed",
                    "failure_stage": "jax_translation",
                    "failure_message": "Failed to translate NumPy model to executable JAX code.",
                }
            else:
                evaluation_log_updates[candidate_key] = {
                    "status": "param_estimator_failed",
                    "failure_stage": "param_estimator",
                    "failure_message": "Failed to generate executable parameter estimator.",
                }

            _set_score(np.inf)
            _flush_log()
            continue

        model_np = load_function_from_source(model_code_string, "model")
        if model_np is None:
            evaluation_log_updates[candidate_key] = {
                "status": "numpy_parse_failed",
                "failure_stage": "numpy_parse",
                "failure_message": "Failed to parse generated NumPy model into a callable.",
            }
            _set_score(np.inf)
            _flush_log("failed to parse NumPy model")
            continue
        try:
            _run_translation_check_on_eval(
                np_func=model_np,
                jax_func=model_new,
                param_estimator=param_est_new,
                data_train_trials=X[0, 0],
                x_eval=X_eval_train,
            )
        except Exception as e:
            evaluation_log_updates[candidate_key] = {
                "status": "translation_check_failed",
                "failure_stage": "translation_check",
                "failure_message": str(e),
            }
            _set_score(np.inf)
            _flush_log(f"JAX translation check failed: {e}")
            continue

        opt_start = time.time()
        initial_loss, initial_params, loss, optimized_params = _call_objective(
            use_simple_objective,
            model=model_new,
            param_estimator=param_est_new,
            data=[X[0,0], X[0,1]],
            loss_fn=loss_fn,
            param_penalty_weight=param_penalty_weight,
            fit_params=fit_params,
            use_param_estimator=use_param_estimator,
            max_iter=max_iter,
            trial_batch_size=trial_batch_size,
            timeout_s=param_estimator_timeout_s,
            objective_timeout_s=objective_timeout_s,
        )
        optimization_time_s = time.time() - opt_start
        if not np.isfinite(float(loss)):
            evaluation_log_updates[candidate_key] = {
                "status": "objective_failed",
                "failure_stage": "objective",
                "failure_message": "objective returned FAILED_PROGRAM_COST.",
            }

            print("Status: objective failed (non-finite loss).", flush=True)
            _set_score(np.inf)
            _flush_log("objective failed (non-finite loss)")
            logging.info('-' * 50)
            continue

        y_eval = utils.compute_evaluation_matrix(
            model_new,
            optimized_params,
            eval_points=X_eval_train,
        )
        _set_score(loss)
        _flush_log()
        logging.info(f"Loss: {loss:.2f}\n")

        train_fit_path = None
        test_fit_path = None
        train_fit_losses = []
        test_fit_losses = []
        if has_spec_plotter:
            initial_params_plot = initial_params
            optimized_params_plot = optimized_params
            flat_init, _ = ravel_pytree(initial_params_plot)
            flat_opt, _ = ravel_pytree(optimized_params_plot)
            param_delta = np.asarray(flat_opt) - np.asarray(flat_init)
            mean_abs_delta = float(np.mean(np.abs(param_delta)))
            max_abs_delta = float(np.max(np.abs(param_delta)))
            if np.allclose(np.asarray(flat_init), np.asarray(flat_opt), equal_nan=True):
                logging.info(
                    f"param_est_vs_gd: initial and optimized params are numerically identical "
                    f"(iter={i}, island={island_idx}, batch={j})."
                )
            else:
                logging.info(
                    f"param_est_vs_gd: param deltas (iter={i}, island={island_idx}, batch={j}) "
                    f"mean_abs={mean_abs_delta:.6g}, max_abs={max_abs_delta:.6g}"
                )
            plot_model_fits(
                data=X[0, 0],
                programs_list=[
                    {
                        "model": model_new,
                        "params": initial_params_plot,
                        "losses": np.full(utils.data_n_samples(X[0, 0]), float(initial_loss)),
                    },
                    {
                        "model": model_new,
                        "params": optimized_params_plot,
                        "losses": np.full(utils.data_n_samples(X[0, 0]), float(loss)),
                    },
                ],
                X_eval=X_eval_train,
                save_path=os.path.join(image_param_est_vs_gd_dir, f'iter_{i}_island_{island_idx}_batch_{j}_param_est_vs_gd.png'),
                labels=['PE', 'GD'],
            )
            train_fit_path = os.path.join(image_family_tree_fits_dir, f'iter_{i}_island_{island_idx}_batch_{j}_train_fit.png')
            train_programs_df = pd.DataFrame({
                "program": [model_new, model_new],
                "params": [initial_params_plot, optimized_params_plot],
            })
            train_programs_list = _programs_df_to_programs_list(
                train_programs_df,
                loss_func=loss_fn,
                data=X[0, 0],
                complexity_penalty=param_penalty_weight,
            )
            train_fit_losses = []
            for entry in train_programs_list:
                if "losses" in entry:
                    train_fit_losses.append(float(np.mean(np.asarray(entry["losses"]))))
                else:
                    train_fit_losses.append(None)
            plot_model_fits(
                data=X[0, 0],
                programs_list=train_programs_list,
                X_eval=X_eval_train,
                save_path=train_fit_path,
                labels=['PE', 'GD'],
                title_prefix="Train fits",
            )
            test_fit_path = os.path.join(image_family_tree_fits_dir, f'iter_{i}_island_{island_idx}_batch_{j}_test_fit.png')
            test_programs_df = pd.DataFrame({
                "program": [model_new, model_new],
                "params": [initial_params_plot, optimized_params_plot],
            })
            test_programs_list = _programs_df_to_programs_list(
                test_programs_df,
                loss_func=loss_fn,
                data=X[0, 1],
                complexity_penalty=param_penalty_weight,
            )
            test_fit_losses = []
            for entry in test_programs_list:
                if "losses" in entry:
                    test_fit_losses.append(float(np.mean(np.asarray(entry["losses"]))))
                else:
                    test_fit_losses.append(None)
            plot_model_fits(
                data=X[0, 1],
                programs_list=test_programs_list,
                X_eval=X_eval_train,
                save_path=test_fit_path,
                labels=['PE', 'GD'],
                title_prefix="Test fits",
            )
        if not has_spec_plotter:
            train_fit_losses = []
            test_fit_losses = []

        train_fit_loss_pe = train_fit_losses[0] if len(train_fit_losses) > 0 else None
        train_fit_loss_gd = train_fit_losses[1] if len(train_fit_losses) > 1 else train_fit_loss_pe
        train_fit_loss = train_fit_loss_gd if train_fit_loss_gd is not None else train_fit_loss_pe
        test_fit_loss_pe = test_fit_losses[0] if len(test_fit_losses) > 0 else None
        test_fit_loss_gd = test_fit_losses[1] if len(test_fit_losses) > 1 else test_fit_loss_pe
        test_fit_loss = test_fit_loss_gd if test_fit_loss_gd is not None else test_fit_loss_pe

        param_summary = utils.params_tree_summary(
            optimized_params,
            n_samples=utils.data_n_samples(X[0, 0]),
            max_lines=16,
        )
        if param_summary:
            logging.info(f"Optimized parameter structure (sample view):\n{param_summary}\n")
        t_added = time.time() - t_start
        new_program_df = pd.DataFrame({'program_code_string': model_code_string,
                                    'program': model_new,
                                    'parameter_estimator_code_string': param_est_code_string,
                                    'parameter_estimator': param_est_new,
                                    'iteration_number': i,
                                    'birth_island': island_idx,
                                    'batch_index': j,
                                    'train_loss': loss,
                                    'test_loss': None,
                                    'optimization_time_s': optimization_time_s,
                                    'llm_name': llm_name,
                                    'params': [optimized_params],
                                    'initial_loss': initial_loss,
                                    'initial_params': [initial_params],
                                    'parent1_id': [parent1_id],
                                    'parent2_id': [parent2_id],
                                    'evaluation_matrix': [y_eval]
                                    })

        islands[island_idx] = pd.concat([islands[island_idx], new_program_df], ignore_index=True)

        n_params = utils.params_numel_per_sample(
            optimized_params,
            n_samples=utils.data_n_samples(X[0, 0]),
        )
        complexity_penalty = float(param_penalty_weight * n_params)
        evaluation_log_updates[candidate_key] = {
            "train_loss": float(loss),
            "initial_loss": float(initial_loss),
            "optimization_time_s": float(optimization_time_s),
            "model_prompt": prompt if log_prompts else None,
            "model_llm_response": model_llm_response if log_prompts else None,
            "model_code_numpy": model_code_string,
            "model_code_jax": model_code_string_jax if log_jax_translations else None,
            "param_est_prompt": pe_metadata.get("initial_prompt") if log_prompts else None,
            "param_est_llm_response": pe_metadata.get("initial_response") if log_prompts else None,
            "param_est_code": param_est_code_string,
            "param_est_refinement_prompts": pe_metadata.get("refinement_prompts", []) if log_prompts else [],
            "param_est_refinement_responses": pe_metadata.get("refinement_responses", []) if log_prompts else [],
            "llm_name": llm_name,
            "temperature": float(temperature),
            "mode": mode,
            "n_params": n_params,
            "complexity_penalty": complexity_penalty,
            "image_prompt_path": model_image_dirs[island_idx, j],
            "train_fit_image_path": train_fit_path,
            "test_fit_image_path": test_fit_path,
            "train_fit_loss": train_fit_loss,
            "test_fit_loss": test_fit_loss,
            "train_fit_loss_pe": train_fit_loss_pe,
            "train_fit_loss_gd": train_fit_loss_gd,
            "test_fit_loss_pe": test_fit_loss_pe,
            "test_fit_loss_gd": test_fit_loss_gd,
            "status": "accepted",
            "failure_stage": None,
            "failure_message": None,
        }

        success_rate += 1 / (n_islands * batch_size)
        print(f"iteration {i}, island {island_idx}, batch {j}, loss: {loss:.2f}", flush=True)
        print('-' * 50, flush=True)
        logging.info("-" * 50)

    return success_rate, evaluation_log_updates
