import asyncio
import logging
import os
import warnings
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from . import utils
from .engine.diagnostics import _align_eval_grid, _programs_df_to_programs_list
from .engine.generation_log import (
    _append_generation_record,
    _apply_removal_reasons_to_log,
    _drop_nonfinite_train_loss_from_islands,
    _drop_nonfinite_train_loss_rows,
    _update_generation_log_records,
)
from .engine.finalize import finalize_run
from .engine.evaluation import evaluate_candidate_batch
from .engine.paths import configure_file_logging, create_run_paths
from .engine.results import (
    CandidateGenerationResult,
    ModelGenerationResult,
    ParamEstimatorGenerationResult,
)
from .evolution import genetic_helpers
from .llm import prompts as prompt_tools
from .llm.candidates import (
    _run_translation_check_on_eval,
    generate_new_model,
    generate_new_parameter_estimator,
    translate_to_jax,
)
from .scoring.jax_objective import _call_objective

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(relativeCreated)dms | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)






























async def hypothesis_engine(
        n_iterations=9, time_limit=60, k_max=2, n_islands=8, batch_size=6, 
        critical_population_size=12, min_wise_population_size=0, n_migrants=2, 
        fit_params=True, use_param_estimator=True, 
        param_penalty_weight=0.01, FAILED_PROGRAM_COST=np.inf, exploit_point=0.5,
        param_estimator_timeout_s: float | None = 5.0,
        objective_timeout_s: float | None = None,
        param_estimator_refinement_rounds=0,
        exploration_topology = [1, 2, 3, 4, 5, 6, 7, 0], exploitation_topology = [1, 2, 3, 4, 5, 6, 7, 0],
        model_llm = None, param_est_llm = None, jax_translator_llm = None,
        max_iter = 1_000, learning_rate = 3e-3,
        penalty_denominator = 1,
        numpy_programs = None, param_estimators = None,
        X = None, X_eval = None,
        plot_model_fits = None, loss_fn = None,
        prompt_config = None, trial_batch_size = None, swear_words = None,
        open_family_tree = False,
        use_simple_objective: bool = False,
        log_prompts: bool = False,
        log_jax_translations: bool = False,
        random_seed = 42, # consider setting up a seed_manager to make behaviours more robustly reproducible.
        ):
    """
    Run the full island-based hypothesis search loop.

    This is the orchestration entrypoint that:
    1) translates seed programs to JAX,
    2) scores/initializes island populations,
    3) iteratively generates new model and parameter-estimator candidates,
    4) evaluates train/test losses,
    5) migrates/prunes island populations,
    6) saves databases and visual diagnostics.

    Args:
        n_iterations (int): Maximum outer-loop iterations.
        time_limit (int | float): Wall-clock budget in minutes.
        k_max (int): Parent count sampled for generation prompts.
        n_islands (int): Number of independent island populations.
        batch_size (int): Number of proposals generated per island per iteration.
        critical_population_size (int): Target max size before pruning.
        min_wise_population_size (int): Minimum retained "wise" programs.
        n_migrants (int): Programs migrated per iteration.
        fit_params (bool): Whether gradient-based parameter fitting is enabled.
        use_param_estimator (bool): Whether to initialize parameters using estimator.
        param_penalty_weight (float): Complexity penalty applied in objective.
        FAILED_PROGRAM_COST (float): Failure sentinel cost used in scoring.
        exploit_point (float): Explore/exploit phase boundary as fraction of run.
        param_estimator_timeout_s (float | None): Per-sample timeout for
            ``param_estimator`` calls inside objective initialization.
        objective_timeout_s (float | None): Hard timeout (seconds) for each full
            objective call (initialization + optimization + evaluation).
        param_estimator_refinement_rounds (int): Refinement rounds for new estimators.
        exploration_topology (list[int]): Migration destination map during explore.
        exploitation_topology (list[int]): Migration destination map during exploit.
        model_llm (str | list[str]): LLM(s) for model generation. Lists are traversed by iteration.
        param_est_llm (str | list[str]): LLM(s) for parameter estimator generation.
        jax_translator_llm (str | list[str]): LLM(s) for JAX translation.
        max_iter (int): Max optimization steps inside ``objective``.
        learning_rate (float): Optimizer learning rate inside ``objective``.
        numpy_programs (list[callable]): Seed NumPy model functions.
        param_estimators (list[callable]): Seed parameter estimators.
        X: Split data container with shape ``(2, 2)`` where each cell is a
            data dict. ``X[0,*]`` is train-sample split, ``X[1,*]`` is test-sample split.
        X_eval (np.ndarray): Evaluation grid used for diagnostics/comparison.
        plot_model_fits (callable | None): Optional plotting callback.
        loss_fn (callable | None): Objective loss function override.
        prompt_config (dict | None): Prompt config for all prompt construction.
        trial_batch_size (int | None): Trial batching size used by ``objective``.
        swear_words (list[str] | None): Blacklist for generated estimator code.
        open_family_tree (bool): Open the combined family tree HTML at the end of the run.
        use_simple_objective (bool): Use the minimal objective implementation everywhere.
        log_prompts (bool): If True, include prompt/response text in logs and generation records.
        log_jax_translations (bool): If True, include JAX translator prompt/response/code in logs.
        random_seed (int): Run seed for deterministic split-dependent operations.

    Returns:
        str: Path to the run output directory containing logs, plots, and program DBs.
    """
    has_spec_plotter = plot_model_fits is not None

    def _normalize_llm_sequence(value, label: str):
        if value is None:
            raise ValueError(f"{label} must be set (string or list).")
        if isinstance(value, (list, tuple)):
            seq = list(value)
        else:
            seq = [value]
        if not seq or any(v is None for v in seq):
            raise ValueError(f"{label} must contain at least one non-null model name.")
        return seq

    model_llm_seq = _normalize_llm_sequence(model_llm, "model_llm")
    param_est_llm_seq = _normalize_llm_sequence(param_est_llm, "param_est_llm")
    jax_llm_seq = _normalize_llm_sequence(jax_translator_llm, "jax_translator_llm")

    # PydanticAI reads provider API keys from the environment.
    load_dotenv()
    client = None

    logging.info("Using independent structured PydanticAI LLM queries")
    print("Using independent structured PydanticAI LLM queries")

    X = np.asarray(X, dtype=object)
    if X.shape != (2, 2):
        raise ValueError(f"X must have shape (2, 2), got {X.shape}.")

    n_training_samples = utils.data_n_samples(X[0, 0])
    n_training_trials = utils.data_n_trials(X[0, 0])
    n_test_samples = utils.data_n_samples(X[1, 1])
    n_test_trials = utils.data_n_trials(X[1, 1])
    X_eval_train = _align_eval_grid(X_eval, n_samples=n_training_samples)
    X_eval_test = _align_eval_grid(X_eval, n_samples=n_test_samples)

    print(f"Using {n_training_trials} training trials and {n_test_trials} test trials.")
    print(f"Using {n_training_samples} samples for training and {n_test_samples} samples for testing.")

    logging.info("Preparing seed models for JAX execution.")
    prompt_config = prompt_tools.with_default_prompts(prompt_config or {})
    model_name = prompt_tools.get_model_name(prompt_config)
    seed_code_strings = [
        utils.format_function_source(program, f'{model_name}_v{i+1}', 'import numpy as np')
        for i, program in enumerate(numpy_programs)
    ]
    seed_jax_llm = jax_llm_seq[0]
    jax_programs = []
    jax_code_strings = []
    for i, (program, code_string) in enumerate(zip(numpy_programs, seed_code_strings)):
        try:
            _run_translation_check_on_eval(
                np_func=program,
                jax_func=program,
                param_estimator=param_estimators[i],
                data_train_trials=X[0, 0],
                x_eval=X_eval_train,
            )
            logging.info("Seed model %d is already JAX-compatible; skipping LLM translation.", i + 1)
            jax_programs.append(program)
            jax_code_strings.append(code_string)
            continue
        except Exception as native_exc:
            logging.info(
                "Seed model %d failed native JAX check (%s); falling back to LLM translation.",
                i + 1,
                native_exc,
            )

        jax_code_string, jax_func, _jax_prompt, _jax_response = await translate_to_jax(
            code_string,
            client,
            prompt_config,
            seed_jax_llm,
        )
        if not callable(jax_func):
            raise RuntimeError(
                "Failed to translate seed model "
                f"{i + 1} to JAX using {seed_jax_llm}. "
                "This is commonly caused by LLM API rate limits (429 RESOURCE_EXHAUSTED). "
                "Please retry after cooldown or lower request pressure."
            )
        _run_translation_check_on_eval(
            np_func=program,
            jax_func=jax_func,
            param_estimator=param_estimators[i],
            data_train_trials=X[0, 0],
            x_eval=X_eval_train,
        )
        jax_programs.append(jax_func)
        jax_code_strings.append(jax_code_string)

    # create a dataframe to store the programs in each island
    islands = []
    for _ in range(n_islands):
        islands.append(pd.DataFrame(columns=['program_code_string', 'program', 'parameter_estimator_code_string', 'parameter_estimator',
                                             'iteration_number', 'birth_island', 'batch_index', 'train_loss', 'test_loss', 'params',
                                             'initial_loss', 'initial_params', 'llm_name', 'parent1_id', 'parent2_id', 'evaluation_matrix']))
    initial_programs = pd.DataFrame([])

    paths = create_run_paths()
    base_dir = paths.base_dir
    date_stamp = paths.date_stamp
    time_stamp = paths.time_stamp
    full_dir = paths.full_dir
    image_prompts_dir = paths.image_prompts_dir
    image_param_est_vs_gd_dir = paths.image_param_est_vs_gd_dir
    image_param_est_refine_dir = paths.image_param_est_refine_dir
    image_family_tree_fits_dir = paths.image_family_tree_fits_dir
    generation_log_path = paths.generation_log_path
    best_loss_log = []  # List of dicts: {iteration, timestamp, best_train_loss, best_island, ...}
    best_loss_path = paths.best_loss_path
    # store and compute loss of 2 initial programs
    t_start = time.time()
    seed_losses = np.zeros(2)
    seed_initial_losses = []
    seed_train_params = []
    seed_model_code_strings = []
    seed_param_est_code_strings = []
    model_name = prompt_tools.get_model_name(prompt_config)
    for i in range(2):
        # get the program, parameter estimator, and jax program
        program_num = numpy_programs[i]
        param_est = param_estimators[i]
        program_jax = jax_programs[i]
        # score the initial program
        seed_opt_start = time.time()
        loss_init, params_init, loss, params = _call_objective(
            use_simple_objective,
            model=program_jax,
            param_estimator=param_est,
            data=[X[0,0], X[0,1]],
            loss_fn=loss_fn,
            fit_params=fit_params,
            param_penalty_weight=param_penalty_weight,
            learning_rate=learning_rate,
            use_param_estimator=use_param_estimator,
            max_iter=max_iter,
            trial_batch_size=trial_batch_size,
            timeout_s=param_estimator_timeout_s,
            # Keep seed scoring unconstrained by objective timeout so good seeds
            # are not discarded due strict per-candidate runtime caps.
            objective_timeout_s=None,
        )
        seed_opt_time = time.time() - seed_opt_start
        print(f"Initial program {i + 1} loss before parameter fitting: {loss_init:.2f} and loss after fitting: {loss:.2f}")

        seed_losses[i] = loss
        seed_initial_losses.append(loss_init)
        seed_train_params.append(params)
        program_code_string = utils.format_function_source(
            program_num, f'{model_name}_v{i+1}', 'import numpy as np'
        )
        parameter_estimator_code_string = utils.format_function_source(
            param_est, f'parameter_estimator_v{i+1}', 'import numpy as np'
        )
        seed_model_code_strings.append(program_code_string)
        seed_param_est_code_strings.append(parameter_estimator_code_string)
        y_eval = utils.compute_evaluation_matrix(
            program_jax,
            params,
            eval_points=X_eval_train,
        )

        new_program_df = pd.DataFrame({'program_code_string': program_code_string,
                                    'program': program_jax,
                                    'parameter_estimator_code_string': parameter_estimator_code_string,
                                    'parameter_estimator': param_est,
                                    'iteration_number': -1,
                                    'birth_island': -1,  # Birth island is set to a special value for initial programs
                                    'batch_index': i,
                                    'train_loss': loss,
                                    'test_loss': None,  # all test losses will be computed at the end
                                    'optimization_time_s': seed_opt_time,
                                    'llm_name': None,
                                    'params': [params],
                                    'initial_loss': loss_init,
                                    'initial_params': [params_init],
                                    'parent1_id': None,
                                    'parent2_id': None,
                                    'evaluation_matrix': [y_eval]})
        initial_programs = pd.concat([initial_programs, new_program_df], ignore_index=True)
        print(f"Initial program {i + 1} loss: {loss:.2f}")

    # Drop invalid seed programs immediately (e.g., timed out/failed objective).
    initial_programs, _ = _drop_nonfinite_train_loss_rows(
        initial_programs,
        context="Seed initialization",
    )
    if len(initial_programs) == 0:
        raise ValueError(
            "All seed programs have non-finite train_loss and were removed. "
            "Cannot start evolutionary loop."
        )

    # Write seed programs to generation log
    n_samples=utils.data_n_samples(X[0, 0])
    for seed_idx, row in initial_programs.iterrows():
        seed_n_params = utils.params_numel_per_sample(row['params'], n_samples=n_samples)
        _append_generation_record(generation_log_path, {
            "iteration_number": -1,
            "birth_island": -1,
            "batch_index": int(seed_idx),
            "parent1_id": None,
            "parent2_id": None,
            "train_loss": float(row['train_loss']),
            "initial_loss": float(row['initial_loss']),
            "n_params": seed_n_params,
            "complexity_penalty": float(param_penalty_weight * seed_n_params),
            "model_code_numpy": row['program_code_string'],
            "param_est_code": row['parameter_estimator_code_string'],
            "model_prompt": None,
            "model_llm_response": None,
            "param_est_prompt": None,
            "param_est_llm_response": None,
            "llm_name": None,
            "temperature": None,
            "mode": "seed",
            "is_seed": True,
        })

    # seed each island with the initial programs
    for i in range(n_islands):
        islands[i] = pd.concat([islands[i], initial_programs], ignore_index=True)

    configure_file_logging(full_dir)
    
    seed_train_fit_losses = [None] * len(jax_programs)
    seed_test_fit_losses = [None] * len(jax_programs)
    seed_train_fit_paths = [None] * len(jax_programs)
    seed_test_fit_paths = [None] * len(jax_programs)
    if has_spec_plotter:
        seed_programs_list = _programs_df_to_programs_list(
            initial_programs,
            loss_func=loss_fn,
            data=X[0, 1],
            complexity_penalty=param_penalty_weight,
        )
        plot_model_fits(
            data=X[0, 0],
            programs_list=seed_programs_list,
            X_eval=X_eval_train,
            save_path=os.path.join(image_prompts_dir, 'initial_programs.png'),
            labels=['seed_1', 'seed_2'],
        )

        # Seed train/test fit plots for diagnostics.
        # Use the train-fitted params for test plots to avoid extra optimization
        # and keep cross-validated evaluation consistent.
        try:
            seed_test_params = list(seed_train_params)
            for idx, program_jax in enumerate(jax_programs):
                seed_label = f"seed_{idx+1}"
                seed_train_df = pd.DataFrame({
                    "program": [program_jax],
                    "params": [seed_train_params[idx]],
                })
                seed_train_programs_list = _programs_df_to_programs_list(
                    seed_train_df,
                    loss_func=loss_fn,
                    data=X[0, 0],
                    complexity_penalty=param_penalty_weight,
                )
                if seed_train_programs_list:
                    seed_train_fit_losses[idx] = float(
                        np.mean(np.asarray(seed_train_programs_list[0].get("losses")))
                    )
                seed_train_path = os.path.join(
                    image_family_tree_fits_dir, f'{seed_label}_train_fit.png'
                )
                seed_train_fit_paths[idx] = seed_train_path
                plot_model_fits(
                    data=X[0, 0],
                    programs_list=seed_train_programs_list,
                    X_eval=X_eval_train,
                    save_path=seed_train_path,
                    labels=[seed_label],
                    title_prefix=f"Train fits ({seed_label})",
                )

                seed_test_df = pd.DataFrame({
                    "program": [program_jax],
                    "params": [seed_test_params[idx]],
                })
                seed_test_programs_list = _programs_df_to_programs_list(
                    seed_test_df,
                    loss_func=loss_fn,
                    data=X[0, 1],
                    complexity_penalty=param_penalty_weight,
                )
                if seed_test_programs_list:
                    seed_test_fit_losses[idx] = float(
                        np.mean(np.asarray(seed_test_programs_list[0].get("losses")))
                    )
                seed_test_path = os.path.join(
                    image_family_tree_fits_dir, f'{seed_label}_test_fit.png'
                )
                seed_test_fit_paths[idx] = seed_test_path
                plot_model_fits(
                    data=X[0, 1],
                    programs_list=seed_test_programs_list,
                    X_eval=X_eval_test,
                    save_path=seed_test_path,
                    labels=[seed_label],
                    title_prefix=f"Test fits ({seed_label})",
                )
        except Exception as e:
            logging.info(f"Seed fit plotting failed: {e}")

    # Patch seed records with fit diagnostics instead of appending duplicates.
    seed_log_updates = {}
    for idx in range(len(jax_programs)):
        seed_n_params = None
        seed_complexity_penalty = None
        if idx < len(seed_train_params):
            seed_n_params = utils.params_numel_per_sample(
                seed_train_params[idx],
                n_samples=n_samples,
            )
            seed_complexity_penalty = float(param_penalty_weight * seed_n_params)
        seed_log_updates[(-1, -1, idx)] = {
            "iteration_number": -1,
            "birth_island": -1,
            "batch_index": idx,
            "train_loss": float(seed_losses[idx]),
            "initial_loss": float(seed_initial_losses[idx]) if idx < len(seed_initial_losses) else None,
            "n_params": seed_n_params,
            "complexity_penalty": seed_complexity_penalty,
            "optimization_time_s": float(initial_programs.iloc[idx].get("optimization_time_s"))
            if idx < len(initial_programs) else None,
            "train_fit_loss": seed_train_fit_losses[idx],
            "test_fit_loss": seed_test_fit_losses[idx],
            "model_prompt": None,
            "model_llm_response": None,
            "model_code_numpy": seed_model_code_strings[idx] if idx < len(seed_model_code_strings) else None,
            "model_code_jax": jax_code_strings[idx] if (log_jax_translations and idx < len(jax_code_strings)) else None,
            "param_est_prompt": None,
            "param_est_llm_response": None,
            "param_est_code": seed_param_est_code_strings[idx] if idx < len(seed_param_est_code_strings) else None,
            "param_est_refinement_prompts": [],
            "param_est_refinement_responses": [],
            "llm_name": None,
            "temperature": None,
            "mode": "seed",
            "image_prompt_path": None,
            "train_fit_image_path": seed_train_fit_paths[idx],
            "test_fit_image_path": seed_test_fit_paths[idx],
            "is_seed": True,
        }
    _update_generation_log_records(generation_log_path, seed_log_updates)

    # -----------------------------
    # HYPOTHESIS ENGINE
    # -----------------------------
    for i in tqdm(range(n_iterations), desc="Hypothesis Engine Iterations"):
        # check if time limit is reached
        if time.time() - t_start > time_limit * 60:
            logging.info(f"Time limit of {time_limit} minutes reached. Stopping iterations.")
            break
        
        logging.info(f"Iteration {i}")
        llm_name = model_llm_seq[i % len(model_llm_seq)]
        logging.info(f"Using model LLM: {llm_name}")
        mode = 'explore' if i < n_iterations * exploit_point else 'exploit'
        temperature = 1 + np.exp(-i / n_iterations)
        model_image_dirs = np.empty((n_islands, batch_size), dtype=object)
        # param_est_image_dirs = np.empty((n_islands, batch_size), dtype=object)
        for island_idx in range(n_islands):
            for j in range(batch_size):
                if has_spec_plotter:
                    model_image_dirs[island_idx, j] = os.path.join(image_prompts_dir, f'iter_{i}_island_{island_idx}_batch_{j}.png')
                    # param_est_image_dirs[island_idx, j] = os.path.join(image_feedback_dir, f'iter_{i}_island_{island_idx}_batch_{j}_param_est.png')
                else:
                    model_image_dirs[island_idx, j] = None
                    # param_est_image_dirs[island_idx, j] = None
        model_generation_tasks = [generate_new_model(islands[island_idx], 
                                                    llm_name=llm_name, 
                                                    client=client, 
                                                    mode=mode, 
                                                    k_max=k_max, 
                                                    temp=temperature,
                                                    data=X[0, 0],
                                                    x_eval=X_eval_train,
                                                    prompt_config=prompt_config,
                                                    img_dir=model_image_dirs[island_idx, j],
                                                    plot_model_fits=plot_model_fits,
                                                    batch_id=j,
                                                    loss_fn=loss_fn,
                                                    loss_data=X[0, 1],
                                                    complexity_penalty=param_penalty_weight) 
                                         for island_idx in range(n_islands) for j in range(batch_size)]
        logging.info(f"Generating {n_islands * batch_size} new programs... LLM Model: {llm_name}, mode: {mode}, temperature: {temperature:.2f}")
        print(f"Generating {n_islands * batch_size} new programs... LLM Model: {llm_name}, mode: {mode}, temperature: {temperature:.2f}")
        raw_model_results = await asyncio.gather(*model_generation_tasks)
        model_results = [
            ModelGenerationResult(
                numpy_code=model_code_string,
                prompt=prompt,
                llm_response=llm_output,
            )
            for model_code_string, prompt, llm_output, _parent_ids in raw_model_results
        ]
        parent_ids = [result[3] for result in raw_model_results]

        model_code_strings = [result.numpy_code for result in model_results]

        for candidate_idx in range(n_islands * batch_size):
            island_idx = candidate_idx // batch_size
            batch_idx = candidate_idx % batch_size
            model_result = model_results[candidate_idx]
            model_code_string = model_result.numpy_code
            model_prompt = model_result.prompt
            model_llm_response = model_result.llm_response
            parent1_id, parent2_id = parent_ids[candidate_idx]
            model_generated = model_code_string is not None
            _append_generation_record(generation_log_path, {
                "iteration_number": i,
                "birth_island": island_idx,
                "batch_index": batch_idx,
                "parent1_id": list(parent1_id) if parent1_id is not None else None,
                "parent2_id": list(parent2_id) if parent2_id is not None else None,
                "model_prompt": model_prompt,
                "model_llm_response": model_llm_response,
                "model_code_numpy": model_code_string,
                "llm_name": llm_name,
                "temperature": float(temperature),
                "mode": mode,
                "image_prompt_path": model_image_dirs[island_idx, batch_idx],
                "status": "numpy_generated" if model_generated else "model_generation_failed",
                "failure_stage": None if model_generated else "model_generation",
                "failure_message": None if model_generated else "No NumPy model code generated.",
                # these will be updated later after evaluation and parameter estimation steps
                "train_loss": None,
                "initial_loss": None,
                "n_params": None,
                "complexity_penalty": None,
                "model_code_jax": None,
                "param_est_prompt": None,
                "param_est_llm_response": None,
                "param_est_code": None,
                "param_est_refinement_prompts": [],
                "param_est_refinement_responses": [],
                "train_fit_image_path": None,
                "test_fit_image_path": None,
            })

        # convert to jax
        jax_llm_name = jax_llm_seq[i % len(jax_llm_seq)]
        model_function_translation_tasks = [translate_to_jax(code_string, client, prompt_config, jax_llm_name) for code_string in model_code_strings]
        jax_results = await asyncio.gather(*model_function_translation_tasks)
        translation_updates = {}
        for candidate_idx, (jax_code_string, jax_func, jax_prompt, jax_response) in enumerate(jax_results):
            model_result = model_results[candidate_idx]
            model_result.jax_code = jax_code_string
            model_result.jax_callable = jax_func
            model_result.jax_prompt = jax_prompt
            model_result.jax_raw_response = jax_response
            if model_code_strings[candidate_idx] is None:
                continue
            island_idx = candidate_idx // batch_size
            batch_idx = candidate_idx % batch_size
            key = (i, island_idx, batch_idx)
            update = {"model_code_jax": jax_code_string}
            if jax_code_string is None:
                update.update({
                    "status": "jax_translation_failed",
                    "failure_stage": "jax_translation",
                    "failure_message": "No JAX code block generated.",
                })
            elif jax_func is None:
                update.update({
                    "status": "jax_translation_failed",
                    "failure_stage": "jax_translation",
                    "failure_message": "Failed to parse translated JAX code into a callable.",
                })
            else:
                update.update({
                    "status": "jax_translated",
                    "failure_stage": None,
                    "failure_message": None,
                })
            translation_updates[key] = update
        _update_generation_log_records(generation_log_path, translation_updates)
        
        # build parameter‑estimator tasks
        if param_estimator_refinement_rounds > 0 and not has_spec_plotter:
            raise ValueError(
                "Parameter estimator refinement requires image diagnostics. "
                "Define plot_model_fits in the project spec or disable refinement."
            )

        # build parameter‑estimator tasks
        if param_estimator_refinement_rounds > 0 and not has_spec_plotter:
            raise ValueError(
                "Parameter estimator refinement requires image diagnostics. "
                "Define plot_model_fits in the project spec or disable refinement."
            )
        param_estimation_tasks = [
            generate_new_parameter_estimator(
                current_island=islands[island_idx],
                model_code_string=model_code_strings[island_idx * batch_size + j],
                model_fn=model_results[island_idx * batch_size + j].jax_callable,
                llm_name=param_est_llm_seq[i % len(param_est_llm_seq)],
                client=client,
                data=[X[0,0], X[0,1]],
                loss_fn=loss_fn,
                prompt_config=prompt_config,
                mode=mode,
                k_max=2,
                temp=temperature,
                param_estimator_max_lines=100,
                refine_rounds=param_estimator_refinement_rounds,
                param_penalty_weight=param_penalty_weight,
                random_seed=random_seed,
                swear_words=swear_words,
                island_id=island_idx,
                batch_id=j,
                iteration=i,
                use_simple_objective=use_simple_objective,
                plot_model_fits=plot_model_fits,
                x_eval=X_eval_train,
                image_refinement_dir=image_param_est_refine_dir,
                param_estimator_timeout_s=param_estimator_timeout_s,
                objective_timeout_s=objective_timeout_s,
            )
            for island_idx in range(n_islands)
            for j in range(batch_size)
        ]

        logging.info(
            f"Generating {n_islands * batch_size} parameter estimators "
            f"(LLM={param_est_llm_seq[i % len(param_est_llm_seq)]}, mode={mode}, T={temperature:.2f})"
        )
        logging.info(
            f"Generating {n_islands * batch_size} new parameter estimators... "
            f"Model: {param_est_llm_seq[i % len(param_est_llm_seq)]}, mode: {mode}, temperature: {temperature:.2f}"
        )
        raw_param_est_results = await asyncio.gather(*param_estimation_tasks)
        param_est_results = [
            ParamEstimatorGenerationResult(param_est_code_string, param_est_new, pe_metadata)
            for param_est_code_string, param_est_new, pe_metadata in raw_param_est_results
        ]

        param_est_llm_name = param_est_llm_seq[i % len(param_est_llm_seq)]
        param_est_translation_tasks = [
            translate_to_jax(
                result.code,
                client,
                prompt_config,
                jax_llm_name,
                entrypoint_name="parameter_estimator",
            )
            for result in param_est_results
        ]
        param_est_jax_results = await asyncio.gather(*param_est_translation_tasks)
        for candidate_idx, (jax_code_string, jax_func, jax_prompt, jax_response) in enumerate(param_est_jax_results):
            result = param_est_results[candidate_idx]
            result.metadata["numpy_code"] = result.code
            result.metadata["jax_prompt"] = jax_prompt
            result.metadata["jax_response"] = jax_response
            result.metadata["jax_code"] = jax_code_string
            result.metadata["llm_name"] = param_est_llm_name
            result.code = jax_code_string
            result.callable_obj = jax_func

        param_est_updates = {}
        for candidate_idx, param_est_result in enumerate(param_est_results):
            if model_code_strings[candidate_idx] is None:
                continue
            param_est_code_string = param_est_result.code
            param_est_new = param_est_result.callable_obj
            pe_metadata = param_est_result.metadata
            island_idx = candidate_idx // batch_size
            batch_idx = candidate_idx % batch_size
            key = (i, island_idx, batch_idx)
            update = {
                "param_est_prompt": pe_metadata.get("initial_prompt"),
                "param_est_llm_response": pe_metadata.get("initial_response"),
                "param_est_code": param_est_code_string,
                "param_est_refinement_prompts": pe_metadata.get("refinement_prompts", []),
                "param_est_refinement_responses": pe_metadata.get("refinement_responses", []),
            }
            if model_results[candidate_idx].jax_callable is not None:
                if param_est_new is None:
                    update.update({
                        "status": "param_estimator_failed",
                        "failure_stage": "param_estimator",
                        "failure_message": "Failed to generate executable parameter estimator.",
                    })
                else:
                    update.update({
                        "status": "ready_for_evaluation",
                        "failure_stage": None,
                        "failure_message": None,
                    })
            param_est_updates[key] = update
        _update_generation_log_records(generation_log_path, param_est_updates)
        # combine results
        island_results = [[
            CandidateGenerationResult(
                model=model_results[island_idx * batch_size + j],
                param_estimator=param_est_results[island_idx * batch_size + j],
            )
            for j in range(batch_size)
        ] for island_idx in range(n_islands)]

        success_rate, evaluation_log_updates = evaluate_candidate_batch(
            iteration=i,
            islands=islands,
            island_results=island_results,
            parent_ids=parent_ids,
            model_name=model_name,
            X=X,
            X_eval_train=X_eval_train,
            loss_fn=loss_fn,
            use_simple_objective=use_simple_objective,
            param_penalty_weight=param_penalty_weight,
            fit_params=fit_params,
            use_param_estimator=use_param_estimator,
            max_iter=max_iter,
            trial_batch_size=trial_batch_size,
            param_estimator_timeout_s=param_estimator_timeout_s,
            objective_timeout_s=objective_timeout_s,
            has_spec_plotter=has_spec_plotter,
            plot_model_fits=plot_model_fits,
            image_param_est_vs_gd_dir=image_param_est_vs_gd_dir,
            image_family_tree_fits_dir=image_family_tree_fits_dir,
            llm_name=llm_name,
            temperature=temperature,
            mode=mode,
            model_image_dirs=model_image_dirs,
            log_prompts=log_prompts,
            log_jax_translations=log_jax_translations,
            t_start=t_start,
        )
        print("Success rate:", success_rate, flush=True)
        _update_generation_log_records(generation_log_path, evaluation_log_updates)

        # Remove invalid-loss programs immediately so they never participate
        # in sorting, deduplication, pruning, or migration.
        islands = _drop_nonfinite_train_loss_from_islands(
            islands,
            context=f"Iteration {i} pre-migration cleanup",
        )

        # sort each island by loss
        for island_idx in range(n_islands):
            islands[island_idx] = islands[island_idx].sort_values(by='train_loss').reset_index(drop=True)
        logging.info(f"Iteration {i} complete. The proportion of programs that successfully ran and received a loss is {success_rate:.2f}.")
        logging.info('-' * 50)
        # migrate and prune programs (better here for temperature to be in [0, 1] range)
        try:
            islands, dedup_events = genetic_helpers.perform_island_deduplication(
                islands,
                overlap_threshold=int(0.75 * critical_population_size),
                iteration=i,
            )
            islands, prune_events = genetic_helpers.perform_population_pruning(
                islands,
                critical_population_size=critical_population_size - n_migrants,
                min_wise_population_size=min_wise_population_size,
                iteration=i,
            )
            _apply_removal_reasons_to_log(generation_log_path, dedup_events + prune_events)
            islands = genetic_helpers.perform_probabilistic_migration(
                islands,
                n_migrants=n_migrants,
                destination_islands=exploration_topology if mode == 'explore' else exploitation_topology,
                temperature=(temperature - 1.0) ** 4,
                iteration=i,
            )
        except Exception as migration_error:
            logging.exception("Migration/pruning failed at iteration %s: %s", i, migration_error)
            print(
                f"Warning: migration/pruning failed at iteration {i}; "
                "continuing without migration for this iteration.",
                flush=True,
            )
        islands = _drop_nonfinite_train_loss_from_islands(
            islands,
            context=f"Iteration {i} post-migration cleanup",
        )

                                                             
        # save diagnostics
        iteration_dir = os.path.join(full_dir, 'iteration_updates', f'iteration_{i}')
        os.makedirs(iteration_dir, exist_ok=True)
        for island_idx in range(n_islands):
            pg_info = islands[island_idx][['iteration_number', 'birth_island', 'batch_index', 'train_loss']].to_string(index=False, header=False)
            print(f"Iter {i}, Island {island_idx} programs:\n{pg_info}\n")
            logging.info(f"Iter {i}, Island {island_idx} programs:\n{pg_info}\n")
        
            # Save plots of top programs
            if has_spec_plotter:
                top_df = islands[island_idx].sort_values(by='train_loss').head(3).reset_index(drop=True)
                top_df = top_df.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
                top_programs_list = _programs_df_to_programs_list(
                    top_df,
                    loss_func=loss_fn,
                    data=X[0, 1],
                    complexity_penalty=param_penalty_weight,
                )
                plot_model_fits(
                    data=X[0, 0],
                    programs_list=top_programs_list,
                    X_eval=X_eval_train,
                    save_path=os.path.join(iteration_dir, f'island_{island_idx}_top_programs.png'),
                )
        
        if has_spec_plotter:
            all_programs = pd.concat([islands[idx] for idx in range(n_islands)], ignore_index=True)
            top_programs = all_programs.sort_values(by='train_loss').head(3).reset_index(drop=True)
            top_programs = top_programs.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
            top_programs_list = _programs_df_to_programs_list(
                top_programs,
                loss_func=loss_fn,
                data=X[0, 1],
                complexity_penalty=param_penalty_weight,
            )
            plot_model_fits(
                data=X[0, 0],
                programs_list=top_programs_list,
                X_eval=X_eval_train,
                save_path=os.path.join(iteration_dir, 'top_programs_overall.png'),
            )
        
        # Log best loss across all islands for live monitoring
        all_programs = pd.concat([islands[idx] for idx in range(n_islands)], ignore_index=True)
        best_program = all_programs.loc[all_programs['train_loss'].idxmin()]
        best_loss_log.append({
            'iteration': i,
            'timestamp': pd.Timestamp.now().isoformat(),
            'best_train_loss': best_program['train_loss'],
            'best_island': best_program['birth_island'],
            'n_programs_total': len(all_programs),
            'elapsed_time': time.time() - t_start
        })
        # Write to CSV after each iteration for live monitoring
        pd.DataFrame(best_loss_log).to_csv(best_loss_path, index=False)

    return finalize_run(
        islands=islands,
        X=X,
        X_eval_test=X_eval_test,
        paths=paths,
        n_islands=n_islands,
        fit_params=fit_params,
        max_iter=max_iter,
        param_penalty_weight=param_penalty_weight,
        use_param_estimator=use_param_estimator,
        trial_batch_size=trial_batch_size,
        param_estimator_timeout_s=param_estimator_timeout_s,
        objective_timeout_s=objective_timeout_s,
        use_simple_objective=use_simple_objective,
        loss_fn=loss_fn,
        has_spec_plotter=has_spec_plotter,
        plot_model_fits=plot_model_fits,
        open_family_tree=open_family_tree,
    )
