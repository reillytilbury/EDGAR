import asyncio
import logging
import os
import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from .. import utils
from ..engine.diagnostics import _programs_df_to_programs_list
from ..llm.code_loading import load_function_from_source
from ..scoring.objective import _call_objective
from ..scoring.param_init import compute_initial_params

# TODO: All this code is outdated. New prompt building functions + prompt schema mean that a single function can build all types of prompt.
# TODO: Discover what each of these funcs is doing apart from building a prompt and requesting code from the LLM, and move that logic to more appropriate places. 
# For example, the logic for selecting parent programs and building the model prompt should be separate from the logic for requesting code from the LLM and normalizing it.

async def generate_new_model(current_island, llm_name, client,
                            data, x_eval, prompt_config,
                            mode='explore', k_max=2, temp=1,
                            thinking_budget=1, img_dir=None,
                            plot_model_fits=None,
                            batch_id: int = 0,
                            loss_fn=None,
                            loss_data=None,
                            complexity_penalty: float = 0.0):
    """
    Propose a new model program by querying the LLM from island context.

    Args:
        current_island (pd.DataFrame): Program population for one island.
        llm_name (str): Model name for structured LLM calls.
        client: Reserved LLM client slot; structured calls read provider keys from the environment.
        data (dict[str, np.ndarray]): Data dict forwarded to plotting.
        x_eval: Evaluation grid used for consistent plotting across projects.
        prompt_config: Prompt config used to build prompts.
        mode (str): Search mode (typically ``"explore"`` or ``"exploit"``).
        k_max (int): Number of parent programs to include in prompt context.
        temp (float): Sampling temperature for LLM decoding.
        thinking_budget (float): Relative budget forwarded to LLM helper.
        img_dir (str | None): Base output path used for refinement-round
            feedback images.
        plot_model_fits (callable | None): Optional plotting callback.
        batch_id (int): Candidate batch id for logging context.
        loss_fn (callable | None): Loss function used for per-sample diagnostics
            when building `programs_list` for plot feedback.
        loss_data (dict | None): Optional data dict to use specifically for
            diagnostics loss computation. Defaults to ``data``.
        complexity_penalty (float): Complexity-penalty multiplier used when
            computing diagnostics losses.

    Returns:
        tuple[str | None, str | None, str | None, tuple]:
            ``(code_string, prompt, llm_output, parent_ids)``.
            ``code_string`` is ``None`` when no valid code block is produced.
    """
    k = min(k_max, len(current_island))
    random_programs = current_island.sample(k, replace=False).reset_index(drop=True)
    random_programs = random_programs.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
    # save parent1_id and parent2_id. These are strings of the form "(iteration_number)_(birth_island)_(batch_index)"
    parent1_id = (random_programs['iteration_number'][0], 
                  random_programs['birth_island'][0], 
                  random_programs['batch_index'][0])
    parent2_id = (random_programs['iteration_number'][1],
                  random_programs['birth_island'][1], 
                  random_programs['batch_index'][1])
    use_image = (
        img_dir is not None
        and plot_model_fits is not None
    )
    model_name = prompt_tools.get_model_name(prompt_config)
    model_prompt = prompt_tools.build_model_prompt(
        prompt_config,
        random_programs,
        mode=mode,
        use_image=use_image,
    )
    if use_image:
        try:
            data_for_loss = data if loss_data is None else loss_data
            programs_list = _programs_df_to_programs_list(
                random_programs,
                loss_func=loss_fn,
                data=data_for_loss,
                complexity_penalty=complexity_penalty,
            )
            plot_model_fits(
                data=data,
                programs_list=programs_list,
                X_eval=x_eval,
                save_path=img_dir,
                labels=[f"v_{i+1}" for i in range(len(random_programs))],
            )
            
            img_path = Path(img_dir)
            with img_path.open("rb") as f:
                img_bytes = f.read()
        except Exception as e:
            logging.info(f"Error generating image for neuron model prompt: {e}")
            img_bytes = None
            # if we can't generate an image, we will just use the text prompt without image
            use_image = False
    else:
        img_bytes = None
    
    try:
        code_string, llm_output = await request_model_module(
            prompt=model_prompt,
            model_name=llm_name,
            image_bytes=img_bytes,
            temperature=temp,
            thinking=True if thinking_budget else None,
        )
    except Exception as exc:
        logging.exception("Structured model generation failed: %s", exc)
        return None, model_prompt, repr(exc), (parent1_id, parent2_id)
    if code_string is None:
        return None, model_prompt, llm_output, (parent1_id, parent2_id)
    code_string = _normalize_generated_model_code(
        code_string,
        model_name=model_name,
        expected_version=k + 1,
    )
    code_string = re.sub(
        rf"^\s*def\s+{re.escape(model_name)}\s*\(",
        "def model(",
        code_string,
        count=1,
        flags=re.MULTILINE,
    )
    code_string = _normalize_generated_model_code(
        code_string,
        model_name="model",
        expected_version=k + 1,
    )

    return code_string, model_prompt, llm_output, (parent1_id, parent2_id)


async def generate_new_parameter_estimator(current_island,
                                           model_code_string: str,
                                           model_fn,
                                           llm_name, client,
                                           data,
                                           prompt_config,
                                           mode='explore', k_max=1, temp=1,
                                           param_estimator_max_lines=100,
                                           swear_words=None,
                                           refine_rounds: int = 0,
                                           param_penalty_weight: float = 0.1,
                                           random_seed: int | None = None,
                                           island_id: int = None,
                                           batch_id: int = 0,
                                           iteration: int | None = None,
                                           use_simple_objective: bool = False,
                                           loss_fn=None,
                                           plot_model_fits=None,
                                           x_eval=None,
                                           image_refinement_dir=None,
                                           param_estimator_timeout_s: float | None = 5.0,
                                           objective_timeout_s: float | None = None,):
    """
    Generate and optionally refine a parameter-estimator function via LLM.

    This function prompts the LLM for a ``parameter_estimator`` implementation,
    validates/parses returned code, and can run iterative refinement rounds where
    each round is scored with ``objective(..., fit_params=False)``.

    Args:
        current_island (pd.DataFrame): Program population for one island.
        model_code_string (str): NumPy model source used in estimator prompt context.
        model_fn (callable): Executable model function used for scoring refinements.
        llm_name (str): Model name for structured LLM calls.
        client: Reserved LLM client slot; structured calls read provider keys from the environment.
        data: Length-2 container ``[data_train_trials, data_test_trials]`` of
            data dicts passed to ``objective`` during scoring.
        prompt_config: Prompt config used to build prompts.
        mode (str): Search mode (typically ``"explore"`` or ``"exploit"``).
        k_max (int): Number of parent programs to include in prompt context.
        temp (float): Sampling temperature for LLM decoding.
        param_estimator_max_lines (int): Soft budget for generated estimator length.
        swear_words (list[str] | None): Token blacklist for generated code.
        refine_rounds (int): Number of iterative refinement rounds.
        param_penalty_weight (float): Parameter-count penalty used during scoring.
        random_seed (int | None): Base RNG seed for parent-program sampling.
        island_id (int | None): Island id for logging/refinement diagnostics.
        batch_id (int): Batch id for logging/refinement diagnostics.
        loss_fn (callable | None): Loss function forwarded to ``objective``.
        use_simple_objective (bool): Use the minimal objective implementation for scoring.
        param_estimator_timeout_s (float | None): Per-sample timeout (seconds)
            for estimator evaluation during refinement scoring.
        objective_timeout_s (float | None): Hard timeout (seconds) for each
            refinement objective call.

    Returns:
        tuple[str | None, callable | None, dict]: Best estimator code string, parsed
            callable, and metadata dict with prompt/response info.
            Returns ``(None, None, pe_metadata)`` when generation/validation fails.
    """
    pe_metadata = {
        "initial_prompt": None,
        "initial_response": None,
        "refinement_prompts": [],
        "refinement_responses": [],
        "refinement_codes": [],
        "status": None,
    }
    if model_code_string is None:
        pe_metadata["status"] = "missing_model_code"
        return None, None, pe_metadata
    if not (isinstance(data, (list, tuple, np.ndarray)) and len(data) == 2):
        logging.info("Parameter estimator generation expects data split as [train_trials, test_trials].")
        return None, None, pe_metadata

    k = min(k_max, len(current_island))
    sample_seed = None
    if random_seed is not None:
        island_offset = 0 if island_id is None else int(island_id)
        sample_seed = int(random_seed) + 10_000 * island_offset + int(batch_id)
    random_programs = current_island.sample(k, replace=False, random_state=sample_seed).reset_index(drop=True)
    # sort from worst to best (loss descending)
    random_programs = random_programs.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
    # Chat mode is not supported for parameter estimator generation/refinement.
    prompt = prompt_tools.build_parameter_estimator_prompt(
        prompt_config,
        random_programs,
        model_code_string=model_code_string,
        max_lines=param_estimator_max_lines,
    )
    if swear_words:
        banned_list = "\n".join(f"- {word}" for word in swear_words)
        prompt = (
            f"{prompt}\n\n"
            "**Banned tokens (do not use in code):**\n"
            f"{banned_list}\n"
        )
    
    try:
        code_string, llm_output = await request_parameter_estimator_module(
            prompt=prompt,
            model_name=llm_name,
            temperature=temp,
            thinking="low",
        )
    except Exception as exc:
        logging.exception("Structured parameter-estimator generation failed: %s", exc)
        pe_metadata["initial_prompt"] = prompt
        pe_metadata["initial_response"] = repr(exc)
        pe_metadata["status"] = "llm_error"
        return None, None, pe_metadata
    pe_metadata["initial_prompt"] = prompt
    pe_metadata["initial_response"] = llm_output
    if code_string is None:
        pe_metadata["status"] = "missing_code"
        return None, None, pe_metadata
    swear_word = _find_banned_token(code_string, swear_words)
    if swear_word is not None:
        pe_metadata["status"] = f"banned_token:{swear_word}"
        return None, None, pe_metadata
    code_string = re.sub(r"def\s+parameter_estimator_v\d+\s*\(", "def parameter_estimator(", code_string)
    code_string = re.sub(r"def\s+parameter_estimator_prev\s*\(", "def parameter_estimator(", code_string)
    func = load_function_from_source(code_string, 'parameter_estimator')

    if func is None:
        pe_metadata["status"] = "parse_failed"
        return None, None, pe_metadata

    if refine_rounds <= 0 or model_fn is None:
        return code_string, func, pe_metadata

    iter_label = "?" if iteration is None else str(iteration)
    print(
        f"Param-est refinement start (iter={iter_label}, island={island_id}, "
        f"batch={batch_id}, rounds={refine_rounds}).",
        flush=True,
    )

    best_code = code_string
    best_func = func
    best_loss = float(jnp.inf)

    current_code = code_string
    current_func = func
    current_loss, current_params, _, _ = _call_objective(
        use_simple_objective,
        model=model_fn,
        param_estimator=current_func,
        data=data,
        loss_fn=loss_fn,
        fit_params=False,  # Don't fit parameters during refinement evaluation
        param_penalty_weight=param_penalty_weight,
        timeout_s=param_estimator_timeout_s,
        objective_timeout_s=objective_timeout_s,
    )

    if current_loss < best_loss:
        best_loss = current_loss
        best_code = current_code
        best_func = current_func

    for r in range(refine_rounds):
        print(
            f"Param-est refinement round {r+1}/{refine_rounds} "
            f"(iter={iter_label}, island={island_id}, batch={batch_id}).",
            flush=True,
        )
        if plot_model_fits is None or x_eval is None or image_refinement_dir is None:
            raise ValueError(
                "Parameter estimator refinement requires image feedback. "
                "Missing plot_model_fits/x_eval/image_refinement_dir."
            )
        img_bytes = None
        try:
            img_path = os.path.join(
                image_refinement_dir,
                f"param_est_refine_island_{island_id}_batch_{batch_id}_r{r+1}.png",
            )
            plot_model_fits(
                data=data[0],
                programs_list=[
                    {
                        "model": model_fn,
                        "params": current_params,
                        "losses": np.full(utils.data_n_samples(data[0]), float(current_loss)),
                    }
                ],
                X_eval=x_eval,
                save_path=img_path,
                labels=['PE'],
            )
            with open(img_path, "rb") as f:
                img_bytes = f.read()
        except Exception as e:
            logging.info(f"Error generating image for parameter estimator refinement: {e}")
            logging.info(f"Model code string was:\n{model_code_string}")
            raise RuntimeError(f"Param-estimator image generation failed: {e}") from e

        # Build refinement prompt using current estimator as the only parent
        refinement_df = pd.DataFrame({
            'train_loss': [current_loss],
            'program_code_string': [model_code_string],
            'parameter_estimator_code_string': [current_code],
        })

        refine_prompt = prompt_tools.build_parameter_estimator_refinement_prompt(
            prompt_config,
            refinement_df,
            model_code_string=model_code_string,
            max_lines=param_estimator_max_lines,
            current_loss=current_loss,
        )
        if swear_words:
            banned_list = "\n".join(f"- {word}" for word in swear_words)
            refine_prompt = (
                f"{refine_prompt}\n\n"
                "**Banned tokens (do not use in code):**\n"
                f"{banned_list}\n"
            )
        # Call LLM for refinement
        try:
            new_code, llm_output = await request_parameter_estimator_module(
                prompt=refine_prompt,
                model_name=llm_name,
                temperature=temp,
                image_bytes=img_bytes,
                thinking="low",
            )
        except Exception as exc:
            logging.exception("Structured parameter-estimator refinement failed: %s", exc)
            pe_metadata["refinement_prompts"].append(refine_prompt)
            pe_metadata["refinement_responses"].append(repr(exc))
            pe_metadata["refinement_codes"].append(None)
            continue
        pe_metadata["refinement_prompts"].append(refine_prompt)
        pe_metadata["refinement_responses"].append(llm_output)

        if new_code is None:
            pe_metadata["refinement_codes"].append(None)
            continue
        if _find_banned_token(new_code, swear_words) is not None:
            pe_metadata["refinement_codes"].append(None)
            continue

        new_code = re.sub(r"def\s+parameter_estimator_v\d+\s*\(", "def parameter_estimator(", new_code)
        new_code = re.sub(r"def\s+parameter_estimator_prev\s*\(", "def parameter_estimator(", new_code)
        pe_metadata["refinement_codes"].append(new_code)
        new_func = load_function_from_source(new_code, 'parameter_estimator')
        if new_func is None:
            continue

        new_loss, _, _, _ = _call_objective(
            use_simple_objective,
            model=model_fn,
            param_estimator=new_func,
            data=data,
            loss_fn=loss_fn,
            fit_params=False,  # Don't fit parameters during refinement evaluation
            param_penalty_weight=param_penalty_weight,
            timeout_s=param_estimator_timeout_s,
            objective_timeout_s=objective_timeout_s,
        )

        print(
            f"Param-est refinement eval (iter={iter_label}, island={island_id}, "
            f"batch={batch_id}, round={r+1}): loss={new_loss:.6g}.",
            flush=True,
        )

        if new_loss < current_loss:
            current_code = new_code
            current_func = new_func
            current_loss = new_loss
            if new_loss < best_loss:
                best_loss = new_loss
                best_code = new_code
                best_func = new_func
        else:
            pass

    return best_code, best_func, pe_metadata


async def translate_to_jax(
    code_string: str,
    client,
    prompt_config,
    llm_name='gemini-2.0-flash-lite',
    entrypoint_name: str = "model",
    max_retries: int = 2,
    retry_delay_s: float = 2.0,
) -> tuple[str, callable, str | None, str | None]:
    """
    Translate a model code string to a JAX-compatible implementation via LLM.

    Args:
        code_string (str): Source code containing the NumPy model definition.
        client: The LLM client.
        prompt_config: Prompt config used to build translation prompts.
        llm_name (str): LLM model name for translation.

    Returns:
        tuple[str | None, callable | None, str | None, str | None]:
            ``(jax_code_string, jax_callable, prompt, raw_response)``.
            Returns ``(None, None)`` when translation cannot be produced/parsed.
    """
    if code_string is None:
        return None, None, None, None
    
    prompt = prompt_tools.build_jax_translator_prompt(prompt_config, code_string)
    if prompt is None:
        return None, None, None, None
    raw_response = None
    jax_code_string = None
    for attempt in range(max_retries + 1):
        try:
            jax_code_string, raw_response = await request_jax_translation(
                prompt=prompt,
                model_name=llm_name,
                entrypoint_name=entrypoint_name,
                temperature=0,
            )
        except Exception as exc:
            raw_response = repr(exc)
            jax_code_string = None
            logging.warning(
                "Structured JAX translation attempt %d/%d failed for model %s: %s",
                attempt + 1,
                max_retries + 1,
                llm_name,
                exc,
            )
        if isinstance(jax_code_string, str) and jax_code_string.strip():
            break
        if attempt < max_retries:
            sleep_s = float(retry_delay_s) * (2 ** attempt)
            logging.warning(
                "JAX translation attempt %d/%d failed for model %s; retrying in %.1fs.",
                attempt + 1,
                max_retries + 1,
                llm_name,
                sleep_s,
            )
            await asyncio.sleep(sleep_s)

    if not (isinstance(raw_response, str) and raw_response.strip()):
        logging.error(
            "JAX translation failed after %d attempts for model %s (empty/None response).",
            max_retries + 1,
            llm_name,
        )
        return None, None, prompt, raw_response

    if not (isinstance(jax_code_string, str) and jax_code_string.strip()):
        logging.error("JAX translation response did not contain code.")
        return None, None, prompt, raw_response

    if entrypoint_name == "model":
        project_model_name = prompt_tools.get_model_name(prompt_config)
        jax_code_string = _normalize_generated_model_code(
            jax_code_string,
            model_name=project_model_name,
            expected_version=None,
        )
        jax_code_string = re.sub(
            rf"^\s*def\s+{re.escape(project_model_name)}\s*\(",
            "def model(",
            jax_code_string,
            count=1,
            flags=re.MULTILINE,
        )
        jax_code_string = _normalize_generated_model_code(
            jax_code_string,
            model_name="model",
            expected_version=None,
        )
    else:
        jax_code_string = re.sub(
            r"def\s+parameter_estimator_v\d+\s*\(",
            "def parameter_estimator(",
            jax_code_string,
        )
        jax_code_string = re.sub(
            r"def\s+parameter_estimator_prev\s*\(",
            "def parameter_estimator(",
            jax_code_string,
        )
    func = load_function_from_source(jax_code_string, entrypoint_name)
    if not callable(func):
        logging.error("Translated JAX code parsed but did not produce callable function %s.", entrypoint_name)
    return jax_code_string, func, prompt, raw_response


def _run_translation_check_on_eval(
    np_func,
    jax_func,
    param_estimator,
    data_train_trials,
    x_eval,
    max_samples: int = 3,
    max_eval_trials: int = 32,
):
    """
    Validate NumPy/JAX numerical agreement on a subset of evaluation points.

    Parameters are first estimated from observed train-trial data. The resulting
    parameter vectors are then used to compare NumPy vs JAX predictions on
    a small data subset.

    Args:
        np_func (callable): Original NumPy model.
        jax_func (callable): Translated JAX model.
        param_estimator (callable): Parameter estimator with signature
            ``param_estimator(data_i) -> params`` for a single sample.
        data_train_trials (dict[str, np.ndarray]): Train-trial data dict with
            sample axis at dim 0.
        x_eval: Evaluation grid (currently unused, kept for API compatibility).
        max_samples (int): Maximum number of samples to check.
        max_eval_trials (int): Maximum eval trials per sample to compare.

    Returns:
        None: Raises on mismatch; otherwise completes silently.
    """
    n_samples = utils.data_n_samples(data_train_trials)
    if n_samples <= 0:
        raise ValueError("No samples available for translation check.")

    n_trials = utils.data_n_trials(data_train_trials)
    n_eval_trials = min(5, int(max_eval_trials), n_trials)
    rng = np.random.default_rng(0)
    if n_eval_trials <= 0:
        raise ValueError("No trials available for translation check.")
    if n_eval_trials == n_trials:
        trial_idx = np.arange(n_trials)
    else:
        trial_idx = rng.choice(n_trials, size=n_eval_trials, replace=False)
    data_subset = utils.slice_data_trials(data_train_trials, trial_idx)

    n_check = min(max_samples, n_samples)
    sample_idx = np.linspace(0, n_samples - 1, num=n_check, dtype=int)
    data_subset = utils.slice_data_samples(data_subset, sample_idx)

    params_subset = compute_initial_params(
        param_estimator,
        np_func,
        data_subset,
    )
    if params_subset is None:
        raise ValueError("Failed to compute parameters for translation check.")

    utils.check_jax_translation(
        np_func=np_func,
        jax_func=jax_func,
        data=data_subset,
        params=params_subset,
        max_eval_trials=max_eval_trials,
    )

