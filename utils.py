from __future__ import annotations

import asyncio
import inspect
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Sequence, Tuple, Union

import google.genai
import jax
import jax.numpy as jnp
import yaml
import numpy as np  # type: ignore
from dotenv import load_dotenv
from google.genai import types

from entities import Program

# Set up logging to suppress warnings from httpx, urllib3, and google.genai
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.ERROR)
load_dotenv()

_TEMPLATE_PATH = Path(__file__).with_name("prompt_templates.txt")
_PROMPT_OVERRIDES: dict[str, str] = {}


def _read_prompt_templates() -> dict[str, str]:
    """Load flat prompt templates from disk (one section per template)."""
    if not _TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Prompt template file not found: {_TEMPLATE_PATH}")

    templates: dict[str, str] = {}
    current_section: str | None = None
    lines: list[str] = []
    with _TEMPLATE_PATH.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                if current_section:
                    templates[current_section] = "".join(lines).lstrip("\n")
                current_section = stripped[1:-1]
                lines = []
            else:
                lines.append(raw_line)
    if current_section:
        templates[current_section] = "".join(lines).lstrip("\n")
    if not templates:
        raise ValueError("No prompt templates could be parsed.")
    return templates


@lru_cache(maxsize=1)
def _prompt_templates() -> dict[str, str]:
    return _read_prompt_templates()


def _format_template(section: str, **context) -> str:
    """Retrieve and format a prompt template by section name."""
    if section in _PROMPT_OVERRIDES:
        template = _PROMPT_OVERRIDES[section]
    else:
        templates = _prompt_templates()
        if section not in templates:
            raise KeyError(f"Missing prompt template for section {section!r}")
        template = templates[section]
    return template.format(**context)


def validate_jax_translation(
    numpy_function: Callable,
    jax_function: Callable,
    X_samples: Union[jnp.ndarray, None] = None,
    params: Union[dict, None] = None,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> bool:
    """Check whether a translated JAX function matches its NumPy source on sample inputs."""
    if X_samples is None:
        X_samples = jnp.linspace(0, 2 * jnp.pi, 64, dtype=jnp.float32)

    if params is None:
        # Use default values of the functions if params is not provided
        params = {}
        if hasattr(numpy_function, "__defaults__") and numpy_function.__defaults__:
            param_names = inspect.signature(numpy_function).parameters
            params = {
                name: default
                for name, default in zip(param_names, numpy_function.__defaults__)
            }

    try:
        numpy_output = numpy_function(X_samples, **params)
        jax_output = jax_function(X_samples, **params)
    except Exception:
        return False

    return bool(jnp.allclose(numpy_output, jax_output, atol=atol, rtol=rtol))


def reset_prompt_overrides() -> None:
    """Clear any user-provided prompt overrides."""
    _PROMPT_OVERRIDES.clear()


def set_prompt_overrides(overrides: Mapping[str, str]) -> None:
    """Apply user-provided prompt overrides (flat section -> text)."""
    for section, text in overrides.items():
        _PROMPT_OVERRIDES[section] = str(text)


def vmap_over_units(model_fn: Callable) -> Callable:
    """Return a version of `model_fn` that accepts
       (theta, params_matrix) and runs one row per unit."""
    def _wrapped(theta, params_row):
        # params_row shape: (k,)  ← one unit’s parameters
        return model_fn(theta, *params_row)   # unpack to scalars
    return jax.vmap(_wrapped, in_axes=(None, 0))   # x shared, params batched

def split_arrays(X: jnp.ndarray, Y: jnp.ndarray, random_seed: int = 0, axis: int = 1) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Split 2D arrays along a given axis into training and testing sets (50% each).
    Args:
        X (jnp.ndarray): input data (n_units, n_points).
        Y (jnp.ndarray): Response data of shape (n_units, n_points).
        random_seed (int): Seed used to shuffle points before splitting.
    Returns:
        tuple: (X_train, Y_train, X_test, Y_test), each with shape (n_units, n_points//2) if axis=1, else (n_units//2, n_points).
    """
    n_units, n_points = X.shape
    key = jax.random.PRNGKey(random_seed)
    shuffled_indices = jax.random.permutation(key, jnp.arange(n_points if axis == 1 else n_units))
    if axis == 1:
        train_indices, test_indices = shuffled_indices[:n_points // 2], shuffled_indices[n_points // 2:]
        X_train, X_test = X[:, train_indices], X[:, test_indices]
        Y_train, Y_test = Y[:, train_indices], Y[:, test_indices]
    else:
        train_indices, test_indices = shuffled_indices[:n_units // 2], shuffled_indices[n_units // 2:]
        X_train, X_test = X[train_indices, :], X[test_indices, :]
        Y_train, Y_test = Y[train_indices, :], Y[test_indices, :]
    return X_train, Y_train, X_test, Y_test

def create_output_directories(use_image: bool = True) -> Tuple[str, str, str, str, str]:
    base_dir = os.path.join(os.getcwd(), 'program_databases')
    print("Base directory:", base_dir)
    os.makedirs(base_dir, exist_ok=True)
    now = datetime.now()
    date_stamp = now.strftime("%m-%d")
    time_stamp = now.strftime("%H-%M-%S")
    full_dir = os.path.join(base_dir, date_stamp, time_stamp)
    os.makedirs(full_dir, exist_ok=True)
    print("Created folder:", full_dir)
    # create a directory for image diagnostics
    image_feedback_dir = os.path.join(full_dir, 'image_feedback') if use_image else None
    if use_image:
        os.makedirs(image_feedback_dir, exist_ok=True)
    print("Created image feedback folder:", image_feedback_dir)
    return base_dir, date_stamp, time_stamp, full_dir, image_feedback_dir

def extract_code_block(text: Union[str, None], start_marker: str = "```python\n", end_marker: str = "```") -> Union[str, None]:
    """
    Extracts a code block from a given text string, using specified start and end markers.
    If the text is None, it returns an empty string.
    If start and end markers not found returns the whole text.
    Args:
        text (str or None): The input text containing the code block.
        start_marker (str): The marker indicating the start of the code block.
        end_marker (str): The marker indicating the end of the code block.
    Returns:
        str: The extracted code block, or an empty string if the text is None.
    """
    if text is None:
        return None
    
    # find the start of the code block
    start = text.find(start_marker)
    if start == -1:
        start = 0
    else:
        # move the start index to the end of the marker
        start += len(start_marker)

    # find the closing fence after that
    end = text.find(end_marker, start)
    if end == -1:
        end = len(text) 

    # return just the code between the fences
    return text[start:end].rstrip()

def call_llm(
    prompt_text: str,
    llm_name: str = "gemini-2.0-flash",
    client: google.genai.Client = None,
    temperature: float = 1.0,
    thinking_budget: float = 1.0) -> Union[str, None]:
    """
    Send one prompt to the GenAI client and return the text result.
    """
    try:
        # create the config for the request (thinking budget for 2.5 flash model)
        if '2.5-flash' in llm_name:
            thinking_budget = int(thinking_budget * 24_576)
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=5_000,
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
            )
        else:
            config = types.GenerateContentConfig(temperature=temperature, max_output_tokens=5_000)
        
        # send the request to the GenAI client
        resp = client.models.generate_content(model=llm_name, contents=[prompt_text], config=config)
        return resp.text
    except Exception as e:
        print(f"ERROR (Gemini): {e}")
        # wait a small amount of time before retrying
        time.sleep(5)
        return None
    
async def call_llm_async(
    prompt_text: Union[str, None],
    client: google.genai.Client,
    llm_name: str = "gemini-2.0-flash",
    temperature: float = 1.0,
    thinking_budget: float = 1,
    img_bytes: Union[bytes, None] = None,
    semaphore: asyncio.Semaphore = None  # <--- NEW ARGUMENT
    ) -> Union[str, None]:
    """
    Send one prompt to the GenAI client and return the text result.
    """
    if prompt_text is None:
        return None

    try:
        # Use the semaphore if provided, otherwise use a dummy context that does nothing
        ctx = semaphore if semaphore else nullcontext()

        async with ctx:  # <--- NEW CONTEXT MANAGER WRAPPER
            # Create the config for the request (thinking budget for 2.5 flash model)
            if '2.5' in llm_name:
                thinking_budget = int(thinking_budget * 24_576) if thinking_budget >= 0 else -1
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
                )
            else:
                config = types.GenerateContentConfig(temperature=temperature)

            # Send the request to the GenAI client
            if img_bytes is not None:
                resp = await client.aio.models.generate_content(
                    model=llm_name,
                    contents=[prompt_text, types.Part.from_bytes(data=img_bytes, mime_type="image/png")],
                    config=config
                )
            else:
                resp = await client.aio.models.generate_content(model=llm_name, contents=[prompt_text], config=config)

            return resp.text
    except Exception as e:
        print(f"Error in GenAI async call: {e}")
        return None

def str_to_func(code_string: Tuple[str, None], needle: str = 'neuron_model') -> Tuple[callable, None]:
    """
    Convert a string containing Python code into a callable function,

    Args:
        code_string (str or None): The string containing the Python function definition.
        needle (str): The name of the function to be extracted from the string.

    Returns:
        function: The callable function defined in the string, or None if not found.
    """
    # check if code sting is None, if so, return None
    if code_string is None:
        return None
    
    # Prepare a namespace dictionary for exec to run in. 
    execution_namespace = {}

    # Execute the code string within the specified namespace
    try:
        exec(code_string, execution_namespace)  # Pass the dictionary
    except Exception as e:
        print(f"Error executing code string: {e}\nCode:\n{code_string}") # Print code on error
        return None
    else:
        # Retrieve the function object from the namespace dictionary
        if needle in execution_namespace:
            return execution_namespace[needle]
        else:
            print(f"Function {needle} not found in executed code.")
            return None


def load_edgar_config(path: str) -> dict:
    """Load an Edgar configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    engine_config = raw.get("engine") or {}
    # Normalize engine keys to match Edgar/_run_engine kwargs
    key_aliases = {"function_name": "func_name"}
    allowed_keys = {
        "n_generations",
        "time_limit",
        "k_max",
        "n_islands",
        "batch_size",
        "critical_population_size",
        "min_wise_population_size",
        "n_migrants",
        "fit_params",
        "exploit_point",
        "param_penalty_weight",
        "FAILED_PROGRAM_COST",
        "exploration_topology",
        "exploitation_topology",
        "seed_functions_numpy",
        "seed_parameter_estimators",
        "func_name",
        "tiny_lm_name",
        "little_lm_name",
        "large_lm_name",
        "use_large_every",
        "diagnostic_image_fn",
        "llm_concurrency",
    }
    config: dict = {}
    for key, value in engine_config.items():
        normalized = key_aliases.get(key, key)
        if normalized in allowed_keys:
            config[normalized] = value
    func_name = config.get("func_name") or "neuron_model"
    config["func_name"] = func_name

    seed_functions: list[Callable] = []
    seed_estimators: list[Callable | None] = []
    for idx, entry in enumerate(raw.get("seed_programs", [])):
        func_code = entry.get("function")
        if not func_code:
            continue
        target_name = entry.get("function_name", func_name)
        func = str_to_func(func_code, target_name)
        if func is None:
            continue
        seed_functions.append(func)
        est_code = entry.get("parameter_estimator")
        if est_code:
            est_name = entry.get("parameter_estimator_name", "parameter_estimator")
            est = str_to_func(est_code, est_name)
        else:
            est = None
        seed_estimators.append(est)

    if seed_functions:
        config["seed_functions_numpy"] = seed_functions
        config["seed_parameter_estimators"] = seed_estimators

    diag_code = raw.get("diagnostic_image_fn") or raw.get("diagnostic_function")
    if diag_code:
        diag_name = raw.get("diagnostic_image_fn_name") or raw.get("diagnostic_function_name") or "diagnostic_image"
        diag_fn = str_to_func(diag_code, diag_name)
        if diag_fn:
            config["diagnostic_image_fn"] = diag_fn

    overrides = raw.get("prompt_overrides") or raw.get("prompt")
    reset_prompt_overrides()
    if isinstance(overrides, Mapping):
        string_overrides = {k: v for k, v in overrides.items() if isinstance(v, str)}
        if string_overrides:
            set_prompt_overrides(string_overrides)

    return config
def _ensure_program_sequence(programs: Sequence[Program]) -> list[Program]:
    if programs is None:
        return []
    return list(programs)


def create_program_prompt(random_programs: Sequence[Program], mode: str,
                          use_image: bool = True, function_name: str = 'neuron_model') -> str:
    """
    Create a prompt to generate a new function based on k existing models.

    Args:
        random_programs (Sequence[Program]): Existing parent programs sorted from highest to lowest loss.
        mode (str): The mode of evolution - 'explore' or 'exploit'.
        use_image (bool): Whether to include an image prompt in the generated prompt.
        function_name (str): The name of the function to be generated.
    Returns:
        prompt (str): The prompt string for the AI to generate a new function.
    """
    # Ensure the mode is valid
    assert mode in ['explore', 'exploit'], "Invalid mode. Choose either 'explore' or 'exploit'."

    parent_programs = _ensure_program_sequence(random_programs)
    k = len(parent_programs)

    context = {"function_name": function_name, "version_index": k + 1}
    prompt = _format_template("program_creation_context", **context)
    prompt += _format_template("explore_instructions" if mode == "explore" else "exploit_instruction", **context)
    if use_image:
        prompt += _format_template("image_analysis_instructions", **context)
    prompt += _format_template("coding_instructions", **context)
    
    # add programs to the prompt
    for i, program in enumerate(parent_programs):
        fn_def = f"def {function_name}("
        fn_version = f"def {function_name}_v{i+1}("
        prompt += f"""
loss of model {i+1}: {program.train_loss: .2f}
{program.function_code_string.replace(fn_def, fn_version)}
\n
"""

    return prompt

def create_parameter_estimator_prompt(random_programs: Sequence[Program], func_code_string: str,
                                      max_lines: int = 100, function_name: str = 'neuron_model') -> str:
    """
    Create a prompt to generate a new parameter estimator based on k existing models.
    Args:
        random_programs (Sequence[Program]): Existing parent programs sorted from highest to lowest loss.
        func_code_string (str): The code for the candidate function whose parameters need to be estimated.
        max_lines (int): Maximum number of lines allowed in the generated estimator code.
        function_name (str): The name of the function whose parameters will be estimated.
    Returns:
        prompt (str): The prompt string for the AI to generate a new parameter estimator.
    """
    parent_programs = _ensure_program_sequence(random_programs)
    k = len(parent_programs)

    context = {"function_name": function_name, "version_index": k + 1, "max_lines": max_lines}
    prompt = _format_template("parameter_estimator_context", **context)

    # loop through the models, and add the relevant code and metadata.
    for i, program in enumerate(parent_programs):
        fn_def = f"def {function_name}("
        fn_version = f"def {function_name}_v{i+1}("
        prompt += f"""
loss of model {i+1}: {program.train_loss: .2f}
{program.function_code_string.replace(fn_def, fn_version)}
\n
{program.parameter_estimator_code_string.replace('def parameter_estimator(', f'def parameter_estimator_v{i+1}(')}
\n
----------------------------
\n
"""
    # add the new function code string to the prompt
    prompt += f"""
{func_code_string.replace(f'def {function_name}(', f'def {function_name}_v{k+1}(')}
\n
"""
    return prompt

def create_jax_translater_prompt(program: str, function_name: str = 'neuron_model') -> str:
    """
    Create a prompt to translate a program to JAX compatible code.
    Args:
        program (str): The string containing the code to be translated.
    Returns:
        prompt (str): The prompt string for the AI to translate the program.
    """
    # Ensure the program is a string
    assert isinstance(program, str), "The program must be a string."
    return _format_template("jax_translation_instructions", program=program, function_name=function_name)

# text = """

# You are an AI scientist. The programs below are biological models of neurons. The models are sorted from highest to lowest loss.

# Your task is to create a new neuron model, neuron_model_v3, that has a lower loss than the models below.

# *Analyze* the progression of the models, *generalize* the improvements, and *create* a new model that is better than *all* previous models.


# Use the models below as inspiration, but be *creative* and *invent* something new. Which features in the models below correlate with lower loss? Find these features and *extrapolate* them. You should also *combine* features from several models, and *experiment* with new ideas.

# **Image Analysis Instructions:**

# Attached is a scatter plot of the neuron models' performance on top of raw neural data. The binned mean is plotted in **sky-blue**, `neuron_model_v1` is plotted in **green**, and `neuron_model_v2` is plotted in **red**. 

# Analyse the models' fits to the data in the image below. Identify systematic weaknesses of the models by observing patterns across multiple cell plots. For instance, consider:
# *   **Model Comparisons:** Which models are better for each cell? That is to say, which models track the blue curve better? Which features of the models are responsible for improving the fit?
# *   **Model Fit:** How well do the models fit the binned data mean? Look for places where even the models (**red** curve for best model, **green** for second best model) deviate most from the binned data mean (**blue** curve). This is where the models are weakest, and where you should focus your improvements.
# *   **Model Shape:** Do the models' shapes (e.g., peak sharpness, width, skewness, amplitude, etc.) align with the binned data mean (**blue**) and raw data scatter points (**black**)? If not, how do they differ? How can you change the model to better match the data shape?
# *   **Parameter Flexibility:** Are there free parameters that could be introduced or modified to better capture the observed response profiles? Utilize your analysis of the shortcomings of the current models' shapes and add free parameters or modify existing ones to address these issues.

# Use this analysis to inform the design of a new neuron model, `neuron_model_v3`, that improves upon the previous models. 

# Include your analysis of the image in the docstring of your new model. Point to specific subplots in the image that illustrate the *strengths* and *weaknesses* of the parent models. Explain how you plan to **fix** the weaknesses of the parent models.

# **Code Generation Guidelines:**

# * Import any packages you use.
# * Do not include any text other than the code.
# * Ensure all free parameters are numeric, not strings.
# * At the beginning of the code, clip the free parameters to a biologically plausible range, e.g., `theta_pref = np.clip(theta_pref, 0, 2 * np.pi)`.

# **Docstring Guidelines:**
# * Begin by listing the parent models and give them a name that describes their key features, e.g., `parent_model_1: simple_exponential_decay-model`, `parent_model_2: double_exponential_decay_model`. Never refer to the models as `neuron_model_v1`, `neuron_model_v2`, etc. Instead, refer to them as `parent_models` or their descriptive names (e.g. `simple_exponential_decay_model`).
# * Do not refer to the current model as `neuron_model_v3`. Instead, refer to it as "this model".
# * Provide a simple equation for the model, including all free parameters.
# * Include a brief description of how the model improves upon the previous models, citing specific features or changes that lead to lower loss.


# loss of model 1:  30.02
# import numpy as np 
# def neuron_model_v1(theta, theta_pref=0.0, baseline=0.0, amplitude=1.0, tuning_width=1.0):
#     # A simple neuron model that computes the response based on a Gaussian tuning curve.
#     # Args:
#     #     theta (np.ndarray): The angle in radians.
#     #     theta_pref (float): Preferred direction of the neuron.
#     #     baseline (float): Baseline firing rate.
#     #     amplitude (float): Maximum firing rate above baseline.
#     #     tuning_width (float): Width of the tuning curve.
#     # Returns:
#     #     np.ndarray: The firing rate of the neuron at angle theta.

#     theta_pref = np.clip(theta_pref, 0, 2 * np.pi)
#     baseline = np.clip(baseline, 0, None)
#     amplitude = np.clip(amplitude, 0, None)
#     tuning_width = np.clip(tuning_width, 0.01, None)

#     circ_dist_rad = lambda theta1, theta2: np.abs(np.arctan2(np.sin(theta1 - theta2), np.cos(theta1 - theta2)))
#     dist = circ_dist_rad(theta, theta_pref)
#     return baseline + amplitude * np.exp(-0.5 * (dist / tuning_width) ** 2)




# loss of model 2:  28.85
# import numpy as np 
# def neuron_model_v2(theta, theta_pref=0.0, baseline=0.0, amplitude_1=1.0, amplitude_2=0.0, tuning_width=1.0):
    
#     # A neuron model that computes the response based on a double peaked gaussian tuning curve, with peaks at theta_pref and (theta_pref + pi) % 2pi.
#     # Args:
#     #     theta (np.ndarray): Input angles in radians.
#     #     theta_pref (float): Preferred angle in radians.
#     #     baseline (float): Baseline firing rate.
#     #     amplitude_1 (float): Amplitude of the first peak.
#     #     amplitude_2_ratio (float): Ratio of the second peak's amplitude to the first peak's amplitude.
#     #     tuning_width (float): Width of the tuning curves around preferred angles.
#     # Returns:
#     #     np.ndarray: The response of the neuron model.

#     theta_pref = np.clip(theta_pref, 0, 2 * np.pi)
#     baseline = np.clip(baseline, 0, None)
#     amplitude_1 = np.clip(amplitude_1, 0, None)
#     amplitude_2 = np.clip(amplitude_2, 0, None)
#     tuning_width = np.clip(tuning_width, 0.01, None)
    
#     circ_dist_rad = lambda theta1, theta2: np.abs(np.arctan2(np.sin(theta1 - theta2), np.cos(theta1 - theta2)))
#     dist_1 = circ_dist_rad(theta, theta_pref)
#     dist_2 = circ_dist_rad(theta, (theta_pref + np.pi) % (2 * np.pi))
#     return baseline + amplitude_1 * np.exp(-0.5 * (dist_1 / tuning_width) ** 2) + amplitude_2 * np.exp(-0.5 * (dist_2 / tuning_width) ** 2)
# """
# # async call llm example
# import asyncio 
# import time
# async def main():
#     client = google.genai.Client()
#     # Create the controller (allow fixed number of concurrent requests)
#     sem = asyncio.Semaphore(10) 
#     tasks = [call_llm_async(prompt, llm_name="gemini-2.5-pro", client=client, semaphore=sem) for prompt in [text]*50]
#     t_start = time.time()
#     responses = await asyncio.gather(*tasks)
#     t_end = time.time()
#     n_failed = sum(1 for resp in responses if resp is None)
#     # print time taken
#     print(f"Time taken: {t_end - t_start} seconds")
#     print(f"Number of failed responses: {n_failed} out of {len(responses)}")
#     # for i, resp in enumerate(responses):
#     #     if resp is None:
#     #         print(f"Response {i}: Error or no response")
#     #         continue
#     #     print(f"Response {i}: {resp[:100]}...")
# asyncio.run(main())
