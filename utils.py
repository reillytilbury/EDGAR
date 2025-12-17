from __future__ import annotations

import configparser
import inspect
import os
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence, Tuple, Union

import google.genai
import jax
import jax.numpy as jnp
try:  # pragma: no cover
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = jnp  # fall back to JAX arrays for validation
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


def _read_prompt_templates() -> configparser.RawConfigParser:
    parser = configparser.RawConfigParser()
    parser.optionxform = str
    with _TEMPLATE_PATH.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)
    return parser


@lru_cache(maxsize=1)
def _prompt_templates() -> configparser.RawConfigParser:
    if not _TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Prompt template file not found: {_TEMPLATE_PATH}")
    return _read_prompt_templates()


def _format_template(section: str, key: str, **context) -> str:
    templates = _prompt_templates()
    if not templates.has_section(section) or key not in templates[section]:
        raise KeyError(f"Missing prompt template [{section}] {key}")
    return templates.get(section, key).format(**context)


def validate_jax_translation(
    numpy_function: Callable,
    jax_function: Callable,
    X_samples: Union[jnp.ndarray, None] = None,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> bool:
    """Check whether a translated JAX function matches its NumPy source on sample inputs."""
    if X_samples is None:
        X_samples = jnp.linspace(0, 2 * jnp.pi, 64, dtype=jnp.float32)

    X_np = np.asarray(X_samples, dtype=np.float32)
    X_jax = jnp.asarray(X_np)
    try:
        numpy_output = np.asarray(numpy_function(X_np), dtype=np.float32)
        jax_output = np.asarray(jax_function(X_jax), dtype=np.float32)
    except Exception:
        return False

    return np.allclose(numpy_output, jax_output, atol=atol, rtol=rtol)


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
    img_bytes: Union[bytes, None] = None
    ) -> Union[str, None]:
    """
    Send one prompt to the GenAI client and return the text result.
    """
    if prompt_text is None:
        return None
    try:
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
    prompt = _format_template("program_creation", "base", **context)
    prompt += _format_template("program_creation", mode, **context)
    if use_image:
        prompt += _format_template("image_analysis", "template", **context)
    prompt += _format_template("coding", "template", **context)
    
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
    prompt = _format_template("parameter_estimator", "template", **context)

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
    return _format_template("jax_translation", "template", program=program, function_name=function_name)

# call llm example
# client = google.genai.Client()
# response = call_llm("What is the capital of France?", model_name="gemini-2.0-flash", client=client)
# print("LLM Response:", response)

# async call llm example
# import asyncio 
# async def main():
#     client = google.genai.Client()
#     tasks = [call_llm_async(prompt, model_name="gemini-2.0-flash", client=client) for prompt in ["what is the capital of france"] * 100]
#     responses = await asyncio.gather(*tasks)
#     for i, resp in enumerate(responses):
#         print(f"Response {i+1}: {resp}")

# asyncio.run(main())
