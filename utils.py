import jax
import time
import pandas as pd
import prompt_templates as prompt_text
from typing import Callable, Union, Tuple
# gemini client
from google import genai
from google.genai import types
# Set up logging to suppress warnings from httpx, urllib3, and google.genai
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.ERROR)
# # load dotenv to load environment variables from .env file
# load_dotenv()

def vmap_over_units(model_fn: Callable) -> Callable:
    """Return a version of `model_fn` that accepts
       (theta, params_matrix) and runs one row per unit."""
    def _wrapped(theta, params_row):
        # params_row shape: (k,)  ← one unit’s parameters
        return model_fn(theta, *params_row)   # unpack to scalars
    return jax.vmap(_wrapped, in_axes=(None, 0))   # x shared, params batched

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
    model_name: str = "gemini-2.0-flash",
    client: genai.Client = None,
    temperature: float = 1.0,
    thinking_budget: float = 1.0) -> Union[str, None]:
    """
    Send one prompt to the GenAI client and return the text result.
    """
    try:
        # create the config for the request (thinking budget for 2.5 flash model)
        if '2.5-flash' in model_name:
            thinking_budget = int(thinking_budget * 24_576)
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=5_000,
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
            )
        else:
            config = types.GenerateContentConfig(temperature=temperature, max_output_tokens=5_000)
        
        # send the request to the GenAI client
        resp = client.models.generate_content(model=model_name, contents=[prompt_text], config=config)
        return resp.text
    except Exception as e:
        print(f"ERROR (Gemini): {e}")
        # wait a small amount of time before retrying
        time.sleep(5)
        return None
    
async def call_llm_async(
    prompt_text: Union[str, None],
    client: genai.Client,
    model_name: str = "gemini-2.0-flash",
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
        if '2.5' in model_name:
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
                model=model_name,
                contents=[prompt_text, types.Part.from_bytes(data=img_bytes, mime_type="image/png")],
                config=config
            )
        else:
            resp = await client.aio.models.generate_content(model=model_name, contents=[prompt_text], config=config)
        
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

def create_program_prompt(random_programs: pd.DataFrame, mode: str,
                          use_image: bool = True, function_name: str = 'neuron_model') -> str:
    """
    Create a prompt to generate a new function based on k existing models.

    Args:
        random_programs (pd.DataFrame): A DataFrame containing the existing models, their losses, and their parameter estimators. (+ more)
            (Assumes that df is sorted from highest loss to lowest loss)
        mode (str): The mode of evolution - 'explore' or 'exploit'.
        use_image (bool): Whether to include an image prompt in the generated prompt. Defaults to True.
        function_name (str): The name of the function to be generated. Defaults to 'neuron_model'.
    Returns:
        prompt (str): The prompt string for the AI to generate a new function.
    """
    # Ensure the mode is valid
    assert mode in ['explore', 'exploit'], "Invalid mode. Choose either 'explore' or 'exploit'."

    # get the number k of parent programs
    k = len(random_programs)

    prompt = prompt_text.program_creation_instructions(k=k, mode=mode, function_name=function_name)
    #  prompt explaining the image
    if use_image:
        prompt += prompt_text.image_analysis_instructions(k=k, function_name=function_name)

    # docstring and coding guidelines
    prompt += prompt_text.coding_instructions(k=k, function_name=function_name)
    
    # add programs to the prompt
    for i in range(k):
        fn_def = f"def {function_name}("
        fn_version = f"def {function_name}_v{i+1}("
        prompt += f"""
loss of model {i+1}: {random_programs.iloc[i]['train_loss']: .2f}
{random_programs.iloc[i]['function_code_string'].replace(fn_def, fn_version)}
\n
"""

    return prompt

def create_parameter_estimator_prompt(random_programs: pd.DataFrame, func_code_string: str,
                                      max_lines: int = 100, function_name: str = 'neuron_model') -> str:
    """
    Create a prompt to generate a new parameter estimator based on k existing models.
    Args:
        random_programs (pd.DataFrame): A DataFrame containing the existing models, their losses, and their parameter estimators. (+ more)
            (Assumes that df is sorted from highest loss to lowest loss)
        func_code_string (str): The code for the candidate function whose parameters need to be estimated.
        max_lines (int): Maximum number of lines allowed in the generated estimator code.
        function_name (str): The name of the function whose parameters will be estimated.
    Returns:
        prompt (str): The prompt string for the AI to generate a new parameter estimator.
    """
    # get the number k of parent programs
    k = len(random_programs)

    prompt = prompt_text.parameter_estimator_creation_instructions(k=k, function_name=function_name, max_lines=max_lines)

    # loop through the models, and add the relevant code and metadata.
    for i in range(k):
        fn_def = f"def {function_name}("
        fn_version = f"def {function_name}_v{i+1}("
        prompt += f"""
loss of model {i+1}: {random_programs.iloc[i]['train_loss']: .2f}
{random_programs.iloc[i]['function_code_string'].replace(fn_def, fn_version)}
\n
{random_programs.iloc[i]['parameter_estimator_code_string'].replace('def parameter_estimator(', f'def parameter_estimator_v{i+1}(')}
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
    return prompt_text.jax_translation_instructions(program, function_name)
