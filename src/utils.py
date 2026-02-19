import os
import asyncio
from . import hypothesis_engine
import ast
import inspect
import jax
import time
import numpy as np
import scipy as sc
import asyncio
import pickle
import jax.numpy as jnp
import pandas as pd
import optax
from dotenv import load_dotenv
from typing import Callable, Dict, Any, Optional, Sequence, Union, Tuple, List
# gemini client
from google import genai
from google.genai import types
# anthropic client
import anthropic
# Set up logging to suppress warnings from httpx, urllib3, and google.genai
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.ERROR)
# load dotenv to load environment variables from .env file


def format_function_source(func: Callable, new_name: str, import_statement: str = "") -> str:
    """
    Format a function's source code with a new name and import statement.
    
    Args:
        func: The function to extract source code from
        new_name: The new name for the function
        import_statement: Import statement to prepend to the source code
        
    Returns:
        Formatted source code string
    """
    source = inspect.getsource(func)
    original_name = func.__name__
    formatted_source = source.replace(f'def {original_name}(', f'def {new_name}(')
    
    if import_statement and not import_statement.endswith('\n'):
        import_statement += '\n'
    
    return import_statement + formatted_source


def vmap_over_samples(model_fn):
    """Return a version of `model_fn` that accepts
       (X, params_matrix) and runs one row per sample.
       
    Args:
        model_fn: A model function with signature:
                  model_fn(X, *params) -> (n_trials,)
                  where X has shape (n_features, n_trials).
    
    Returns:
        A vmapped function that accepts:
        - X: shape (n_samples, n_features, n_trials)
        - params_matrix: shape (n_samples, n_params)
        Returns shape (n_samples, n_trials).
    """
    def _wrapped(X_cell, params_row):
        # X_cell shape: (n_features, n_trials) for one cell
        # params_row shape: (k,) - one cell's parameters
        return model_fn(X_cell, *params_row)   # unpack to scalars
    return jax.vmap(_wrapped, in_axes=(0, 0))   # both X and params batched over cells


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


def str_to_func(code_string: Union[str, None], needle: str) -> Union[Callable, None]:
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


# response, angles = load_data(data_dir = ['/home/reilly/Desktop/ali data/stim_sequence.npy','/home/reilly/Desktop/ali data/stim_resps.npy'],
#                                    data_type='ali',
#                                    conc_thresh=0.4,
#                                    activity_thresh=0.0,
#                                    signal_fraction_thresh=0.8,
#                                    n_bins=90, min_repeats=6)

# async def main():
#     # load api keys
#     load_dotenv()
#     client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
#     # client = anthropic.AsyncClient(api_key=os.getenv("ANTHROPIC_API_KEY"))
#     print("API key loaded")

#     prompts = ["Return **ONLY** the numerical answer (to 2 decimal places), do **NOT** show your reasoning. A thin hoop of diameter d=0.3 is thrown on to an infinitely large chessboard with squares of side L=1.0. What is the chance of the hoop enclosing two colours?"] * 30
#     t1 = time.time()
#     # model_name = 'claude-3-5-haiku-latest'
#     # model_name = 'gemini-2.0-flash'
#     # model_name = 'gemini-1.5-flash-8b'
#     # model_name = 'gemini-2.5-flash'
#     model_name = 'gemini-2.5-flash-lite-preview-06-17'

#     tasks = [call_llm_async(p, client=client, model_name=model_name, thinking_budget=0.1) for p in prompts]
#     results = await asyncio.gather(*tasks)
#     for i, result in enumerate(results):
#         print(f"Result {i + 1}:")
#         print(result)
#         print('--------------------------------')
#     t2 = time.time()
#     print(f"Time taken: {t2 - t1:.2f} seconds")

# if __name__ == "__main__":
#     asyncio.run(main()) 

# from pathlib import Path
# # call llm with image prompt
# load_dotenv()
# client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
# img_path = Path('program_databases/06-05/13-57-13/combined/top_model_fit_1.png')

# # ---------- OPTION 1 : inline bytes (best for files < 20 MB) ----------
# with img_path.open("rb") as f:
#     img_bytes = f.read()
    

# response = client.models.generate_content(
#     model="gemini-2.5-pro-preview-05-06",
#     contents=[
#         types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
#         text_prompt
#     ]
# )
# print(response.text)

# from pathlib import Path

# prompt = """"
# You are an AI scientist. The programs below are biological models of neurons. The models are sorted from highest to lowest loss.

# Your task is to create a new neuron model, neuron_model_v3, that has a lower loss than the models below.

# *Analyze* the progression of the models, *generalize* the improvements, and *create* a new model that is better than *all* previous models.


# Use the models below as a *template* to create an improved, simpler model.
# Focus on *exploiting* the strengths of the existing models and *eliminating* their weaknesses or *redundancies*.
# You will be *penalized* for complexity, so make the new model as *simple* as possible while still being better than the previous models.

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

# **Docstring Guidelines:**
# * Begin by listing the parent models and give them a name that describes their key features, e.g., `parent_model_1: simple_exponential_decay-model`, `parent_model_2: double_exponential_decay_model`. Never refer to the models as `neuron_model_v1`, `neuron_model_v2`, etc. Instead, refer to them as `parent_models` or their descriptive names (e.g. `simple_exponential_decay_model`).
# * Do not refer to the current model as `neuron_model_v3`. Instead, refer to it as "this model".
# * Provide a simple equation for the model, including all free parameters.
# * Include a brief description of how the model improves upon the previous models, citing specific features or changes that lead to lower loss.


# loss of model 1:  21.90
# import numpy as np

# def neuron_model_v1(theta, theta_pref_1=0.0, baseline=0.0, amplitude_1=1.0, 
#                     tuning_width_1_left=1.0, tuning_width_1_right=1.0, 
#                     amplitude_2=0.0, offset_2_peak=np.pi, 
#                     tuning_width_2_left=1.0, tuning_width_2_right=1.0):
#     Parent model 1: simple_gaussian_model
#         Equation: F = baseline + amplitude * exp(-0.5 * (circ_dist(theta, theta_pref) / tuning_width)^2)
#         Strengths: Basic tuning curve representation, effectively capturing a single preferred direction.
#         Weaknesses:
#             - Cannot adequately model multi-peaked responses. This limitation is clearly evident in cells such as Cell 228 (subplot row 2, col 2), Cell 41 (subplot row 2, col 3), or Cell 81 (subplot row 1, col 2), where the binned mean (sky-blue curve) unequivocally shows a second distinct peak approximately pi radians from the main preferred direction. The simple_gaussian_model (green curve) largely fails to capture these secondary peaks or results in an overly broad, imprecise fit.
#             - Assumes perfect symmetric tuning around the preferred direction. This rigidity prevents the model from accurately representing the observed skewness or asymmetric decay often present in actual neural firing profiles. For example, in Cell 165 (subplot row 1, col 1) and Cell 226 (subplot row 2, col 1), the primary peak in the mean data (sky-blue) displays subtle, yet significant, asymmetry that the symmetric Gaussian component (green) cannot precisely reproduce.

#     Parent model 2: double_gaussian_model
#         Equation: F = baseline + amplitude_1 * exp(-0.5 * (circ_dist(theta, theta_pref)^2 / tuning_width^2)) + amplitude_2 * exp(-0.5 * (circ_dist(theta, (theta_pref + pi))^2 / tuning_width^2))
#         Strengths: Demonstrates a significant improvement in fit by introducing a second Gaussian peak fixed at 180 degrees (pi radians) opposite the primary preferred direction. This effectively addresses the core multi-modal distribution weakness of the simple_gaussian_model. This advancement is particularly noticeable in cells like Cell 228, Cell 41, and Cell 210 (subplot row 3, col 3), where the double_gaussian_model (red curve) provides a much closer approximation to the binned mean (sky-blue) compared to its predecessor.
#         Weaknesses:
#             - Both peaks are constrained to share a single 'tuning_width' parameter. This is a considerable limitation, as visually confirmed in Cell 81 (subplot row 1, col 2), where the primary peak (situated around 5.5-6 radians) is noticeably wider than the secondary peak (located around 2.5 radians). The shared `tuning_width` forces a compromise, resulting in a suboptimal fit for at least one of the peaks. Similar width discrepancies are also apparent in Cell 224 (subplot row 1, col 3) and Cell 41 (subplot row 2, col 3).
#             - Despite the inclusion of a second peak, each individual Gaussian component within the model still presumes symmetric tuning. Consequently, actual response profiles (e.g., those in Cell 165, Cell 226, or the secondary peak in Cell 81) frequently exhibit inherent asymmetry that cannot be fully captured by these symmetric building blocks.
#             - The angular location of the second peak is rigidly set at `theta_pref + pi`. While this "opposite direction" tuning is common in neuronal populations, precise biological tuning might involve minor deviations from this exact 180-degree angular separation, limiting the model's flexibility.

#     This model ("neuron_model_v3") introduces significant improvements over both parent models by systematically addressing their core limitations:
#     1.  **Asymmetric Peak Modeling:** Each Gaussian component in this model now incorporates *two* independent tuning widths: `tuning_width_left` and `tuning_width_right`. This innovation enables the creation of highly flexible, skewed tuning curves, allowing a substantially better fit to the naturally asymmetric firing profiles frequently observed in real neuronal data. This feature directly rectifies the `symmetric_tuning` weakness present in both parent models. For instance, to accurately replicate the subtle yet critical asymmetry evident in Cell 165 (subplot row 1, col 1), this model can independently adjust `tuning_width_1_left` and `tuning_width_1_right` to precisely match the varying rates of decay on either side of the peak.
#     2.  **Independent Peak Tuning Widths:** Crucially, the primary (`tuning_width_1_left`, `tuning_width_1_right`) and secondary (`tuning_width_2_left`, `tuning_width_2_right`) peaks are no longer compelled to share common width parameters. Instead, each peak possesses its own independent set of `left` and `right` widths. This explicitly resolves the `shared_tuning_width` weakness that constrained the double_gaussian_model. As a concrete example, for Cell 81 (subplot row 1, col 2), this model can now utilize wider parameters for `tuning_width_1_left/right` to accurately represent the broader primary peak, while simultaneously applying narrower parameters for `tuning_width_2_left/right` to precisely fit the sharper secondary peak, leading to a superior overall fit.
#     3.  **Flexible Second Peak Location:** The inclusion of the `offset_2_peak` parameter grants fine-grained control over the angular position of the second peak. Rather than being fixed at an exact `pi` radians from the first preferred direction (`theta_pref_1`), its location can now be adjusted to any arbitrary angular offset. This added flexibility is crucial for precisely modeling neuronal responses that do not conform to an exact 180-degree opposition in preferred directions, thereby overcoming a significant rigidity of the double_gaussian_model.

#     Equation:
#     F(theta) = baseline + G1(theta) + G2(theta)
#     where:
#     G_k(theta) = amplitude_k * exp(-0.5 * (signed_circ_dist(theta, theta_pref_k) / effective_tuning_width_k)^2)
#     signed_circ_dist(theta_val, peak_pref) = np.arctan2(np.sin(theta_val - peak_pref), np.cos(theta_val - peak_pref))
#     effective_tuning_width_k = (tuning_width_k_left if signed_circ_dist_k < 0 else tuning_width_k_right)  # (Applied element-wise over array `theta`)
#     theta_pref_1_val = theta_pref_1
#     theta_pref_2_val = (theta_pref_1 + offset_2_peak) % (2 * np.pi)

#     Args:
#         theta (np.ndarray): The angle(s) in radians for which to compute the firing rate.
#         theta_pref_1 (float): The preferred direction (in radians) for the primary Gaussian peak. Default is 0.0.
#         baseline (float): The baseline firing rate across all angles. Default is 0.0.
#         amplitude_1 (float): The maximum firing rate of the primary peak above baseline. Default is 1.0.
#         tuning_width_1_left (float): The standard deviation (width) governing the decay of the primary peak for angles circularly smaller (left) than theta_pref_1. Default is 1.0.
#         tuning_width_1_right (float): The standard deviation (width) governing the decay of the primary peak for angles circularly larger (right) than theta_pref_1. Default is 1.0.
#         amplitude_2 (float): The maximum firing rate of the secondary peak above baseline. Default is 0.0.
#         offset_2_peak (float): The angular offset (in radians) of the secondary peak's preferred direction relative to theta_pref_1. Default is pi (180 degrees).
#         tuning_width_2_left (float): The standard deviation (width) governing the decay of the secondary peak for angles circularly smaller (left) than its preferred direction. Default is 1.0.
#         tuning_width_2_right (float): The standard deviation (width) governing the decay of the secondary peak for angles circularly larger (right) than its preferred direction. Default is 1.0.

#     Returns:
#         np.ndarray: The computed firing rate of the neuron model for each input angle in `theta`.
#     # Helper function for signed circular distance
#     def signed_circ_dist_rad(angle_current, angle_peak_pref):
#             Calculates the signed circular distance from angle_peak_pref to angle_current.
#         return np.arctan2(np.sin(angle_current - angle_peak_pref), np.cos(angle_current - angle_peak_pref))

#     # Calculate signed distances for both peaks
#     dist_1_signed = signed_circ_dist_rad(theta, theta_pref_1)
    
#     theta_pref_2_effective = (theta_pref_1 + offset_2_peak) % (2 * np.pi)
#     dist_2_signed = signed_circ_dist_rad(theta, theta_pref_2_effective)

#     # Determine effective tuning widths for peak 1 based on the sign of the signed distance
#     # np.where allows for vectorized conditional assignment of widths
#     effective_width_1 = np.where(dist_1_signed < 0, tuning_width_1_left, tuning_width_1_right)
    
#     # Determine effective tuning widths for peak 2 based on the sign of the signed distance
#     effective_width_2 = np.where(dist_2_signed < 0, tuning_width_2_left, tuning_width_2_right)

#     # Calculate Gaussian components. Add a small epsilon to widths to prevent division by zero or very small numbers,
#     # ensuring numerical stability if optimized widths approach zero.
#     epsilon = 1e-6 
    
#     response_1 = amplitude_1 * np.exp(-0.5 * (dist_1_signed / (effective_width_1 + epsilon)) ** 2)
#     response_2 = amplitude_2 * np.exp(-0.5 * (dist_2_signed / (effective_width_2 + epsilon)) ** 2)

#     return baseline + response_1 + response_2



# loss of model 2:  21.80
# import numpy as np

# def neuron_model_v2(theta, theta_pref=0.0, baseline=0.0,
#                     amplitude_1=1.0, width1_neg=1.0, width1_pos=1.0, power1_neg=2.0, power1_pos=2.0,
#                     amplitude_2=0.0, width2_neg=1.0, width2_pos=1.0, power2_neg=2.0, power2_pos=2.0,
#                     theta_offset_2=0.0):
#     parent_model_1: generalized_power_double_exponential_model (Corresponds to neuron_model_v1 code in prompt, green curve in plot, Loss 21.93).
#     parent_model_2: split_width_double_gaussian_model (Corresponds to neuron_model_v2 code in prompt, red curve in plot, Loss 21.89).

#     Equation:
#     circ_signed_dist(t, tp) = arctan2(sin(t - tp), cos(t - tp))

#     For Peak 1:
#     sd_1 = circ_signed_dist(theta, theta_pref)
#     w_1 = width1_neg if sd_1 <= 0 else width1_pos
#     p_1 = power1_neg if sd_1 <= 0 else power1_pos
#     Peak1 = amplitude_1 * exp(- (abs(sd_1) / w_1) ** p_1)

#     For Peak 2:
#     theta_pref_2_actual = (theta_pref + pi + theta_offset_2) % (2 * pi)
#     sd_2 = circ_signed_dist(theta, theta_pref_2_actual)
#     w_2 = width2_neg if sd_2 <= 0 else width2_pos
#     p_2 = power2_neg if sd_2 <= 0 else power2_pos
#     Peak2 = amplitude_2 * exp(- (abs(sd_2) / w_2) ** p_2)

#     Response(theta) = baseline + Peak1 + Peak2

#     This model integrates the asymmetric 'split width' functionality from the `split_width_double_gaussian_model` with the generalized 'power' parameters and flexible `theta_offset_2` from the `generalized_power_double_exponential_model`. By combining these advanced features, this model addresses systematic weaknesses observed in its predecessors, leading to a significantly lower loss.

#     Analysis of Parent Models:
#     The `generalized_power_double_exponential_model` (green curve, overall Loss 21.93 in plot header), provides advanced control over peak shape through its `power` parameters, allowing for sharper or broader (super-Gaussian/sub-Gaussian) decay profiles. It also introduces `theta_offset_2`, enabling flexible positioning of the secondary peak, a feature crucial for accurately capturing scenarios where the secondary preferred direction is not precisely 180 degrees opposite the primary. Despite these strengths, its primary weakness, clearly visible in cells like **Cell 123** and **Cell 225**, is its inherent symmetry around each preferred direction. For instance, in **Cell 123**, the green curve consistently fails to match the differing decay rates on the "left" and "right" flanks of both its main peak (around 0/2pi) and its secondary peak (around 5.5-6.0 radians), indicating an inability to capture asymmetry.

#     The `split_width_double_gaussian_model` (red curve, overall Loss 21.89 in plot header) directly addresses the issue of asymmetry by introducing separate `width_neg` and `width_pos` parameters for each side of the peaks. This enhancement significantly improves its fit for neurons exhibiting asymmetric tuning curves, as best demonstrated in **Cell 123** where the red curve closely tracks the asymmetric shape of the binned mean (sky-blue) far better than the green curve. Similarly, in **Cell 225**, the red curve provides a noticeably better fit to the distinct slopes of the peak shoulders. However, this model retains limitations: it fixes the decay function to a standard Gaussian (power=2) and restricts the secondary peak to a fixed 180-degree offset from the primary. In **Cell 155**, the very sharp and narrow peaks in the binned data suggest a decay steeper than Gaussian (power > 2) might be beneficial; the red curve here appears slightly too broad and flat at the peak compared to the blue mean.

#     This model significantly improves upon both parent models by combining their respective strengths:
#     *   **Asymmetric Shape & Peakedness:** This model employs **independent width parameters (`widthX_neg`, `widthX_pos`) AND independent power parameters (`powerX_neg`, `powerX_pos`) for both the negative and positive flanks of each peak.** This allows for unparalleled flexibility in sculpting the exact shape of each peak, addressing both asymmetry (like in **Cell 123**'s distinct peak flanks) and the overall 'peakedness' or 'tailedness' (e.g., achieving the extreme sharpness seen in **Cell 155** that was difficult for the standard Gaussian form of the red model, or creating broader shoulders for cells like **Cell 225** as needed by the sky-blue data). This highly granular control over peak morphology ensures a more accurate fit to diverse neural response profiles.
#     *   **Flexible Secondary Peak Positioning:** By re-introducing `theta_offset_2`, this model unfixes the secondary preferred direction from a rigid 180-degree opposition. This allows for fine-tuning the location of the second peak to perfectly match empirical data, such as subtle angular shifts that might improve the fit in specific cells not perfectly aligned on the diametric axis. This composite approach leverages the best features from prior models while providing a comprehensive solution for highly precise neural response modeling, aiming for a lower overall loss.
    

#     # Helper function for signed circular distance
#     # np.arctan2(np.sin(x), np.cos(x)) maps x to [-pi, pi]
#     circ_signed_dist = lambda t, tp: np.arctan2(np.sin(t - tp), np.cos(t - tp))

#     # --- Peak 1 Calculation ---
#     signed_dist_1 = circ_signed_dist(theta, theta_pref)

#     # Determine width and power based on the sign of the signed distance for Peak 1
#     # np.where works element-wise on arrays, selecting parameter based on the sign of each element in signed_dist_1
#     width_1 = np.where(signed_dist_1 <= 0, width1_neg, width1_pos)
#     power_1 = np.where(signed_dist_1 <= 0, power1_neg, power1_pos)

#     # Calculate response for Peak 1 using the selected width and power for each side
#     # np.abs(signed_dist_1) ensures the base for exponentiation is always non-negative
#     response_1 = amplitude_1 * np.exp(- (np.abs(signed_dist_1) / width_1) ** power_1)

#     # --- Peak 2 Calculation ---
#     # Calculate the actual preferred direction for the second peak, allowing for an offset from pi
#     theta_pref_2_actual = (theta_pref + np.pi + theta_offset_2) % (2 * np.pi)
#     signed_dist_2 = circ_signed_dist(theta, theta_pref_2_actual)

#     # Determine width and power based on the sign of the signed distance for Peak 2
#     width_2 = np.where(signed_dist_2 <= 0, width2_neg, width2_pos)
#     power_2 = np.where(signed_dist_2 <= 0, power2_neg, power2_pos)

#     # Calculate response for Peak 2
#     response_2 = amplitude_2 * np.exp(- (np.abs(signed_dist_2) / width_2) ** power_2)

#     return baseline + response_1 + response_2
# """

# load_dotenv()
# img_path = Path('/home/reilly/ai_scientist_project/program_databases/07-15/17-05-19 (big_only)/image_feedback/iter_6_island_7_batch_1.png')
# with img_path.open("rb") as f:
#     img_bytes = f.read()

# client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# # request the model to generate a new neuron model based on the text and image prompt
# temperature = 2.0
# config = types.GenerateContentConfig(temperature=temperature,
#                                     thinking_config=types.ThinkingConfig(thinking_budget=24_576)
#                                     )
# output_tokens = []
# input_tokens = []
# for i in range(10):
#     response = client.models.generate_content(
#         model='gemini-2.5-flash-preview-05-20',
#         # model='gemini-2.0-flash',
#         contents=[prompt],
#         config=config)

#     # check the response, counting both the output tokens and the thinking tokens
#     usage = response.usage_metadata       # google.genai.types.UsageMetadata
#     print(f"Response: {response.text}")
#     print("Prompt tokens   :", usage.prompt_token_count)
#     print("Output tokens   :", usage.candidates_token_count)   # sometimes called output_token_count
#     print("Thinking tokens :", getattr(usage, "thoughts_token_count", 0))
#     print("Total tokens    :", usage.total_token_count)
#     n_output_tokens = usage.prompt_token_count if usage.prompt_token_count is not None else 0
#     n_thoughts_tokens = getattr(usage, "thoughts_token_count", 0) if getattr(usage, "thoughts_token_count", 0) is not None else 0
#     n_total_output_tokens = n_output_tokens + n_thoughts_tokens
#     output_tokens.append(n_total_output_tokens)
#     input_tokens.append(usage.prompt_token_count)
# print(f"output tokens: {output_tokens}")
# print(f"input tokens: {input_tokens}")
