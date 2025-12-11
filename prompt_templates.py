"""Utility functions for assembling reusable LLM prompt preambles."""

from __future__ import annotations

def _validate_mode(mode: str) -> None:
    if mode not in {"explore", "exploit"}:
        raise ValueError(f"Unsupported mode '{mode}'. Expected 'explore' or 'exploit'.")

def program_creation_instructions(k: int, mode: str, function_name: str) -> str:
    """Prompt creation preamble for program generation tasks.
    Args:
        k: number of parent programs
        mode: either "explore" or "exploit"
        function_name: name of the function being optimized
    Returns:
        prompt guidance string
    """
    _validate_mode(mode)
    prompt = f"""
You are an AI scientist. The programs below are biological models describing system behaviour. The models are sorted from highest to lowest loss.

Your task is to create a new {function_name}, {function_name}_v{k+1}, that has a lower loss than the models below.

*Analyze* the progression of the models, *generalize* the improvements, and *create* a new model that is better than *all* previous models.

"""
    if mode == "explore":
        prompt += """
Use the models below as inspiration, but be *creative* and *invent* something new. Which features in the models below correlate with lower loss? Find these features and *extrapolate* them. You should also *combine* features from several models, and *experiment* with new ideas.
"""
    else:
        prompt += """
Use the models below as a *template* to create a new model.
Which features in the models below correlate with lower loss? Find these features and *extrapolate* them.
Focus on *exploiting* the strengths of the existing models and *eliminating* their weaknesses or *redundancies*.
Are the parameter ranges correct? If not, adjust them to be more appropriate.
You will be *penalized* for complexity, so make the new model as *simple* as possible while still being better than the previous models.
"""
    return prompt

def image_analysis_instructions(k: int, function_name: str) -> str:
    """ Image analysis guidance for program generation tasks.
    Args:
        k: number of parent programs
        function_name: name of the function being optimized
    Returns:
        prompt guidance string"""
    return f"""
**Image Analysis Instructions:**

Attached is a scatter plot of the models' performance on top of raw data. The binned mean is plotted in **sky-blue**, `{function_name}_v1` is plotted in **green**, and `{function_name}_v2` is plotted in **red**.

Analyse the models' fits to the data in the image below. Identify systematic weaknesses of the models by observing patterns across multiple cell plots. For instance, consider:
*   **Model Comparisons:** Which models are better for each cell? Which features of the models are responsible for improving the fit?
*   **Model Fit:** How well do the models fit the binned data mean? Look for places where even the models (red curve for best model, green for second best model) deviate most from the binned data mean (blue curve).
*   **Model Shape:** Do the models' shapes (e.g., peak sharpness, width, skewness, amplitude, etc.) align with the binned data mean and raw data scatter points? If not, how do they differ? How can you change the model to better match the data shape?
*   **Parameter Flexibility:** Are there free parameters that could be introduced or modified to better capture the observed response profiles? Utilize your analysis of the shortcomings and add or modify free parameters to address these issues.

Use this analysis to inform the design of a new {function_name}, `{function_name}_v{k+1}`, that improves upon the previous models.
Include your analysis of the image in the docstring of your new model. Point to specific subplots in the image that illustrate the *strengths* and *weaknesses* of the parent models. Explain how you plan to **fix** the weaknesses of the parent models.
"""

def coding_instructions(k: int, function_name: str) -> str:
    """ Docstring guidelines for program generation tasks.
    Args:
        k: number of parent programs
        function_name: name of the function being optimized
    Returns:
        prompt guidance string"""
    return f"""
**Code Generation Guidelines:**

* Import any packages you use.
* Do not include any text other than the code.
* Ensure all free parameters are numeric, not strings.
* At the beginning of the code, clip the free parameters to a plausible range, e.g., `A = np.clip(A, 0, None)`.

**Docstring Guidelines:**
* Begin by listing the parent models and give them a name that describes their key features, e.g., `parent_model_1: simple_exponential_decay-model`. Never refer to the models as `{function_name}_v1`, `{function_name}_v2`, etc. Instead, refer to them as `parent_models` or descriptive names.
* Do not refer to the current model as `{function_name}_v{k+1}`. Instead, refer to it as "this model".
* Provide a simple equation for the model, including all free parameters.
* Include a brief description of how the model improves upon the previous models, citing specific features or changes that lead to lower loss.

"""

def parameter_estimator_creation_instructions(k: int, function_name: str, max_lines: int) -> str:
    return f"""
You are an AI scientist. Your task is to create a simple parameter estimator function, `parameter_estimator_v{k+1}`, to estimate the free parameters of the latest model, `{function_name}_v{k+1}`.

The parameters should be estimated directly, using statistical principles and knowledge of what the parameters represent in context.

*Analyze* the progression of the parameter estimators, *generalize* the improvements, and *create* a new parameter estimator that is better than *all* previous estimators.

**Code Generation Guidelines:**
* Import any packages you use.
* Do not include any text other than the code.
* The only arguments to the function should be the stimuli and the spike count.
* Your response **must** be less than {max_lines} lines (including imports). If it is longer, it will be immediately rejected. 
* Do not attempt to fit the parameters using complex fitting functions like `curve_fit`, `least_squares` or `minimize`. This function should be a simple starting point for the parameter estimation.
"""

def jax_translation_instructions(program: str, function_name: str) -> str:
    return f"""Convert the following function (`{function_name}`) to a JAX-compatible function.

Include all necessary imports, and ensure that the function is compatible with JAX transformations like `jax.jit`, `jax.grad`, and `jax.vmap`.

Do not include any text other than the code. 

Here is the code to translate:

{program}
"""