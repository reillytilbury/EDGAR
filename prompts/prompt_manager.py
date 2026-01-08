import yaml
import pandas as pd
from pathlib import Path

class PromptManager:
    def __init__(self, config_path="prompts.yaml", validate=True):
        if validate:
            validator = PromptValidator(config_path)
            try:
                validator.test_required_prompts_exist()
                validator.test_no_invalid_variables()
            except AssertionError as e:
                raise ValueError(f"Invalid prompt configuration:\n{e}")

        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.prompts = self.config['prompts']

    def _load_config(self, path):
        with open(path) as f:
            return yaml.safe_load(f)
    
    def get_program_prompt(self, programs_df : pd.DataFrame, mode : str , use_image=True) -> str:
        """Build program generation prompt from config.
        
        Args :
            programs_df (pd.DataFrame): DataFrame of existing programs.
            mode (str): 'exploit' or 'explore'.
            use_image (bool): Whether to include image analysis section.

        Returns:
            prompt (str): The prompt string for the AI to generate a new neuron model.

        """
        # Ensure the mode is valid
        assert mode in ['explore', 'exploit'], "Invalid mode. Choose either 'explore' or 'exploit'."

        k = len(programs_df)
        templates = self.config['prompts']['program_prompt']

        # Format with variables
        prompt = templates['base'].format(k=k, next_version=k+1)

        if mode == 'exploit':
            prompt = prompt + templates['exploit'].format(k=k, next_version=k+1)
        else:        
            prompt = prompt + templates['explore'].format(k=k, next_version=k+1)
        
        # Add optional sections
        if use_image:
            prompt += templates['image_analysis'].format(k=k, next_version=k+1)
        
        prompt += templates['code_guidelines'].format(max_lines=100)
        prompt += templates['docstring_guidelines'].format(next_version=k+1)
        
        for i in range(k):
            model_idx = i + 1 
            train_loss = programs_df.iloc[i]['train_loss']
            program_code_string = programs_df.iloc[i]['program_code_string'].replace('def neuron_model(', f'def neuron_model_v{model_idx}(')
            per_model_prompt = templates['per_model_detail'].format(model_idx=model_idx, train_loss=f'{train_loss}: .2f', program_code_string=program_code_string)
            prompt += per_model_prompt
        
        return prompt
    
    def get_parameter_estimator_prompt(self, programs_df : pd.DataFrame, neuron_model_code_string : str, max_lines : int = 100, use_image : bool = True) -> str:
        k = len(programs_df)
        templates = self.config['prompts']['parameter_estimator']
        prompt = templates[llm_type].format(k=k, next_version=k+1)

        if use_image:
            prompt += templates['image_analysis'].format(k=k, next_version=k+1)
        
        prompt += templates['code_guidelines'].format(max_lines=max_lines)
        prompt += templates['docstring_guidelines'].format(next_version=k+1)

        for i in range(k):
            model_idx = i + 1 
            train_loss = programs_df.iloc[i]['train_loss']
            program_code_string = programs_df.iloc[i]['program_code_string'].replace('def neuron_model(', f'def neuron_model_v{model_idx}(')
            parameter_estimator_code_string = programs_df.iloc[i]['parameter_estimator_code_string'].replace('def parameter_estimator(', f'def parameter_estimator_v{model_idx}(')
            per_model_prompt = templates['per_model_detail'].format(model_idx=model_idx, train_loss=f'{train_loss}: .2f', program_code_string=program_code_string, parameter_estimator_code_string=parameter_estimator_code_string)
            prompt += per_model_prompt

        # add the neuron model code string to the prompt
        prompt += neuron_model_code_string + "\n"

        return prompt

    def get_jax_translator_prompt(self, function_code):
        template = self.config['prompts']['jax_translator_prompt']
        return template.format(function_code=function_code)