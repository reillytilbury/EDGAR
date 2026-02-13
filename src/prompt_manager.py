import yaml
import pandas as pd
from pathlib import Path
from .prompt_test import PromptValidator

class PromptManager:
    def __init__(self, config_path="config.yaml", config: dict = None, validate=True):
        """Initialize PromptManager with either a config path or a pre-loaded config dict.
        
        Args:
            config_path: Path to config file (used if config is None)
            config: Pre-loaded config dict (takes precedence over config_path)
            validate: Whether to validate prompt configuration
        """
        if config is not None:
            self.config = config
        else:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        
        if validate:
            validator = PromptValidator(config=self.config)
            try:
                validator.test_required_prompts_exist()
                validator.test_no_invalid_variables()
            except AssertionError as e:
                raise ValueError(f"Invalid prompt configuration:\n{e}")

        self.prompts = self.config['prompts']

    def _load_config(self, path):
        with open(path) as f:
            return yaml.safe_load(f)

    def get_model_name(self):
        """Get the model name from config. Raise error if not found."""
        model_name = self.config.get('model_name')        
        if model_name is None:
            raise ValueError("Model name not found in config. Add 'model_name: your_model_name' at the top level of prompts.yaml")
        
        return model_name
    
    # ---------------------------------------------------------------
    # Legacy prompt functions (full self-contained prompts)
    # ---------------------------------------------------------------
    
    def get_program_prompt_legacy(self, programs_df : pd.DataFrame, mode : str, use_image=True) -> str:
        """Build full program generation prompt from config (legacy mode).
        
        This creates a self-contained prompt with all guidelines included.
        Use this when NOT using chat mode.
        
        Args :
            programs_df (pd.DataFrame): DataFrame of existing programs.
            mode (str): 'exploit' or 'explore'.
            use_image (bool): Whether to include image analysis section.

        Returns:
            prompt (str): The full prompt string for the AI to generate a new model.
        """
        # Ensure the mode is valid
        assert mode in ['explore', 'exploit'], "Invalid mode. Choose either 'explore' or 'exploit'."

        k = len(programs_df)
        model_name = self.get_model_name()
        templates = self.config['prompts']['program_prompt']

        # Format with variables
        prompt = templates['base'].format(k=f"{k}", next_version=f"{k+1}")
        max_lines = 100

        if mode == 'exploit':
            prompt = prompt + templates['exploit'].format(k=f"{k}", next_version=f"{k+1}")
        else:        
            prompt = prompt + templates['explore'].format(k=f"{k}", next_version=f"{k+1}")
        
        # Add optional sections
        if use_image:
            prompt += templates['image_analysis'].format(k=f"{k}", next_version=f"{k+1}")
        
        prompt += templates['code_guidelines'].format(max_lines=f"{max_lines}")
        prompt += templates['function_signature'].format(next_version=f"{k+1}")
        prompt += templates['docstring_guidelines'].format(next_version=f"{k+1}")
        
        for i in range(k):
            model_idx = i + 1 
            train_loss = programs_df.iloc[i]['train_loss']
            program_code_string = programs_df.iloc[i]['program_code_string'].replace(f'def {model_name}(', f'def {model_name}_v{model_idx}(')
            per_model_prompt = templates['per_model_detail'].format(model_idx=f"{model_idx}", train_loss=f'{train_loss}: .2f', program_code_string=program_code_string)
            prompt += per_model_prompt
        
        return prompt
    
    def get_parameter_estimator_prompt_legacy(self, programs_df : pd.DataFrame, model_code_string : str, max_lines : int = 100, use_image : bool = True) -> str:
        """Build full parameter estimator prompt from config (legacy mode).
        
        This creates a self-contained prompt with all guidelines included.
        Use this when NOT using chat mode.

        Args :
            programs_df (pd.DataFrame): DataFrame of existing parameter estimators.
            model_code_string (str): The code string of the model to be used.
            max_lines (int): Maximum number of lines for the generated code.
            use_image (bool): Whether to include image analysis section.

        Returns:
            prompt (str): The full prompt string for the AI to generate a new parameter estimator.
        """
        k = len(programs_df)
        model_name = self.get_model_name()
        templates = self.config['prompts']['parameter_estimator']
        prompt = templates['base'].format(k=f"{k}", next_version=f"{k+1}")

        if use_image:
            prompt += templates['image_analysis'].format(k=f"{k}", next_version=f"{k+1}")
        
        prompt += templates['code_guidelines'].format(max_lines=f"{max_lines}")
        prompt += templates['function_signature'].format(next_version=f"{k+1}")
        prompt += templates['docstring_guidelines'].format(next_version=f"{k+1}")

        for i in range(k):
            model_idx = i + 1 
            train_loss = programs_df.iloc[i]['train_loss']
            program_code_string = programs_df.iloc[i]['program_code_string'].replace(f'def {model_name}(', f'def {model_name}_v{model_idx}(')
            parameter_estimator_code_string = programs_df.iloc[i]['parameter_estimator_code_string'].replace('def parameter_estimator(', f'def parameter_estimator_v{model_idx}(')
            per_model_prompt = templates['per_model_detail'].format(model_idx=f"{model_idx}", train_loss=f'{train_loss}: .2f', program_code_string=program_code_string, parameter_estimator_code_string=parameter_estimator_code_string)
            prompt += per_model_prompt

        # add the model code string to the prompt
        prompt += model_code_string + "\n"

        return prompt

    # ---------------------------------------------------------------
    # Chat mode prompt functions (dynamic content only)
    # ---------------------------------------------------------------
    
    def get_program_prompt(self, programs_df : pd.DataFrame, mode : str, use_image=True) -> str:
        """Build program generation prompt for chat mode (dynamic content only).
        
        This creates a shorter prompt containing only the dynamic parts.
        Static guidelines are assumed to be in the system instruction.
        Use this when using chat mode with IslandChatManager.
        
        Args :
            programs_df (pd.DataFrame): DataFrame of existing programs.
            mode (str): 'exploit' or 'explore'.
            use_image (bool): Whether to include image analysis section.

        Returns:
            prompt (str): The dynamic prompt for generating a new model.
        """
        assert mode in ['explore', 'exploit'], "Invalid mode. Choose either 'explore' or 'exploit'."

        k = len(programs_df)
        model_name = self.get_model_name()
        templates = self.config['prompts']['program_prompt']
        
        prompt_parts = []
        
        # Brief context line using model_name from config
        prompt_parts.append(f"Generate {model_name}_v{k+1} based on the {k} parent models below.")

        # Mode-specific instructions
        if mode == 'exploit':
            prompt_parts.append(templates['exploit'].format(k=f"{k}", next_version=f"{k+1}"))
        else:        
            prompt_parts.append(templates['explore'].format(k=f"{k}", next_version=f"{k+1}"))
        
        # Image analysis if applicable
        if use_image:
            prompt_parts.append(templates['image_analysis'].format(k=f"{k}", next_version=f"{k+1}"))
        
        # Parent model details
        prompt_parts.append("\n**Parent Models:**\n")
        for i in range(k):
            model_idx = i + 1 
            train_loss = programs_df.iloc[i]['train_loss']
            program_code_string = programs_df.iloc[i]['program_code_string'].replace(f'def {model_name}(', f'def {model_name}_v{model_idx}(')
            per_model_prompt = templates['per_model_detail'].format(model_idx=f"{model_idx}", train_loss=f'{train_loss}: .2f', program_code_string=program_code_string)
            prompt_parts.append(per_model_prompt)
        
        return "\n".join(prompt_parts)
    
    def get_parameter_estimator_prompt(self, programs_df : pd.DataFrame, model_code_string : str, max_lines : int = 100, use_image : bool = True) -> str:
        """Build parameter estimator prompt for chat mode (dynamic content only).
        
        This creates a shorter prompt containing only the dynamic parts.
        Static guidelines are assumed to be in the system instruction.
        Use this when using chat mode with IslandChatManager.

        Args :
            programs_df (pd.DataFrame): DataFrame of existing parameter estimators.
            model_code_string (str): The code string of the model to be used.
            max_lines (int): Maximum number of lines for the generated code.
            use_image (bool): Whether to include image analysis section.

        Returns:
            prompt (str): The dynamic prompt for generating a new parameter estimator.
        """
        k = len(programs_df)
        model_name = self.get_model_name()
        templates = self.config['prompts']['parameter_estimator']
        
        prompt_parts = []
        
        # Brief context line
        prompt_parts.append(f"Now create parameter_estimator_v{k+1} for the new {model_name} below.")

        if use_image:
            prompt_parts.append(templates['image_analysis'].format(k=f"{k}", next_version=f"{k+1}"))

        # Parent model details
        prompt_parts.append("\n**Parent Models and Estimators:**\n")
        for i in range(k):
            model_idx = i + 1 
            train_loss = programs_df.iloc[i]['train_loss']
            program_code_string = programs_df.iloc[i]['program_code_string'].replace(f'def {model_name}(', f'def {model_name}_v{model_idx}(')
            parameter_estimator_code_string = programs_df.iloc[i]['parameter_estimator_code_string'].replace('def parameter_estimator(', f'def parameter_estimator_v{model_idx}(')
            per_model_prompt = templates['per_model_detail'].format(model_idx=f"{model_idx}", train_loss=f'{train_loss}: .2f', program_code_string=program_code_string, parameter_estimator_code_string=parameter_estimator_code_string)
            prompt_parts.append(per_model_prompt)

        # Add the new model code string
        prompt_parts.append(f"\n**New {model_name} to create parameter estimator for:**\n")
        prompt_parts.append(model_code_string)

        return "\n".join(prompt_parts)

    def get_jax_translator_prompt(self, function_code):
        template = self.config['prompts']['jax_translator_prompt']
        return template.format(function_code=function_code)

    def get_system_instruction(self, mode: str = 'explore') -> str:
        """
        Build the system instruction for chat-based LLM sessions.
        
        This contains the static guidelines that don't change per query:
        - Role description for model generation
        - Mode-specific guidance (explore vs exploit)
        - Code guidelines
        - Function signature requirements
        - Docstring guidelines
        - Parameter estimator guidelines
        
        Args:
            mode: Either 'explore' or 'exploit'. Determines the strategy guidance.
        
        Returns:
            str: The system instruction for the chat session.
        """
        assert mode in ['explore', 'exploit'], f"Invalid mode: {mode}. Choose 'explore' or 'exploit'."
        
        program_templates = self.config['prompts']['program_prompt']
        param_est_templates = self.config['prompts']['parameter_estimator']
        
        # Build system instruction from static sections
        model_name = self.get_model_name()
        system_parts = [
            f"# {model_name.upper()} GENERATION GUIDELINES",
            program_templates['base'].format(k="N", next_version="N+1"),
            f"\n# CURRENT MODE: {mode.upper()}",
            program_templates[mode].format(k="N", next_version="N+1"),
            program_templates['code_guidelines'].format(max_lines="100"),
            program_templates['function_signature'].format(next_version="N+1"),
            program_templates['docstring_guidelines'].format(next_version="N+1"),
            "\n# PARAMETER ESTIMATOR GUIDELINES",
            param_est_templates['base'].format(k="N", next_version="N+1"),
            param_est_templates['code_guidelines'].format(max_lines="100"),
            param_est_templates['function_signature'].format(next_version="N+1"),
            param_est_templates['docstring_guidelines'].format(next_version="N+1"),
        ]
        
        return "\n\n".join(system_parts)