import yaml
import re
import pandas as pd
from pathlib import Path

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
            # Lazy import avoids pulling pytest into unrelated subprocesses.
            from .prompt_test import PromptValidator
            validator = PromptValidator(config=self.config)
            try:
                validator.test_required_prompts_exist()
                validator.test_no_invalid_variables()
            except AssertionError as e:
                raise ValueError(f"Invalid prompt configuration:\n{e}")

        self.prompts = self.config['prompts']

    @staticmethod
    def _normalize_template_text(text: str) -> str:
        """
        Normalize escaped newlines from YAML strings and trim edges.
        """
        if text is None:
            return ""
        return str(text).replace("\\n", "\n").strip()

    def _render_template(self, template_text: str, **kwargs) -> str:
        """
        Format a template section and normalize whitespace.
        """
        normalized = self._normalize_template_text(template_text)
        if not normalized:
            return ""
        if "{model_name}" in normalized and "model_name" not in kwargs:
            kwargs["model_name"] = self.get_model_name()
        return normalized.format(**kwargs)

    @staticmethod
    def _format_loss(train_loss) -> str:
        try:
            return f"{float(train_loss):.2f}"
        except Exception:
            return str(train_loss)

    @staticmethod
    def _format_seconds(value) -> str:
        try:
            return f"{float(value):.2f}s"
        except Exception:
            return "n/a"

    @staticmethod
    def _normalize_model_code_for_prompt(code_string: str, model_name: str, model_idx: int) -> str:
        if not code_string:
            return code_string
        lines = code_string.splitlines()
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("def "):
                indent = line[:len(line) - len(stripped)]
                open_paren = stripped.find("(")
                if open_paren != -1:
                    lines[i] = f"{indent}def {model_name}_v{model_idx}{stripped[open_paren:]}"
                    return "\n".join(lines)
        replacement = f"def {model_name}_v{model_idx}("
        return re.sub(r"def\\s+\\w+\\s*\\(", replacement, code_string, count=1)

    def _render_image_analysis(self, templates: dict, prompt_key: str, k: int, use_image: bool) -> str:
        """
        Render image-analysis prompt section or raise a clear error if missing.
        """
        if not use_image:
            return ""
        image_analysis = templates.get('image_analysis')
        if image_analysis is None or str(image_analysis).strip() == "":
            raise ValueError(
                "Image diagnostics are enabled, but prompt template "
                f"`prompts.{prompt_key}.image_analysis` is missing.\n"
                "This run is attaching diagnostic plots to the LLM call, so the prompt must "
                "explain how to interpret the image.\n"
                f"Add `prompts.{prompt_key}.image_analysis` to your config."
            )
        return self._render_template(image_analysis, k=f"{k}", next_version=f"{k+1}")

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
        max_lines = 100
        next_version = f"{k+1}"
        sections = [
            self._render_template(templates['base'], k=f"{k}", next_version=next_version),
            self._render_template(templates['exploit'] if mode == 'exploit' else templates['explore'], k=f"{k}", next_version=next_version),
            self._render_image_analysis(templates, "program_prompt", k, use_image),
            self._render_template(templates['code_guidelines'], max_lines=f"{max_lines}"),
            self._render_template(templates['docstring_guidelines'], next_version=next_version),
            "**Parent Models:**",
        ]
        prompt = "\n\n".join([s for s in sections if s])

        for i in range(k):
            model_idx = i + 1 
            train_loss = programs_df.iloc[i]['train_loss']
            optimization_time_s = programs_df.iloc[i].get('optimization_time_s')
            program_code_string = self._normalize_model_code_for_prompt(
                programs_df.iloc[i]['program_code_string'],
                model_name,
                model_idx,
            )
            per_model_prompt = self._render_template(
                templates['per_model_detail'],
                model_idx=f"{model_idx}",
                train_loss=self._format_loss(train_loss),
                optimization_time_s=self._format_seconds(optimization_time_s),
                program_code_string=program_code_string,
            )
            prompt += "\n\n" + per_model_prompt
        
        return prompt
    
    def get_parameter_estimator_prompt(self, programs_df : pd.DataFrame, model_code_string : str, max_lines : int = 100) -> str:
        """Build full parameter estimator prompt from config (non-chat mode).
        
        This creates a self-contained prompt with all guidelines included.
        Use this when NOT using chat mode.

        Args :
            programs_df (pd.DataFrame): DataFrame of existing parameter estimators.
            model_code_string (str): The code string of the model to be used.
            max_lines (int): Maximum number of lines for the generated code.
        Returns:
            prompt (str): The full prompt string for the AI to generate a new parameter estimator.
        """
        k = len(programs_df)
        model_name = self.get_model_name()
        templates = self.config['prompts']['parameter_estimator']
        next_version = f"{k+1}"
        sections = [
            self._render_template(templates.get('generation_base', templates.get('base', '')), k=f"{k}", next_version=next_version),
            self._render_template(templates['code_guidelines'], max_lines=f"{max_lines}"),
            self._render_template(templates['docstring_guidelines'], next_version=next_version),
            "**Parent Models and Estimators:**",
        ]
        prompt = "\n\n".join([s for s in sections if s])

        for i in range(k):
            model_idx = i + 1 
            train_loss = programs_df.iloc[i]['train_loss']
            optimization_time_s = programs_df.iloc[i].get('optimization_time_s')
            program_code_string = self._normalize_model_code_for_prompt(
                programs_df.iloc[i]['program_code_string'],
                model_name,
                model_idx,
            )
            parameter_estimator_code_string = programs_df.iloc[i]['parameter_estimator_code_string'].replace('def parameter_estimator(', f'def parameter_estimator_v{model_idx}(')
            per_model_prompt = self._render_template(
                templates['per_model_detail'],
                model_idx=f"{model_idx}",
                train_loss=self._format_loss(train_loss),
                optimization_time_s=self._format_seconds(optimization_time_s),
                program_code_string=program_code_string,
                parameter_estimator_code_string=parameter_estimator_code_string,
            )
            prompt += "\n\n" + per_model_prompt

        # add the model code string to the prompt
        prompt += "\n\n**New Model:**\n\n" + model_code_string + "\n"

        return prompt

    def get_linked_prompt_legacy(self, programs_df: pd.DataFrame, mode: str, use_image: bool = True, max_lines: int = 120) -> str:
        """
        Build a single prompt that asks for both a new model and a new parameter estimator.

        This is a separate prompt family from program_prompt/parameter_estimator.
        """
        assert mode in ['explore', 'exploit'], "Invalid mode. Choose either 'explore' or 'exploit'."

        k = len(programs_df)
        model_name = self.get_model_name()
        templates = self.config['prompts'].get('linked_prompt')
        if not templates:
            raise ValueError("Missing prompts.linked_prompt in config.")

        next_version = f"{k+1}"
        sections = [
            self._render_template(templates['base'], k=f"{k}", next_version=next_version),
            self._render_template(templates['exploit'] if mode == 'exploit' else templates['explore'], k=f"{k}", next_version=next_version),
            self._render_image_analysis(templates, "linked_prompt", k, use_image),
            self._render_template(templates['code_guidelines'], max_lines=f"{max_lines}", next_version=next_version),
            self._render_template(templates['docstring_guidelines'], next_version=next_version),
            "**Parent Models and Estimators:**",
        ]
        prompt = "\n\n".join([s for s in sections if s])

        for i in range(k):
            model_idx = i + 1
            train_loss = programs_df.iloc[i]['train_loss']
            optimization_time_s = programs_df.iloc[i].get('optimization_time_s')
            program_code_string = self._normalize_model_code_for_prompt(
                programs_df.iloc[i]['program_code_string'],
                model_name,
                model_idx,
            )
            parameter_estimator_code_string = programs_df.iloc[i]['parameter_estimator_code_string'].replace(
                'def parameter_estimator(', f'def parameter_estimator_v{model_idx}('
            )
            per_model_prompt = self._render_template(
                templates['per_model_detail'],
                model_idx=f"{model_idx}",
                train_loss=self._format_loss(train_loss),
                optimization_time_s=self._format_seconds(optimization_time_s),
                program_code_string=program_code_string,
                parameter_estimator_code_string=parameter_estimator_code_string,
            )
            prompt += "\n\n" + per_model_prompt

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
            prompt_parts.append(self._render_template(templates['exploit'], k=f"{k}", next_version=f"{k+1}"))
        else:        
            prompt_parts.append(self._render_template(templates['explore'], k=f"{k}", next_version=f"{k+1}"))
        
        # Image analysis if applicable
        image_analysis = self._render_image_analysis(templates, "program_prompt", k, use_image)
        if image_analysis:
            prompt_parts.append(image_analysis)
        
        # Parent model details
        prompt_parts.append("**Parent Models:**")
        for i in range(k):
            model_idx = i + 1 
            train_loss = programs_df.iloc[i]['train_loss']
            program_code_string = self._normalize_model_code_for_prompt(
                programs_df.iloc[i]['program_code_string'],
                model_name,
                model_idx,
            )
            per_model_prompt = self._render_template(
                templates['per_model_detail'],
                model_idx=f"{model_idx}",
                train_loss=self._format_loss(train_loss),
                program_code_string=program_code_string,
            )
            prompt_parts.append(per_model_prompt)
        
        return "\n\n".join([p for p in prompt_parts if p])
    
    def get_parameter_estimator_prompt_chat(self, programs_df : pd.DataFrame, model_code_string : str, max_lines : int = 100) -> str:
        """Build parameter estimator prompt for chat mode (dynamic content only).
        
        This creates a shorter prompt containing only the dynamic parts.
        Static guidelines are assumed to be in the system instruction.
        Use this when using chat mode with IslandChatManager.

        Args :
            programs_df (pd.DataFrame): DataFrame of existing parameter estimators.
            model_code_string (str): The code string of the model to be used.
            max_lines (int): Maximum number of lines for the generated code.
        Returns:
            prompt (str): The dynamic prompt for generating a new parameter estimator.
        """
        k = len(programs_df)
        model_name = self.get_model_name()
        templates = self.config['prompts']['parameter_estimator']
        
        prompt_parts = []
        
        # Brief context line
        prompt_parts.append(f"Now create parameter_estimator_v{k+1} for the new {model_name} below.")

        # Parent model details
        prompt_parts.append("**Parent Models and Estimators:**")
        for i in range(k):
            model_idx = i + 1 
            train_loss = programs_df.iloc[i]['train_loss']
            program_code_string = self._normalize_model_code_for_prompt(
                programs_df.iloc[i]['program_code_string'],
                model_name,
                model_idx,
            )
            parameter_estimator_code_string = programs_df.iloc[i]['parameter_estimator_code_string'].replace('def parameter_estimator(', f'def parameter_estimator_v{model_idx}(')
            per_model_prompt = self._render_template(
                templates['per_model_detail'],
                model_idx=f"{model_idx}",
                train_loss=self._format_loss(train_loss),
                program_code_string=program_code_string,
                parameter_estimator_code_string=parameter_estimator_code_string,
            )
            prompt_parts.append(per_model_prompt)

        # Add the new model code string
        prompt_parts.append(f"**New {model_name} to create parameter estimator for:**")
        prompt_parts.append(model_code_string)

        return "\n\n".join([p for p in prompt_parts if p])

    def get_parameter_estimator_refinement_prompt(
        self,
        programs_df: pd.DataFrame,
        model_code_string: str,
        max_lines: int,
        refine_round: int,
        refine_rounds: int,
        current_loss: float,
    ) -> str:
        """Build refinement prompt for parameter estimator updates (chat mode)."""
        templates = self.config['prompts']['parameter_estimator']
        prompt_parts = []
        base = self._render_template(templates.get('base', ''), next_version="2")
        if base:
            prompt_parts.append(base)
        image_instructions = self._render_template(templates.get('refinement_image_instructions', ''))
        if image_instructions:
            stripped = image_instructions.lstrip()
            if stripped.startswith("**"):
                image_block = image_instructions
            else:
                image_block = "**Image analysis instructions:**\n" + image_instructions
            prompt_parts.append(image_block)
        prompt_parts.append(self._render_template(templates['code_guidelines'], max_lines=f"{max_lines}"))
        if not model_code_string or not str(model_code_string).strip():
            raise ValueError("Refinement prompt requires current model code string.")
        prompt_parts.append("**Current model:**")
        prompt_parts.append(model_code_string)
        if programs_df is not None and len(programs_df) > 0:
            row = programs_df.iloc[0]
            prev_loss = self._format_loss(row.get('train_loss', current_loss))
            prev_est = row.get('parameter_estimator_code_string', '').replace(
                'def parameter_estimator(', 'def parameter_estimator_prev('
            )
            prompt_parts.append("**Previous parameter estimator:**")
            prompt_parts.append(f"Previous estimator loss: {prev_loss}")
            if prev_est:
                prompt_parts.append(prev_est)
        else:
            raise ValueError("Refinement prompt requires previous parameter estimator code.")
        return "\n\n".join([p for p in prompt_parts if p])

    def get_parameter_estimator_refinement_prompt_legacy(
        self,
        programs_df: pd.DataFrame,
        model_code_string: str,
        max_lines: int,
        refine_round: int,
        refine_rounds: int,
        current_loss: float,
    ) -> str:
        """Build refinement prompt for parameter estimator updates (legacy mode)."""
        templates = self.config['prompts']['parameter_estimator']
        prompt_parts = []
        base = self._render_template(templates.get('base', ''), k="1", next_version="2")
        if base:
            prompt_parts.append(base)
        image_instructions = self._render_template(templates.get('refinement_image_instructions', ''))
        if image_instructions:
            stripped = image_instructions.lstrip()
            if stripped.startswith("**"):
                image_block = image_instructions
            else:
                image_block = "**Image analysis instructions:**\n" + image_instructions
            prompt_parts.append(image_block)
        prompt_parts.append(self._render_template(templates['code_guidelines'], max_lines=f"{max_lines}"))
        if not model_code_string or not str(model_code_string).strip():
            raise ValueError("Refinement prompt requires current model code string.")
        prompt_parts.append("**Current model:**")
        prompt_parts.append(model_code_string)
        if programs_df is not None and len(programs_df) > 0:
            row = programs_df.iloc[0]
            prev_loss = self._format_loss(row.get('train_loss', current_loss))
            prev_est = row.get('parameter_estimator_code_string', '').replace(
                'def parameter_estimator(', 'def parameter_estimator_prev('
            )
            prompt_parts.append("**Previous parameter estimator:**")
            prompt_parts.append(f"Previous estimator loss: {prev_loss}")
            if prev_est:
                prompt_parts.append(prev_est)
        else:
            raise ValueError("Refinement prompt requires previous parameter estimator code.")
        return "\n\n".join([p for p in prompt_parts if p])

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
            self._render_template(program_templates['code_guidelines'], max_lines="100"),
            program_templates['docstring_guidelines'].format(next_version="N+1"),
            "\n# PARAMETER ESTIMATOR GUIDELINES",
            param_est_templates['base'].format(k="N", next_version="N+1"),
            self._render_template(param_est_templates['code_guidelines'], max_lines="100"),
            param_est_templates['docstring_guidelines'].format(next_version="N+1"),
        ]
        
        return "\n\n".join(system_parts)
