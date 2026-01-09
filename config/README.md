# prompts.yaml : Customizing Prompts

1. Copy `prompts.yaml` to `prompts_custom.yaml`
2. Edit `prompts_custom.yaml` with your preferred text editor
3. Modify the text while keeping the structure intact
4. Variables like {k+1} will be automatically filled in

## Variables Available:
- {k} - number of existing models
- {next_version} - k+1
- {max_lines} - line limit for code
- {current_program_code_string} - current program in str

## experiment.yaml : Specifying seed programs (function + parameter_estimator)
1. Create a seed_programs.py in your specific task folder 
2. Update the paths 