import yaml
import re
from pathlib import Path
import pytest

class PromptValidator:
    def __init__(self, config_path="prompts.yaml"):
        self.config_path = Path(config_path)
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.metadata = self.config.get('_metadata', {})
        self.prompts = self.config.get('prompts', {})
        

    def _get_nested_value(self, data, path):
        """Get value from nested dict using dot notation path."""
        keys = path.split('.')
        value = data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def _get_allowed_variables_for_path(self, prompt_path):
        """
        Get allowed variables for a prompt path.
        Supports hierarchical lookup with inheritance from _default.
        
        Example paths:
          - "prompts.program_prompt.gemini.explore" 
          - "prompts.parameter_estimator.claude.base"
        """
        # Remove 'prompts.' prefix if present
        if prompt_path.startswith('prompts.'):
            prompt_path = prompt_path[8:]  # len('prompts.') = 8
        
        path_parts = prompt_path.split('.')
        allowed_vars_config = self.metadata.get('allowed_variables', {})
        
        # Traverse the path, collecting allowed variables
        current = allowed_vars_config
        collected_vars = set()
        
        for i, part in enumerate(path_parts):
            if not isinstance(current, dict):
                break
            
            # Check for _default at this level
            if '_default' in current and isinstance(current['_default'], list):
                collected_vars.update(current['_default'])
            
            # Move to next level
            if part in current:
                next_level = current[part]
                
                # If it's a list, we've reached the leaf - use these variables
                if isinstance(next_level, list):
                    collected_vars = set(next_level)  # Override with specific
                    break
                else:
                    current = next_level
            else:
                # Path doesn't exist in allowed_variables
                # Use whatever we've collected so far (from _default)
                break
        
        return list(collected_vars)

  
    def test_required_prompts_exist(self):
        """Test that all required prompts are present in the config."""
        required = self.metadata.get('required_prompts', [])
        missing = []
        
        for prompt_path in required:
            value = self._get_nested_value(self.config, prompt_path)
            if value is None:
                missing.append(prompt_path)
        
        if missing:
            raise AssertionError(
                f"Missing required prompts:\n" + 
                "\n".join(f"  - {p}" for p in missing)
            )
        
        return True
    
    def test_no_invalid_variables(self):
        """Test that prompts only use allowed variables."""
        errors = []
        
        def check_prompt(prompt_text, prompt_path):
            """Extract variables from prompt and check if allowed."""
            if not isinstance(prompt_text, str):
                return
            
            # Find all {variable} patterns
            used_vars = re.findall(r'\{(\w+)\}', prompt_text)
            
            if not used_vars:
                return  # No variables used
            
            # Get allowed variables for this specific path
            allowed = self._get_allowed_variables_for_path(prompt_path)
            
            if not allowed:
                # No allowed variables defined for this path
                errors.append(
                    f"{prompt_path}:\n"
                    f"  Uses variables {used_vars} but no allowed_variables defined for this path"
                )
                return
            
            # Check for invalid variables
            invalid = [v for v in used_vars if v not in allowed]
            
            if invalid:
                errors.append(
                    f"{prompt_path}:\n"
                    f"  Invalid variables: {invalid}\n"
                    f"  Allowed variables: {allowed}\n"
                    f"  Used variables: {used_vars}"
                )
        
        def traverse_prompts(data, path="prompts"):
            """Recursively traverse prompt structure."""
            if isinstance(data, str):
                check_prompt(data, path)
            elif isinstance(data, dict):
                for key, value in data.items():
                    new_path = f"{path}.{key}"
                    traverse_prompts(value, new_path)
        
        traverse_prompts(self.prompts)
        
        if errors:
            raise AssertionError(
                "Invalid variables found in prompts:\n\n" + 
                "\n\n".join(errors)
            )
        
        return True        
    
  
    def test_optional_prompts_documented(self):
        """Verify that optional prompts are clearly marked in metadata."""
        optional = self.metadata.get('optional_prompts', [])
        
        if not optional:
            print("Warning: No optional prompts specified in metadata.")
            print("Consider adding _metadata.optional_prompts section.")
        
        return True
    
    def test_yaml_is_valid(self):
        """Basic YAML syntax validation."""
        assert self.config is not None
        assert isinstance(self.config, dict)
        return True
    
    def test_metadata_section_exists(self):
        """Check that metadata section exists with proper structure."""
        assert '_metadata' in self.config, "Missing _metadata section"
        
        required_meta_keys = ['required_prompts', 'allowed_variables']
        for key in required_meta_keys:
            assert key in self.metadata, f"Missing _metadata.{key}"
        
        return True
    
    def get_validation_report(self):
        """Run all tests and return a report."""
        tests = [
            ("YAML syntax", self.test_yaml_is_valid),
            ("Metadata structure", self.test_metadata_section_exists),
            ("Required prompts", self.test_required_prompts_exist),
            ("Variable validation", self.test_no_invalid_variables),
            ("Optional prompts", self.test_optional_prompts_documented),
        ]
        
        results = []
        all_passed = True
        
        for test_name, test_func in tests:
            try:
                test_func()
                results.append(f"✓ {test_name}: PASSED")
            except AssertionError as e:
                results.append(f"✗ {test_name}: FAILED\n  {str(e)}")
                all_passed = False
            except Exception as e:
                results.append(f"✗ {test_name}: ERROR\n  {str(e)}")
                all_passed = False
        
        return "\n".join(results), all_passed


# Pytest integration
def test_prompts_yaml():
    """Main test function for pytest."""
    validator = PromptValidator("prompts.yaml")
    validator.test_yaml_is_valid()
    validator.test_metadata_section_exists()
    validator.test_required_prompts_exist()
    validator.test_no_invalid_variables()
    validator.test_optional_prompts_documented()


# Standalone script
if __name__ == "__main__":
    print("=" * 60)
    print("PROMPT CONFIGURATION VALIDATOR")
    print("=" * 60)
    print()
    
    try:
        validator = PromptValidator("prompts.yaml")
        report, passed = validator.get_validation_report()
        
        print(report)
        print()
        print("=" * 60)
        
        if passed:
            print("✓ ALL TESTS PASSED")
            exit(0)
        else:
            print("✗ SOME TESTS FAILED")
            print("\nPlease fix the issues above and run again.")
            exit(1)
    
    except FileNotFoundError:
        print("✗ ERROR: prompts.yaml not found")
        print("\nPlease ensure prompts.yaml exists in the current directory.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"✗ ERROR: Invalid YAML syntax\n\n{e}")
        exit(1)
    except Exception as e:
        print(f"✗ UNEXPECTED ERROR: {e}")
        exit(1)