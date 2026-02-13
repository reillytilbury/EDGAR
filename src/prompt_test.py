import yaml
import re
from pathlib import Path
import pytest

class PromptValidator:
    def __init__(self, config_path="prompts.yaml", config: dict = None):
        """Initialize PromptValidator with either a config path or a pre-loaded config dict.
        
        Args:
            config_path: Path to config file (used if config is None)
            config: Pre-loaded config dict (takes precedence over config_path)
        """
        if config is not None:
            self.config = config
            self.config_path = None
        else:
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


class ExperimentValidator:
    def __init__(self, config_path="experiment.yaml"):
        self.config_path = Path(config_path)
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
    
    def test_yaml_is_valid(self):
        """Basic YAML syntax validation."""
        assert self.config is not None
        assert isinstance(self.config, dict)
        return True
    
    def test_required_fields(self):
        """Test that all required fields are present."""
        required_fields = ['task', 'seed_programs', 'experiment_params']
        missing = []
        
        for field in required_fields:
            if field not in self.config:
                missing.append(field)
        
        if missing:
            raise AssertionError(
                f"Missing required fields in experiment.yaml:\n" + 
                "\n".join(f"  - {f}" for f in missing)
            )
        
        return True
    
    def test_seed_programs_structure(self):
        """Test that seed_programs has the correct structure."""
        seed_programs = self.config.get('seed_programs', {})
        
        required = ['module', 'function_seeds', 'parameter_estimator_seeds']
        missing = []
        
        for field in required:
            if field not in seed_programs:
                missing.append(f"seed_programs.{field}")
        
        if missing:
            raise AssertionError(
                f"Missing required fields:\n" + 
                "\n".join(f"  - {f}" for f in missing)
            )
        
        # Check that seeds are lists
        if not isinstance(seed_programs.get('function_seeds', []), list):
            raise AssertionError("seed_programs.function_seeds must be a list")
        
        if not isinstance(seed_programs.get('parameter_estimator_seeds', []), list):
            raise AssertionError("seed_programs.parameter_estimator_seeds must be a list")
        
        return True
    
    def test_experiment_params_structure(self):
        """Test that experiment_params has the essential fields."""
        exp_params = self.config.get('experiment_params', {})
        
        essential_fields = [
            'n_iterations', 'time_limit', 'k_max', 'n_islands', 
            'batch_size', 'critical_population_size'
        ]
        missing = []
        
        for field in essential_fields:
            if field not in exp_params:
                missing.append(f"experiment_params.{field}")
        
        if missing:
            raise AssertionError(
                f"Missing essential fields:\n" + 
                "\n".join(f"  - {f}" for f in missing)
            )
        
        return True
    
    def test_experiment_params_types(self):
        """Test that experiment_params values have correct types."""
        exp_params = self.config.get('experiment_params', {})
        
        type_checks = {
            'n_iterations': int,
            'time_limit': (int, float),
            'k_max': int,
            'n_islands': int,
            'batch_size': int,
            'critical_population_size': int,
            'n_migrants': int,
            'fit_params': bool,
            'tol': float,
            'exploit_point': float,
            'param_penalty_weight': float,
            'FAILED_PROGRAM_COST': float,  # Must be float, not string!
            'use_image_feedback': bool,
            'use_param_estimator': bool,
        }
        
        errors = []
        for field, expected_type in type_checks.items():
            if field not in exp_params:
                continue  # Skip missing fields (handled by other test)
            
            value = exp_params[field]
            if not isinstance(value, expected_type):
                errors.append(
                    f"experiment_params.{field}: expected {expected_type.__name__ if isinstance(expected_type, type) else expected_type}, "
                    f"got {type(value).__name__} (value: {value!r})"
                )
        
        if errors:
            raise AssertionError(
                f"Type errors in experiment_params:\n" + 
                "\n".join(f"  - {e}" for e in errors)
            )
        
        return True
    
    def get_validation_report(self):
        """Run all tests and return a report."""
        tests = [
            ("YAML syntax", self.test_yaml_is_valid),
            ("Required fields", self.test_required_fields),
            ("Seed programs structure", self.test_seed_programs_structure),
            ("Experiment params structure", self.test_experiment_params_structure),
            ("Experiment params types", self.test_experiment_params_types),
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


class DataValidator:
    def __init__(self, config_path="data.yaml"):
        self.config_path = Path(config_path)
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
    
    def test_yaml_is_valid(self):
        """Basic YAML syntax validation."""
        assert self.config is not None
        assert isinstance(self.config, dict)
        return True
    
    def test_required_fields(self):
        """Test that all required fields are present."""
        required_fields = ['task', 'load_data']
        missing = []
        
        for field in required_fields:
            if field not in self.config:
                missing.append(field)
        
        if missing:
            raise AssertionError(
                f"Missing required fields in data.yaml:\n" + 
                "\n".join(f"  - {f}" for f in missing)
            )
        
        return True
    
    def test_load_data_format(self):
        """Test that load_data is in the correct format (module.function)."""
        load_data = self.config.get('load_data', '')
        
        if not isinstance(load_data, str):
            raise AssertionError("load_data must be a string")
        
        if not load_data:
            raise AssertionError("load_data cannot be empty")
        
        # Check that it looks like a module path
        if '.' not in load_data:
            raise AssertionError(
                f"load_data should be in format 'module.path.function', got: {load_data}"
            )
        
        return True
    
    def get_validation_report(self):
        """Run all tests and return a report."""
        tests = [
            ("YAML syntax", self.test_yaml_is_valid),
            ("Required fields", self.test_required_fields),
            ("Load data format", self.test_load_data_format),
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


def test_experiment_yaml():
    """Test function for experiment.yaml."""
    validator = ExperimentValidator("experiment.yaml")
    validator.test_yaml_is_valid()
    validator.test_required_fields()
    validator.test_seed_programs_structure()
    validator.test_experiment_params_structure()


def test_data_yaml():
    """Test function for data.yaml."""
    validator = DataValidator("data.yaml")
    validator.test_yaml_is_valid()
    validator.test_required_fields()
    validator.test_load_data_format()


# Standalone script
if __name__ == "__main__":
    print("=" * 60)
    print("CONFIGURATION VALIDATOR")
    print("=" * 60)
    print()
    
    all_configs_passed = True
    
    # Test prompts.yaml
    print("Testing prompts.yaml...")
    print("-" * 60)
    try:
        config_dir = Path(__file__).parent
        validator = PromptValidator(config_dir / "prompts.yaml")
        report, passed = validator.get_validation_report()
        print(report)
        all_configs_passed = all_configs_passed and passed
    except FileNotFoundError:
        print("✗ ERROR: prompts.yaml not found")
        all_configs_passed = False
    except yaml.YAMLError as e:
        print(f"✗ ERROR: Invalid YAML syntax\n\n{e}")
        all_configs_passed = False
    except Exception as e:
        print(f"✗ UNEXPECTED ERROR: {e}")
        all_configs_passed = False
    
    print()
    
    # Test experiment.yaml
    print("Testing experiment.yaml...")
    print("-" * 60)
    try:
        validator = ExperimentValidator(config_dir / "experiment.yaml")
        report, passed = validator.get_validation_report()
        print(report)
        all_configs_passed = all_configs_passed and passed
    except FileNotFoundError:
        print("✗ ERROR: experiment.yaml not found")
        all_configs_passed = False
    except yaml.YAMLError as e:
        print(f"✗ ERROR: Invalid YAML syntax\n\n{e}")
        all_configs_passed = False
    except Exception as e:
        print(f"✗ UNEXPECTED ERROR: {e}")
        all_configs_passed = False
    
    print()
    
    # Test data.yaml
    print("Testing data.yaml...")
    print("-" * 60)
    try:
        validator = DataValidator(config_dir / "data.yaml")
        report, passed = validator.get_validation_report()
        print(report)
        all_configs_passed = all_configs_passed and passed
    except FileNotFoundError:
        print("✗ ERROR: data.yaml not found")
        all_configs_passed = False
    except yaml.YAMLError as e:
        print(f"✗ ERROR: Invalid YAML syntax\n\n{e}")
        all_configs_passed = False
    except Exception as e:
        print(f"✗ UNEXPECTED ERROR: {e}")
        all_configs_passed = False
    
    print()
    print("=" * 60)
    
    if all_configs_passed:
        print("✓ ALL TESTS PASSED")
        exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the issues above and run again.")
        exit(1)