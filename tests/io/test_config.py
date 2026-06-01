import pytest
from edgar.io.config import Config
from pydantic import ValidationError
from edgar.io.task_spec import TaskSpec

def test_load_perfect_config():
    """
        Tests that a config with all fields specified loads correctly, and that all fields are correctly read from the config file.
    """
    config = Config.from_yaml("tests/io/test_configs/perfect/config.yaml", default_path = "tests/io/test_configs/perfect/default_config.yaml")
    #All fields should be from perfect/config.yaml
    assert config.io.data_path == ""
    assert config.io.save_path == ""
    assert config.evolution.n_generations == 12
    assert config.evolution.n_islands == 8
    assert config.evolution.batch_size == 6
    assert config.evolution.critical_population_size == 12
    assert config.evolution.n_migrants == 2
    assert config.evolution.topology == [1, 2, 3, 4, 5, 6, 7, 0]
    assert config.llms.num_parents == 2
    assert config.llms.retry.max_retries == 3
    assert config.llms.retry.initial_delay == 1.0
    assert config.llms.retry.backoff_multiplier == 2.0
    assert config.llms.retry.max_delay == 60.0
    assert config.llms.retry.retryable_status_codes == [500, 503]
    assert config.llms.model_llm == 3*["gemini-2.5-flash"]
    assert config.llms.param_est_llm == "gemini-2.5-flash"
    assert config.llms.jax_model_translator_llm == "gemini-2.5-flash-lite"
    assert not config.llms.log_raw_llm_response
    assert config.llms.max_tokens == 10000
    assert config.llms.max_lines == 50
    assert config.llms.swear_words == ['lstsq', 'scipy.optimize', 'optimize.minimize', 'curve_fit', 'sklearn']
    assert config.scoring.param_penalty_weight == 0.01
    assert config.scoring.timeout_s == 120.0
    assert config.scoring.gradient_descent.max_iter == 1000
    assert config.scoring.gradient_descent.learning_rate == 0.01
    #All prompt fields should be from perfect/prompts.yaml
    assert config.prompts.model.base == 'default_base1'
    assert config.prompts.model.explore == 'default_explore1'
    assert config.prompts.model.exploit == 'default_exploit1'
    assert config.prompts.model.code_guidelines == 'default_code_guidelines1'
    assert config.prompts.model.docstring_guidelines == 'default_docstring_guidelines1'
    assert config.prompts.model.image_analysis_instructions is None
    assert config.prompts.model.parent_program_template == 'default_program_detail_template1'
    assert config.prompts.parameter_estimator.base == 'default_param_est_base1'
    assert config.prompts.parameter_estimator.explore is None
    assert config.prompts.parameter_estimator.exploit is None
    assert config.prompts.parameter_estimator.code_guidelines == 'default_param_est_code_guidelines1'
    assert config.prompts.parameter_estimator.docstring_guidelines == 'default_param_est_docstring_guidelines1'
    assert config.prompts.parameter_estimator.image_analysis_instructions is None
    assert config.prompts.parameter_estimator.parent_program_template == 'default_param_est_program_detail_template1'
    assert config.prompts.jax_translator_model.base == 'default_jax_base1'
    assert config.prompts.jax_translator_model.explore is None
    assert config.prompts.jax_translator_model.exploit is None
    assert config.prompts.jax_translator_model.code_guidelines == 'default_jax_code_guidelines1'
    assert config.prompts.jax_translator_model.docstring_guidelines == 'default_jax_docstring_guidelines1'
    assert config.prompts.jax_translator_model.image_analysis_instructions is None
    assert config.prompts.jax_translator_model.parent_program_template == 'default_jax_program_detail_template1'

def test_load_missing_field_config_perfect_default():
    """
        Tests that if a config is missing a field, the missing values are filled in from the default.
    """
    config = Config.from_yaml("tests/io/test_configs/missing_field/config.yaml", default_path = "tests/io/test_configs/perfect/default_config.yaml")
    assert config.io.data_path == "a"
    assert config.io.save_path == "b"

def test_load_missing_field_config_missing_default():
    """
        Tests that if a config is missing a field, and the default is also missing the field, an error is raised.
    """
    with pytest.raises(ValidationError):
        config = Config.from_yaml("tests/io/test_configs/missing_field/config.yaml", default_path = "tests/io/test_configs/missing_field/default_config.yaml")

def test_load_missing_subfield_config_perfect_default():
    """
        Tests that if a config is missing a subfield, the missing value is filled in from the default.
    """
    config = Config.from_yaml("tests/io/test_configs/missing_subfield/config.yaml", default_path = "tests/io/test_configs/perfect/default_config.yaml")
    assert config.evolution.n_generations == 10

def test_load_missing_subfield_config_missing_default():
    """
        Tests that if a config is missing a subfield, and the default is also missing the subfield. 
    """
    with pytest.raises(ValidationError):
        config = Config.from_yaml("tests/io/test_configs/missing_subfield/config.yaml", default_path = "tests/io/test_configs/missing_subfield/default_config.yaml")

def test_load_invalid_subfield_config_perfect_default():
    """
        Tests that if a config has an invalid subfield, the error is raised, even though the default has a valid value.
    """
    with pytest.raises(ValidationError):
        config = Config.from_yaml("tests/io/test_configs/invalid_subfield/config.yaml", default_path = "tests/io/test_configs/perfect/default_config.yaml")

def test_load_invalid_topology_config():
    """
        Tests that if a config has an invalid topology, the error is raised, even though the default has a valid topology
    """
    with pytest.raises(ValidationError):
        config = Config.from_yaml("tests/io/test_configs/invalid_topology/config_invalidset.yaml", default_path = "tests/io/test_configs/perfect/default_config.yaml")
    with pytest.raises(ValidationError):
        config = Config.from_yaml("tests/io/test_configs/invalid_topology/config_invalidlength.yaml", default_path = "tests/io/test_configs/perfect/default_config.yaml")

def test_load_additional_field():
    """
        Tests that if a config has an additional field, a warning is raised, printing out the unexpected field.
    """
    with pytest.warns(UserWarning, match="additional_field"):
        config = Config.from_yaml("tests/io/test_configs/additional_field/config.yaml", default_path = "tests/io/test_configs/perfect/default_config.yaml")

def test_load_additional_subfield():
    """
        Tests that if a config has an additional subfield, a warning is raised, printing out the unexpected subfield.
    """
    with pytest.warns(UserWarning, match="additional_subfield"):
        config = Config.from_yaml("tests/io/test_configs/additional_subfield/config.yaml", default_path = "tests/io/test_configs/perfect/default_config.yaml")

def test_load_missing_prompt_field():
    """
        Tests that if a config is missing a prompt field, the missing value is filled in from the default.
    """
    config = Config.from_yaml("tests/io/test_configs/missing_field/config.yaml", default_path = "tests/io/test_configs/perfect/default_config.yaml")
    assert config.prompts.jax_translator_model.base == 'default_jax_base'

def test_load_missing_prompt_field_no_default():
    """
        Tests that if a config is missing a prompt field, and the default is also missing the field, an error is raised.
    """
    with pytest.raises(ValidationError):
        config = Config.from_yaml("tests/io/test_configs/missing_field/config.yaml", default_path = "tests/io/test_configs/missing_field/prompt_defaults.yaml")

def test_load_missing_prompt_subfield():
    """
        Tests that if a config is missing a prompt subfield, the missing value is filled in from the default.
    """
    config = Config.from_yaml("tests/io/test_configs/missing_subfield/config.yaml", default_path = "tests/io/test_configs/perfect/default_config.yaml")
    assert config.prompts.model.code_guidelines == 'default_code_guidelines'

def test_load_missing_prompt_subfield_no_default():
    """
        Tests that if a config is missing a prompt subfield, and the default is also missing the subfield, an error is raised.
    """
    with pytest.raises(ValidationError):
        config = Config.from_yaml("tests/io/test_configs/missing_subfield/config.yaml", default_path = "tests/io/test_configs/missing_subfield/prompt_defaults.yaml")

def test_load_invalid_prompt_subfield():
    """
        Tests that if a config has an invalid prompt subfield, an error is raised, even if the default has a valid value.
    """
    with pytest.raises(ValidationError):
        config = Config.from_yaml("tests/io/test_configs/invalid_subfield/config.yaml", default_path = "tests/io/test_configs/perfect/default_config.yaml")


def test_config_taskspec_round_trip(tmp_path):
    """
        Tests that the config can be converted to a TaskSpec and back without losing any information.
    """
    config = Config.from_yaml("tests/io/test_task/config.yaml")
    taskspec = TaskSpec.from_config(config)
    saved_taskspec = taskspec.save(tmp_path)
    config_from_taskspec = Config.from_taskspec(saved_taskspec)
    a, b = config.model_dump(), config_from_taskspec.model_dump()
    assert a == b, "\n".join(
        f"  {k}: {a[k]!r} != {b[k]!r}" for k in a if a[k] != b[k]
    )
