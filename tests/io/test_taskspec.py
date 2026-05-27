from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from edgar.llm.prompt_schema import PromptSchema


def test_fromconfig():
    """
    Tests that a TaskSpec can be created from a perfect config
    """
    config = Config.from_yaml("tests/system/test_task/config.yaml")
    taskspec = TaskSpec.from_config(config)
    assert taskspec.task_name == "test_task"
    # assert taskspec.git_sha == "abc123"
    # assert taskspec.git_dirty == False
    # Check config parameters correctly loaded in
    assert taskspec.run == {"random_seed": 42}
    assert taskspec.io == {"data_path": "", "save_path": ""}
    assert taskspec.evolution == {
        "n_generations": 2,
        "n_islands": 2,
        "batch_size": 3,
        "critical_population_size": 3,
        "n_migrants": 1,
        "topology": [1, 0],
    }
    assert taskspec.llms == {
        "num_parents": 2,
        "retry": {
            "max_retries": 3,
            "initial_delay": 10.0,
            "backoff_multiplier": 2.0,
            "max_delay": 60.0,
            "retryable_status_codes": [500, 502, 503, 504],
        },
        "model_llm": "gemini-2.5-flash-lite",
        "param_est_llm": "gemini-2.5-flash-lite",
        "jax_model_translator_llm": "gemini-2.5-flash-lite",
        "log_raw_llm_response": False,
        "max_tokens": 10000,
        "max_lines": 50,
        "swear_words": [
            "lstsq",
            "scipy.optimize",
            "optimize.minimize",
            "curve_fit",
            "sklearn",
        ],
    }
    assert taskspec.scoring == {
        "param_penalty_weight": 0.01,
        "timeout_s": 120.0,
        "gradient_descent": {"max_iter": 100, "learning_rate": 0.01},
    }

    # Check prompt schemas correctly loaded in
    assert isinstance(taskspec.model_prompt_schema, PromptSchema)
    assert taskspec.model_prompt_schema.base == "test_base"
    assert taskspec.model_prompt_schema.explore == "test_explore"
    assert taskspec.model_prompt_schema.exploit == "test_exploit"
    assert taskspec.model_prompt_schema.code_guidelines == "test_code_guidelines"
    assert (
        taskspec.model_prompt_schema.docstring_guidelines == "test_docstring_guidelines"
    )
    assert taskspec.model_prompt_schema.image_analysis_instructions is None
    assert (
        taskspec.model_prompt_schema.program_detail_template
        == "test_program_detail_template"
    )
    assert isinstance(taskspec.param_est_prompt_schema, PromptSchema)
    assert taskspec.param_est_prompt_schema.base == "test_param_est_base"
    assert taskspec.param_est_prompt_schema.explore is None
    assert taskspec.param_est_prompt_schema.exploit is None
    assert (
        taskspec.param_est_prompt_schema.code_guidelines
        == "test_param_est_code_guidelines"
    )
    assert (
        taskspec.param_est_prompt_schema.docstring_guidelines
        == "test_param_est_docstring_guidelines"
    )
    assert taskspec.param_est_prompt_schema.image_analysis_instructions is None
    assert (
        taskspec.param_est_prompt_schema.program_detail_template
        == "test_param_est_program_detail_template"
    )
    assert isinstance(taskspec.jax_model_prompt_schema, PromptSchema)
    assert taskspec.jax_model_prompt_schema.base == "test_jax_base"
    assert taskspec.jax_model_prompt_schema.explore is None
    assert taskspec.jax_model_prompt_schema.exploit is None
    assert (
        taskspec.jax_model_prompt_schema.code_guidelines == "test_jax_code_guidelines"
    )
    assert (
        taskspec.jax_model_prompt_schema.docstring_guidelines
        == "test_jax_docstring_guidelines"
    )
    assert taskspec.jax_model_prompt_schema.image_analysis_instructions is None
    assert (
        taskspec.jax_model_prompt_schema.program_detail_template
        == "test_jax_program_detail_template"
    )

    assert callable(taskspec.load_data_fn)
    assert callable(taskspec.loss_fn)
