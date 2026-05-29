from pathlib import Path

import numpy as np
from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from edgar.llm.prompt_schema import PromptSchema


def test_fromconfig():
    """
    Tests that a TaskSpec can be created from a perfect config
    """
    config = Config.from_yaml("tests/io/test_task/config.yaml")
    taskspec = TaskSpec.from_config(config)
    assert taskspec.task_name == "test_task"
    assert taskspec.project_dir == Path("tests/io/test_task").resolve()
    # Check config parameters correctly loaded in
    assert taskspec.run == {"random_seed": 42}
    assert taskspec.io == {"data_path": "", "save_path": ""}
    assert taskspec.evolution == {
        "n_generations": 4,
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
        "model_llm": ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
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
        taskspec.model_prompt_schema.parent_program_template
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
        taskspec.param_est_prompt_schema.parent_program_template
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
        taskspec.jax_model_prompt_schema.parent_program_template
        == "test_jax_program_detail_template"
    )

    assert callable(taskspec.load_data_fn)
    assert callable(taskspec.loss_fn)
    assert callable(taskspec.plot_fn)
    seed_model_src = (Path("tests/io/test_task/seed_programs/model1.py")).read_text()
    assert taskspec.seed_programs[0].code.model in seed_model_src
    seed_model_src = (Path("tests/io/test_task/seed_programs/model2.py")).read_text()
    assert taskspec.seed_programs[1].code.model in seed_model_src
    assert taskspec.rng.integers(0, 2**31) == np.random.default_rng(42).integers(0, 2**31) #check rng correctly seeded with 42 (see config.yaml)

def test_fromconfig_no_plot_fn():
    """Tests that plot_fn is None when image_feedback/plot.py is absent."""
    config = Config.from_yaml("tests/io/test_task_no_image/config.yaml")
    taskspec = TaskSpec.from_config(config)
    assert taskspec.plot_fn is None

def test_schedule_model_list():
    """
        Tests that the schedule produces the expected mode, temperature and model_llms for each generation
    """
    config = Config.from_yaml("tests/io/test_task/config.yaml")
    taskspec = TaskSpec.from_config(config)
    n_generations = taskspec.evolution["n_generations"]
    expected_modes = 2*["explore"] + 2*["exploit"]
    expected_model_llms = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"]
    for gen in range(n_generations):
        mode, temperature, llms = taskspec.schedule(gen)
        assert mode == expected_modes[gen]
        assert temperature == 1 + np.exp(-gen / n_generations)
        assert llms.model == expected_model_llms[gen]
        assert llms.param_est == "gemini-2.5-flash-lite"
        assert llms.model_jax == "gemini-2.5-flash-lite"

def test_schedule_single_model():
    """
        Tests that the schedule produces the expected mode, temperature and model_llms for each generation
    """
    config = Config.from_yaml("tests/io/test_task/config.yaml")
    config.llms.model_llm = "gemini-2.5-flash-lite" #Override with single model instead of list
    taskspec = TaskSpec.from_config(config)
    n_generations = taskspec.evolution["n_generations"]
    expected_modes = 2*["explore"] + 2*["exploit"]
    for gen in range(n_generations):
        mode, temperature, llms = taskspec.schedule(gen)
        assert mode == expected_modes[gen]
        assert temperature == 1 + np.exp(-gen / n_generations)
        assert llms.model == "gemini-2.5-flash-lite"
        assert llms.param_est == "gemini-2.5-flash-lite"
        assert llms.model_jax == "gemini-2.5-flash-lite"