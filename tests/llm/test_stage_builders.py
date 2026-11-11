"""Stage builders bind generation arguments without owning timing."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from edgar.io.config import RetryConfig
from edgar.io.task_spec import LLMs
from edgar.llm.generate import build_generation_stages


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "names",
    [
        ["generate_models"],
        ["generate_models", "generate_param_ests"],
        ["generate_models", "translate_programs"],
        ["generate_models", "generate_param_ests", "translate_programs"],
    ],
)
async def test_build_generation_stages_binds_arguments(names):
    prompts = SimpleNamespace(model=object(), param_est=object(), jax_model=object())
    responses = SimpleNamespace(model=object(), param_est=object(), jax_model=object())
    spec = SimpleNamespace(prompt_schemas=prompts, response_schemas=responses)
    llms = LLMs(model="model", param_est="estimator", model_jax="translator")
    retry = RetryConfig()
    config = {"max_tokens": 123, "retry_config": retry}
    population, data = object(), {"x": object()}

    with (
        patch("edgar.llm.generate.generate_models", autospec=True) as models,
        patch("edgar.llm.generate.generate_param_ests", autospec=True) as estimators,
        patch("edgar.llm.generate.translate_programs", autospec=True) as translate,
    ):
        stages = build_generation_stages(names, spec, llms, "exploit", config, retry)
        assert [stage.name for stage in stages] == names
        models.assert_not_called()
        estimators.assert_not_called()
        translate.assert_not_called()

        for stage in stages:
            await stage.execute(population, data=data)

        models.assert_awaited_once_with(
            population,
            prompts.model,
            llms.model,
            "exploit",
            config=config,
            spec=spec,
            data=data,
            output_schema=responses.model,
        )
        if "generate_param_ests" in names:
            estimators.assert_awaited_once_with(
                population,
                prompts.param_est,
                llms.param_est,
                config,
                output_schema=responses.param_est,
            )
        else:
            estimators.assert_not_called()
        if "translate_programs" in names:
            translate.assert_awaited_once_with(
                population,
                prompts.jax_model,
                llms.model_jax,
                retry_config=retry,
                max_tokens=123,
                output_schema=responses.jax_model,
            )
        else:
            translate.assert_not_called()


def test_build_generation_stages_rejects_unknown_stage():
    with pytest.raises(ValueError, match="Unknown generation stage: unknown"):
        build_generation_stages(["unknown"], None, None, "explore", {}, RetryConfig())
