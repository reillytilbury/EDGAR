import re
from pathlib import Path

import pandas as pd
import yaml


def with_default_prompts(config: dict) -> dict:
    if config.get("prompts"):
        return config
    defaults = Path(__file__).resolve().parents[2] / "projects" / "prompt_defaults.yaml"
    if defaults.exists():
        merged = dict(config)
        with defaults.open() as handle:
            merged["prompts"] = yaml.safe_load(handle) or {}
        return merged
    return config


def get_model_name(config: dict) -> str:
    model_name = config.get("model_name")
    if model_name is None:
        raise ValueError("Config must include top-level `model_name`.")
    return model_name


def _text(value) -> str:
    if isinstance(value, dict):
        value = value.get("text")
    if value is None:
        return ""
    return str(value).replace("\\n", "\n").strip()


def _render(template, config: dict, **kwargs) -> str:
    text = _text(template)
    if not text:
        return ""
    if "{model_name}" in text and "model_name" not in kwargs:
        kwargs["model_name"] = get_model_name(config)
    return text.format(**kwargs)


def _section(config: dict, old_name: str, new_name: str | None = None) -> dict:
    prompts = config.get("prompts", {})
    if old_name in prompts:
        return prompts[old_name] or {}
    if new_name and new_name in prompts:
        section = prompts[new_name] or {}
        return {
            "base": section.get("base", ""),
            "explore": section.get("explore", ""),
            "exploit": section.get("exploit", ""),
            "code_guidelines": section.get("code_guidelines", ""),
            "docstring_guidelines": section.get("docstring_guidelines", ""),
            "image_analysis": section.get("image_analysis_instructions", ""),
            "per_model_detail": section.get("parent_detail_template", ""),
        }
    return {}


def _format_loss(value) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def _format_seconds(value) -> str:
    try:
        return f"{float(value):.2f}s"
    except Exception:
        return "n/a"


def _version_model_code(code_string: str, model_name: str, model_idx: int) -> str:
    if not code_string:
        return code_string
    lines = str(code_string).splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("def "):
            indent = line[: len(line) - len(stripped)]
            open_paren = stripped.find("(")
            if open_paren != -1:
                lines[i] = f"{indent}def {model_name}_v{model_idx}{stripped[open_paren:]}"
                return "\n".join(lines)
    return re.sub(r"def\s+\w+\s*\(", f"def {model_name}_v{model_idx}(", str(code_string), count=1)


def build_model_prompt(config: dict, programs_df: pd.DataFrame, mode: str, use_image: bool) -> str:
    assert mode in ["explore", "exploit"]
    config = with_default_prompts(config)
    templates = _section(config, "model")
    k = len(programs_df)
    next_version = f"{k + 1}"
    image = templates.get("image_analysis")
    parts = [
        _render(templates.get("base", ""), config, k=f"{k}", next_version=next_version),
        _render(templates.get(mode, ""), config, k=f"{k}", next_version=next_version),
        _render(image, config, k=f"{k}", next_version=next_version) if use_image else "",
        _render(templates.get("code_guidelines", ""), config, max_lines="100", next_version=next_version),
        _render(templates.get("docstring_guidelines", ""), config, next_version=next_version),
        "**Parent Models:**",
    ]
    prompt = "\n\n".join([p for p in parts if p])
    detail = templates.get("per_model_detail", "")
    model_name = get_model_name(config)
    for i in range(k):
        row = programs_df.iloc[i]
        prompt += "\n\n" + _render(
            detail,
            config,
            model_idx=f"{i + 1}",
            train_loss=_format_loss(row["train_loss"]),
            loss=_format_loss(row["train_loss"]),
            optimization_time_s=_format_seconds(row.get("optimization_time_s")),
            program_code_string=_version_model_code(row["program_code_string"], model_name, i + 1),
        )
    return prompt


def build_parameter_estimator_prompt(
    config: dict,
    programs_df: pd.DataFrame,
    model_code_string: str,
    max_lines: int = 100,
) -> str:
    config = with_default_prompts(config)
    templates = _section(config, "parameter_estimator")
    k = len(programs_df)
    next_version = f"{k + 1}"
    parts = [
        _render(templates.get("generation_base", templates.get("base", "")), config, k=f"{k}", next_version=next_version),
        _render(templates.get("code_guidelines", ""), config, max_lines=f"{max_lines}", next_version=next_version),
        _render(templates.get("docstring_guidelines", ""), config, next_version=next_version),
        "**Parent Models and Estimators:**",
    ]
    prompt = "\n\n".join([p for p in parts if p])
    detail = templates.get("per_model_detail", "")
    model_name = get_model_name(config)
    for i in range(k):
        row = programs_df.iloc[i]
        prompt += "\n\n" + _render(
            detail,
            config,
            model_idx=f"{i + 1}",
            train_loss=_format_loss(row["train_loss"]),
            loss=_format_loss(row["train_loss"]),
            optimization_time_s=_format_seconds(row.get("optimization_time_s")),
            program_code_string=_version_model_code(row["program_code_string"], model_name, i + 1),
            parameter_estimator_code_string=str(row.get("parameter_estimator_code_string", "")).replace(
                "def parameter_estimator(", f"def parameter_estimator_v{i + 1}("
            ),
        )
    return f"{prompt}\n\n**New Model:**\n\n{model_code_string}\n"


def build_parameter_estimator_refinement_prompt(
    config: dict,
    programs_df: pd.DataFrame,
    model_code_string: str,
    max_lines: int,
    current_loss: float,
) -> str:
    config = with_default_prompts(config)
    templates = _section(config, "parameter_estimator")
    parts = [
        _render(templates.get("base", ""), config, k="1", next_version="2"),
        _render(templates.get("refinement_image_instructions", ""), config),
        _render(templates.get("code_guidelines", ""), config, max_lines=f"{max_lines}", next_version="next"),
        "**Current model:**",
        model_code_string,
    ]
    if programs_df is not None and len(programs_df) > 0:
        row = programs_df.iloc[0]
        parts.extend(
            [
                "**Previous parameter estimator:**",
                f"Previous estimator loss: {_format_loss(row.get('train_loss', current_loss))}",
                str(row.get("parameter_estimator_code_string", "")).replace(
                    "def parameter_estimator(", "def parameter_estimator_prev("
                ),
            ]
        )
    return "\n\n".join([p for p in parts if p])


def build_jax_translator_prompt(config: dict, function_code: str) -> str:
    config = with_default_prompts(config)
    prompts = config.get("prompts", {})
    template = prompts.get("jax_translator_prompt")
    if template is None:
        section = prompts.get("jax_translator", {})
        template = "\n\n".join(
            [
                _text(section.get("base", "")),
                _text(section.get("code_guidelines", "")),
                _text(section.get("docstring_guidelines", "")),
                _text(section.get("parent_detail_template", "{function_code}")),
            ]
        )
    return str(template).format(function_code=function_code)
