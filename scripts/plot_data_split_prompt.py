"""Visualise any EDGAR project's train/test/discover/validate split.

Works for any project that has a ``data_loader/load_data.py`` — you don't need the
data-loader-helper agent. Given a project name (or a config path), this:
  1. resolves the project's config -> TaskSpec (same machinery as check_loader.py),
  2. feeds the project's load_data.py source through plot_split_prompt.md to have Claude
     generate a project-specific plot_split function,
  3. runs the real load_data with the project's configured params,
  4. renders the split figure.

Usage:
    python scripts/plot_data_split_prompt.py particle_eom
    python scripts/plot_data_split_prompt.py projects/retinotopy_map/config.yaml

Pass a second argument to skip the Claude generate step and use an existing
plot_split script instead (e.g. one you've hand-fixed):
    python scripts/plot_data_split_prompt.py retinotopy_map test_output/plot_split_test/retinotopy_map_plot_split.py
"""

import os
import re
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

PROMPT_PATH = REPO_ROOT / ".claude/skills/data-loader-helper/plot_split_prompt.md"
OUTPUT_DIR = REPO_ROOT / "test_output/plot_split_test"

os.chdir(REPO_ROOT)

from edgar.io.config import Config  # noqa: E402
from edgar.io.task_spec import TaskSpec  # noqa: E402


def _strip_fences(text: str) -> str:
    """Extracts code from a model response, tolerating a missing closing fence.

    Prefers a complete ```python ... ``` block; if the response was truncated mid-block
    (no closing fence), keeps everything after the opening fence instead of failing.
    """
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```(?:python)?\n(.*)", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


def _resolve_config_path(project: str) -> Path:
    """Accepts a project name or a config path and returns the config.yaml path."""
    p = Path(project)
    if p.suffix in (".yaml", ".yml"):
        config_path = p if p.is_absolute() else REPO_ROOT / p
    else:
        config_path = REPO_ROOT / "projects" / project / "config.yaml"
    if not config_path.exists():
        raise SystemExit(f"No config found for {project!r} (looked at {config_path})")
    return config_path


def _generate_plot_split_code(name: str, load_data_path: Path) -> str:
    """Calls Claude to generate a project-specific plot_split function."""
    prompt_template = PROMPT_PATH.read_text()
    load_data_source = load_data_path.read_text()
    prompt = prompt_template.replace("{load_data_source}", load_data_source)

    print(f"Calling Claude to generate plot_split for {name}...")
    client = anthropic.Anthropic()
    message = client.messages.create(
        # model="claude-opus-4-8",
        max_tokens=8192,
        model="claude-sonnet-4-6",
        # max_tokens=16384,
        # thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )

    generated = next((b.text for b in message.content if b.type == "text"), None)
    if generated is None:
        raise SystemExit(
            f"No text block in response (stop_reason={message.stop_reason}). "
            "Likely hit max_tokens during thinking — raise max_tokens or reduce thinking budget."
        )

    print("--- Generated code ---")
    print(generated)
    print("----------------------")
    if message.stop_reason == "max_tokens":
        print("WARNING: response truncated at max_tokens; generated code may be incomplete.")

    return _strip_fences(generated)


def main(project: str, override_script: str | None = None):
    config_path = _resolve_config_path(project)
    config = Config.from_yaml(config_path)
    spec = TaskSpec.from_config(config)
    name = config.task_name

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if override_script is not None:
        override_path = Path(override_script)
        if not override_path.is_absolute():
            override_path = REPO_ROOT / override_path
        if not override_path.exists():
            raise SystemExit(f"Override script not found: {override_path}")
        print(f"Using override plot_split script: {override_path}")
        code = override_path.read_text()
    else:
        load_data_path = config.project_dir / "data_loader" / "load_data.py"
        code = _generate_plot_split_code(name, load_data_path)
        # Save the generated code for inspection
        code_path = OUTPUT_DIR / f"{name}_plot_split.py"
        code_path.write_text(code)
        print(f"\nGenerated code saved to {code_path}")

    # Execute the generated functions
    ns = {}
    exec(code, ns)
    plot_split = ns["plot_split"]

    # Load real project data using the project's configured params.
    print(f"\nLoading {name} data with project_params={spec.project_params}...")
    out = spec.load_data_fn(spec.io["data_path"], **spec.project_params)

    save_path = str(OUTPUT_DIR / f"{name}_split.png")
    print(f"\nGenerating plot -> {save_path}")
    plot_split(out, save_path=save_path)
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(
            "Usage: python scripts/plot_data_split_prompt.py "
            "<project_name|config.yaml> [override_plot_split.py]",
            file=sys.stderr,
        )
        sys.exit(2)
    main(*sys.argv[1:])
