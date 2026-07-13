# ruff: noqa F841
import pytest
from edgar.cli import _apply_overrides
from edgar.io.config import Config

PERFECT_CONFIG = "tests/io/test_configs/perfect/config.yaml"
DEFAULT_CONFIG = "tests/io/test_configs/perfect/default_config.yaml"


def test_apply_overrides_success():
    config = Config.from_yaml(PERFECT_CONFIG, default_path=DEFAULT_CONFIG)

    # Check baseline values before overrides
    assert config.evolution.n_generations == 12
    assert config.io.save_path == ""
    assert config.scoring.timeout_s == 120.0
    assert config.run.random_seed == 42

    _apply_overrides(
        config,
        [
            "--evolution.n_generations=24",
            "--io.save_path=/tmp/new_save_path",
            "--scoring.timeout_s=240.0",
            "--run.random_seed=12345",
        ],
    )

    # Verify overridden values are applied to Config
    assert config.evolution.n_generations == 24
    assert config.io.save_path == "/tmp/new_save_path"
    assert config.scoring.timeout_s == 240.0
    assert config.run.random_seed == 12345


def test_apply_overrides_rejects_default_provider():
    config = Config.from_yaml(PERFECT_CONFIG, default_path=DEFAULT_CONFIG)
    with pytest.raises(ValueError, match="default_provider"):
        _apply_overrides(config, ["--llms.default_provider=anthropic"])


def test_run_dashboard_roots_with_target(tmp_path):
    from unittest.mock import patch
    from edgar.cli import _run_dashboard

    # Create a temporary folder as target
    target = tmp_path / "custom_run_output"
    target.mkdir()

    with (
        patch("edgar.dashboard.server.build_app") as mock_build_app,
        patch("uvicorn.run") as mock_run,
        patch("webbrowser.open") as mock_open,
    ):
        # Call _run_dashboard with the target
        _run_dashboard(target=str(target), port=8765, host="127.0.0.1", no_open=True)

        # Verify that build_app was called with roots containing ONLY target,
        # and NOT including the default program_databases
        mock_build_app.assert_called_once()
        roots_arg = mock_build_app.call_args[0][0]

        assert len(roots_arg) == 1
        assert roots_arg[0] == target.resolve()


def test_run_dashboard_roots_without_target():
    from unittest.mock import patch
    from edgar.cli import _run_dashboard

    with (
        patch("edgar.dashboard.server.build_app") as mock_build_app,
        patch("uvicorn.run") as mock_run,
        patch("webbrowser.open") as mock_open,
    ):
        # Call _run_dashboard without a target
        _run_dashboard(target=None, port=8765, host="127.0.0.1", no_open=True)

        mock_build_app.assert_called_once()
        roots_arg = mock_build_app.call_args[0][0]

        assert len(roots_arg) == 1
        assert "program_databases" in str(roots_arg[0])
