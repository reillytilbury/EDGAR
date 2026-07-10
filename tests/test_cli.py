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
