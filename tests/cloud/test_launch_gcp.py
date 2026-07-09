import hashlib
import re
from types import SimpleNamespace

import pytest

from edgar.cli import _apply_overrides
from edgar.cloud.launch_gcp import (
    CODE_EXCLUDE,
    GCP_DEFAULTS,
    build_overrides,
    data_object_uri,
    flatten_runs,
    validate_spec,
)
from edgar.cloud.startup_script import render

SYNTH_CONFIG = "projects/synthetic_data/config.yaml"


def _spec(**gcp_extra):
    return {
        "gcp": {"project_id": "p", "bucket": "b", "zone": "us-central1-a", **gcp_extra},
        "runs": [{"config": SYNTH_CONFIG}],
    }


# ── validate_spec ──


def test_validate_spec_fills_gcp_defaults():
    spec = validate_spec(_spec())
    assert spec["gcp"]["machine_type"] == GCP_DEFAULTS["machine_type"]
    assert spec["gcp"]["spot"] is True
    assert spec["gcp"]["secret_name"] == "edgar-env"
    assert spec["runs"][0]["run_name"] == "synthetic-data"  # from dir name, normalized


@pytest.mark.parametrize("missing", ["project_id", "bucket", "zone"])
def test_validate_spec_requires_core_gcp_fields(missing):
    raw = _spec()
    del raw["gcp"][missing]
    with pytest.raises(ValueError, match="missing required"):
        validate_spec(raw)


def test_validate_spec_rejects_no_runs():
    with pytest.raises(ValueError, match="at least one run"):
        validate_spec({"gcp": {"project_id": "p", "bucket": "b", "zone": "z"}})


def test_validate_spec_rejects_missing_config():
    raw = _spec()
    raw["runs"] = [{"config": "projects/does_not_exist/config.yaml"}]
    with pytest.raises(ValueError, match="config not found"):
        validate_spec(raw)


def test_validate_spec_rejects_unknown_override_section():
    raw = _spec()
    raw["runs"] = [{"config": SYNTH_CONFIG, "overrides": {"bogus.key": 1}}]
    with pytest.raises(ValueError, match="section in"):
        validate_spec(raw)


def test_validate_spec_merges_defaults_then_run_overrides():
    raw = _spec()
    raw["defaults"] = {
        "overrides": {"evolution.n_generations": 1, "scoring.timeout_s": 5}
    }
    raw["runs"] = [
        {"config": SYNTH_CONFIG, "overrides": {"evolution.n_generations": 9}}
    ]
    spec = validate_spec(raw)
    ovr = spec["runs"][0]["overrides"]
    assert ovr["evolution.n_generations"] == 9  # run wins
    assert ovr["scoring.timeout_s"] == 5  # default carried through


# ── flatten_runs ──


def test_flatten_runs_expands_replicas_with_seeds():
    raw = _spec()
    raw["defaults"] = {"base_seed": 10}
    raw["runs"] = [{"config": SYNTH_CONFIG, "run_name": "foo", "n_replicas": 3}]
    flat = flatten_runs(validate_spec(raw))
    assert [f["run_name"] for f in flat] == ["foo-r0", "foo-r1", "foo-r2"]
    assert [f["seed"] for f in flat] == [10, 11, 12]


def test_flatten_runs_single_replica_keeps_name():
    flat = flatten_runs(validate_spec(_spec()))
    assert len(flat) == 1
    assert flat[0]["run_name"] == "synthetic-data"


def test_flatten_runs_detects_duplicate_names():
    raw = _spec()
    raw["runs"] = [
        {"config": SYNTH_CONFIG, "run_name": "dup"},
        {"config": SYNTH_CONFIG, "run_name": "dup"},
    ]
    with pytest.raises(ValueError, match="duplicate run names"):
        flatten_runs(validate_spec(raw))


# ── data_object_uri ──


def test_data_object_uri_is_content_hashed(tmp_path):
    f = tmp_path / "data.npy"
    f.write_bytes(b"hello world")
    sha = hashlib.sha256(b"hello world").hexdigest()
    uri, basename = data_object_uri("mybucket", str(f))
    assert uri == f"gs://mybucket/data/{sha}/data.npy"
    assert basename == "data.npy"


# ── build_overrides ──


def _flat(**kw):
    base = {"run_name": "synth", "seed": 7, "overrides": {}}
    base.update(kw)
    return base


def test_build_overrides_order_and_data_path():
    ovr = build_overrides(_flat(overrides={"evolution.n_generations": 4}), "data.npy")
    assert ovr == [
        "--io.save_path=/opt/edgar/out/synth",
        "--io.data_path=/opt/edgar_data/data.npy",
        "--run.random_seed=7",
        "--evolution.n_generations=4",
    ]


def test_build_overrides_omits_data_path_when_none():
    ovr = build_overrides(_flat(), None)
    assert not any(o.startswith("--io.data_path=") for o in ovr)
    assert "--run.random_seed=7" in ovr


def test_build_overrides_strips_spaces_in_list_values():
    ovr = build_overrides(_flat(overrides={"evolution.topology": [1, 2, 3, 0]}), None)
    assert "--evolution.topology=[1,2,3,0]" in ovr


# ── exclude regex ──


@pytest.mark.parametrize(
    "path",
    [
        ".git/config",
        "edgar/__pycache__/x.pyc",
        "sub/foo.pyc",
        "program_databases/x.jsonl",
        "test_output/run",
        ".env",
        "edgar.egg-info/PKG-INFO",
        "docs/build/index.html",
    ],
)
def test_code_exclude_matches_junk(path):
    assert re.search(CODE_EXCLUDE, path)


@pytest.mark.parametrize(
    "path",
    [
        "uv.lock",
        "pyproject.toml",
        "projects/synthetic_data/config.yaml",
        "edgar/cli.py",
    ],
)
def test_code_exclude_keeps_source(path):
    assert not re.search(CODE_EXCLUDE, path)


# ── startup script ──


def test_render_contains_key_steps():
    script = render()
    for token in [
        "until nvidia-smi",
        "uv sync --frozen",
        "trap delete_vm EXIT",
        "timeout",
        "mapfile -t OVERRIDES",
        "gcloud secrets versions access latest",
        'gsutil cp - "${RESULTS_URI}/STATUS"',
        "gcloud compute instances delete",
    ]:
        assert token in script, token
    assert "@@CODE_DIR@@" not in script  # tokens substituted


# ── CLI override regression (needs the new "run" section) ──


def test_apply_overrides_accepts_run_random_seed():
    spec = SimpleNamespace(
        io={}, evolution={}, llms={}, scoring={}, project_params={}, run={}
    )
    _apply_overrides(spec, ["--run.random_seed=42"])
    assert spec.run["random_seed"] == 42


def test_apply_overrides_rejects_default_provider():
    spec = SimpleNamespace(
        io={}, evolution={}, llms={}, scoring={}, project_params={}, run={}
    )
    with pytest.raises(ValueError, match="default_provider"):
        _apply_overrides(spec, ["--llms.default_provider=anthropic"])


def test_validate_spec_rejects_default_provider_override():
    raw = _spec()
    raw["runs"] = [
        {"config": SYNTH_CONFIG, "overrides": {"llms.default_provider": "anthropic"}}
    ]
    with pytest.raises(ValueError, match="default_provider"):
        validate_spec(raw)
