"""Launch EDGAR runs on Google Cloud Platform, one GPU VM per run.

``launch_gcp`` reads a self-contained launch spec (see
``projects/gcp_launch.example.yaml``), syncs the local working tree + data to a GCS
bucket, and creates one spot GPU VM per run. Each VM builds its environment with
``uv sync --frozen`` and runs ``edgar run`` under a watchdog, syncing results back to
the bucket and self-deleting. The launcher only shells out to ``gcloud``/``gsutil``; all
GCP auth is the caller's ``gcloud auth login`` plus the VM's default service account, so
no keys are transmitted for GCP itself.

The provider only needs ``GOOGLE_API_KEY`` (and optionally ``ANTHROPIC_API_KEY``) in a
local ``.env``, which is uploaded to a private bucket path and pulled by each VM.
"""

from __future__ import annotations

import getpass
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import yaml

from ..io.config import REPO_ROOT, Config
from .startup_script import CODE_DIR, DATA_DIR, render

# Override sections accepted by edgar/cli.py:_apply_overrides (kept in sync with it).
OVERRIDE_SECTIONS = {"io", "evolution", "llms", "scoring", "project_params", "run"}

GCP_DEFAULTS = {
    "machine_type": "g2-standard-8",
    "gpu_type": "nvidia-l4",
    "gpu_count": 1,
    "spot": True,
    "boot_disk_size_gb": 200,
    "image_family": "common-cu123",
    "image_project": "deeplearning-platform-release",
    "name_prefix": "edgar",
    "max_hours": 12,
}
REQUIRED_GCP = ("project_id", "bucket", "zone")

# Single regex passed to `gsutil rsync -x`; matched against paths under the repo root.
CODE_EXCLUDE = "|".join(
    [
        r"(^|.*/)\.git/.*",
        r"(^|.*/)__pycache__/.*",
        r".*\.pyc$",
        r"(^|.*/)program_databases/.*",
        r"(^|.*/)test_output.*",
        r"(^|.*/)figures/.*",
        r"(^|.*/)\.venv/.*",
        r"(^|.*/)\.env$",
        r".*\.egg-info/.*",
        r"(^|.*/)docs/build/.*",
        r"(^|.*/)sample_plots/.*",
        r"(^|.*/)\.vscode/.*",
    ]
)


# ── small helpers ──


def _run(cmd, dry_run=False, capture=False):
    """Run a command, or print it under ``--dry-run``.

    Args:
        cmd: Command as an argv list.
        dry_run: If True, print the command instead of executing it.
        capture: If True, capture stdout/stderr (text mode).

    Returns:
        The ``subprocess.CompletedProcess`` (a stub with empty output under dry-run).
    """
    if dry_run:
        print("[dry-run] " + " ".join(shlex.quote(c) for c in cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, capture_output=capture, text=True, check=True)


def _resolve_repo_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _repo_relative(path: Path, what: str) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as e:
        raise ValueError(
            f"{what} must live inside the repo ({REPO_ROOT}) so the VM can find it: {path}"
        ) from e


def _normalize_name(name: str) -> str:
    """GCP-safe instance/run name: lowercase, ``[a-z0-9-]``, no leading/trailing dash."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _label(value: str) -> str:
    """GCP-safe label value: lowercase, ``[a-z0-9_-]``, <=63 chars."""
    return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")[:63]


def _fmt_value(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (list, tuple)):
        return str(list(v)).replace(" ", "")
    return str(v)


def _resolve_data_path(data_path: str) -> Path:
    p = Path(data_path)
    return p if p.is_absolute() else (REPO_ROOT / p)


# ── spec loading / validation ──


def load_spec(path: str) -> dict:
    """Load a launch spec YAML into a dict."""
    return yaml.safe_load(Path(path).read_text()) or {}


def _validate_override_keys(overrides: dict) -> None:
    for key in overrides:
        if "." not in key or key.split(".", 1)[0] not in OVERRIDE_SECTIONS:
            raise ValueError(
                f"override key '{key}' must be '<section>.<name>' with section in "
                f"{sorted(OVERRIDE_SECTIONS)}"
            )


def validate_spec(spec: dict) -> dict:
    """Validate and normalize a launch spec, filling defaults.

    Args:
        spec: Raw spec dict from ``load_spec``.

    Returns:
        A dict ``{"gcp": <infra dict>, "runs": [<run dict>, ...]}`` where each run has
        ``config_rel``, ``config_path``, ``run_name``, ``n_replicas``, ``seed``,
        ``overrides``.

    Raises:
        ValueError: On missing required fields, absent configs, or bad override keys.
    """
    gcp = {**GCP_DEFAULTS, **(spec.get("gcp") or {})}
    missing = [k for k in REQUIRED_GCP if not gcp.get(k)]
    if missing:
        raise ValueError(f"gcp spec missing required fields: {missing}")

    defaults = spec.get("defaults") or {}
    base_overrides = defaults.get("overrides") or {}
    base_seed = defaults.get("base_seed", 0)

    raw_runs = spec.get("runs") or []
    if not raw_runs:
        raise ValueError("spec must define at least one run under 'runs'")

    runs = []
    for i, r in enumerate(raw_runs):
        config = r.get("config")
        if not config:
            raise ValueError(f"run #{i} missing required 'config'")
        config_path = _resolve_repo_path(config)
        if not config_path.exists():
            raise ValueError(f"run #{i} config not found: {config}")
        config_rel = _repo_relative(config_path, "config")
        overrides = {**base_overrides, **(r.get("overrides") or {})}
        _validate_override_keys(overrides)
        runs.append(
            {
                "config_rel": config_rel,
                "config_path": config_path,
                "run_name": _normalize_name(
                    r.get("run_name") or config_path.parent.name
                ),
                "n_replicas": int(r.get("n_replicas", 1)),
                "seed": int(r.get("seed", base_seed)),
                "overrides": overrides,
            }
        )
    return {"gcp": gcp, "runs": runs}


def flatten_runs(spec: dict) -> list[dict]:
    """Expand ``n_replicas`` into individual runs with unique names and per-replica seeds."""
    flat = []
    for r in spec["runs"]:
        n = r["n_replicas"]
        for i in range(n):
            name = r["run_name"] if n == 1 else f"{r['run_name']}-r{i}"
            flat.append(
                {
                    "run_name": _normalize_name(name),
                    "config_rel": r["config_rel"],
                    "config_path": r["config_path"],
                    "seed": r["seed"] + i,
                    "overrides": r["overrides"],
                }
            )
    names = [f["run_name"] for f in flat]
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise ValueError(f"duplicate run names after normalization: {dups}")
    return flat


# ── data staging ──


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_object_uri(bucket: str, data_path: str) -> tuple[str, str]:
    """Return ``(gs_uri, basename)`` for a data file, keyed by content hash."""
    p = _resolve_data_path(data_path)
    return f"gs://{bucket}/data/{_sha256_file(p)}/{p.name}", p.name


def _gcs_exists(uri: str) -> bool:
    try:
        subprocess.run(["gsutil", "-q", "stat", uri], check=True, capture_output=True)
        return True
    except Exception:
        return False


def ensure_data_uploaded(bucket: str, data_path: str, dry_run: bool) -> tuple[str, str]:
    """Upload the data file to the bucket only if the content-hashed object is absent."""
    p = _resolve_data_path(data_path)
    if dry_run and not p.exists():
        print(f"[dry-run] data file missing locally: {p} (would hash + upload)")
        return f"gs://{bucket}/data/<sha256>/{p.name}", p.name
    uri, basename = data_object_uri(bucket, data_path)
    if dry_run:
        print(f"[dry-run] ensure {uri} (upload from {p} if missing)")
        return uri, basename
    if _gcs_exists(uri):
        print(f"Data already present: {uri}")
    else:
        print(f"Uploading data {p} -> {uri}")
        _run(["gsutil", "cp", str(p), uri])
    return uri, basename


# ── code / secrets / provenance ──


def rsync_code(bucket: str, dry_run: bool) -> None:
    """Mirror the local working tree (minus excludes) to ``gs://BUCKET/code``."""
    _run(
        [
            "gsutil",
            "-m",
            "rsync",
            "-r",
            "-d",
            "-x",
            CODE_EXCLUDE,
            str(REPO_ROOT),
            f"gs://{bucket}/code",
        ],
        dry_run=dry_run,
    )


def build_manifest() -> str:
    """Provenance record: HEAD sha, dirty flag, and the full working-tree diff."""

    def git(*args) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        except Exception as e:  # noqa: BLE001 - provenance is best-effort
            return f"<git {' '.join(args)} failed: {e}>\n"

    sha = git("rev-parse", "HEAD").strip()
    dirty = bool(git("status", "--porcelain").strip())
    return (
        f"git_sha: {sha}\n"
        f"dirty: {dirty}\n"
        f"generated: {datetime.now().isoformat()}\n\n"
        f"=== git diff --stat ===\n{git('diff', '--stat')}\n"
        f"=== git diff ===\n{git('diff')}\n"
    )


def upload_manifest(bucket: str, dry_run: bool) -> None:
    content = build_manifest()
    if dry_run:
        print(
            f"[dry-run] upload MANIFEST.txt ({len(content)} bytes) -> gs://{bucket}/code/MANIFEST.txt"
        )
        return
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(content)
        tmp = f.name
    try:
        _run(["gsutil", "cp", tmp, f"gs://{bucket}/code/MANIFEST.txt"])
    finally:
        os.unlink(tmp)


def upload_env(bucket: str, dry_run: bool) -> None:
    env = REPO_ROOT / ".env"
    if not env.exists():
        print("WARN: no local .env found; remote runs will have no API keys")
        return
    _run(["gsutil", "cp", str(env), f"gs://{bucket}/secrets/.env"], dry_run=dry_run)


# ── VM creation ──


def build_overrides(flat_run: dict, data_basename: str | None) -> list[str]:
    """Build the ordered ``--section.key=value`` override list for a run.

    Pins ``io.save_path`` to a unique VM-side dir (so the whole subtree is syncable),
    overrides ``io.data_path`` to the downloaded file when data is present, sets the
    per-replica seed, then appends the user overrides.
    """
    overrides = [f"--io.save_path={CODE_DIR}/out/{flat_run['run_name']}"]
    if data_basename:
        overrides.append(f"--io.data_path={DATA_DIR}/{data_basename}")
    overrides.append(f"--run.random_seed={flat_run['seed']}")
    for key, value in flat_run["overrides"].items():
        overrides.append(f"--{key}={_fmt_value(value)}")
    return overrides


def create_vm(gcp, flat_run, data_uri, launch_id, user, dry_run) -> str:
    """Create one GPU VM for a flattened run and return its instance name."""
    run_name = flat_run["run_name"]
    vm_name = _normalize_name(f"{gcp['name_prefix']}-{run_name}")[:63].strip("-")
    metadata = ",".join(
        [
            "install-nvidia-driver=True",
            f"edgar-bucket={gcp['bucket']}",
            f"edgar-run-name={run_name}",
            f"edgar-config={flat_run['config_rel']}",
            f"edgar-data-uri={data_uri or ''}",
            f"edgar-max-hours={gcp['max_hours']}",
        ]
    )
    overrides_text = "\n".join(flat_run["overrides_list"]) + "\n"
    with (
        tempfile.NamedTemporaryFile("w", suffix="-startup.sh", delete=False) as sf,
        tempfile.NamedTemporaryFile("w", suffix="-overrides.txt", delete=False) as of,
    ):
        sf.write(render())
        startup_file = sf.name
        of.write(overrides_text)
        overrides_file = of.name
    try:
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "create",
            vm_name,
            f"--project={gcp['project_id']}",
            f"--zone={gcp['zone']}",
            f"--machine-type={gcp['machine_type']}",
            f"--accelerator=type={gcp['gpu_type']},count={gcp['gpu_count']}",
            f"--image-family={gcp['image_family']}",
            f"--image-project={gcp['image_project']}",
            f"--boot-disk-size={gcp['boot_disk_size_gb']}GB",
            "--maintenance-policy=TERMINATE",
            "--scopes=cloud-platform",
            f"--labels=edgar-launch={launch_id},edgar-user={user}",
            f"--metadata={metadata}",
            f"--metadata-from-file=edgar-overrides={overrides_file},startup-script={startup_file}",
        ]
        if gcp["spot"]:
            cmd += ["--provisioning-model=SPOT", "--instance-termination-action=DELETE"]
        _run(cmd, dry_run=dry_run)
    finally:
        os.unlink(startup_file)
        os.unlink(overrides_file)
    return vm_name


# ── preflight / teardown / fetch ──


def preflight(spec: dict, dry_run: bool) -> None:
    """Validate local tooling, auth, bucket, configs, and data files before launching.

    Under ``--dry-run`` problems are warnings so the plan prints on any machine;
    otherwise they raise.
    """
    problems = []
    for tool in ("gcloud", "gsutil"):
        if not shutil.which(tool):
            problems.append(f"'{tool}' not found on PATH")

    gcp = spec["gcp"]
    if shutil.which("gcloud"):
        try:
            acct = subprocess.run(
                [
                    "gcloud",
                    "auth",
                    "list",
                    "--filter=status:ACTIVE",
                    "--format=value(account)",
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            if not acct:
                problems.append("no active gcloud account (run: gcloud auth login)")
        except Exception as e:  # noqa: BLE001
            problems.append(f"gcloud auth check failed: {e}")
    if shutil.which("gsutil"):
        try:
            subprocess.run(
                ["gsutil", "ls", f"gs://{gcp['bucket']}"],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception:  # noqa: BLE001
            problems.append(
                f"bucket gs://{gcp['bucket']} not accessible (create it / check access)"
            )

    for r in spec["runs"]:
        try:
            data_path = Config.from_yaml(r["config_rel"]).io.data_path
        except Exception as e:  # noqa: BLE001
            problems.append(f"config {r['config_rel']} failed to load: {e}")
            continue
        if data_path and not _resolve_data_path(data_path).exists():
            problems.append(
                f"data file not found locally: {data_path} (config {r['config_rel']})"
            )

    if problems:
        msg = "preflight found problems:\n  - " + "\n  - ".join(problems)
        if dry_run:
            warnings.warn(msg, stacklevel=2)
        else:
            raise RuntimeError(msg)


def _teardown(spec: dict, dry_run: bool) -> int:
    """Delete all VMs labelled with the current user (only your launches)."""
    gcp = spec["gcp"]
    user = _label(getpass.getuser())
    result = _run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            f"--project={gcp['project_id']}",
            f"--filter=labels.edgar-user={user}",
            "--format=value(name,zone)",
        ],
        dry_run=dry_run,
        capture=True,
    )
    if dry_run:
        print(f"[dry-run] would delete instances labelled edgar-user={user}")
        return 0
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        print(f"No EDGAR instances found for user '{user}'.")
        return 0
    for line in lines:
        name, zone = line.split()
        zone = zone.split("/")[-1]
        _run(
            [
                "gcloud",
                "compute",
                "instances",
                "delete",
                name,
                f"--zone={zone}",
                f"--project={gcp['project_id']}",
                "--quiet",
            ]
        )
    return 0


def fetch_results(spec: dict, dry_run: bool) -> int:
    """Rsync each run's results from the bucket into local ``program_databases/``."""
    gcp = spec["gcp"]
    for f in flatten_runs(spec):
        src = f"gs://{gcp['bucket']}/results/{f['run_name']}"
        _run(["gsutil", "-m", "rsync", "-r", src, "program_databases"], dry_run=dry_run)
    return 0


def _print_summary(summary, gcp, dry_run) -> None:
    tag = "[dry-run] " if dry_run else ""
    print(f"\n{tag}Launched {len(summary)} run(s):")
    for run_name, vm, results in summary:
        print(f"  {run_name}: vm={vm}  results={results}")
    print(
        f"\nMonitor:  gcloud compute ssh <vm> --zone={gcp['zone']} "
        "--command='tail -f /var/log/edgar-startup.log'"
    )
    print("Fetch:    uv run edgar launch-gcp <spec> --fetch")
    print("Teardown: uv run edgar launch-gcp <spec> --teardown")


# ── entry point ──


def launch_gcp(spec_path: str, *, teardown=False, dry_run=False, fetch=False) -> int:
    """Launch (or tear down / fetch) an EDGAR sweep on GCP from a launch spec.

    Args:
        spec_path: Path to the launch spec YAML.
        teardown: Delete this user's EDGAR VMs instead of launching.
        dry_run: Print the gcloud/gsutil commands and startup script without executing.
        fetch: Download results from the bucket instead of launching.

    Returns:
        Process exit code (0 on success).
    """
    spec = validate_spec(load_spec(spec_path))
    if teardown:
        return _teardown(spec, dry_run)
    if fetch:
        return fetch_results(spec, dry_run)

    preflight(spec, dry_run)
    gcp = spec["gcp"]
    bucket = gcp["bucket"]

    rsync_code(bucket, dry_run)
    upload_manifest(bucket, dry_run)
    upload_env(bucket, dry_run)

    # Upload each unique config's data file once (skip-if-present).
    data_cache: dict[str, tuple[str | None, str | None]] = {}
    for r in spec["runs"]:
        cr = r["config_rel"]
        if cr in data_cache:
            continue
        data_path = Config.from_yaml(cr).io.data_path
        data_cache[cr] = (
            ensure_data_uploaded(bucket, data_path, dry_run)
            if data_path
            else (None, None)
        )

    flat = flatten_runs(spec)
    launch_id = _label(datetime.now().strftime("%Y%m%d-%H%M%S"))
    user = _label(getpass.getuser())

    if dry_run:
        print("=== rendered startup script ===")
        print(render())

    summary = []
    for f in flat:
        _uri, basename = data_cache[f["config_rel"]]
        f["overrides_list"] = build_overrides(f, basename)
        data_uri = data_cache[f["config_rel"]][0]
        vm = create_vm(gcp, f, data_uri, launch_id, user, dry_run)
        summary.append((f["run_name"], vm, f"gs://{bucket}/results/{f['run_name']}"))

    _print_summary(summary, gcp, dry_run)
    return 0
