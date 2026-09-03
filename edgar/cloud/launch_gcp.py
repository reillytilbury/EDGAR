"""Launch EDGAR runs on Google Cloud Platform, one GPU VM per run.

``launch_gcp`` reads a self-contained launch spec (see
``projects/gcp_launch.example.yaml``), syncs the local working tree + data to a GCS
bucket, and creates one spot GPU VM per run. Each VM builds its environment with
``uv sync --frozen`` and runs ``edgar run`` under a watchdog, syncing results back to
the bucket and self-deleting. The launcher only shells out to ``gcloud``/``gsutil``; all
GCP auth is the caller's ``gcloud auth login`` plus the VM's default service account, so
no keys are transmitted for GCP itself.

The provider only needs ``GOOGLE_API_KEY`` (and optionally ``ANTHROPIC_API_KEY``) in a
local ``.env``, which is stored in Secret Manager (never in the bucket) and fetched by
each VM's service account at runtime.
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
import time
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
    "image_family": "common-cu129-ubuntu-2204-nvidia-580",
    "image_project": "deeplearning-platform-release",
    "name_prefix": "edgar",
    "max_hours": 12,
    "secret_name": "edgar-env",
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
    """Resolve a path relative to the repository root.

    Args:
        p: The path string to resolve. Can be absolute or relative.

    Returns:
        A `Path` object representing the absolute path within the repository.
    """
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _repo_relative(path: Path, what: str) -> str:
    """Get the repository-relative path for a given Path object.

    Args:
        path: The absolute path to convert to a repository-relative path.
        what: A descriptive string for the path's purpose, used in error messages.

    Returns:
        A string representing the path relative to the repository root.

    Raises:
        ValueError: If the provided path is not within the repository.
    """
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
    """Format a Python value as a string suitable for command-line arguments.

    Args:
        v: The value to format. Can be a boolean, list, tuple, or other type.

    Returns:
        A string representation of the value. Booleans are 'True'/'False', lists/tuples
        are `str(list(v))` with no spaces, and other types are `str(v)`.
    """
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (list, tuple)):
        return str(list(v)).replace(" ", "")
    return str(v)


def _resolve_data_path(data_path: str) -> Path:
    """Resolve a data path, ensuring it's absolute or relative to the repository root.

    Args:
        data_path: The data path string.

    Returns:
        A `Path` object representing the resolved data path.
    """
    p = Path(data_path)
    return p if p.is_absolute() else (REPO_ROOT / p)


# ── spec loading / validation ──


def load_spec(path: str) -> dict:
    """Load a launch spec YAML into a dict.

    Args:
        path: Path to the launch spec YAML file.

    Returns:
        A dictionary parsed from the YAML file. Returns an empty dict if the file
        is empty or contains no YAML.
    """
    return yaml.safe_load(Path(path).read_text()) or {}


def _validate_override_keys(overrides: dict) -> None:
    """Validate that override keys are in the correct '<section>.<name>' format.

    Ensures that override keys target known configuration sections and do not attempt
    to modify `llms.default_provider`, which has no effect after config loading.

    Args:
        overrides: A dictionary of override keys and their values.

    Raises:
        ValueError: If an override key is malformed or attempts to modify
            'llms.default_provider'.
    """
    for key in overrides:
        if "." not in key or key.split(".", 1)[0] not in OVERRIDE_SECTIONS:
            raise ValueError(
                f"override key '{key}' must be '<section>.<name>' with section in "
                f"{sorted(OVERRIDE_SECTIONS)}"
            )
        # Rejected by edgar/cli.py:_apply_overrides too; caught here so a launch fails
        # locally instead of on the VM after provisioning.
        if key == "llms.default_provider":
            raise ValueError(
                "override 'llms.default_provider' has no effect (it only fills unset "
                "roles at config load); set the role models directly, e.g. "
                "llms.model_llm=..., or change default_provider in config.yaml."
            )


def validate_spec(spec: dict) -> dict:
    """Validate and normalize a launch spec, filling defaults.

    This function takes a raw launch specification, applies default GCP settings,
    validates required fields, resolves configuration paths, normalizes run names,
    and expands run replicas.

    Args:
        spec: Raw spec dict from ``load_spec``.

    Returns:
        A validated and normalized dictionary with two top-level keys:
        - "gcp": A dictionary of Google Cloud Platform infrastructure settings.
        - "runs": A list of dictionaries, each representing a single EDGAR run,
          including 'config_rel', 'config_path', 'run_name', 'n_replicas', 'seed',
          and 'overrides'.

    Raises:
        ValueError: On missing required GCP fields, if no runs are defined, if a
            run's configuration file is missing, or if override keys are invalid.
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
    """Expand ``n_replicas`` into individual runs with unique names and per-replica seeds.

    If `n_replicas` for a run is greater than 1, new run entries are created with
    suffixes (e.g., '-r0', '-r1') and incremented seeds.

    Args:
        spec: The validated launch specification dictionary containing "gcp" and "runs".

    Returns:
        A list of dictionaries, where each dictionary represents a single,
        fully specified EDGAR run (no ``n_replicas`` field). Each run has a unique
        'run_name' and 'seed'.

    Raises:
        ValueError: If duplicate run names are generated after normalization and expansion.
    """
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
    """Compute the SHA256 hash of a file.

    Reads the file in chunks to efficiently handle large files.

    Args:
        path: The `Path` object of the file to hash.

    Returns:
        A hexadecimal string representing the SHA256 hash of the file's content.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_object_uri(bucket: str, data_path: str) -> tuple[str, str]:
    """Return ``(gs_uri, basename)`` for a data file, keyed by content hash.

    The GCS URI includes the SHA256 hash of the file's content, ensuring that
    only unique data files are stored and uploaded when content changes.

    Args:
        bucket: The name of the GCS bucket.
        data_path: The local path to the data file.

    Returns:
        A tuple containing:
        - gs_uri (str): The Google Cloud Storage URI for the data object.
        - basename (str): The base name of the data file.
    """
    p = _resolve_data_path(data_path)
    return f"gs://{bucket}/data/{_sha256_file(p)}/{p.name}", p.name


def _gcs_exists(uri: str) -> bool:
    """Check if a Google Cloud Storage URI exists.

    Args:
        uri: The Google Cloud Storage URI to check.

    Returns:
        True if the URI exists, False otherwise.
    """
    try:
        subprocess.run(["gsutil", "-q", "stat", uri], check=True, capture_output=True)
        return True
    except Exception:
        return False


def ensure_data_uploaded(bucket: str, data_path: str, dry_run: bool) -> tuple[str, str]:
    """Upload the data file to the bucket only if the content-hashed object is absent.

    If `dry_run` is True, it will print the intended action without performing the upload.
    If the data file is missing locally, a warning is issued during dry-run.

    Args:
        bucket: The name of the GCS bucket.
        data_path: The local path to the data file to be uploaded.
        dry_run: If True, simulate the upload without executing `gsutil` commands.

    Returns:
        A tuple containing:
        - uri (str): The GCS URI where the data would be or is stored.
        - basename (str): The base name of the data file.
    """
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
    """Mirror the local working tree (minus excludes) to ``gs://BUCKET/code``.

    Uses `gsutil rsync` to synchronize the local repository with the GCS bucket,
    excluding specified paths like `.git/`, `__pycache__`, and `program_databases/`.

    Args:
        bucket: The name of the GCS bucket to sync code to.
        dry_run: If True, print the `gsutil` command without executing it.
    """
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
    """Provenance record: HEAD sha, dirty flag, and the full working-tree diff.

    Captures the current git state (commit SHA, dirty status) and the full
    `git diff --stat` and `git diff` outputs.

    Returns:
        A string containing the git SHA, dirty flag, generation timestamp, and
        the full git diff, providing a detailed provenance record. Returns
        error messages if git commands fail.
    """

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
    """Upload the provenance manifest to ``gs://BUCKET/code/MANIFEST.txt``.

    The manifest includes git SHA, dirty status, and a full diff of the working tree.

    Args:
        bucket: The name of the GCS bucket to upload the manifest to.
        dry_run: If True, print the upload command without executing it.
    """
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


def _wait_secretmanager_ready(project: str, attempts: int = 12, delay: int = 8) -> None:
    """Poll until the Secret Manager API is usable after enabling it.

    Enabling an API returns before it is fully propagated, so the first ``secrets``
    call can still fail with ``SERVICE_DISABLED``. Poll a cheap read until it succeeds
    (or a different error surfaces, which the real command will then report).

    Args:
        project: The GCP project ID.
        attempts: The number of times to poll for API readiness.
        delay: The delay in seconds between polling attempts.
    """
    for _ in range(attempts):
        r = subprocess.run(
            ["gcloud", "secrets", "list", f"--project={project}", "--limit=1"],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0 or "SERVICE_DISABLED" not in (r.stderr or ""):
            return
        time.sleep(delay)
    print("WARN: Secret Manager API still not ready after enabling; continuing anyway")


def _compute_service_account(project_id: str) -> str | None:
    """Return the project's default Compute Engine service account email, or None.

    Args:
        project_id: The Google Cloud Project ID.

    Returns:
        The email address of the default Compute Engine service account for the
        given project, or None if it cannot be determined.
    """
    try:
        num = subprocess.run(
            [
                "gcloud",
                "projects",
                "describe",
                project_id,
                "--format=value(projectNumber)",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return None
    return f"{num}-compute@developer.gserviceaccount.com" if num else None


def ensure_secret(gcp: dict, dry_run: bool) -> str | None:
    """Store the local ``.env`` in Secret Manager and grant the VM's SA read access.

    The key never lands in the bucket: the secret is created if missing, a new version
    is added only when the local ``.env`` differs from the stored one, and the project's
    default Compute Engine service account (which each VM runs as) is granted
    ``secretAccessor`` on the secret. The VM fetches it at runtime with
    ``gcloud secrets versions access``.

    Args:
        gcp: Validated gcp infra dict (uses ``project_id`` and ``secret_name``).
        dry_run: If True, print what would happen and return the secret name.

    Returns:
        The name of the secret in Secret Manager that the VM should fetch.
        Returns None if there is no local ``.env`` file.

    Raises:
        subprocess.CalledProcessError: If any `gcloud` command fails during secret
            creation, versioning, or IAM binding.
    """
    env = REPO_ROOT / ".env"
    secret = gcp["secret_name"]
    project = gcp["project_id"]
    if not env.exists():
        print("WARN: no local .env found; remote runs will have no API keys")
        return None
    if dry_run:
        print(
            f"[dry-run] ensure Secret Manager secret '{secret}' holds .env and grant the "
            "compute service account secretAccessor"
        )
        return secret

    _run(
        [
            "gcloud",
            "services",
            "enable",
            "secretmanager.googleapis.com",
            f"--project={project}",
        ]
    )
    _wait_secretmanager_ready(project)

    exists = (
        subprocess.run(
            ["gcloud", "secrets", "describe", secret, f"--project={project}"],
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )
    if not exists:
        _run(
            [
                "gcloud",
                "secrets",
                "create",
                secret,
                "--replication-policy=automatic",
                f"--project={project}",
            ]
        )

    # Add a new version only if the stored value differs (avoids version bloat).
    try:
        current = subprocess.run(
            [
                "gcloud",
                "secrets",
                "versions",
                "access",
                "latest",
                f"--secret={secret}",
                f"--project={project}",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except Exception:  # noqa: BLE001
        current = None
    if current is None or current.strip() != env.read_text().strip():
        _run(
            [
                "gcloud",
                "secrets",
                "versions",
                "add",
                secret,
                f"--data-file={env}",
                f"--project={project}",
            ]
        )
    else:
        print(f"Secret '{secret}' already up to date.")

    sa = _compute_service_account(project)
    if sa:
        _run(
            [
                "gcloud",
                "secrets",
                "add-iam-policy-binding",
                secret,
                f"--member=serviceAccount:{sa}",
                "--role=roles/secretmanager.secretAccessor",
                f"--project={project}",
            ]
        )
    else:
        print(
            "WARN: could not resolve the default compute service account; grant secretAccessor manually"
        )
    return secret


# ── VM creation ──


def build_overrides(flat_run: dict, data_basename: str | None) -> list[str]:
    """Build the ordered ``--section.key=value`` override list for a run.

    Pins ``io.save_path`` to a unique VM-side dir (so the whole subtree is syncable),
    overrides ``io.data_path`` to the downloaded file when data is present, sets the
    per-replica seed, then appends the user overrides.

    Args:
        flat_run: A dictionary representing a single flattened EDGAR run, containing
            'run_name', 'seed', and 'overrides'.
        data_basename: The basename of the data file, if applicable. None if no data.

    Returns:
        A list of strings, where each string is a command-line override in the format
        '--section.key=value'.
    """
    overrides = [f"--io.save_path={CODE_DIR}/out/{flat_run['run_name']}"]
    if data_basename:
        overrides.append(f"--io.data_path={DATA_DIR}/{data_basename}")
    overrides.append(f"--run.random_seed={flat_run['seed']}")
    for key, value in flat_run["overrides"].items():
        overrides.append(f"--{key}={_fmt_value(value)}")
    return overrides


def create_vm(
    gcp: dict, flat_run: dict, data_uri: str, secret_name: str | None, launch_id: str, user: str, dry_run: bool
) -> str:
    """Create one GPU VM for a flattened run and return its instance name.

    The VM is configured with specified machine type, GPU, boot disk size, and image.
    It includes metadata for EDGAR's startup script and applies provisioning model
    and instance termination actions based on the 'spot' configuration.

    Args:
        gcp: A dictionary of Google Cloud Platform infrastructure settings.
        flat_run: A dictionary representing a single flattened EDGAR run,
            including 'run_name', 'config_rel', 'overrides_list'.
        data_uri: The GCS URI of the data file, or an empty string if no data.
        secret_name: The name of the secret in Secret Manager, or None if no secret.
        launch_id: A unique identifier for this launch operation.
        user: The normalized username of the launching user.
        dry_run: If True, print the `gcloud` command without executing it.

    Returns:
        The name of the created (or planned) VM instance.
    """
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
            f"edgar-secret-name={secret_name or ''}",
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

    Args:
        spec: The validated launch specification dictionary.
        dry_run: If True, issues are reported as warnings instead of raising errors.

    Raises:
        RuntimeError: If `dry_run` is False and preflight checks reveal problems
            with local tooling, GCP authentication, bucket accessibility, or
            missing data files.
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
    """Delete all VMs labelled with the current user (only your launches).

    Args:
        spec: The validated launch specification dictionary. Used to get GCP details.
        dry_run: If True, print the delete commands without executing them.

    Returns:
        Process exit code (always 0 on success, even if no VMs are found).
    """
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
    """Download each run's results into ``program_databases/<project>/<run_name>/YYYY-MM-DD/HH-MM-SS/``.

    On the bucket a run is stored at ``results/<run_name>/<project>/<YYYY-MM-DD>/<HH-MM-SS>/`` —
    edgar's own ``<save_path>/<project>/<date>/<time>`` layout under the VM's per-run save_path.
    Locally the project comes first, matching where local runs land, with the ``<run_name>/``
    dir below it so different runs don't collide; the date/time subdirs are preserved, so the
    same run name launched several times in a day stays separated by timestamp.

    Args:
        spec: The validated launch specification dictionary. Used to get bucket details
            and flattened run configurations.
        dry_run: If True, print the `gsutil rsync` commands without executing them.

    Returns:
        Process exit code (0 on success).
    """
    bucket = spec["gcp"]["bucket"]
    for f in flatten_runs(spec):
        run_name = f["run_name"]
        project = Path(f["config_rel"]).parent.name
        src = f"gs://{bucket}/results/{run_name}/{project}"
        dest = f"program_databases/{project}/{run_name}"
        dest_path = _resolve_repo_path(dest)
        if not dry_run:
            dest_path.mkdir(parents=True, exist_ok=True)
        _run(["gsutil", "-m", "rsync", "-r", src, str(dest_path)], dry_run=dry_run)
    return 0


def _print_summary(summary: list[tuple[str, str, str]], gcp: dict, dry_run: bool) -> None:
    """Prints a summary of the launched or planned runs.

    Includes details on each run's name, VM instance, and results bucket path.
    Also provides instructions for monitoring, fetching, and tearing down runs.

    Args:
        summary: A list of tuples, where each tuple contains (run_name, vm_name, results_uri).
        gcp: A dictionary of Google Cloud Platform infrastructure settings, used for zone.
        dry_run: If True, indicates that the summary is for a dry run.
    """
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
    secret_name = ensure_secret(gcp, dry_run)

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
        vm = create_vm(gcp, f, data_uri, secret_name, launch_id, user, dry_run)
        summary.append((f["run_name"], vm, f"gs://{bucket}/results/{f['run_name']}"))

    _print_summary(summary, gcp, dry_run)
    return 0