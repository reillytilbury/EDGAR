"""Remote startup script for EDGAR GCP VMs.

The VM's Deep Learning image runs ``STARTUP_TEMPLATE`` as root on first boot. Per-run
values (bucket, run name, config path, data URI, overrides, runtime cap) are read from
the instance metadata server, so the rendered script is byte-identical across every VM
in a launch. The script waits for the GPU driver, pulls the code/secrets/data from GCS,
builds the environment with ``uv sync --frozen``, runs ``edgar run`` under a wall-clock
watchdog, and self-deletes via an ``EXIT`` trap that always flushes results and writes a
``SUCCESS``/``FAILED``/``TIMEOUT`` sentinel to the bucket.
"""

CODE_DIR = "/opt/edgar"
DATA_DIR = "/opt/edgar_data"

# ``@@TOKEN@@`` placeholders (not str.format braces) so the bash ``${...}`` and ``$(...)``
# below need no escaping. ``render`` substitutes them.
STARTUP_TEMPLATE = r"""#!/usr/bin/env bash
set -uo pipefail
exec > >(tee -a /var/log/edgar-startup.log) 2>&1
echo "=== edgar startup $(date -u) ==="

CODE_DIR="@@CODE_DIR@@"
DATA_DIR="@@DATA_DIR@@"

md() {
  curl -s -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/$1"
}

BUCKET=$(md attributes/edgar-bucket)
RUN_NAME=$(md attributes/edgar-run-name)
CONFIG=$(md attributes/edgar-config)
DATA_URI=$(md attributes/edgar-data-uri)
MAX_HOURS=$(md attributes/edgar-max-hours)
VM_NAME=$(md name)
ZONE=$(basename "$(md zone)")

SAVE_ROOT="${CODE_DIR}/out/${RUN_NAME}"
RESULTS_URI="gs://${BUCKET}/results/${RUN_NAME}"

RUN_RC=""
SYNC_PID=""

# Guaranteed final flush + self-delete. Fires on normal exit and on error, so an early
# failure can never leak a billing GPU VM.
delete_vm() {
  set +e
  [ -n "$SYNC_PID" ] && kill "$SYNC_PID" 2>/dev/null
  [ -d "$SAVE_ROOT" ] && gsutil -m rsync -r "$SAVE_ROOT" "$RESULTS_URI"
  gsutil cp /var/log/edgar-startup.log "${RESULTS_URI}/startup.log"
  local status="FAILED"
  [ "$RUN_RC" = "0" ] && status="SUCCESS"
  [ "$RUN_RC" = "124" ] && status="TIMEOUT"
  echo "${status} rc=${RUN_RC} $(date -u)" | gsutil cp - "${RESULTS_URI}/STATUS"
  gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
}
trap delete_vm EXIT

# Wait for the NVIDIA driver the Deep Learning image installs on first boot.
until nvidia-smi; do echo "waiting for GPU driver..."; sleep 5; done

# Pull code, secrets, and (optionally) the data file.
mkdir -p "$CODE_DIR" "$DATA_DIR"
gsutil -m rsync -r "gs://${BUCKET}/code" "$CODE_DIR"
gsutil cp "gs://${BUCKET}/secrets/.env" "${CODE_DIR}/.env" \
  || echo "WARN: no .env in bucket; LLM calls will fail"
if [ -n "$DATA_URI" ]; then
  gsutil cp "$DATA_URI" "${DATA_DIR}/$(basename "$DATA_URI")"
fi

# Overrides -> bash array (newline-delimited preserves spaces in list values).
mapfile -t OVERRIDES < <(md attributes/edgar-overrides)

# Build the environment from the shipped lockfile with uv (no system pip, fully pinned).
export HOME=/root
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"
cd "$CODE_DIR"
uv sync --frozen

# Periodic sync of the whole save-root to the bucket (spot-preemption safety).
(
  while true; do
    sleep 120
    [ -d "$SAVE_ROOT" ] && gsutil -m rsync -r "$SAVE_ROOT" "$RESULTS_URI"
  done
) &
SYNC_PID=$!

# Run under a wall-clock watchdog. XLA env guards are set in edgar/run.py at import;
# do not set them here.
timeout "${MAX_HOURS}h" uv run edgar run "$CONFIG" "${OVERRIDES[@]}"
RUN_RC=$?
echo "=== edgar run exited rc=${RUN_RC} ==="
# trap delete_vm runs on EXIT: final rsync, sentinel, self-delete.
"""


def render(code_dir: str = CODE_DIR, data_dir: str = DATA_DIR) -> str:
    """Render the startup script with the VM-side code and data directories.

    Args:
        code_dir: Absolute path the repo is synced to on the VM.
        data_dir: Absolute path the run's data file is downloaded to on the VM.

    Returns:
        The bash startup script as a string, ready to pass as instance metadata.
    """
    return STARTUP_TEMPLATE.replace("@@CODE_DIR@@", code_dir).replace(
        "@@DATA_DIR@@", data_dir
    )
