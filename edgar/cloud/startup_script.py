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
set -euo pipefail
exec > >(tee -a /var/log/edgar-startup.log) 2>&1
echo "=== edgar startup $(date -u) ==="

CODE_DIR="@@CODE_DIR@@"
DATA_DIR="@@DATA_DIR@@"

md() {
  curl -fsS -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/$1"
}

BUCKET=$(md attributes/edgar-bucket)
RUN_NAME=$(md attributes/edgar-run-name)
CONFIG=$(md attributes/edgar-config)
DATA_URI=$(md attributes/edgar-data-uri)
MAX_HOURS=$(md attributes/edgar-max-hours)
SECRET_NAME=$(md attributes/edgar-secret-name)
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

  # Dynamically find the YYYY-MM-DD/HH-MM-SS subdirectory inside SAVE_ROOT
  local ts_dir=""
  if [ -d "$SAVE_ROOT" ]; then
    for d in "$SAVE_ROOT"/*/*; do
      if [ -d "$d" ] && [[ "$(basename "$(dirname "$d")")" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] && [[ "$(basename "$d")" =~ ^[0-9]{2}-[0-9]{2}-[0-9]{2}$ ]]; then
        ts_dir="$(basename "$(dirname "$d")")/$(basename "$d")"
        break
      fi
    done
  fi

  #copy startup.log to the timestamped subdirectory
  if [ -n "$ts_dir" ]; then
    gsutil cp /var/log/edgar-startup.log "${RESULTS_URI}/${ts_dir}/startup.log"
  fi

  local status="FAILED"
  [ "$RUN_RC" = "0" ] && status="SUCCESS"
  [ "$RUN_RC" = "124" ] && status="TIMEOUT"
  echo "${status} rc=${RUN_RC} $(date -u)" | gsutil cp - "${RESULTS_URI}/STATUS"

  # Send optional completion/failure notification to Slack
  if [ -f "${CODE_DIR}/.env" ]; then
    local WEBHOOK
    WEBHOOK=$(grep -E "^SLACK_WEBHOOK_URL=" "${CODE_DIR}/.env" | cut -d'=' -f2- | tr -d '"' | tr -d "'" | tr -d '[:space:]')
    if [ -n "$WEBHOOK" ]; then
      local icon="🚀"
      [ "$status" = "SUCCESS" ] && icon="✅"
      [ "$status" = "FAILED" ] && icon="❌"
      [ "$status" = "TIMEOUT" ] && icon="⚠️"

      local results_gcs_path="$RESULTS_URI"
      if [ -n "$ts_dir" ]; then
        results_gcs_path="${RESULTS_URI}/${ts_dir}"
      fi

      curl -s -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"${icon} *EDGAR Run Finished!*\\n• *Run Name:* \`${RUN_NAME}\`\\n• *Status:* \`${status}\` (rc=${RUN_RC})\\n• *Timestamp:* \`${ts_dir:-N/A}\`\\n• *GCS Location:* \`${results_gcs_path}\`\"}" \
        "$WEBHOOK" || true
    fi
  fi

  gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
}
trap delete_vm EXIT

# Wait for the NVIDIA driver the Deep Learning image installs on first boot.
until nvidia-smi; do echo "waiting for GPU driver..."; sleep 5; done

# Pull code, secrets, and (optionally) the data file.
mkdir -p "$CODE_DIR" "$DATA_DIR"
gsutil -m rsync -r "gs://${BUCKET}/code" "$CODE_DIR"
# API keys come from Secret Manager (never staged in the bucket), fetched via the VM's
# service account. The secret holds the .env contents; write them back to disk for dotenv.
if [ -n "$SECRET_NAME" ]; then
  gcloud secrets versions access latest --secret="$SECRET_NAME" > "${CODE_DIR}/.env" \
    || echo "WARN: could not access secret ${SECRET_NAME}; LLM calls will fail"
else
  echo "WARN: no secret configured; LLM calls will fail"
fi
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
