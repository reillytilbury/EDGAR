#!/usr/bin/env bash
# Launch N GCP VMs, each running one config from RUN_CONFIGS in run_configs.py.
#
# One-time setup (do this once before first launch):
#   1. gcloud auth login && gcloud config set project YOUR_PROJECT_ID
#   2. Create a GCS bucket and upload code + data + .env:
#        gsutil mb -l us-central1 gs://YOUR_BUCKET
#        gsutil cp /home/reilly/Documents/code/EDGAR-main/.env gs://YOUR_BUCKET/.env
#      (datasets are uploaded separately — see README)
#   3. Make sure your project has GPU quota in the chosen zone (Compute Engine API > Quotas).
#
# Usage:
#   ./launch_gcp.sh                 # launches one VM per entry in RUN_CONFIGS
#   ./launch_gcp.sh teardown        # deletes all VMs created by this script
#
# Results are uploaded by each VM to gs://YOUR_BUCKET/results/<timestamp>_<dataset>_<idx>/
# Pull them back with:
#   gsutil -m rsync -r gs://YOUR_BUCKET/results ./program_databases_cloud
#
# Cost guide (us-central1, on-demand): n1-standard-4 + T4 ~ $0.45/hr. 4 VMs * 1hr ~ $1.80.
# Use --preemptible/--provisioning-model=SPOT to cut ~70%.

set -euo pipefail

# ---- EDIT THESE ----
PROJECT_ID="reilly-462416"
BUCKET="edgar-revisions-reilly"
ZONE="us-central1-a"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
USE_SPOT=true
# --------------------

INSTANCE_PREFIX="edgar"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Read NUM_RUNS and per-config run_names from run_configs.py (no heavy deps needed)
SCRIPT_DIR="$(dirname "$0")"
NUM_RUNS=$(python3 -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from run_configs import RUN_CONFIGS; print(len(RUN_CONFIGS))")
mapfile -t RUN_NAMES < <(python3 -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from run_configs import RUN_CONFIGS; [print(cfg['run_name']) for cfg in RUN_CONFIGS]")
echo "Detected ${NUM_RUNS} remote configs."

if [[ "${1:-}" == "teardown" ]]; then
  echo "Tearing down all VMs for timestamp ${TIMESTAMP}..."
  gcloud compute instances list \
    --filter="name~^${INSTANCE_PREFIX}-" \
    --format="value(name,zone)" | while read -r name zone; do
      gcloud compute instances delete "${name}" --zone="${zone}" --quiet || true
  done
  exit 0
fi

# Sync local codebase to bucket before launching so VMs always get the latest code.
echo "Syncing codebase to gs://${BUCKET}/code ..."
gsutil -m rsync -r -x 'program_databases/.*|program_databases_cloud/.*|logs_cloud/.*|__pycache__/.*|figures/.*|equation_formatting/.*|\.git/.*|\.env' "${SCRIPT_DIR}" "gs://${BUCKET}/code"
echo "Sync complete."

# Startup script that runs on each VM at boot.
# It downloads code+data, installs deps, runs one config, uploads results, then shuts down.
read -r -d '' STARTUP_SCRIPT <<'EOF' || true
#!/bin/bash
set -euxo pipefail
exec > >(tee /var/log/edgar-startup.log) 2>&1

# Wait for NVIDIA driver from Deep Learning VM image to be ready
until nvidia-smi; do sleep 5; done

RUN_IDX=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/run-idx" -H "Metadata-Flavor: Google")
RUN_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/run-name" -H "Metadata-Flavor: Google")
BUCKET=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket" -H "Metadata-Flavor: Google")

mkdir -p /opt/edgar
cd /opt/edgar
gsutil -m rsync -r "gs://${BUCKET}/code" .
mkdir -p "/home/reilly/datasets/jacob data"
gsutil -m rsync -r "gs://${BUCKET}/datasets/jacob data" "/home/reilly/datasets/jacob data"
mkdir -p "/home/reilly/datasets/stringer_2021"
gsutil -m rsync -r "gs://${BUCKET}/datasets/stringer_2021" "/home/reilly/datasets/stringer_2021"
mkdir -p "/home/reilly/datasets/ali data"
gsutil -m rsync -r "gs://${BUCKET}/datasets/ali data" "/home/reilly/datasets/ali data"
mkdir -p "/home/reilly/datasets/hayley_data"
gsutil -m rsync -r "gs://${BUCKET}/datasets/hayley_data" "/home/reilly/datasets/hayley_data"
gsutil cp "gs://${BUCKET}/.env" .env

# Bootstrap pip if missing, then install all deps including JAX with CUDA support.
apt-get update -qq && apt-get install -y -qq python3-pip
python3 -m pip install -q --upgrade pip
python3 -m pip install -q "jax[cuda12]==0.6.0" \
    timeout-decorator jaxopt optax google-genai python-dotenv tqdm \
    pandas matplotlib pillow scipy numpy anthropic

# Quick JAX sanity check; log it loudly so we can see if CUDA is broken.
python3 -c "import jax; print('JAX_BACKEND:', jax.default_backend()); print('JAX_DEVICES:', jax.devices())" 2>&1 || true

python3 hypothesis_engine.py --run-idx "${RUN_IDX}" || true

# Upload results and the full log no matter what, then self-terminate.
gsutil -m rsync -r program_databases "gs://${BUCKET}/results/${RUN_NAME}" || true
gsutil cp /var/log/edgar-startup.log "gs://${BUCKET}/logs/${RUN_NAME}.log" || true
NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
ZONE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | awk -F/ '{print $NF}')
gcloud compute instances delete "${NAME}" --zone="${ZONE}" --quiet
EOF

SPOT_FLAGS=()
if $USE_SPOT; then
  SPOT_FLAGS=(--provisioning-model=SPOT --instance-termination-action=DELETE)
fi

for i in $(seq 0 $((NUM_RUNS - 1))); do
  DATASET="${RUN_NAMES[$i]}"
  RUN_NAME="${TIMESTAMP}_${DATASET}_${i}"
  INSTANCE_NAME="${INSTANCE_PREFIX}-${TIMESTAMP}-${DATASET}-${i}"
  echo "Launching ${INSTANCE_NAME} (run-idx=${i})..."
  gcloud compute instances create "${INSTANCE_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --accelerator="type=${GPU_TYPE},count=1" \
    --image-family="common-cu129-ubuntu-2204-nvidia-580" \
    --image-project="deeplearning-platform-release" \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=100GB \
    --scopes=cloud-platform \
    --metadata="run-idx=${i},run-name=${RUN_NAME},bucket=${BUCKET},install-nvidia-driver=True" \
    --metadata-from-file="startup-script=/dev/stdin" \
    "${SPOT_FLAGS[@]}" <<<"${STARTUP_SCRIPT}"
done

echo
echo "Launched ${NUM_RUNS} VMs. Watch logs with:"
echo "  gcloud compute ssh ${INSTANCE_PREFIX}-${TIMESTAMP}-${RUN_NAMES[0]}-0 --zone=${ZONE} -- tail -f /var/log/edgar-startup.log"
echo
echo "VMs self-delete on completion. To force teardown: ./launch_gcp.sh teardown"
echo "Pull results when done:"
echo "  gsutil -m rsync -r gs://${BUCKET}/results ./program_databases_cloud"
