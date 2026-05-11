#!/usr/bin/env bash
# Launch N GCP VMs, each running one config from RUN_CONFIGS in hypothesis_engine.py.
#
# One-time setup (do this once before first launch):
#   1. gcloud auth login && gcloud config set project YOUR_PROJECT_ID
#   2. Create a GCS bucket and upload code + data + .env:
#        gsutil mb -l us-central1 gs://YOUR_BUCKET
#        gsutil -m rsync -r /home/reilly/Documents/code/EDGAR-main \
#                          gs://YOUR_BUCKET/code \
#                          -x 'program_databases/.*|__pycache__/.*|figures/.*|\.git/.*'
#        gsutil -m rsync -r "/home/reilly/datasets/jacob data" \
#                          "gs://YOUR_BUCKET/datasets/jacob data"
#        gsutil cp /home/reilly/Documents/code/EDGAR-main/.env gs://YOUR_BUCKET/.env
#   3. Make sure your project has GPU quota in the chosen zone (Compute Engine API > Quotas).
#
# Usage:
#   ./launch_gcp.sh                 # launches NUM_RUNS VMs
#   ./launch_gcp.sh teardown        # deletes all VMs created by this script
#
# When done, results are uploaded by each VM to gs://YOUR_BUCKET/results/run_<idx>/
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
START_IDX=0
NUM_RUNS=8
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
USE_SPOT=true
# --------------------

INSTANCE_PREFIX="edgar-run"

if [[ "${1:-}" == "teardown" ]]; then
  for i in $(seq 0 $((NUM_RUNS - 1))); do
    gcloud compute instances delete "${INSTANCE_PREFIX}-${i}" \
      --zone="${ZONE}" --quiet || true
  done
  exit 0
fi

# Startup script that runs on each VM at boot.
# It downloads code+data, installs deps, runs one config, uploads results, then shuts down.
read -r -d '' STARTUP_SCRIPT <<'EOF' || true
#!/bin/bash
set -euxo pipefail
exec > >(tee /var/log/edgar-startup.log) 2>&1

# Wait for NVIDIA driver from Deep Learning VM image to be ready
until nvidia-smi; do sleep 5; done

RUN_IDX=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/run-idx" -H "Metadata-Flavor: Google")
BUCKET=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket" -H "Metadata-Flavor: Google")

mkdir -p /opt/edgar
cd /opt/edgar
gsutil -m rsync -r "gs://${BUCKET}/code" .
mkdir -p "/home/reilly/datasets/jacob data"
gsutil -m rsync -r "gs://${BUCKET}/datasets/jacob data" "/home/reilly/datasets/jacob data"
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
gsutil -m rsync -r program_databases "gs://${BUCKET}/results/run_${RUN_IDX}" || true
gsutil cp /var/log/edgar-startup.log "gs://${BUCKET}/logs/run_${RUN_IDX}.log" || true
NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
ZONE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | awk -F/ '{print $NF}')
gcloud compute instances delete "${NAME}" --zone="${ZONE}" --quiet
EOF

SPOT_FLAGS=()
if $USE_SPOT; then
  SPOT_FLAGS=(--provisioning-model=SPOT --instance-termination-action=DELETE)
fi

for i in $(seq "${START_IDX}" $((NUM_RUNS - 1))); do
  echo "Launching ${INSTANCE_PREFIX}-${i} for run-idx=${i}..."
  gcloud compute instances create "${INSTANCE_PREFIX}-${i}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --accelerator="type=${GPU_TYPE},count=1" \
    --image-family="common-cu129-ubuntu-2204-nvidia-580" \
    --image-project="deeplearning-platform-release" \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=100GB \
    --scopes=cloud-platform \
    --metadata="run-idx=${i},bucket=${BUCKET},install-nvidia-driver=True" \
    --metadata-from-file="startup-script=/dev/stdin" \
    "${SPOT_FLAGS[@]}" <<<"${STARTUP_SCRIPT}"
done

echo
echo "Launched ${NUM_RUNS} VMs. Watch logs with:"
echo "  gcloud compute ssh ${INSTANCE_PREFIX}-0 --zone=${ZONE} -- tail -f /var/log/edgar-startup.log"
echo
echo "VMs self-delete on completion. To force teardown: ./launch_gcp.sh teardown"
echo "Pull results when done:"
echo "  gsutil -m rsync -r gs://${BUCKET}/results ./program_databases_cloud"
