# GCP Cloud Runs

Run EDGAR on Google Cloud with `edgar launch-gcp` — one GPU VM per run, each building its
own environment with `uv sync --frozen`, running `edgar run`, syncing results to a Cloud
Storage bucket, and deleting itself. This page explains how the workflow is put together
and how to operate it. For the quick version, see the "GCP Cloud Runs" section of the
`README`.

## How it works

The most important thing to understand: **your machine never talks directly to the VM.**
Everything is store-and-forward through a Cloud Storage (GCS) bucket, which acts as a
shared drop box:

```
   your laptop  ──HTTPS──▶  GCS bucket  ──HTTPS──▶  the VM
   (launcher)              (gs://…)               (startup script)
     hop 1                  drop box                 hop 2
```

- **Hop 1 (laptop):** `launch-gcp` uploads your working tree to `gs://<BUCKET>/code` and
  your dataset to `gs://<BUCKET>/data/<sha256>/…`. The VM doesn't exist yet.
- **Hop 2 (VM):** at boot the VM downloads code + data *from* the bucket to its local disk.

GCS is an object store (like S3), not a filesystem or a VM. The transfer tools (`gsutil` /
`gcloud storage`) are just HTTPS clients for the GCS API — they upload raw bytes, with no
compilation, no compression, and no archiving (each file is one object; `rsync` only moves
what changed). The only "build" step, `uv sync`, happens **on the VM** from `pyproject.toml`
+ `uv.lock`, after the code arrives.

**There is no SSH in the data path.** SSH is used only when *you* interactively tail a log.
Two things carry the workflow instead:

- **Instructions travel as instance metadata.** The launcher attaches the startup script
  (and per-run values: bucket, config path, data URI, secret name, seed) as VM metadata.
  At boot the VM's guest agent reads them from the internal metadata server and runs the
  script as root.
- **Identity is the VM's service account.** The VM authenticates to GCS and Secret Manager
  with its attached default Compute Engine service account (granted by the `cloud-platform`
  scope) — never your credentials, never SSH keys.

What the startup script does, in order: wait for the NVIDIA driver → pull code from the
bucket → fetch the API keys from Secret Manager → download the dataset (if any) →
`uv sync --frozen` → run `edgar run` under a `timeout` watchdog, with a background job
syncing the output directory to the bucket every ~2 minutes → on exit, a `trap` does a
final sync, writes a `STATUS` sentinel (`SUCCESS`/`FAILED`/`TIMEOUT`), and deletes the VM.
The `trap` guarantees the VM tears itself down even on error, so a failure can never leave a
GPU billing.

## One-time setup

```bash
gcloud auth login && gcloud config set project <PROJECT_ID>
gcloud services enable compute.googleapis.com storage.googleapis.com secretmanager.googleapis.com
gcloud storage buckets create gs://<BUCKET> --location=<REGION>   # private by default
```

Also confirm:
- **GPU quota** in your target region (see [Troubleshooting](#troubleshooting) — this is the
  most common blocker on a fresh project).
- Your `.env` at the repo root holds `GOOGLE_API_KEY` (and optionally `ANTHROPIC_API_KEY`).
- Keep the **bucket and the VM zone in the same region** so the VM's download is free (see
  [Storage & cost](#storage-and-cost)).

## Writing a launch spec

Copy the committed example and edit the `gcp:` block:

```bash
cp projects/gcp_launch.example.yaml gcp_launch.yaml   # gitignored
```

A spec has three parts:

```yaml
gcp:
  project_id: <required>
  bucket: <required>            # existing bucket name, no gs://
  zone: <required>             # must have GPU quota + your GPU type
  machine_type: g2-standard-8   # default; must match the GPU family (see below)
  gpu_type: nvidia-l4           # default
  gpu_count: 1
  spot: true                    # default; cheap + preemptible
  boot_disk_size_gb: 200
  image_family: common-cu129-ubuntu-2204-nvidia-580   # Deep Learning VM, CUDA 12
  image_project: deeplearning-platform-release
  name_prefix: edgar            # VM names = <prefix>-<run-name>
  max_hours: 12                 # wall-clock watchdog per VM
  secret_name: edgar-env        # Secret Manager secret holding your .env

defaults:
  overrides: {}                 # applied to every run; flat dotted keys
  base_seed: 0                  # replica i gets base_seed + i

runs:
  - config: projects/<name>/config.yaml   # required, repo-relative
    run_name: <name>                       # optional; default = project dir name
    n_replicas: 1                          # optional; fans out to distinct VMs + seeds
    overrides: {evolution.n_generations: 20}   # optional; merged over defaults
```

Only `project_id`, `bucket`, `zone`, and each run's `config` are required; everything else
has a default. Overrides use the same `section.key: value` form as `edgar run`, validated
against `{io, evolution, llms, scoring, project_params, run}`. Each `runs` entry becomes one
VM; `n_replicas: N` fans out to `N` VMs named `<run>-r0…-r(N-1)` with seeds `base_seed + i`.

**GPU / machine pairing:** older GPUs (e.g. `nvidia-tesla-t4`) attach to the **N1** family
(`n1-standard-4`); `nvidia-l4` requires the **G2** family (`g2-standard-8`). A mismatch is
rejected at create time.

## Launching

```bash
uv run edgar launch-gcp gcp_launch.yaml --dry-run   # prints every command + the startup script; runs nothing
uv run edgar launch-gcp gcp_launch.yaml             # real launch
```

The launcher runs a local **preflight** (tooling, auth, bucket reachable, configs parse,
data files exist), then: rsyncs the working tree to the bucket, uploads a provenance
manifest, stores your `.env` in Secret Manager, uploads each unique dataset (skip-if-present
by content hash), and creates one VM per run. Under `--dry-run` preflight problems become
warnings so the plan prints on any machine, even without gcloud.

## Monitoring

```bash
gcloud compute instances list                        # which run VMs are still alive (self-delete when done)
gcloud storage ls -r gs://<BUCKET>/results/           # watch files land (each VM syncs ~every 2 min)
gcloud storage cat gs://<BUCKET>/results/*/STATUS     # SUCCESS / FAILED / TIMEOUT per finished run
gcloud compute ssh <vm> --zone=<ZONE> --command='tail -f /var/log/edgar-startup.log'   # tail one run live
```

For a sweep: VMs still shown by `instances list` are still running; an empty list plus a
`STATUS` per run means everything finished. (An interactive SSH session drops when the VM
self-deletes at the end — that's expected, not an error.)

## Fetching results

```bash
uv run edgar launch-gcp gcp_launch.yaml --fetch       # rsync results -> program_databases/
uv run edgar dashboard                                # view the fetched run
```

Fetched runs land in `program_databases/MM-DD/HH-MM-SS/`, so `edgar dashboard` and
`edgar resume` work on them unchanged.

## Secrets (Secret Manager)

API keys are never staged in the bucket. The launcher stores your `.env` in the `edgar-env`
Secret Manager secret (overridable via `gcp.secret_name`), adds a new version only when the
local `.env` changes, and grants the VM's default compute service account the
`secretAccessor` role. Each VM fetches the value at runtime with `gcloud secrets versions
access` and writes it to `/opt/edgar/.env` for `python-dotenv`. Access is revocable and
auditable, and the secret is reused across runs — no per-run cleanup needed.

## Storage and cost

The bucket keeps everything by design — datasets (content-hashed, so reused across runs with
no re-upload), results, and the synced code. Nothing is auto-deleted. The cost model:

| Item | Charged? |
|---|---|
| Upload laptop → GCS (ingress) | **Free** |
| VM download GCS → VM, same region | **Free** (why bucket + zone must share a region) |
| Storage | ~$0.023/GB/month (London Standard; varies by region) |
| `--fetch` download GCS → laptop (egress) | ~$0.12/GB |
| Operations (per-file API calls) | fractions of a cent |

So keeping a dataset costs a few cents a month and saves re-uploading it; the only real money
mover is `--fetch` egress and the fact that **`results/` grows fastest** (one `population.jsonl`
per run) — prune that, not `data/`.

See usage and a rough monthly cost:

```bash
gcloud storage du --summarize gs://<BUCKET> \
  | awk '{printf "%.2f GB  (~$%.3f/month at $0.023/GB)\n", $1/1e9, $1/1e9*0.023}'
gcloud storage du --summarize --readable-sizes gs://<BUCKET>/results gs://<BUCKET>/data gs://<BUCKET>/code
```

Delete what you no longer need — keep-and-pay or fetch-once-then-delete, your call:

```bash
gcloud storage ls gs://<BUCKET>/results/             # list runs first
gcloud storage rm -r gs://<BUCKET>/results/<run>     # one run's results
gcloud storage rm -r gs://<BUCKET>/results           # all results
gcloud storage rm -r gs://<BUCKET>/data              # all cached datasets (re-uploaded next launch)
```

Optional auto-expiry (e.g. delete results older than 14 days):

```bash
echo '{"rule":[{"action":{"type":"Delete"},"condition":{"age":14,"matchesPrefix":["results/"]}}]}' > lifecycle.json
gcloud storage buckets update gs://<BUCKET> --lifecycle-file=lifecycle.json
```

## Provenance and reproducibility

Because `.git/` is excluded from the upload, the remote `task_spec.yaml` records
`git_sha: unknown` **by design** — that's not a bug. Real provenance is captured separately
in `gs://<BUCKET>/code/MANIFEST.txt` (HEAD SHA, dirty flag, and the full working-tree diff at
launch time). The environment is reproduced exactly via `uv sync --frozen` against the
committed `uv.lock`.

## Troubleshooting

**`instances create` fails: image family not found.** Google periodically retires Deep
Learning VM image families (e.g. `common-cu123`). List the current ones and set
`gcp.image_family`:

```bash
gcloud compute images list --project=deeplearning-platform-release \
  --filter="family~common-cu12 AND status=READY" --format="table(name,family)"
```

**`instances create` fails: `STOCKOUT` / resources not available.** The zone has no **spot**
capacity for your GPU right now (spot uses spare capacity). Options: set `spot: false` for a
reliable on-demand VM (a few cents more, no preemption), try another zone, or retry later.
Spot is ~60–90% cheaper but interruptible; the periodic 2-minute sync + `edgar resume` exist
to tolerate preemption on real runs.

**GPU quota is 0 (create fails or you want to check first).** Quota is per-region. Check what
your project has, and which zones offer your GPU:

```bash
gcloud compute regions describe <REGION> --flatten="quotas[]" \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" | grep -Ei "T4|L4|PREEMPTIBLE"
gcloud compute accelerator-types list --filter="zone ~ <REGION>" --format="table(name,zone)"
```

A fresh project usually has GPU quota 0 — request an increase in IAM & Admin → Quotas.

**VM boots but LLM calls fail with an auth error.** The VM couldn't read the secret. Confirm
the `edgar-env` secret exists and that the default compute service account has
`roles/secretmanager.secretAccessor` (the launcher grants this automatically; a restrictive
org policy can block it).

**`ssh … exited with return code [255]` while tailing.** Expected when the VM self-deletes
at the end of a run — your SSH client lost its host, nothing failed. Check the `STATUS`
sentinel to confirm the run's actual outcome.
