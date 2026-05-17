# EDGAR

## GCP Cloud Runs

### One-time dataset upload
Datasets only need to be uploaded once (or when they change). Run these manually:

```bash
# Stringer 2021 (drifting gratings only)
gsutil -m cp "/home/reilly/datasets/stringer_2021/gratings_drifting_GT1_2019_04_12_1.npy" "/home/reilly/datasets/stringer_2021/gratings_drifting_GT2_2019_04_05_1.npy" "/home/reilly/datasets/stringer_2021/gratings_drifting_GT3_2019_04_05_1.npy" gs://edgar-revisions-reilly/datasets/stringer_2021/

# Jacob data
gsutil -m rsync -r "/home/reilly/datasets/jacob data" "gs://edgar-revisions-reilly/datasets/jacob data"

# Ali data
gsutil -m rsync -r "/home/reilly/datasets/ali data" "gs://edgar-revisions-reilly/datasets/ali data"

# Hayley data
gsutil -m rsync -r "/home/reilly/datasets/hayley_data" "gs://edgar-revisions-reilly/datasets/hayley_data"
```

### Launch runs
The codebase is automatically synced to the bucket before VMs are launched.

```bash
./launch_gcp.sh
```

#### Configurable variables (top of `launch_gcp.sh`)

| Variable | Default | Description |
|---|---|---|
| `PROJECT_ID` | `reilly-462416` | GCP project ID |
| `BUCKET` | `edgar-revisions-reilly` | GCS bucket name for code, data, and results |
| `ZONE` | `us-central1-a` | GCP zone to launch VMs in — must have GPU quota |
| `MACHINE_TYPE` | `n1-standard-4` | VM machine type |
| `GPU_TYPE` | `nvidia-tesla-t4` | GPU type attached to each VM |
| `USE_SPOT` | `true` | Use SPOT (preemptible) pricing — ~70% cheaper but VMs can be interrupted |

### Kill/teardown VMs
```bash
./launch_gcp.sh teardown
```

To kill a single VM:
```bash
gcloud compute instances delete edgar-run-0 --zone=us-central1-a --quiet
```

### Monitor a running VM
```bash
gcloud compute ssh edgar-run-0 --zone=us-central1-a -- tail -f /var/log/edgar-startup.log
```

List all running VMs:
```bash
gcloud compute instances list --filter="name~edgar-run"
```

### Pull results when done
```bash
gsutil -m rsync -r gs://edgar-revisions-reilly/results ./program_databases_cloud
```

Pull logs:
```bash
gsutil -m rsync -r gs://edgar-revisions-reilly/logs ./logs_cloud
```
