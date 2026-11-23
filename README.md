# EDGAR × OpenCode Go

This fork runs [EDGAR](https://github.com/reillytilbury/EDGAR)'s published equation-discovery loop unchanged, with OpenCode Go models in place of Gemini. The upstream modules stay byte-identical to commit `0943922`; `edgar_study/` swaps only the LLM call layer at runtime.

- [docs/PHASE_0.md](docs/PHASE_0.md): the like-for-like replication, its protocol, and results.
- [docs/ROADMAP.md](docs/ROADMAP.md): Phase 1 (agentic harness) and Phase 2 (custom harness) ideas.
- [METHOD_DIFFS.md](METHOD_DIFFS.md): every known difference from the paper and from upstream.

Use [uv](https://docs.astral.sh/uv/) with Python 3.12. The upstream `requirements.txt` is kept for reference only; `pyproject.toml` and `uv.lock` are authoritative.

## Setup

```bash
uv sync --locked
```

```bash
mkdir -p data
curl --fail --location https://ndownloader.figshare.com/files/15479246 \
  --output data/gratings_drifting_GT1_2019_04_12_1.npy
```

Dataset 1 is from [Stringer et al. on Janelia Figshare](https://janelia.figshare.com/articles/dataset/8279387) (859 MB, CC BY-NC 4.0). Every command that loads it checks it against a pinned SHA-256.

Store the OpenCode Go key in a file outside the repository and point `OPENCODE_API_KEY_FILE` to it. `OPENCODE_API_KEY` also works. Request logs never contain the key.

## Offline baseline

Fits the single- and double-Gaussian seeds with the upstream JAX evaluator. Needs no key.

```bash
JAX_PLATFORMS=cpu uv run --locked python -m edgar_study.cli baseline \
  --dataset data/gratings_drifting_GT1_2019_04_12_1.npy --output results/dataset1
```

`--synthetic` replaces `--dataset` for a quick check. The default `strict-holdout-v1` protocol splits the 2,110 eligible cells into 1,055 discovery, 527 model-selection and 528 locked test cells. Discovery cells are identical to upstream's training half.

| Gaussian seed | Discovery MSE | Selection MSE | Discovery penalized score | Selection penalized score |
| --- | ---: | ---: | ---: | ---: |
| Single | 2.997599 | 2.998546 | 30.015987 | 30.025467 |
| Double | 2.880491 | 2.877630 | 28.854914 | 28.826298 |

## Search runs

```bash
uv run --locked python -m edgar_study.cli preflight --model hy4-flash-free
```

```bash
JAX_PLATFORMS=cpu uv run --locked python -m edgar_study.cli run --model hy4-flash-free \
  --dataset data/gratings_drifting_GT1_2019_04_12_1.npy --iterations 12
```

```bash
JAX_PLATFORMS=cpu uv run --locked python -m edgar_study.cli rescore \
  --run results/runs/<run-id> --dataset data/gratings_drifting_GT1_2019_04_12_1.npy
```

Each run directory holds:
- `manifest.json`: models, settings, git SHAs and spend before and after.
- `api_calls.jsonl`: one row per request attempt with role, status, latency and the provider's `usage` object.
- `translations.jsonl`: NumPy → JAX translations, used by `rescore`.
- upstream's own `program_databases/` output.

`rescore` writes `rescore.json`.

### Spending safeguards

- Only models priced at zero (`hy4-flash-free`, `beam-free`, `iceman-contributor-free`) can be called unless you pass `--allow-paid`.
- Every request's list-equivalent cost is appended to `results/spend.jsonl`. New requests are refused once that ledger reaches $9.50.

The paid run is the same command with the models swapped:

```bash
JAX_PLATFORMS=cpu uv run --locked python -m edgar_study.cli run --allow-paid \
  --model deepseek-v4.1-flash --image-model muse-spark-1.3-contributor \
  --dataset data/gratings_drifting_GT1_2019_04_12_1.npy --iterations 12
```

## Tests

```bash
uv run --locked python tests.py
```

```bash
uv run --locked python -m edgar_study.test_go_client
```
