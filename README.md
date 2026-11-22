# EDGAR × DeepSeek

This fork starts with the offline baseline for the DeepSeek V4.1 Flash study. It fits EDGAR's original single- and double-Gaussian seeds with the upstream JAX evaluator, records Dataset 1 provenance, and saves exact cell and trial indices. Model integration and evolutionary search are pending.

## Run the offline baseline

From a Git clone of this fork, with Git, Python 3.12 and uv available:

```bash
uv sync --locked
JAX_PLATFORMS=cpu uv run --locked python -m edgar_study.cli baseline \
  --synthetic --output results/synthetic
```

For the original Stringer et al. recording, download the 859 MB file from [Janelia Figshare](https://janelia.figshare.com/articles/dataset/8279387). Figshare lists the dataset under CC BY-NC 4.0.

```bash
mkdir -p data
curl --fail --location https://ndownloader.figshare.com/files/15479246 \
  --output data/gratings_drifting_GT1_2019_04_12_1.npy

JAX_PLATFORMS=cpu uv run --locked python -m edgar_study.cli baseline \
  --dataset data/gratings_drifting_GT1_2019_04_12_1.npy \
  --output results/dataset1
```

The runner checks the recording against the pinned SHA-256 before loading it. It verifies the six upstream Python modules against Git commit `0943922dcb73b12dbee8e440b3e1ff700cb49c10` before importing them. Use `--upstream /path/to/EDGAR` to point at a separate Git checkout.

Each new output directory contains `baseline.json` and `splits.npz`. The default `strict-holdout-v1` protocol reserves the locked test cells and scores discovery and model-selection cells. Pass `--protocol paper-fidelity` for the upstream 50/50 cell partition. See [METHOD_DIFFS.md](METHOD_DIFFS.md) for the preprocessing and scoring choices.

## Dataset 1 result

The pinned recording contains 20,616 cells and 4,282 trials. The upstream filters retained 2,110 cells and 1,712 trials per eligible cell. The default protocol assigned 1,055 discovery, 527 model-selection, and 528 locked test cells.

| Gaussian seed | Discovery MSE | Selection MSE | Discovery penalized score | Selection penalized score |
| --- | ---: | ---: | ---: | ---: |
| Single | 2.997599 | 2.998546 | 30.015987 | 30.025467 |
| Double | 2.880491 | 2.877630 | 28.854914 | 28.826298 |

The baseline uses Linux CPU, Python 3.12.14 and JAX 0.6.0. No provider credentials or LLM requests are needed.
