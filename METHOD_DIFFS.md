# Method differences

The code reference is [EDGAR at 0943922](https://github.com/reillytilbury/EDGAR/tree/0943922dcb73b12dbee8e440b3e1ff700cb49c10). The paper reference is [Tilbury et al., bioRxiv v3](https://doi.org/10.1101/2025.11.12.688086).

| Item | Pinned implementation and baseline |
| --- | --- |
| Cell eligibility | `hypothesis_engine.main` uses strict concentration >0.55 and activity >0.4. The study plan identifies paper thresholds of ≥0.4 for both. |
| Response preprocessing | `n_pcs=0` with no spontaneous mean subtraction or z-scoring. Take the first `floor(0.4 × recording trial count)` positive-response trials per eligible cell and scale each retained response vector to L2 norm 100. |
| Cell partition | Pinned code permutes eligible cells with JAX seed 42 and splits them 50/50. The default study protocol uses that permutation with 50% discovery, 25% model selection, and the remainder locked for final evaluation. |
| Trial partition | `objective` uses JAX seed 0 and a 50/50 split within each retained trial vector. This baseline preserves preprocessing across both trial halves, including response-dependent cropping and normalization. The cell holdout therefore differs from a protocol that fits preprocessing exclusively on fitting trials. |
| Score | The pinned loss is `10 × MSE + 0.01 × number of parameters`. The run records both raw MSE and the penalized score. |
| Optimizer | The upstream objective uses Adam for up to 1,000 steps. Its `tol` argument does not drive early stopping in the active Adam loop. |
| Search | The pinned code uses eight islands, six proposals per island and `1 + exp(-i / n_iterations)` in the search loop. The baseline invokes only the evaluator on the two fixed seeds. |

No explicit code license file is present in the pinned upstream tree. This fork uses GitHub's native fork relationship and keeps its baseline additions separate from the original source files.
