# Phase 0: like-for-like EDGAR replication on OpenCode Go

## Goal

Run EDGAR's published discovery process unchanged (8 islands × 6 proposals × 12 generations = 576 programs, same prompts, scorer and selection), with only the LLMs swapped. Then check whether the selected equation beats the double Gaussian on locked test cells, and whether a peaky exponent (fitted p < 2, as in [Tilbury et al.](../papers/README.md)) emerges.

| Run | Code-writing model | Image reading | Budget | Status |
| --- | --- | --- | --- | --- |
| Free | `hy4-flash-free` (all roles) | same model, directly | $0 | see [Results](#results-free-run-hy4-flash-free) |
| Paid | `deepseek-v4.1-flash` (all roles) | `muse-spark-1.3-contributor` describes each image | ≤ $10 total | not started |

## Protocol

1. `preflight`: confirm the model is listed, its price, and that it accepts text and images.
2. `run --iterations 1`: smoke run, 48 proposals.
3. `run --iterations 12`: full run, 576 proposals.
4. `rescore`:
   - candidates: the top 10 unique learned programs by discovery loss;
   - choose the winner by penalized score on the 527 selection cells;
   - score the winner and both Gaussian seeds once on the 528 locked test cells.

Stop rules:
- an auth or eligibility error stops everything;
- more than half of the smoke run's programs failing triggers inspection before the full run;
- the spend ledger refuses requests at $9.50;
- for the paid run, the full run goes ahead only if the smoke run's cost per generation × 12 fits within the remaining budget.

Scores are upstream's: `10 × MSE + 0.01 × parameters` on held-out trials, lower is better.

## Why this plan is smaller than the original study plan

The original plan was sound on scientific caution but rebuilt infrastructure EDGAR already has. What changed:

- **One integration point.** Every LLM call in upstream goes through `utils.call_llm_async`. Replacing that function (plus forcing the Gemini prompt family and stubbing the Google client) is the whole model swap, about 60 lines. Search, scoring, persistence and plots are upstream's. No new provider, protocol, sandbox, event-log or reporting modules.
- **One protocol during search.** Upstream's own 50/50 discovery/evaluation split is kept. The strict holdout is applied afterwards by `rescore`. Discovery cells match upstream's training half exactly, so this is possible without changing the search. Upstream's last step did see the locked cells, which this report discloses.
- **Logging proportional to need.** One JSONL row per request (role, status, latency, provider `usage`) plus upstream's `programs_db.csv`, census and log. The 20-field candidate schema and 15-way failure taxonomy can wait until there are failures to classify.
- **No container sandbox.** The run happens in a disposable cloud container, and upstream already executes generated code with timeouts. The key is read from a file rather than the environment and never logged.
- **No seed series.** Upstream's randomness is already fixed (`PRNGKey(42)` for cells, `PRNGKey(0)` for trials). The real variation is LLM sampling, which no seed controls. Repeat runs are labelled by run ID.
- **Shorter run ladder.** Preflight → 1 generation → one paper-sized run. Decide on repeats after seeing it.
- **rcvPCA deferred.** Reproducing the eigenspectrum slopes (4.26 / 4.34) is a separate project. The cheap first check is whether the winning equation has a fitted shape exponent below 2.
- **Added threat: contamination.** The paper (Nov 2025) may be in a model's training data. Proposing the asymmetric generalized-Gaussian form in generation 0 would suggest recall rather than discovery, so the report notes when a `|x|^p` term first appears.

## Results: free run (`hy4-flash-free`)

_Pending._
