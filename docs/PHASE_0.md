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

Everything below cost $0.00 in list-equivalent spend, according to the provider's `usage.opencode.list_cost` and the ledger.

### Getting the free model to work

The first smoke attempt ran upstream's settings directly and failed in two ways, both visible in `api_calls.jsonl`:

- **Reasoning on (provider default):** every proposal spent its whole 16,000-token budget on hidden reasoning and returned no code (`finish_reason: length`, `content_chars: 0`).
- **Reasoning off at upstream's Gemini temperatures (1–2):** the reply turned into multilingual word salad after about 30 lines of code.

The fix keeps upstream's structure and changes only the model settings (see [METHOD_DIFFS.md](../METHOD_DIFFS.md)):

- upstream's non-thinking Gemini 2.0 tiers map to reasoning `none`;
- its every-third-generation Gemini 2.5 thinking tier maps to the same model with reasoning `high`;
- all temperatures are halved, so the annealing schedule runs over (0.5, 1].

### Smoke run: 1 generation, 48 proposals

Artifacts are in [artifacts/smoke](artifacts/smoke). The run took 3.9 h of wall time; most of that was upstream's final refit of 63 programs on CPU.

| | Count | Succeeded | Median latency | Input tokens | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Proposals (reasoning `high`) | 48 | 48 | 69 s | 137,856 | 468,733 |
| Translations | 48 | 48 | 4 s | 71,453 | 37,918 |
| Estimators | 48 | 48 | 7 s | 160,962 | 64,353 |

31 of 48 programs fitted and scored (65%). The rest failed in execution, translation or fitting, which upstream counts as failed programs.

| Program | Parameters | Discovery score | Selection score (527 cells) | Locked test score (528 cells) | Locked test MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single Gaussian seed | 4 | 30.02 | 30.06 | 29.73 | — |
| Double Gaussian seed | 5 | 28.85 | 28.86 | 28.56 | 2.851 |
| **Selected program** (gen 0, island 2, batch 4) | 13 | 28.72 | 28.70 | **28.35** | **2.822** |

The selected program improves the locked-test penalized score by 0.22 over the double Gaussian, and unpenalized MSE by 1.0%. [Its code](artifacts/smoke/winner.py) uses two generalized-Gaussian lobes, each with its own width, sharpness exponent and skew, plus a second lobe near (not fixed at) the antipode and a broad sinusoidal background. That is already most of the paper's structure. Its fitted sharpness exponents have a median of about 2.1, so this particular program is Gaussian-like rather than peaky.

Shape exponents appeared immediately: 26 of the 31 scored generation-0 programs fit one (`** shape`, `np.power(|d|/σ, p)` and similar). The paper's model emerged over several generations; here it is the model's first guess. That points to prior knowledge, from this paper or from generalized Gaussians in general, rather than discovery driven by the search.
