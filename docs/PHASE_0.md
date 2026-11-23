# Phase 0: like-for-like EDGAR replication on OpenCode Go

## Goal

Run EDGAR's published discovery process unchanged (8 islands × 6 proposals × 12 generations = 576 programs, same prompts, scorer and selection), with only the LLMs swapped. Then check whether the selected equation beats the double Gaussian on locked test cells, and whether a peaky exponent (fitted p < 2, as in [Tilbury et al.](../papers/README.md)) emerges.

| Run | Code-writing model | Image reading | Budget | Status |
| --- | --- | --- | --- | --- |
| Free | `hy4-flash-free` (all roles) | same model, directly | $0 | done; see [Results](#results-free-run-hy4-flash-free) and [Conclusions](#conclusions) |
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

### Full run up to Phase 0.1 (stopped by decision after 4 of 12 generations)

Artifacts are in [artifacts/phase0_1_full](artifacts/phase0_1_full). The 12-generation free run was paused after generation 4 (240 proposals attempted, all logged) and that point declared the end of Phase 0.1. On 4 CPU cores each generation took 1.3–1.6 h, nearly all of it per-cell fitting, so the full paper-sized run would have needed about 24 h in total.

- **Requests:** 720 of 724 succeeded and none were truncated.
- **Valid programs per generation:** 75%, 77%, 73%, 75%.
- **Best discovery score:** 28.72, against the double Gaussian's 28.85, essentially flat after generation 0.

The best program so far (generation 3, island 7) has the paper's structure: two peaks of the form `A·exp(−(|θ−φ|/σ±)^p)`, a separate exponent and left/right width per peak, and a free peak separation (median fit 169°). On top of that it adds a 4-parameter structured baseline, for 15 parameters against the paper's 11. A check fit on 300 discovery cells gave min(p₁, p₂) a median of **1.20** (IQR 0.88–1.64), with 63% of cells having a dominant-peak exponent below 2. That is peaky, like the paper's p ≈ 1. Because this form was proposed from generation 0, it looks like recall more than discovery. This program has not been through the locked-cell rescore.

## Fitting speed on CPU

Where the time goes, for one 1,055-cell fit of the double Gaussian: the estimator takes 0.1 s and upstream's checks 1.6 s. Each Adam value-and-gradient step takes about 80–106 ms, mostly `sin`, `cos`, `arctan2` and `exp` over 451,540 points. XLA keeps only about 2.3 of 4 cores busy. The time is in the steps themselves, not Python overhead: compiling upstream's loop into one XLA call saved only 15–25%.

Fitters compared on all 1,055 discovery cells with a 1,000-step budget, 4 CPU cores:

| Program | Fitter | Seconds | Train loss | Score |
| --- | --- | ---: | ---: | ---: |
| double Gaussian (5 params) | upstream | 89.0 | 28.2144 | 28.8550 |
| double Gaussian (5 params) | adam (compiled, sharded) | 42.3 | 28.2144 | 28.8549 |
| double Gaussian (5 params) | lm | 8.4 | 28.1960 | 28.8837 |
| gen-3 best (15 params) | upstream | 38.4 | 27.5517 | 28.7230 |
| gen-3 best (15 params) | adam (compiled, sharded) | 26.0 | 27.5517 | 28.7230 |
| gen-3 best (15 params) | lm | 29.8 | 27.4733 | 28.9196 |

- **Compiled, sharded Adam** gives the same answer 1.5–2.1× faster, so `--fitter auto` now uses it whenever JAX has only a CPU.
- **Levenberg–Marquardt** is the better curve fitter: lower training loss, and 10× faster for small models. But it scores *worse* on held-out trials. Upstream's 1,000 small Adam steps never reach convergence, so they act as early stopping. EDGAR's score therefore rewards equations that fit well from the estimator's starting point under that budget. LM removes that regularisation and penalises flexible models more, which would change which equations win. It stays opt-in. A later phase could pair LM with explicit regularisation or validation-based stopping, as the paper's own post-hoc fits do.
- For larger speed-ups, use a GPU. The paper's run took 45 minutes for 576 programs.

## Fast run: under 30 minutes end to end

A full-size search cannot finish in 30 minutes on 4 CPU cores. So the fast run scales the search down and keeps the testing at full size:

- **Search data:** a random 25% of the discovery cells (264 cells). They are written to a reduced recording, which upstream splits 132/132 for its search and its own final evaluation. Selection and locked-test cells are never in that file.
- **Search size:** 4 islands × 6 proposals × 2 generations = 48 proposals. Generation 0 is upstream's thinking tier (reasoning `high`) and generation 1 a routine tier (reasoning `none`).
- **Fitting:** the compiled, core-sharded Adam (`--fitter auto`), with upstream's 1,000 and 2,000 step budgets unchanged.
- **Test:** `rescore` on the full 527 selection and 528 locked cells, exactly as for the other runs.

```bash
JAX_PLATFORMS=cpu uv run --locked python -m edgar_study.cli run --model hy4-flash-free \
  --dataset data/gratings_drifting_GT1_2019_04_12_1.npy \
  --iterations 2 --islands 4 --batch-size 6 --cell-fraction 0.25
```

Artifacts are in [artifacts/fast](artifacts/fast).

- **Search:** 23.6 min, $0. 38 of 48 programs were valid: 67% in generation 0, 92% in generation 1.
- **Rescore:** about 10 min.
- **Request totals** are in [report.json](artifacts/fast/report.json).

| Program | Parameters | Selection score (527 cells) | Locked test score (528 cells) | Locked test MSE |
| --- | ---: | ---: | ---: | ---: |
| Single Gaussian seed | 4 | 30.06 | 29.73 | 2.969 |
| Double Gaussian seed | 5 | 28.86 | 28.56 | 2.851 |
| **Selected program** (gen 1, island 2, batch 5) | 9 | 28.68 | **28.36** | **2.827** |

The [selected program](artifacts/fast/winner.py) is two generalized-Gaussian lobes `A·exp(−½·|d/w|^s)`:
- one sharpness exponent `s`, shared by both lobes;
- a second-lobe width ratio;
- a second-lobe offset from the antipode;
- a skew that makes widths asymmetric.

That is the paper's form with p₁ = p₂ and a coupled width. The LLM itself clipped `s` to [1.4, 3.2], so it could not reach the paper's p ≈ 1. The fitted median still sat at **1.77**, near that floor, which suggests the data wants sharper peaks than the model allowed.

## Conclusions

1. **The pipeline works end to end at $0.** EDGAR's loop runs unchanged on `hy4-flash-free` through the OpenCode Go route. That covers proposals, translation, estimators, fitting, migration, upstream's final evaluation, and a strict validation-then-locked-test rescore.
   - The full run had 720 of 724 requests succeed; the smoke and fast runs had none fail.
   - Valid-program rates were 65–92% per generation.
   - The paid run is the same command with `--allow-paid --model deepseek-v4.1-flash --reasoning-effort low`, plus optionally `--image-model muse-spark-1.3-contributor`, under the $9.50 spend cap.
2. **Every winner beats the double Gaussian on cells it never saw.**
   - Smoke run: locked score 28.35 against 28.56 (Δ +0.22).
   - Fast run: 28.36 against 28.56 (Δ +0.21), with 9 parameters.

   Unpenalized MSE improves by about 1%, consistent with the paper's "relatively modest improvement in its single-cell-based objective".
3. **The paper's equation is recalled more than discovered.**
   - Asymmetric generalized-Gaussian peaks with free exponents appear in generation 0 of every run, in 26 of 31, 34 of 38 and most of 144 scored programs.
   - The best structural match (Phase 0.1, generation 3) fits min(p₁, p₂) ≈ 1.2 on held-in cells, like the paper's p ≈ 1.
   - The *selected* winners are less peaky: p ≈ 2.1 for the smoke run, and 1.77 against a self-imposed floor of 1.4 for the fast run.

   Detecting the paper's population-geometry result would need the rcvPCA analysis, which this phase deferred.
4. **Speed is a compute problem, not a model problem.** On 4 CPU cores, fitting is over 90% of run time. The compiled, core-sharded Adam gives the same scores 1.5–2.1× faster. A GPU is the right host for paper-size runs.
5. **Settings matter for model swaps.** Upstream's Gemini temperatures (1–2) and default reasoning broke the free model. Halving temperatures and mapping Gemini tiers to reasoning efforts fixed it. Any new model should go through `preflight` and a 1-generation smoke run first.
