# How to run EDGAR

Copy-paste reference for invoking the pipeline from the CLI. Three modes, with
escalating cost and realism, then a full kitchen-sink invocation that exposes
every config knob as a CLI override.

If you've never run this repo before, read **Prerequisites** and run **Mode 3**
(`test-fake`) first. It exercises the full pipeline in ~30 seconds without
making any API calls.

---

## Prerequisites

```bash
# 1. Activate the conda env (one-time setup; see README for env creation)
conda activate edgar

# 2. Install the repo as a package (one-time)
pip install -e .

# 3. Verify your API keys are in .env (gitignored)
#    The default project uses Claude (ANTHROPIC_API_KEY); GOOGLE_API_KEY is only
#    needed if you override the model to a `gemini-*` one.
grep -E "ANTHROPIC_API_KEY|GOOGLE_API_KEY" .env

# 4. Sanity-check a project's structure before running it
edgar validate orientation_tuning
```

The default project uses Anthropic Claude. Costs are pay-per-token, no rate
limit that bites at this scale. If you override to `gemini-*` models, be aware
free-tier Gemini is **5 req/min on `gemini-2.5-flash`** — a default run issues
~1,700 LLM calls and won't finish on free tier even with overrides; either
enable billing or stay on Claude.

---

## Mode 3 — fake LLM dry run (no API calls)

Use this whenever you've touched pipeline code and want to catch plumbing bugs
without spending tokens. ~30 seconds.

```bash
edgar test-fake
```

Pulls fake responses from `tests/system/fake_runner.py`. Catches config-loading,
file-I/O, and scoring-sandbox issues. Does **not** validate prompt quality or
LLM-output parsing (the responses are pre-canned).

---

## Mode 2 — real-LLM smoke run (Recommended first real invocation)

Runs end-to-end with real API calls and reduced settings: 1 generation × 2
islands × 2 batches. ~5-10 min. Output lands in `./test_output/` (separate
from `program_databases/`) so it doesn't pollute real runs.

```bash
edgar test projects/orientation_tuning/config.yaml
```

The reduced settings are hard-coded in `edgar/cli.py:TEST_OVERRIDES`. You can
still pass `--section.key=value` overrides on top of them.

This catches:

- API key works
- Seed programs compile and score on real data
- `plot_fn` runs without errors (images appear under `test_output/image_feedback/`)
- `population.jsonl`, `family_tree.html`, `island_census.jsonl` all written

---

## Mode 1 — full run

```bash
edgar run projects/orientation_tuning/config.yaml
```

Uses every setting from `projects/orientation_tuning/config.yaml` merged on top
of `projects/config_default.yaml`. Output lands in
`program_databases/YYYY-MM-DD/HH-MM-SS/`. `Ctrl-C` is safe — the population is
written to `population.jsonl` as the run progresses, so a killed run still
leaves a partial inspectable run dir.

### With ad-hoc overrides

Override any config key with `--section.key=value`. Values are parsed as Python
literals where possible (`int`, `float`, `bool`, list).

```bash
edgar run projects/orientation_tuning/config.yaml \
  --evolution.n_generations=20 \
  --llms.model_llm=gemini-2.5-pro \
  --io.data_path=/data/other_recording.npy
```

### Re-run from a saved task_spec

Every run writes `task_spec.yaml` to its output dir **at run start** (right
after the dir is created), so the file exists for *finished, killed, and
crashed runs alike*. Passing that file instead of `config.yaml` reproduces the
run **exactly** — same config, same seeds, same LLM names:

```bash
edgar run program_databases/2026-05-24/17-17-45/task_spec.yaml
```

The relaunch gets a fresh `creation_timestamp`, so it lands in a new dir
(`program_databases/YYYY-MM-DD/HH-MM-SS/`); the original run dir is never
overwritten.

**Important:** this is "re-run from scratch using the same recipe", not
"continue where we left off". The killed run's `population.jsonl`,
`island_census.jsonl`, and `image_feedback/` are **not** loaded; generation 0
starts over. If you want to keep the evolved programs from a killed run, use
`edgar resume` instead (see next section).

#### Use cases

- **Reproducibility check.** Re-run a finished experiment against the same
config to confirm determinism, or against a new code version to see if a
refactor changed the result.
- **Cold relaunch when you don't want the prior population.** Useful if you
want to retry a config with a different random seed or different LLM choice
and start fresh.

### Resume from a checkpoint (continue an interrupted run)

If a run was killed mid-flight (Ctrl-C, network drop, process crashed, laptop
closed) and you want to **continue from where it stopped**, use:

```bash
python -m edgar.cli resume program_databases/05-26/14-54-15/
```

This reads the run's `task_spec.yaml`, restores the population and island
membership from disk, skips the seed phase (already done), and picks up at
the next unfinished generation. Outputs are appended back into the same
directory; `run.log` gets a `──── RESUMED ────` banner so you can tell where
the original run ended and the resumed work began.

Refuses to resume in a few obvious cases (with clear error messages):
already-completed runs, empty populations (seed phase never finished), and
runs that already reached the validation phase.

Caveat: `spec.rng` state is not restored across resume, so spawning/migration
draws diverge from what a continuous run would have produced. LLM responses
are non-deterministic anyway, so bit-reproducibility was never on offer.

### Logging verbosity

```bash
edgar run projects/orientation_tuning/config.yaml --log-level compact   # default — one line per generation
edgar run projects/orientation_tuning/config.yaml --log-level code      # also print LLM-generated code
edgar run projects/orientation_tuning/config.yaml --log-level prompts   # firehose: every prompt sent to the LLM
```

---

## Kitchen-sink invocation (every key exposed)

Copy this block, prune what you don't need to override. Every key here has a
default in `projects/config_default.yaml` and almost all of them are optional
on the CLI — but having them all in one place means you never need to remember
which section a key lives in.

```bash
edgar run projects/orientation_tuning/config.yaml \
  --log-level compact \
  \
  --run.random_seed=42 \
  \
  --io.data_path=data/gratings_drifting_GT1_2019_04_12_1.npy \
  --io.save_path=program_databases \
  \
  --evolution.n_generations=12 \
  --evolution.n_islands=8 \
  --evolution.batch_size=6 \
  --evolution.critical_population_size=12 \
  --evolution.n_migrants=2 \
  --evolution.topology="[1, 2, 3, 4, 5, 6, 7, 0]" \
  \
  --llms.num_parents=2 \
  --llms.model_llm=claude-sonnet-4-6 \
  --llms.param_est_llm=claude-sonnet-4-6 \
  --llms.jax_model_translator_llm=claude-haiku-4-5 \
  --llms.max_tokens=10000 \
  --llms.log_raw_llm_response=False \
  --llms.max_lines=50 \
  \
  --scoring.param_penalty_weight=0.01 \
  --scoring.timeout_s=120.0 \
  --scoring.gradient_descent.max_iter=1000 \
  --scoring.gradient_descent.learning_rate=0.003 \
  \
  --project_params.activity_threshold=0.4 \
  --project_params.conc_threshold=0.55 \
  --project_params.random_seed=42 \
  --project_params.n_eval_samples=100
```

### Notes on individual keys


| Section          | Key                              | What it controls                                                                                                                                                |
| ---------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run`            | `random_seed`                    | Seeds spawn/migration RNG; `null` for non-deterministic                                                                                                         |
| `io`             | `data_path`                      | Input data file (`.npy`) consumed by `load_data`                                                                                                                |
| `io`             | `save_path`                      | Output base dir; runs land in `<save_path>/YYYY-MM-DD/HH-MM-SS/`                                                                                                     |
| `evolution`      | `n_generations`                  | Outer loop iterations                                                                                                                                           |
| `evolution`      | `n_islands`                      | Independent populations evolving in parallel                                                                                                                    |
| `evolution`      | `batch_size`                     | Children per island per generation (= LLM call fan-out)                                                                                                         |
| `evolution`      | `critical_population_size`       | Hard cap per island; pruned down to this minus `n_migrants`                                                                                                     |
| `evolution`      | `n_migrants`                     | Programs swapped between islands each generation                                                                                                                |
| `evolution`      | `topology`                       | List defining migration graph; must be a permutation of `0..n_islands-1`                                                                                        |
| `llms`           | `num_parents`                    | Parents shown to LLM per prompt                                                                                                                                 |
| `llms`           | `model_llm`                      | Model that writes the numpy `model()` code. `claude-*` → Anthropic, `gemini-*` → Google (provider inferred from prefix). Can be a list (cycled per generation). |
| `llms`           | `param_est_llm`                  | Model that writes `parameter_estimator()` code. Same naming rules as `model_llm`.                                                                               |
| `llms`           | `jax_model_translator_llm`       | Model that translates numpy → JAX. Translation is mechanical — keep this on a small/cheap model (Haiku, Flash-Lite).                                            |
| `llms`           | `max_tokens`                     | Per-call output limit (Anthropic requires this; Google uses server default if `None`)                                                                           |
| `llms`           | `log_raw_llm_response`           | Print raw model response parts after every call (debug only)                                                                                                    |
| `llms`           | `max_lines`                      | Max lines allowed in a parameter estimator response                                                                                                             |
| `llms`           | `retry.*`                        | HTTP retry policy — best edited in the project's YAML, not on the CLI                                                                                           |
| `scoring`        | `param_penalty_weight`           | Complexity penalty: `final_loss += weight * n_params`                                                                                                           |
| `scoring`        | `timeout_s`                      | Wall-clock kill for each scoring subprocess                                                                                                                     |
| `scoring`        | `gradient_descent.max_iter`      | Max optimizer steps when fitting parameters                                                                                                                     |
| `scoring`        | `gradient_descent.learning_rate` | Adam LR for parameter fitting                                                                                                                                   |
| `project_params` | (any)                            | Forwarded as `**kwargs` to `load_data` — keys are project-specific                                                                                              |


`llms.retry` is a nested dict (`max_retries`, `initial_delay`, `backoff_multiplier`,
`max_delay`, `retryable_status_codes`). The CLI override format flattens awkwardly
for nested dicts — easier to edit it directly in the project's `config.yaml`.

---

## Output anatomy

Every successful run writes:

```
program_databases/YYYY-MM-DD/HH-MM-SS/
├── task_spec.yaml         # frozen config + git sha + seed source (re-runnable, read-only)
├── population.jsonl       # one Program per line: birth, code, losses, fitted params
├── island_census.jsonl    # per-generation per-island membership (single JSON doc despite the name)
├── family_tree.html       # interactive parent-child graph, open in Chrome
├── run.log                # human-readable per-generation summary
└── image_feedback/        # plot_fn outputs shown to the LLM (one per spawn)
    └── gen_NNN/island_NNN/batch_NNN/image.png
```

Inspect a finished (or partial) run with `tutorials/inspect_outputs.py` —
change `RUN_DIR` at the top to point at the new timestamp.

---

## Watching a run in flight

In a second terminal:

```bash
# Latest real run, live (mode 1)
tail -f "$(ls -td program_databases/*/*/ | head -1)run.log"

# Latest test run, live (mode 2 — note: lands in ./test_output/, not program_databases/)
tail -f "$(ls -td test_output/*/*/ | head -1)run.log"

# Family tree (rewritten every generation — refresh the browser tab)
open -a "Google Chrome" "$(ls -td program_databases/*/*/ | head -1)family_tree.html"
```

Run directories are nested two deep (`<save_path>/YYYY-MM-DD/HH-MM-SS/`), so any
manual glob needs **two** stars: `program_databases/*/*/run.log`, not `*/run.log`.
zsh errors on no-match by default — quote the `$(...)` substitution as shown
above to sidestep that.

`population.jsonl` and `island_census.jsonl` are written **incrementally**: a
snapshot lands at the end of every generation (atomic rename, so a polling
reader can never see a torn file). `status.json` tracks the run state
(`starting` / `running` / `complete` / `failed`). The live dashboard
(`python -m edgar.cli dashboard <run_dir>`) polls these files; `run.log`
(per-generation summary) and `image_feedback/*.png`
(per-spawn plots).

`Ctrl-C` is safe — the `finally` block catches `KeyboardInterrupt` and still
saves all of the above. `kill -9` skips `finally`; you lose everything.

---

## Common operations

```bash
# Create a new project scaffold
python -m edgar.cli init-project my_new_task

# Validate that a project has all required files / entry points
python -m edgar.cli validate my_new_task

# List the projects you have
ls projects/
```

`init-project` creates seed-program stubs, `load_data` / `loss_fn` stubs, a
`plot_fn` stub, and a `config.yaml`. Fill in the stubs, then `validate`, then
run Mode 3 → Mode 2 → Mode 1.

---

## Troubleshooting


| Symptom                                                                  | Likely cause                                                      | Fix                                                                                 |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `UserError: ANTHROPIC_API_KEY is not set` / `GOOGLE_API_KEY is not set`  | Missing key in `.env` for the provider implied by your model name | Add to `.env`; `load_dotenv()` reads it on `call_llm` import                        |
| `[call_llm] HTTP 429` repeatedly on `gemini-*`                           | Free-tier Gemini rate limit (5 req/min on flash)                  | Stay on Claude (default), enable Gemini billing, or reduce `n_islands * batch_size` |
| Scoring hangs then `inf` loss                                            | LLM-generated code hit `scoring.timeout_s`                        | Expected for pathological programs; check `code.model_jax` in the offender          |
| `RuntimeError: asyncio.run() cannot be called from a running event loop` | You called the CLI from a Jupyter cell                            | Run from a terminal, or use the notebook-mode walkthrough                           |
| `ValueError: topology length (...) must equal n_islands (...)`           | `topology` and `n_islands` out of sync after override             | Pass both or neither                                                                |
| Output goes to wrong dir                                                 | Forgot `--io.save_path` override                                  | Defaults to `program_databases/`; for tests use Mode 2 (lands in `./test_output/`)  |


