---
name: Refactor session handoff
description: Summary of scoring/monitoring refactor progress and outstanding decisions for next session
type: project
---

## Working style
- User has ADHD — be decisive, give one clear recommendation rather than listing options, keep responses short
- Don't add complexity beyond what's asked
- One thing at a time

## What was completed this session

### src/scoring/
- Rewrote `objective.py` → renamed to `scoring.py`. Clean flat structure: `score_with_timeout`, `score_program`, `_get_params`, `_optimize`, `_eval_loss`
- Deleted `param_init.py`, `evaluation.py`, `finalize.py`
- `score_with_timeout` runs `score_program` in a spawned subprocess — subprocess kill IS the timeout, no internal timeout machinery needed
- `score_program(program, data, loss_fn, config)` — `data = (data_train, data_test)` JAX dicts, returns scalar loss + complexity penalty
- param estimator is now JAX and vmapped over samples; `DEFAULT_PARAMS` is the fallback
- loss_fn contract: `(output, data) -> scalar` over full batch (all projects updated to `jnp.mean(...)`)
- Config consolidated: two separate `timeout_s` (param est + GD) replaced with single `scoring.timeout_s`
- Tests added: `src/scoring/tests/test_scoring.py` — 4 tests including XLA subprocess timeout test

### src/evolution/program.py
- Added 3 fields: `mode: str | None`, `temperature: float | None`, `removal_reason: dict | None`
- These replace monitoring-only JSONL fields that couldn't be computed post-hoc

### projects/prompt_defaults.yaml
- `jax_translator` prompt updated: "function" → "code block", removed "update function name", `{function_code}` → `{code_block}`

### projects/config_default.yaml + project configs
- Removed `parameter_estimation` subsection entirely (param est is now JAX, no timeout needed)
- Single `scoring.timeout_s: 120.0`

---

## Outstanding decisions / next steps

### Immediate
- **`monitoring/diagnostics.py`** — delete it, it's dead code (DataFrame-based loss computation replaced by `score_program`)
- **`progress_gd_effect.html`** — unclear whether to keep. Requires `initial_loss` (pre-GD) which `score_program` no longer tracks. Decision: keep and add `initial_loss` to Program, OR drop the plot entirely
- **`monitoring/log.py`** — doesn't belong in monitoring (it's JSONL record-keeping, not visualization). Where does it move? TBD. Also uses DataFrames throughout — needs updating

### Medium term
- **Monitoring refactor (Option 2)**: `family_tree.py` and `progress_monitor.py` should accept a `Population` object instead of a JSONL file path. Population becomes the single source of truth, eliminating the separate generation JSONL entirely. Prompts reconstructable post-hoc via `build_prompt` + parent_ids + mode. Sidebar shows `model_code`/`param_est_code` instead of raw LLM responses
- **`candidates.py`** — grossly outdated. References old prompt formatting system, imports from wrong path (`..engine.diagnostics` should be `..monitoring.diagnostics`). Needs rewrite once monitoring is settled
- **`hypothesis_engine.py`** — still uses old DataFrame-based island API, not migrated to new `island.py` or `Program`/`Population`. Big job, save for later
- **`finalize.py` logic** — was deleted but its non-scoring responsibilities (CSV save, family tree HTML, dedup) need a new home

### Long-term goals
- Single `Program` dataclass as the one representation of a candidate (down from ~4 different representations)
- `Population` as the single persistence layer (replace monitoring JSONL)
- Clean config routing: each module only receives its own subsection (`config["scoring"]`, `config["llms"]`, etc.)
- JAX translator applies to both model AND param estimator codeblocks (prompt already updated)
- `candidates.py` rewrite to match new prompt system

## Current points of tension
- `monitoring/log.py` home: it handles JSONL I/O but doesn't belong in monitoring
- `progress_gd_effect.html`: requires `initial_loss` we no longer track — keep or drop?
- How aggressively to rewrite `candidates.py` vs patch it
