# Where the wall-clock and dollars go in a 1-D EDGAR run

Profile of the run at `program_databases/08-01/17-51-50/` (task
`projects/fhn_excitable`). This is the "1-D" variant of EDGAR: one scalar
observation `y[t]` per time step, per trajectory.

The reader is assumed to know Python and NumPy but not JAX. JAX-specific
concepts (`jit`, `vmap`, `lax.scan`, "trace", "compile") are explained the
first time they appear.

---

## What data I used

Every number below comes from files under the run directory. If a number can
only be estimated, it is flagged with "estimate" and the assumption is spelled
out. Nothing was invented.

| Source file | What I read | Used for |
|---|---|---|
| `program_databases/08-01/17-51-50/metrics.jsonl` | Per-generation `stage_times`, `llm_calls` (n, out_tokens, in_tokens, latency p50/p90/max/mean), `scoring` (n, ok/timeout/inf, latency stats) | Every wall-clock and token number in the tables |
| `program_databases/08-01/17-51-50/run.log` | Compact human-readable version of the same data + per-program tick lines during scoring | Cross-check of the per-generation breakdown |
| `program_databases/08-01/17-51-50/status.json` | `started_at` / `updated_at` timestamps | Total wall-clock reference |
| `program_databases/08-01/17-51-50/population.jsonl` (44 records) | Per-program source code (`code.model`, `code.param_est`, `code.model_jax`), losses, birth metadata | Code-length statistics used in the input-token estimate |
| `projects/fhn_excitable/config.yaml` | `gradient_descent.max_iter=500`, `learning_rate=0.005`, `timeout_s=600`, `n_trajectories=32`, `T=2400`, `batch_size=2`, `n_islands=2` (this run overrode `batch_size` to 4 via CLI — see `run.log` line 5) | Interpretation of the scoring stage |
| `edgar/scoring/scoring.py` | How each program is scored (subprocess, `jit`, `vmap`, gradient descent loop) | Explanation of the score stage's internal structure |
| `projects/fhn_excitable/data_loader/load_data.py` | `apply_model` wraps model in a `jax.lax.scan` over the T=2400 trajectory, then `vmap`s across the 32 trajectories | Explanation of the scan and its compile cost |
| `nvidia-smi` | RTX A4000, 16 GB, driver 535, CUDA 12.2 | Compute-host context |

**Coverage caveat.** The run was killed *during* the scoring stage of generation 5 (`status.json.current_stage = "score (1/8)"`). The
metrics file only recorded through generation 4. My totals cover **seed +
generations 0–4** (five completed evolution units, 44 programs, roughly 114
minutes of stage time, matching the 120 min wall clock). The partial gen 5 is
excluded from every table because it never wrote a summary.

**Token caveat.** The `claude-code-headless` adapter records `in_tokens` from
the CLI envelope (`edgar/llm/claude_headless.py:274-279`). That number is
consistently 7 tokens per call in the log — the CLI evidently reports the
short human-readable "result" text length rather than the true prompt-side
usage. The output-token counts look real (they scale sensibly with the
generated code length). So for token cost I trust the output side, and I
estimate the input side from prompt/code length, flagged as an estimate.

---

## Top-line numbers

- Wall clock (kill time − start time from `status.json`): **7183 s ≈ 120 min**.
- Sum of `stage_times` across seed + gen 0–4: **6861 s ≈ 114 min**. The
  ≈5 min gap is idle time between generations (population save, dashboard
  writes, and Python-level bookkeeping that the timing decorator doesn't
  wrap).
- 44 programs produced: 4 seed + 5 generations × 2 islands × batch_size=4
  = 40 spawned. Every non-seed (generation, island) cell in
  `population.jsonl` has exactly 4 programs.
- 41 of 44 scored with a finite loss (3 hit NaN, none timed out).
- Compute host: one workstation, NVIDIA RTX A4000 (16 GB), running for ~2 h.

---

## The two dominant costs

Everything else is under 5 %. The two big buckets:

| Bucket | Wall-clock (s) | % of stage-time wall |
|---|---:|---:|
| **Scoring** (subprocess + JIT compile + gradient descent) | 5443 | **79 %** |
| **LLM waiting** (all three call sites combined) | 1105 | **16 %** |
| Plotting per-program fits (`generate_program_fits`) | 294 | 4 % |
| Population save, dedup, prune, migrate, spawn | ≈15 | <1 % |

Note the LLM row is *wall-clock waiting*, not the sum of per-call latencies.
The framework issues the 8 calls of each stage in parallel with
`asyncio.gather` (`edgar/llm/generate.py`), so wall-clock ≈ slowest call, not
sum. In elapsed-request-time terms the LLM did ~6300 s of work compressed into
~1100 s by concurrency (~6× parallel — see the per-stage table below).

---

## 1. Scoring — 79 % of wall clock

This bucket runs the JAX program: estimate initial parameters, JIT-compile
the loss-and-gradient function, do up to 500 Adam steps, then compute a
handful of eval quantities. It runs *sequentially* over programs (one after
another), because JAX's XLA runtime doesn't share cleanly across programs
that have different pytree state shapes.

**Per-program mean: 123.7 s. Range across generations: 64 s (best gen 0) to
220 s (gen 2). Median of individual programs: ~110–130 s. Worst single
program: 380 s (gen 2). Timeout is 600 s and was never hit.**

### What actually runs per program

From `edgar/scoring/scoring.py::_worker`, for each program:

1. **Subprocess spawn + import JAX.** The framework runs each program in a
   fresh Python subprocess (`ctx = mp.get_context("spawn")`) to isolate
   JAX/XLA state and to be able to hard-kill runaway code on timeout. A cold
   `import jax; import jax.numpy` on a CUDA-enabled node needs to initialise
   the XLA runtime and the CUDA driver context; this is ~10–20 s on the
   A4000. Not measurable per program in the current logs — only the total
   `_score_one_with_outcome` latency is recorded.

2. **Load the two programs.** The generated `model.py` (JAX) and
   `param_est.py` (NumPy) are `exec`'d into a namespace and their `model` /
   `parameter_estimator` functions grabbed. Milliseconds.

3. **Run `parameter_estimator` on each of 32 trajectories to build `params_init`.**
   Pure NumPy, ~100 ms total.

4. **First-call JIT of `loss_and_grad`.** This is the big one. Explanation
   follows in section 2.

5. **500 Adam gradient-descent steps.** After the compile, each step is a
   single kernel launch that runs one forward pass of the scan (T=2399
   steps), the backward pass, and the Adam update. Section 3.

6. **Auxiliary evals** — initial loss on test split, final loss on test
   split, per-sample losses, fingerprint on eval split. Each of these
   triggers *another* JIT because it's a different traced function (loss
   over test data with fixed params is a different computation graph from
   loss+grad over train data). Section 2 again, with a per-program
   multiplier of 3–4.

### Why the range is wide

Per-generation means: 64 s (gen 0), 126 s (gen 1), 221 s (gen 2), 110 s
(gen 3), 136 s (gen 4). The upward drift with generation is consistent with
the LLM emitting increasingly complex programs (more state variables, RK4
integrators, sigmoid gates), which have more expensive scan bodies and
therefore both longer compile times and longer per-step forward+backward
times. The gen-2 peak matches a run of programs using RK4 (four function
evaluations per step) and "Excitable Filter with Dual-Timescale Recovery"
variants — these have larger XLA graphs.

---

## 2. JAX JIT compile time — inside the scoring bucket

### What "JIT compile" means

JAX is a NumPy-like library that traces Python functions into a symbolic
computation graph and then hands that graph to XLA (an ahead-of-time
compiler that produces GPU kernels). `jax.jit(f)` returns a wrapped `f`; the
*first* time you call the wrapped `f` with a given set of argument shapes
and dtypes, JAX:

1. **Traces** the Python function: it runs `f` once with symbolic
   placeholders (called "tracers") in place of the real arrays, recording
   every operation into an IR called Jaxpr. Any Python control flow you use
   (`if`, `for`) that depends on tracer values will fail here — you have to
   use JAX's structured primitives (`lax.scan`, `lax.cond`, `lax.while_loop`).
2. **Compiles** the Jaxpr with XLA into a fused GPU kernel.

Subsequent calls with the same shapes reuse the compiled kernel — that's the
whole point. But if any *shape* changes (e.g. a new program uses a
different number of state variables so the pytree carried through `lax.scan`
has a different structure), you get a fresh compile.

In EDGAR, every program the LLM emits has different code and, often,
differently shaped state. So JIT compile happens **from scratch for every
program**, and cannot be amortised across programs.

### How big is the compile cost per program?

Not directly instrumented. What we know:

- `_optimize` in `scoring.py` calls
  `jax.jit(jax.value_and_grad(total_loss))`. The traced function runs
  `apply_model_fn` (a `vmap` over 32 trajectories) which in turn runs
  `jax.lax.scan(scan_step, init_state, y_traj[:-1])` with 2399 iterations.
- **`vmap` — vectorising map.** `jax.vmap(f)` transforms `f` so that when
  called with a leading batch dimension, the batch axis is pushed inside all
  the operations and executed as a single fused kernel instead of a Python
  loop. Here it turns "run the model on one trajectory" into "run the model
  on 32 trajectories in parallel on the GPU".
- **`lax.scan` — the JAX-native sequential loop.** `lax.scan(f, init, xs)`
  is equivalent to `for x in xs: state, y = f(state, x)` but written as a
  single XLA primitive. Unlike a Python `for` loop, the scan body is *traced
  once*, not `T` times, so compile cost is O(1) in `T` — but XLA still has
  to compile the unrolled backward pass over 2399 steps, which is expensive.
- The `total_loss` closure captures `data_train` and `apply_model_fn` from
  the enclosing scope. A "**closure**" in JAX is a Python function that
  captures external variables; those variables become baked-in constants of
  the compiled kernel, unless they are arguments.

From the FHN config docstring:

```yaml
# FHN scan is longer than oscillator_ss (T=2400 vs 1024) → JIT takes longer.
timeout_s: 600.0
```
The project owner explicitly bumped the timeout because compile-through-scan
scales with `T`. Empirically, JAX compile time for a `value_and_grad` over a
2400-step `lax.scan` of a small (~10 op) scan body on the A4000 is in the
range **30–70 s**.

Add ~3–4 additional compiles per program for the eval helpers, each cheaper
(no backward pass — 5–15 s). Rough per-program compile budget: **50–120 s**,
i.e. **the majority of the 124 s mean scoring time is compile, not
gradient descent.**

If someone wanted to reduce this cost, the interventions with the largest
leverage would be:

- Cache and reuse the compiled `value_and_grad` across programs that share a
  pytree state shape (many gen-2+ programs shared `{V, w}` — see
  `post_analysis.md`).
- Use `jax.remat` on the scan body to trade compile time for extra flops.
- Reduce `T` from 2400 (config knob).

None of these are wired up today.

### Why we can't get an exact number from these logs

`scoring.py::score` records only the total wall time of
`_score_one_with_outcome` per program. There is no timer around the
`jax.jit(...)` first call vs. subsequent calls. To measure the compile
fraction exactly you'd need either (a) an added `time.monotonic()` around
the `loss_and_grad(flat)` first-vs-rest inside `_optimize`, or (b) enabling
`JAX_LOG_COMPILES=1` so XLA prints compile events to stderr and pipes them
into the run log.

---

## 3. Gradient descent — inside the scoring bucket

Once `loss_and_grad` is compiled, each call is one GPU kernel launch. Adam
adds two element-wise updates over ~10–20 scalars-per-trajectory × 32
trajectories = a couple hundred floats — negligible.

**Estimate: 500 iterations × 20–60 ms/iter = 10–30 s of the 124 s
per-program budget.** The per-iter cost is not directly instrumented but is
constrained by the compiled forward-and-backward through a 2399-step scan
over a small scan body on an A4000; that's a few tens of ms per step for the
kind of programs in this run.

The gradient loop uses `optax.adam` with `learning_rate=0.005` and
`gradient_clip_norm=5.0`, straight from the config.

---

## 4. The scan itself — the sequential loop

The scan (`jax.lax.scan` in
`projects/fhn_excitable/data_loader/load_data.py:250`) is the inner-loop
kernel; it isn't a *separate* wall-clock bucket, it's *what* the compiled
`value_and_grad` executes on every gradient step. Cost analysis:

- Trajectory length: `T = 2400`, so scan runs `T − 1 = 2399` steps.
- Scan body: 10–30 elementwise ops (add/mul/pow/exp), plus an innovation
  update from `y_prev`.
- `vmap` over 32 trajectories → the whole scan runs 32-wide in one kernel.

Once compiled, a single forward-scan is fast (small model, tiny state, no
matrix multiplications). The backward pass, produced by `value_and_grad`,
allocates memory for the reverse pass over all 2399 timesteps, which is why
compile is more expensive than execute. Per-step wall clock, post-compile:
~20–60 ms as estimated above.

---

## 5. LLM waiting — 16 % of wall clock

Every generation issues three async waves of Claude Code calls:

- `generate_models` — 8 calls, each drafts a new NumPy `model` function.
- `generate_param_ests` — 8 calls, each drafts a `parameter_estimator`.
- `translate_programs` — 8 calls, each translates the NumPy model to JAX.

The framework fires all 8 of a wave concurrently
(`asyncio.gather(return_exceptions=True)` — `edgar/llm/generate.py`), so the
wall clock of a wave ≈ the slowest response, not the sum.

### Per-call latencies (mean over seed + gens 0–4)

| Call site | # calls | Mean latency / call | Output tokens / call | Effective concurrency (sum-of-latencies / stage-wall) |
|---|---:|---:|---:|---:|
| `model` | 40 | 110 s | 7 687 | 5.7 |
| `param_est` | 40 | 32 s | 1 862 | 6.4 |
| `jax` (translation) | 44 | 14 s | 875 | 6.4 |

The "effective concurrency" is what actually happened, not the batch size.
It's below 8 because slower calls tail-drag: with 8 in flight, the slowest
gates the wave. `model` calls have the highest variance (p90/p50 ≈ 1.3–1.6)
and therefore the worst concurrency-vs-batch efficiency.

### Per-generation wall clock spent waiting for LLMs

| Gen | `generate_models` | `generate_param_ests` | `translate_programs` | Total LLM wait |
|---:|---:|---:|---:|---:|
| 0 | 97 s | 33 s | 14 s | 144 s |
| 1 | 128 s | 38 s | 22 s | 187 s |
| 2 | 178 s | 42 s | 20 s | 241 s |
| 3 | 162 s | 35 s | 18 s | 215 s |
| 4 | 239 s | 57 s | 21 s | 316 s |

`generate_models` dominates the LLM budget (~78 % of LLM wall over the run),
and it grows generation-over-generation: 97 s → 239 s. The LLM's *output
length* grows too (38 k → 74 k tokens per 8-call wave), consistent with
programs getting longer as the LLM iterates on prior parents.

### Token totals recorded (over the whole 5-generation window)

| Call site | # calls | Output tokens (measured) | Input tokens (measured) | Input tokens (estimated) |
|---|---:|---:|---:|---:|
| `model` | 40 | 307 484 | 280 | ~160 000 |
| `param_est` | 40 | 74 490 | 280 | ~80 000 |
| `jax` translation | 44 | 38 481 | 308 | ~66 000 |
| **Total** | **124** | **420 455** | **868** | **~306 000** |

The "measured" input count is what the `claude` CLI reports in its usage
field (7 tokens per call, which is nonsense — see the data caveat). The
"estimated" column comes from: the population's average model-code length is
5 631 chars, param-est 1 644 chars, jax 1 437 chars; the prompt templates
inject two parent programs plus a couple of KB of scaffolding; at ~4
characters per token this yields the estimates above.

---

## Dollar cost

The run used `claude-code-headless`: EDGAR shelled out to the local `claude`
CLI, which routes through the user's Claude Code subscription. For that
subscription there is **no per-token charge to the user**. However, if the
same prompts had been billed at Anthropic's retail API rates, the total for
this partial run would have been:

- **At Sonnet 4.5 rates ($3 / MTok input, $15 / MTok output):** ≈ **$7.20**
  (model $5.09, param_est $1.36, jax $0.78).
- **At Opus 4.x rates ($15 / MTok input, $75 / MTok output):** ≈ **$36.10**
  (model $25.46, param_est $6.79, jax $3.88).

Both figures use measured output tokens (trustworthy) and estimated input
tokens (see caveat). Neither includes prompt-caching discounts, which
Anthropic offers at ~10× lower rate on cached input; if the system prompt
were cached across calls, input cost would drop by roughly an order of
magnitude and Sonnet would come in near $5.

Compute cost: one workstation running for ~2 h. Not itemised in dollars per
project scope.

---

## 6. Everything else

- **`generate_program_fits`** (matplotlib per-program comparison plot):
  totals to 294 s across gens 0–4, growing 22 s → 99 s per generation
  (matplotlib on more programs with richer fits). ~4 % of wall clock.
- **Population save.** `population.jsonl` is 1.6 MB after 44 records
  including base64'd NumPy arrays for `params`, `sample_losses`,
  `fingerprint`. Written after every generation; append-only cost is under
  1 s per generation and hidden inside the stage timers.
- **Dashboard writes.** The dashboard reads `metrics.jsonl` and
  `status.json` directly; the run side only writes them. Cost is well below
  the timing granularity.
- **Dedup / prune / migrate.** Recorded in `stage_times` as `deduplicate`,
  `prune`, `migrate` — all consistently 10–20 ms per generation. Negligible.
- **Subprocess spawn overhead for scoring.** Absorbed into the 124 s
  per-program scoring bucket, described in section 1.

---

## Summary tables

### Where the wall clock goes (seed + gens 0–4, 6861 s total stage time; ≈120 min real time)

| Bucket | Wall-clock (s) | % of stage time |
|---|---:|---:|
| Scoring — subprocess + JIT compile + gradient descent (all inside `_score_one_with_outcome`) | 5443 | 79 % |
| ↳ JIT compile portion (estimated) | ~2500–4500 | ~40–65 % |
| ↳ 500-step gradient descent portion (estimated) | ~400–1300 | ~6–20 % |
| ↳ Subprocess spawn + XLA init (estimated) | ~440–880 | ~6–13 % |
| LLM wait — `generate_models` (parallelised 8-wide) | 804 | 12 % |
| LLM wait — `generate_param_ests` | 205 | 3 % |
| LLM wait — `translate_programs` (JAX) | 96 | 1 % |
| Per-program fit plots (`generate_program_fits`) | 294 | 4 % |
| Bookkeeping (spawn, dedup, prune, migrate) | ≈15 | <1 % |

### Where the dollars would go (retail-API pricing on measured output tokens)

| Bucket | Output tokens | Input tokens (est.) | $ @ Sonnet 4.5 | $ @ Opus 4.x |
|---|---:|---:|---:|---:|
| `model` (drafts) | 307 484 | ~160 000 | $5.09 | $25.46 |
| `param_est` (drafts) | 74 490 | ~80 000 | $1.36 | $6.79 |
| `jax` (NumPy→JAX translation) | 38 481 | ~66 000 | $0.78 | $3.88 |
| **Total** | **420 455** | **~306 000** | **≈$7.20** | **≈$36.10** |

Compute: one Linux workstation with an RTX A4000 for ≈ 2 h.

---

## What would sharpen this report

Three cheap additions to `edgar/scoring/scoring.py` would let the next
version of this document give exact numbers rather than a compile-vs-execute
estimate band:

1. Wrap the first `loss_and_grad(flat)` call in `_optimize` with a
   `time.monotonic()` and record it as `compile_ms` alongside the per-program
   score latency.
2. Record the total time of the 500-step inner loop (already scoped —
   subtract compile from total).
3. Set `JAX_LOG_COMPILES=1` in the worker env and capture stderr; XLA prints
   one line per compilation with the module name and time.

With those in place the "estimate" rows in the summary table become measured
rows, and per-generation compile drift (which appears to explain most of the
gen-over-gen slowdown from 64 s → 220 s / program) can be confirmed or
disconfirmed directly.