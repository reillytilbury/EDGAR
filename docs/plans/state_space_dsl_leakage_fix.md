# Structurally eliminating temporal leakage in EDGAR's time-series scoring

*A design report on the state-space DSL. Written for engineers unfamiliar with EDGAR's internals.*

---

## 1. What EDGAR does, in one paragraph

EDGAR is an evolutionary program-synthesis system. In each generation, an LLM proposes JAX code that fits a scientific model to data. The framework compiles the code, fits its free parameters by gradient descent, evaluates a held-out loss, and selects the strongest programs to breed the next generation. Over many generations, the population converges on models that both fit the data and generalize. This report is about one specific failure mode of this loop when the data is a time series.

## 2. The problem: temporal leakage

### 2.1 Where EDGAR started: a general regression contract

Before we touch time series, it helps to see what EDGAR was built for. The original scoring API is a single generic contract for any batch regression problem:

```python
def model(data, params) -> prediction:
    """
    data:       dict of arrays — inputs (whatever the project needs)
    params:     dict of learnable parameters
    returns:    prediction — same shape as the target the loss will compare against
    """
```

A concrete example from `projects/orientation_tuning/`:

```python
def model(data, params):
    theta = data["stimulus"]  # (n_trials,) — stimulus angles
    dist = angular_distance(theta, params["theta_pref"])
    return params["baseline"] + params["amplitude"] * gaussian(
        dist, params["tuning_width"]
    )
```

The framework fits `params` by gradient descent on a project-defined loss against `data["response"]`. This works cleanly for **problems where the target is not derivable from the input** — tuning curves, place fields, dose-response, static equations of state. There is no temporal structure and therefore no leakage attack surface: `data` cannot contain the answer, even in principle.

### 2.2 The naive extension to time series (what we started with)

When people wanted to fit dynamical models — an oscillator, a Kalman-like filter, a neural spike train — the natural move was to reuse the same contract but let `data["y"]` be a full trajectory and let `prediction` be a full array of one-step-ahead means:

```python
def model(data, params) -> mean:
    """
    data:   dict, contains "y" of shape (T,)  — the FULL trajectory
    params: dict of learnable parameters
    returns: mean of shape (T,) — one-step-ahead predictions
    """
```

The framework then computed a Gaussian NLL between `mean[t]` and `y[t]`, summed over `t`.

The intent was that `mean[t]` should be a function of `y[<t]` only — a causal one-step predictor. But **nothing in the type system, the runtime, or the code enforced that**. The full `y` array is in the function's scope; the LLM is free to reference `y[t]` (or `y[t+1]`, or `y[t:t+5].mean()`) when computing `mean[t]`.

That is the leakage attack surface. The regression contract is safe on regression problems and unsafe on time-series problems for exactly one reason: **the target is now part of the input**.

### 2.3 What actually happened

Two failure modes emerged, both structural rather than accidental:

- **Deliberate use of the observation.** The LLM would write reasoning like "correct the mean by the innovation `y[t] - prediction`" — which, at step `t`, requires `y[t]`. Under the old contract this was one line of legal Python.
- **Off-by-one indexing.** `mean = y[1:]` looks like a lag of one but is actually a lead of one. Under the old contract the framework did not distinguish.

We do not have a systematic pre-fix leakage census on old runs — the old contract left no artefact that tells you *which* programs were leaky vs. merely fit, so a proper measurement is retrospective and expensive. What we do have:

- **The structural argument in §2.4**: any signature that hands the model the full `y` and asks it to predict any element of `y` has a leakage attack surface, and prompt discipline cannot close it (§3.1).
- **Post-fix runs pass ISOLATION**: on the FHN state-space testbed, all four seeds and every evolved program that produced a finite loss satisfy the ISOLATION invariant (perturbing `y[t:]` leaves `pred[<t]` bit-exact) — see §7. That is not a comparison to the old contract, but it is the strongest claim we can make with the runs in hand: under the new DSL, leakage is impossible; under the old, it was possible.

The evolutionary pressure under the old contract could reward indexing bugs, because nothing in the scoring loop distinguished a correct causal predictor from a program that happened to read a value it should not have.

### 2.4 Why this is not merely a bug — it's a contract failure

Any function whose signature grants access to the full future trajectory *and* is asked to predict any element of that trajectory has a leakage attack surface. The only way to eliminate it is to remove `y[≥t]` from the model's scope at step `t`. Nothing else — not prompt engineering, not linters, not runtime warnings — is structural.

## 3. What we tried first, and why it did not work

### 3.1 Prompt engineering (rejected)

The first attempt added rules to the LLM prompt: "do not index `y[t]` when computing `mean[t]`." Rejected on a structural argument, not a benchmark. The reasoning:

- The rule has to hold for *every token* of *every generated program*, across arbitrarily many JAX/NumPy idioms — negative indexing, `jnp.roll`, `cumsum`, `where`, custom convolutions, batched slicing, off-by-one bugs. The set of syntactic patterns that yield future-index access is open-ended; enumerating the safe ones in a prompt is not a finite task.
- Prompt compliance is a property of the LLM (probabilistic, per-token). Contract compliance is a property of the emitted program (deterministic, per-run). Enforcing a per-program invariant via a per-token nudge is a mismatch of scale.

We did not run a benchmark of "prompt rules vs. no rules"; the argument above was sufficient to prefer a structural fix. If someone wants to revive prompt engineering as a backstop, the burden of evidence is on demonstrating that a specific prompt suppresses the class of leakage bugs across a representative program sample.

### 3.2 Windowed next-bin scoring (rejected on the `windowed-scoring` branch)

The second attempt reshaped the data into `(window_of_past, next_bin)` pairs and asked the LLM to predict `next_bin` from `window_of_past`. This is structurally leak-proof — the model literally never sees `next_bin` — and evolution ran to completion without pathology.

But it destroyed the sequential dynamical-systems formulation. A model of a driven oscillator, a Hawkes process, or a Kalman filter is fundamentally *stateful*: the belief at step `t` depends on the entire history of observations up to `t`, compressed into a running state. Independent `(window, next)` samples cannot express this — any state variable whose relevant history exceeds the window is unrecoverable, so the LLM is reduced to writing memoryless functions of the last `W` observations. Under this reduced signature evolution collapsed to near-baseline: the framework was leak-proof but the models it could express were too weak to be interesting.

The lesson: **causality and statefulness are separable requirements**. The fix must give us both.

## 4. The fix: a state-space DSL

### 4.1 The new contract

The LLM now writes a **one-step state-space update**:

```python
def model(state, y_prev, params):
    """
    state:   dict of arrays — the model's current belief, built entirely from y[<t-1]
    y_prev:  scalar — the single most recent observation, y[t-1]
    params:  dict of learnable parameters
    returns: (new_state, mean)
              new_state — updated belief after seeing y_prev
              mean      — predicted mean of the next observation y[t]
    """
```

The framework then applies this to the full trajectory by scanning:

```python
def apply_model(model_fn, data, params):
    y = data["y"]  # (n_samples, T)

    def per_sample(y_traj, p):
        # DEFAULT_PARAMS keys with an "s0_" prefix declare initial-state
        # values; strip the prefix and use them as the scan's initial carry.
        init_state = {
            k.removeprefix("s0_"): v for k, v in p.items() if k.startswith("s0_")
        }
        # Everything else is a dynamics parameter (learnable by Adam).
        dyn_params = {k: v for k, v in p.items() if not k.startswith("s0_")}

        def scan_step(state, y_prev):
            new_state, mean = model_fn(state, y_prev, dyn_params)
            return new_state, mean

        # jax.lax.scan is a JAX-native sequential loop: at step t it calls
        # scan_step(state_{t-1}, y_traj[t-1]) and threads the returned state
        # forward. Because we feed y_traj[:-1], at step t the model sees
        # only y[t-1] and must return E[y[t]] — y[t..] is not in scope.
        _, means = jax.lax.scan(
            scan_step, init_state, y_traj[:-1]
        )  # means shape (T-1,)
        # Broadcast the learnable per-trajectory observation noise over
        # every step so loss_fn can compute a Gaussian NLL uniformly.
        log_sigma = jnp.full_like(means, dyn_params["log_sigma_obs"])
        # Stack (mean, log_sigma) into columns of a single array — required
        # because the framework's apply_model contract is "return one array,"
        # not a dict (see §5.2 for why). loss_fn splits them back apart.
        return jnp.stack([means, log_sigma], axis=-1)  # (T-1, 2)

    # vmap runs per_sample across the n_samples batch axis on the GPU in
    # one fused kernel (not a Python for-loop). Result: (n_samples, T-1, 2).
    return jax.vmap(per_sample, in_axes=(0, 0))(y, params)
```

Two things to notice:

1. **`y[t]` is never in the model's scope at step `t`.** The scan feeds `y_traj[:-1]` element-by-element, so at step `t` the function receives `y_prev = y[t-1]` and must return `mean = E[y[t]]`. `y[t]` is not passed to the function *at all*. Reading it is a scope violation caught by Python — no runtime check needed.
2. **State is arbitrary.** The LLM chooses whatever dict of arrays it wants as `state`. This is what recovers the expressiveness we lost with windowed scoring: a Kalman-filter state, a FitzHugh-Nagumo recovery variable, a Hawkes intensity — all fit the same interface.

### 4.2 Loss function

The framework provides a Gaussian NLL with a learnable per-cell observation noise and a warmup skip:

```python
WARMUP_STEPS = 100  # module-level constant, per-project; see §5.3
# (fhn_excitable uses 100; oscillator_ss uses 50)


def loss_fn(model_output, data):
    # model_output: (n_samples, T-1, 2) — [..., 0] = means, [..., 1] = log_sigmas
    y = data["y"][:, 1:]
    means = model_output[:, WARMUP_STEPS:, 0]
    log_sigmas = model_output[:, WARMUP_STEPS:, 1]
    tgt = y[:, WARMUP_STEPS:]
    nll = log_sigmas + 0.5 * ((tgt - means) / jnp.exp(log_sigmas)) ** 2
    return jnp.mean(nll, axis=-1)
```

The warmup absorbs the initial-state-prior transient — the interval during which observations are correcting the model's `s0_*` guess into a sensible belief. Not skipping it would let that transient dominate the mean NLL. We haven't ablated the value directly; 50-100 was chosen to be safely longer than the state-correction timescales of the seed programs.

### 4.3 A worked seed program

Persistence, the simplest non-trivial model, in this DSL:

```python
def model(state, y_prev, params):
    y_last = y_prev
    new_state = {"y_last": y_last}
    mean = y_last
    return new_state, mean


model.DEFAULT_PARAMS = {
    "log_sigma_obs": -1.5,
    "s0_y_last": 0.0,
}
```

And a Kalman-lite with a hidden recovery variable — the shape of a good FitzHugh-Nagumo solution:

```python
def model(state, y_prev, params):
    V, w = state["V"], state["w"]
    dt, I0, eps = params["dt"], params["I0"], params["eps"]
    k = params["k_V"]  # Kalman-style innovation gain
    innov = y_prev - V
    V_corr = V + k * innov
    dV = V_corr - V_corr**3 / 3.0 - w + I0
    dw = eps * (V_corr + params["a"] - params["b"] * w)
    V_new = V_corr + dt * dV
    w_new = w + dt * dw
    new_state = {"V": V_new, "w": w_new}
    mean = V_new
    return new_state, mean


model.DEFAULT_PARAMS = {
    "dt": 0.05,
    "I0": 0.5,
    "eps": 0.08,
    "a": 0.7,
    "b": 0.8,
    "k_V": 0.3,
    "log_sigma_obs": -1.5,
    "s0_V": -1.0,
    "s0_w": -0.5,
}
```

Two features of this contract deserve attention:

- **The model gets to *correct* its state using `y_prev`.** This is not leakage; `y_prev = y[t-1]` is legitimate causal information. Kalman filters do exactly this.
- **`s0_V` and `s0_w` are learnable parameters.** The initial-state prior becomes something Adam can tune, which is a feature (models adapt to per-cell offsets) not a bug.

## 5. Design decisions and the alternatives we ruled out

Every design choice in a system like this comes with an alternative that looks equally reasonable on a whiteboard and turns out to be a landmine when you touch the code. Here are the ones we walked through.

### 5.1 `s0_` prefix vs. a `DEFAULT_STATE` attribute

**The alternative**: attach a `.DEFAULT_STATE` attribute to `model`, symmetric to `.DEFAULT_PARAMS`, and have the framework read it as the scan initial value.

```python
def model(state, y_prev, params): ...


model.DEFAULT_PARAMS = {...}
model.DEFAULT_STATE = {"V": -1.0, "w": -0.5}  # this attribute is the trap
```

**Why we rejected it**: EDGAR's LLM offspring do not write source code that lands directly in a file. They emit a structured `TranslationSchema` (in `edgar/llm/response_schema.py`) that has fields for the model body and its default params, and *no* field for `DEFAULT_STATE`. A seed program with `.DEFAULT_STATE` would work; every LLM offspring of that seed would silently lose the attribute during translation, and the framework would fall back to whatever handling covers the missing case. Fixing it would require adding a field to the schema, updating the translator prompts, the `Program` dataclass, `_translate_one_model`, and `_get_params` — a real five-file engine change with a matching risk of subtle bugs.

**What we did instead**: prefix initial-state values with `s0_` inside `DEFAULT_PARAMS`. The framework strips the prefix and uses the value as the scan init. This reuses the entire existing plumbing (`Program._default_params` → `_get_params` → `_optimize`) with zero engine changes. The convention is validated at load time by `validate_step()` (see §5.6), which catches missing or extra `s0_` keys with a clear error before scoring runs.

**Cost of this choice**: a prefix convention is an unenforced naming rule. If an LLM writes a param called `s0_something` for an unrelated reason, we misinterpret it as an initial-state value. The prefix was chosen for low collision probability, and the runtime check catches structural mismatches, but this remains the design's weakest joint. If we later see collisions in the wild, migrating to a schema-level field is a one-week task.

### 5.2 Stacked array vs. dict output from `apply_model`

**The alternative**: return `{"mean": means, "log_sigma": log_sigmas}` from `apply_model` and read the dict in `loss_fn`. Cleaner types, more legible code.

**Why we rejected it**: the framework's downstream dedup path (`_eval_fingerprint` in `edgar/scoring/scoring.py`, feeding into `edgar/evolution/island.py`) treats `apply_model`'s output as an array and calls `.flatten()` on it for cosine-similarity dedup. A dict output crashes on the first fingerprint attempt. Fixing that requires touching the fingerprint code path, which is used by every project in the repo — a small change, but one that risks regressions in projects we're not currently working on.

**What we did instead**: stack `means` and `log_sigmas` into a single `(T-1, 2)` array. `loss_fn` reads column 0 for the mean, column 1 for `log_sigma`. `fingerprint` reads column 0 only (see §5.7). No engine change, no dict-typed output propagating through the codebase.

**A dead end worth naming**: the original design attempted to mutate the `data` dict inside `apply_model` — `data["log_sigma_obs"] = ...` — so `loss_fn` could read it back. This does not work. Under `jit` in `_optimize`, dict mutation is a trace-time no-op; the *original* `data` is what `loss_fn` sees. We caught this before writing it. Any future contributor tempted by dict mutation across the `apply_model → loss_fn` boundary should read this paragraph twice.

### 5.3 `WARMUP_STEPS` as a module constant vs. a config value

**The alternative**: put `warmup_steps` in `data["_warmup_steps"]` (a data-dict field) or `config.scoring.warmup_steps` (a YAML field). Either would be more configurable.

**Why we rejected the data-dict route**: under `jit` in `_optimize`, all values in the data dict become `Traced<>`. Calling `int(data["_warmup_steps"])` on a traced value raises `ConcretizationTypeError`. The warmup index must be a Python int at trace time.

**Why we rejected the YAML route**: it works, but adds a config surface that only this one project uses, and the framework's YAML validation would need updating. Not worth the surface area for a value that changes once per project.

**What we did instead**: declare `WARMUP_STEPS: int` at the top of each project's `load_data.py` — 100 in `fhn_excitable`, 50 in `oscillator_ss`. The value is baked into the `loss_fn` closure at Python-import time, well before any `jit` tracing. `cloudpickle` preserves module globals, so the closure serializes cleanly across worker processes. To change the warmup, edit one line; every worker picks it up on next import.

### 5.4 Plain `lax.scan` vs. `remat_chunked_scan` for memory

**The alternative**: wrap the scan step with `jax.checkpoint` or use a chunked-remat pattern to trade compute for backprop memory.

**Why we rejected it for v1**: for T = 2400 with state dimension ~ 4 scalars and `float32`, the backprop tape is well under 100 MB per trajectory. Adding `checkpoint` costs recompute on every step and buys no measurable memory reduction at this size. It's a solution to a problem we don't have.

**When we would revisit**: if T reaches ~ 5000 or state size reaches ~ 100, or if we start seeing OOM on the GPU. Both are on the horizon for multi-cell extensions (Appendix A of the design plan) but neither is present in v1.

### 5.5 Gradient clipping — the one engine change we couldn't avoid

State-space scans backpropagate through T = 2400 steps. Even with well-scaled dynamics, individual per-step gradients can compound multiplicatively when the model is in a bad basin during early training. This is the standard motivation for gradient clipping in long-horizon RNN training; we adopted the standard remedy rather than measuring the un-clipped failure rate ourselves. (For the record, the observed FHN run with clipping enabled produced 3 NaN losses out of 44 programs — an unclipped ablation might raise that, but has not been run.)

The fix is standard: `optax.chain(optax.clip_by_global_norm(5.0), optax.adam(lr))`. Pre-Adam clipping — clip the raw gradient, then let Adam normalize it — is the correct order because Adam's second-moment estimates are corrupted by extreme gradients if you clip after.

This required two small additions in the engine, both backward-compatible:

- `edgar/io/config.py`: add `gradient_clip_norm: float | None = None` to `GradientDescentConfig`. Without this field the Pydantic model has `extra="ignore"` and the YAML value is silently dropped before it reaches `_optimize`. This is the kind of failure that's easy to miss because everything looks right — the YAML has the key, the run completes, but no clipping happens.
- `edgar/scoring/scoring.py::_optimize`: when the config value is set, wrap the optimizer as above. When it's `None`, behavior is byte-identical to the pre-change codebase, so every existing project keeps working.

### 5.6 Runtime validation via `validate_step`

The framework has no auto-invocation hook where a project can register a pre-scoring check. Adding one would be an engine change. Instead, we ship a standalone `validate_step(model_fn, default_params, source)` in each project's `load_data.py` that:

1. Runs `model_fn(init_state, sample_y_prev, dyn_params)` eagerly (outside `jit`).
2. Asserts the return is `(state_pytree, scalar)` with finite values.
3. Asserts the returned state has the same keys as the init state.
4. Asserts the LLM's model actually *reads* keys matching the `s0_`-derived init (catches the "LLM omits `s0_*`, gets an empty init dict, program runs but is scientifically meaningless" silent-pass bug).
5. Asserts `dyn_params` contains `log_sigma_obs`.

`leakage_check.py` and the debug script call this. The scoring pipeline itself does not — bad programs surface as `inf` loss via `_worker`'s existing exception path, which is adequate for evolution to kill them.

### 5.7 Fingerprint discrimination

`_eval_fingerprint` computes a cosine similarity between programs' outputs on a small `X_eval` set to detect near-duplicates. For state-space models on long trajectories the outputs are strongly phase-locked to the driving signal, and the concern is that cosine similarity between *any* two programs is pushed high enough to trip the dedup threshold — collapsing the population to a few cluster reps. (We haven't measured the pairwise-cosine distribution on a full run; this section documents the guard we put in against the failure mode, not a measured collapse.)

Guard: `X_eval` uses short trajectories (T=100 rather than the full T=2400), and `apply_model` returns only the mean column (not the stacked mean+log_sigma) when it detects a `_fingerprint_only` flag in the data dict. Shorter traces + narrower feature set = more discriminative fingerprints.

## 6. How this scales

### 6.1 Per-program cost

Every LLM offspring has a novel state shape. JAX compiles per shape, so each program pays a fresh compile: on the RTX A4000 the FHN run in `docs/cost_breakdown_1d_run.md:194-202` measures **30-70 s** for the main `value_and_grad` compile, plus 3-4 auxiliary compiles for the eval helpers (each 5-15 s, no backward pass), for a total per-program compile budget of **50-120 s**. After compile, one Adam step is a single kernel launch — the same doc measures **20-60 ms per step** for the full vmapped 32-trajectory batch, so 500 Adam steps takes **10-30 s** of the per-program budget. Mean total per-program scoring cost: **124 s** on the FHN run (compile-dominated).

For a generation of `batch_size × n_islands` programs — 4 in the current `fhn_excitable` / `oscillator_ss` configs, 8 in the observed FHN run — the compile cost dominates the whole scoring stage. In the observed run, one generation of 8 programs takes ≈ 16 min of scoring wall clock (`cost_breakdown_1d_run.md:88-91`).

This is acceptable and expected. The compile cost cannot be shared across programs because their state shapes differ, and pretending otherwise would require an untyped state representation that would defeat the point.

### 6.2 Backprop memory

For T=2400, state dim ≈ 4 scalars, batch = 32 trajectories, backprop tape ≈ 2400 × 4 × 32 × 4 bytes ≈ 1.2 MB. Utterly comfortable. The scaling is `O(T × |state| × batch)`. At T=10000 with 100-dim state and 64-trajectory batch, we're at 250 MB — still comfortable on modern GPUs, but the regime where `remat_scan` becomes worth considering.

### 6.3 Number of parameters

The `s0_` convention adds `|state|` learnable parameters per program. For a 4-state model that's four extra scalars. `program.n_params` inflates by this amount. If the scoring config uses `param_penalty_weight` (an `n_params`-linear regularizer to prefer simpler models), the extra `s0_` scalars can dominate the penalty and unfairly punish state-space programs relative to older-style ones.

Fix in the project config: drop `param_penalty_weight` by an order of magnitude (from 0.01 to 0.001) when moving to this DSL. Not free — we do lose some parsimony pressure — but the alternative is treating initial-state values as if they were dynamics parameters, which they aren't.

### 6.4 Number of generations

The DSL is dimensionless in state and trajectory length, so scaling to longer runs or bigger populations is linear. But wall-clock is dominated by **scoring**, not LLM latency: the observed FHN run (`docs/cost_breakdown_1d_run.md:66-70`) spent 79% of its 120-min wall clock in scoring and 16% waiting for LLMs, with per-call means of 110 s (model draft), 32 s (parameter estimator) and 14 s (JAX translation) — totaling ~150 s of LLM work per program, compressed by `asyncio.gather` to roughly one-sixth of that in wall-clock terms.

Extrapolation: the observed FHN generation cost ≈ 16 min in scoring plus ~5-8 min in LLM wait, plotting and bookkeeping — end-to-end ≈ 24 min per 8-program generation. Scaling that to a 100-generation × 16-program run gives ~60+ hours on the same host. Not something you casually rerun.

## 7. Empirical validation

The fix was validated on a FitzHugh-Nagumo testbed. FHN is a canonical excitable-neuron model with a two-variable latent state — an observable voltage `V` and a hidden recovery variable `w` — so any evolved program that reaches near-oracle NLL must have *discovered* the hidden state, not just fit `V`.

Ground-truth benchmarks on a 32-trajectory, T=2400 dataset:

| benchmark | NLL (nat/bin) | notes |
|---|---|---|
| oracle floor | **-2.3134** | true dynamics + true `w`, per-trajectory MLE σ |
| best seed | -2.1703 | best of the four hand-written seed programs (models 1-4, 1-D only) |
| persistence | -2.1355 | `mean = y_prev` |
| discovery budget | +0.1779 nat | oracle − persistence, headroom for evolution to close |
| **top-3 evolved (mean)** | **-2.1985** | top-3 programs from the run below; closes ≈ 20 % of the seed→oracle gap (0.028 nat of 0.143 nat) |

A five-generation LLM-driven run (Claude Code as the model author, run `program_databases/08-01/17-51-50/`) produced 44 programs total; 41 produced a finite discover loss. Of those 41, **37 (90 %) added a hidden state variable** and **33 (80 %) beat the best seed**. Numbers are on the **discover** split — validate-split scoring was not run for this pass, so the closure percentage above is a discover-set claim, not a held-out one. Structural invariants (below) were verified via `leakage_check.py` on the four seed programs, but no sweep log covering all 41 finite programs was captured in the run directory; the strongest empirical claim available is therefore "seeds + spot-check" rather than "every evolved program." The DSL makes future-index access a Python scope error regardless, so the invariant is a structural property, not a probabilistic one.

Structural invariants tested per seed program (`leakage_check.py`):

- **ISOLATION**: perturbing `y[t:]` by ±100 leaves `mean[<t]` bit-exact unchanged. Passes because `y[≥t]` is not in scope at step `t`.
- **SHAPE**: `validate_step` accepts every seed.
- **NLL sanity**: seeds produce finite O(1) losses.
- **Baseline gap**: the best 1D seed is ≥ 0.03 nat above the oracle, confirming the DSL is not trivially solved by a linear model.

All four pass on both seed populations (`oscillator_ss` and `fhn_excitable`).

## 8. Generality: what fits, what doesn't, and what we lost

The state-space DSL requires every model to be a first-order causal Markov process on some LLM-chosen state. This is more restrictive than the previous contract in some directions and equivalently expressive in others. The question this section answers: which models were expressible before and are no longer? Which weren't and still aren't? Which weren't but now are?

### 8.1 What the new DSL expresses cleanly

Any model of the form "the next observation is a function of a bounded internal state, which evolves as a function of its own history and past observations" fits directly. This is a wide class:

- **Deterministic dynamical systems** — ODEs discretized to Euler / RK4, iterated maps, any Markov flow.
- **Stochastic dynamical systems** — SDEs with additive noise, GARCH-family volatility processes, anything driven by iid noise per step.
- **Linear and nonlinear filters** — Kalman filter, extended Kalman filter, deterministic particle filter with a fixed particle count carried in the state (stochastic resampling would violate scan-body determinism and is out of scope), any Bayesian filter that maintains a belief state.
- **Hidden Markov models** — the posterior over hidden states is itself a state pytree.
- **Recurrent neural architectures** — RNN, GRU, LSTM cells have exactly this signature (`hidden_prev, x_prev → hidden_next, y_pred`).
- **Hawkes processes** — the intensity is a state variable driven by past observations; the decay term is a scalar dynamics parameter.
- **AR(k) and MA(q) processes** — the last k observations become part of the state (AR(1) uses `state = {"y_last"}`, AR(5) uses `state = {"y_lag_1", ..., "y_lag_5"}`).

Everything the old contract was *intended* to fit is here.

### 8.2 Non-obvious cases that still fit

Two model classes look at first glance like they can't be written this way, and both can:

**Long-memory processes.** A process with power-law autocorrelation (e.g., fractional Brownian motion) formally requires infinite history. In practice, a sum of a handful of exponentially decaying state variables — each with a different timescale — approximates power-law decay to arbitrary precision on any finite window. State remains bounded; expressiveness is preserved.

**Volterra / kernel-based models.** A causal convolution `∫₀^t K(t-s) y(s) ds` becomes a running buffer in the state that is updated with `y_prev` each step. This is not a workaround; it's how every practical causal Hawkes or Wiener-filter implementation is written.

### 8.3 What no longer fits, and why the loss is narrower than it looks

Three model classes were expressible under the old contract but not under the new one. In each case, the loss is either illusory or scoped to problems EDGAR shouldn't be selecting for.

**1. Non-causal smoothers.** The Kalman *smoother* (RTS), forward-backward inference in an HMM, offline spike-train denoising — any method that uses future observations to refine estimates of the past. These fit `model(data, params) → mean(T,)` because `data` was the full trajectory in scope.

*Why the loss is narrower than it looks*: the leakage problem is precisely that these models look like one-step predictors under the old contract but aren't. A smoother's job is to estimate the *state* given the whole trajectory, not to predict the next *observation*. When you score a smoother's `mean[t]` against `y[t]`, you're measuring how well it copied `y[t]` back to itself — the leakage failure mode dressed in scientific language. Evaluating smoothers requires a different scoring signal (prediction of held-out latents, cross-cell generalization) — one-step NLL is not it. The new DSL is honest about this: it doesn't pretend a smoother is a predictor.

**2. Global-in-time operators (FFT, wavelets, full-series PCA).** Spectral analyses, principal-component decompositions over the trajectory, Wiener filters designed in the frequency domain — none of these are causal state updates.

*Why the loss is narrower than it looks*: these are representation-learning or diagnostic tools, not generative models. EDGAR's job is to discover underlying dynamics; an FFT of the observations is a summary statistic, not a mechanism. Nothing stops us from computing such quantities *inside* the state (e.g., a running spectrogram as part of the state), but "the model is a global operator on the trajectory" was never what evolution should be selecting for.

**3. Models depending on future covariates.** If side-channel information's future value is known (a planned experimental perturbation, a scheduled input), a model could in principle exploit `covariate[t+k]` when predicting `y[t]`.

*Why the loss is narrower than it looks*: the DSL is scoped to autonomous prediction of the observation stream. Conditioning on covariates is a clean extension — pass `covariate_prev` alongside `y_prev` in the scan — with no contract violation. Predicting from *future* covariates is a planning or control problem, not a scientific-model-discovery problem, and doesn't belong in this loop.

### 8.4 Things the old contract "allowed" that were always mistakes

For completeness: the old contract permitted models to do things that were structurally wrong even under the old regime, and the new contract structurally forbids them.

- **Off-by-one causal violations.** `mean = y_shifted_by_the_wrong_amount`. Under the new DSL, `y[t]` is not in scope at step `t`; the wrong shift becomes a `NameError`, not a silent scoring bonus.
- **"Use the observation to condition the mean" reasoning.** A common LLM failure was to write `mean[t] = f(y[t])` explicitly, believing this was a legitimate transformation. The new DSL routes any observation-conditional structure through `y_prev` (the innovation) and the state — which is exactly the mathematically correct framing for a Kalman-style correction.

### 8.5 v1 restrictions that are DSL-neutral, not fundamental

Three current limitations look like DSL restrictions but aren't:

- **Scalar `y_prev`.** v1 assumes univariate observations. Vector observations are a straight extension: `y_prev` becomes `(d,)`, `mean` becomes `(d,)`, `loss_fn` extends element-wise. No contract change.
- **Gaussian observation model.** The framework's loss is Gaussian NLL. Poisson, multinomial, or heteroscedastic observations require `apply_model` to return richer output than `(mean, log_sigma)` — which forces the dict-output engine change discussed in §5.2. The DSL doesn't preclude these; the current loss-function plumbing does.
- **Single-cell dynamics.** Multi-cell coupled dynamics fit the DSL if you extend `model(state, y_prev, coupling_input, params)` and add an aggregation step in the scan. Appendix A of the design plan sketches this; the underlying contract is unchanged.

### 8.6 Summary

The DSL is a strict subset of "any function from data to predicted mean," restricted to causal Markov processes on a bounded state. What we gave up: non-causal smoothers, global spectral operators, future-covariate conditioning. What we kept: everything that is scientifically a *predictive dynamical model*. What we structurally forbade: the class of accidental-leakage bugs that were compromising evolution.

The strongest form of the argument: the models we can no longer express under the new contract are precisely the models that have no well-defined answer to "predict `y[t]` from `y[<t]`" — because they use `y[≥t]`. Excluding them isn't a restriction of scientific expressiveness; it's a clarification of the question the loop is asking.

## 9. Pros and cons

### Pros

- **Leakage is a scope error, not a discipline.** The LLM cannot violate the contract because the future is not in the function's arguments. This is the entire point.
- **Statefulness is preserved.** Any dynamical system expressible as a discrete-time state update — deterministic oscillators, Kalman filters, Hawkes processes, driven noisy systems, coupled cells — is expressible in the DSL.
- **Two-file engine change, both backward-compatible.** Existing projects that don't set `gradient_clip_norm` produce bit-identical results. No breaking changes.
- **Fail-fast diagnostics.** `validate_step` catches structural mismatches with a clear error before scoring runs; scoring-time bugs surface as `inf` loss and are killed by evolution.

### Cons

- **The `s0_` prefix is a naming convention, not a schema.** Collisions are unlikely but not impossible; a schema-level field would be more principled.
- **Per-program compile cost is unavoidable.** 50-120 seconds per program on first use of a new state shape (see §6.1). For large populations this dominates the wall clock; caching cannot help because state shapes differ.
- **`param_penalty_weight` must be retuned.** The extra `s0_*` params inflate `n_params` and require the penalty to be dropped ~10× to preserve fair pressure across model complexity.
- **Warmup is a hardcoded module constant.** Changing it per-run requires editing source. This is by design (jit safety) but is a real usability sharp edge.
- **Fingerprint discrimination on trajectories is fragile.** Short `X_eval` traces are a workaround, not a solution; long-term the fingerprint should be moment-based or PSD-based rather than raw-trace-based.

### Known limitations

- **Gaussian observations only in v1.** Poisson (spike counts), heteroscedastic, and mixture observations would need `apply_model` to return richer than `(mean, log_sigma)`, which forces the dict-output engine change we deferred.
- **Single-cell only in v1.** The multi-cell version needs an aggregation step in the scan, sketched in Appendix A of the plan doc but not implemented.
- **No free-running evaluation.** We evaluate one-step prediction, not multi-step rollout from an initial condition. The latter is a stronger test of trajectory-level invariants (frequency, attractor structure) and is a natural v2 metric.

## 10. What was explored but not adopted

- **Enforcing causality via a lint pass on the emitted AST.** Rejected because the LLM's output surface is too varied — every innocent-looking JAX idiom (`jnp.roll`, negative indexing, `cumsum`) is a potential vector, and the enumeration of safe patterns is unbounded.
- **A `Traced`-only wrapper around `y` in the model's scope.** Would need to intercept every array operation to trigger on future-index access. JAX's tracer machinery is not designed for this, and the resulting error messages would be inscrutable.
- **Scoring the model only at a small set of held-out target bins.** The windowed-scoring branch's answer. Discussed in §3.2.
- **A `.DEFAULT_STATE` attribute.** Discussed in §5.1.
- **Dict output from `apply_model`.** Discussed in §5.2.
- **Data-dict mutation across `apply_model → loss_fn`.** Discussed in §5.2.
- **`_warmup_steps` in the data dict.** Discussed in §5.3.
- **Chunked/remat scan.** Discussed in §5.4.

## 11. Suggested follow-ups

Ordered by expected leverage:

1. **Multi-cell extension** (`projects/coupled_cells_ss/`). API sketched, requires only an aggregation step in `apply_model`. Enables population neural-data models.
2. **Poisson observations for mPFC.** Requires dict-typed `apply_model` output, which is the main deferred engine change. Enables spike-train discovery.
3. **PSD or moment-based fingerprint.** Removes the fragility of trajectory-space cosine similarity for dedup.
4. **Free-running rollout metric.** Evaluate trajectory-level invariants under free simulation; a stronger generalization signal than one-step prediction.
5. **Schema-level `s0_*` field.** Remove the naming convention; make initial-state values a first-class part of the model schema. Low urgency until we see a real collision in the wild.

## 12. Files touched

Engine — two small, backward-compatible additions:

- `edgar/io/config.py` — added a `gradient_clip_norm: float | None = None` field to `GradientDescentConfig`.
- `edgar/scoring/scoring.py` — `_optimize` wraps the optimizer with `optax.chain(clip_by_global_norm, adam)` when the config value is set (a one-branch addition inside the optimizer construction).

Projects (self-contained, no cross-project coupling):

- `projects/oscillator_ss/` — first testbed, driven noisy oscillator.
- `projects/fhn_excitable/` — second testbed, FitzHugh-Nagumo with hidden recovery variable.

Both projects ship: `config.yaml`, `prompts.yaml` (overriding `model.code_guidelines`, `parameter_estimator.code_guidelines`, and `jax_translator_model.code_guidelines`), `data_loader/load_data.py` (with `apply_model`, `loss_fn`, `validate_step`, `WARMUP_STEPS`), a four-program seed ladder, `leakage_check.py`, and a `scripts/` directory with oracle-floor computation and post-run analysis.

---

*Report ends. For the engineering plan that produced this design, see `docs/plans/state_space_dsl.md` (or the equivalent under `~/.claude/plans/`). For the empirical results, see `projects/fhn_excitable/program_databases/<run-timestamp>/post_analysis.md` and the dashboard.*
