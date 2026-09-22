# Extending EDGAR's state-space DSL to d-dimensional observations

*A design plan. Written as the multivariate follow-on to `docs/state_space_dsl_leakage_fix.md`. Same audience: an engineer approaching EDGAR's scoring loop for the first time.*

---

## 1. Scope in one paragraph

EDGAR's state-space DSL (v1, September 2026) fixed temporal leakage by forcing each LLM-authored model to be a one-step causal update `model(state, y_prev, params) → (new_state, mean)` on a scalar observation. This plan extends that contract to a d-dimensional observation vector `y_prev: (d,)`, `mean: (d,)` so we can score models on jointly-observed systems — Lorenz-63, coupled Van der Pol, multi-channel EEG, small neural populations. The extension is designed to preserve every structural guarantee of v1 (leakage is still a scope error, fingerprint dedup still works, `WARMUP_STEPS` is still a Python constant) and to require *zero* changes to the `edgar/` engine. It is a project-local change plus prompt work. That is not accidental — it is the design goal.

> **Status update (since writing).** The d-dimensional contract described here is no longer purely
> prospective. `wilson_cowan` is a working multivariate state-space project on the current base — a
> coupled 2-channel `(E, I)` model whose `mean` is a 2-vector, with a dict-structured `y_prev` and a
> heteroscedastic Gaussian NLL whose variance scales with each channel's own predicted mean through
> a single noise coefficient shared by both channels — and it required **no `edgar/` engine
> change**, confirming this plan's central design goal. Three ways WC differs from the plan below:
> (a) it packs channels into a wider stacked array `(n, n_stim, T-1, 3)` = `(E, I, log_noise_coef)`
> — one *shared* noise coefficient — rather than the `(T-1, d, 2)` layout of §3, which would carry a
> *per-channel* `log_sigma` (i.e. `(T-1, 2, 2)` at d=2); (b) it evaluates on a free rollout, not
> one-step-ahead (see §11.2); and (c) it *sums* the per-channel NLLs (`nll_E + nll_I`) rather than
> taking the mean-over-channels that §7.1 recommends (low impact at d=2, where sum = 2×mean, but a
> deliberate deviation to note). The Lorenz-63 / coupled-VdP testbeds specified below remain
> unbuilt.

## 2. The new contract

### 2.1 Signature

The LLM writes:

```python
def model(state, y_prev, params):
    """
    state:   dict of arrays — the model's current belief.
             Values may be scalars OR vectors of shape (d,) or (d, d);
             the LLM chooses the state pytree per program.
    y_prev:  jnp array of shape (d,) — the previous observation vector.
    params:  dict of learnable scalars OR arrays.
    returns: (new_state, mean)
             new_state — same pytree structure as state
             mean      — jnp array of shape (d,)
    """
```

Concretely: `y_prev` is not a Python float any more, and `mean` is not a Python float any more. Everything else about the contract is unchanged.

### 2.2 State dict

State remains an LLM-authored pytree. In the multivariate case, useful state entries typically include:

- **Vector "position"** — e.g. `state["x"]` of shape `(d,)`, the model's belief about the observation.
- **Vector "velocity" / recovery** — e.g. `state["v"]` of shape `(d,)`, a hidden variable per channel.
- **Matrix transition** — e.g. `state["A"]` of shape `(d, d)` if the LLM wants to *learn* the transition on the fly (unusual — normally A is in `params`, not `state`).
- **Kalman covariance** — e.g. `state["P"]` of shape `(d, d)` for an EKF.

The framework does not care what shapes the LLM picks; `lax.scan` requires only that the pytree structure and the shape of each leaf are invariant across iterations, and that requirement is unchanged from v1.

### 2.3 log_sigma_obs: diagonal vs full covariance

This is the first real design decision, with two plausible answers:

**Option A — per-channel `log_sigma_obs`, diagonal covariance (RECOMMENDED for v1).**

```python
model.DEFAULT_PARAMS = {
    ...,
    "log_sigma_obs": [-1.5, -1.5, -1.5],   # length d
    "s0_x": [0.0, 0.0, 0.0],               # length d
}
```

The loss becomes a sum of d independent Gaussian NLLs. Under this choice the noise model assumes channels are conditionally independent given the state — which is the standard assumption when the "channels" are physically separate sensors, and is a first-order-correct assumption when they are cell voltages in a small population.

Cost: fails to capture correlated noise (e.g. a shared electrical artifact across EEG channels).

**Option B — full lower-triangular Cholesky factor `L` of shape (d, d), giving covariance `Σ = L Lᵀ`.**

```python
model.DEFAULT_PARAMS = {
    ...,
    "log_sigma_diag": [-1.5, -1.5, -1.5],
    "sigma_offdiag":  [0.0, 0.0, 0.0],   # d*(d-1)/2 entries below diag
}
```

The loss becomes a proper multivariate Gaussian NLL:
`nll = 0.5 * (y - mean)ᵀ Σ⁻¹ (y - mean) + 0.5 * log|Σ|`.

Cost: adds `d*(d+1)/2` parameters for a "boring" reason (observation noise), which competes for the `param_penalty_weight` budget and inflates `n_params` in a way that makes cross-program comparison noisier. Also, gradient-descent stability on the Cholesky is a real pain — the diagonal must be positive, which means either exponentiating a `log_diag` (fine) or parameterising with softplus (slower). And the LLM has to name and organise `d*(d-1)/2` off-diagonal scalars in `DEFAULT_PARAMS`, which is a legibility burden for the model author.

**Recommendation for v1**: diagonal (Option A). The evidence:

1. In every scientific testbed we care about first — Lorenz-63, coupled VdP, small HH populations — the observation noise is either independent per-channel (sensor noise) or so structured (shared mains hum) that a full covariance is a poor model of it anyway. A full-Σ model would waste parameters chasing a nuisance.
2. Adding a full-Σ observation model *also* makes the fingerprint code path harder to reason about (means become the discriminative signal; per-program covariance structure is an uninformative confound).
3. The diagonal case reduces to v1 exactly at `d=1`, which lets us reuse every unit test and the entire leakage-check machinery unchanged. Full-Σ would need new tests.

**v2 escape hatch**: if a real dataset shows meaningful cross-channel noise (fMRI physiological artifacts, EEG line noise), add a `full_covariance: bool` flag to the project's `load_data.py` module that switches `loss_fn` and `apply_model`'s log_sigma packing. This is a one-project change, not an engine change.

### 2.4 Dead-end: heteroscedastic per-step log_sigma

One temptation is to let the LLM emit a per-step log_sigma from `model_fn`:

```python
return new_state, mean, log_sigma  # 3-tuple — DON'T
```

This does not fit the current `apply_model → loss_fn` handshake without breaking v1's `(mean, log_sigma)` stacking. It also inflates fingerprint dimensionality unpredictably and forces the LLM to reason about noise per-step, which is a distraction from the actual scientific task (getting the mean right). If we ever need heteroscedastic noise, do it as a v2 project that overrides `apply_model` and `loss_fn` — not as a v1 DSL change.

## 3. Framework hooks — what actually changes in each

All changes below are in the *project's* `data_loader/load_data.py`. Engine unchanged.

### 3.1 `apply_model`

The current univariate `apply_model` (from `projects/fhn_excitable/data_loader/load_data.py`):

- Scans `y_traj[:-1]` (a 1-D array of scalars) through `model_fn`.
- Collects `means` of shape `(T-1,)`.
- Broadcasts a scalar `log_sigma_obs` into shape `(T-1,)`.
- Stacks: `jnp.stack([means, log_sigma], axis=-1)` → shape `(T-1, 2)`.
- Fingerprint path returns `targets - means` of shape `(T-1,)`.

The multivariate version scans `y_traj[:-1]` where `y_traj` is now shape `(T, d)`; each scan iteration feeds `y_prev` of shape `(d,)` and receives `mean` of shape `(d,)`.

```python
def apply_model(model_fn, data, params):
    y = data["y"]  # (n_samples, T, d)
    fingerprint_only = bool(data.get("_fingerprint_only", False))

    def per_sample(y_traj, p):
        init_state, dyn_params = _split_params_s0(p)

        def scan_step(state, y_prev):
            new_state, mean = model_fn(state, y_prev, dyn_params)
            return new_state, mean  # mean: (d,)

        _, means = jax.lax.scan(scan_step, init_state, y_traj[:-1])
        # means: (T-1, d)

        if fingerprint_only:
            targets = y_traj[1:]  # (T-1, d)
            return targets - means  # (T-1, d)

        # log_sigma_obs is a length-d vector in dyn_params; broadcast to (T-1, d)
        log_sigma_vec = dyn_params["log_sigma_obs"]  # (d,)
        log_sigma = jnp.broadcast_to(log_sigma_vec, means.shape)  # (T-1, d)
        # Pack as (T-1, d, 2): axis -1 is [mean, log_sigma], axis -2 is channel.
        return jnp.stack([means, log_sigma], axis=-1)  # (T-1, d, 2)

    return jax.vmap(per_sample, in_axes=(0, 0))(y, params)
    # returns (n_samples, T-1, d, 2)
```

**Design note on the stacked-array shape.** The v1 layout was `(T-1, 2)` — axis 0 is time, axis 1 is [mean, log_sigma]. The natural v2 layout is `(T-1, d, 2)` — insert a channel axis in the middle. This has three virtues:

1. `x[..., 0]` still selects means and `x[..., 1]` still selects log_sigmas, unchanged from v1. `loss_fn` gets an extra channel axis to average over but the column-selection idiom is preserved.
2. At `d=1`, we get shape `(T-1, 1, 2)`, which `loss_fn` handles by axis-average; the numeric result is bit-identical to v1's `(T-1, 2)` output. This means the multivariate `apply_model` can drop into the univariate project unchanged.
3. `flatten()` on `(T-1, d, 2)` gives a length-`2*(T-1)*d` vector for fingerprint dedup — see §4.

### 3.2 `loss_fn`

```python
def loss_fn(model_output, data):
    # model_output: (n_samples, T-1, d, 2)
    y = data["y"][:, 1:, :]  # (n_samples, T-1, d)
    means = model_output[:, WARMUP_STEPS:, :, 0]
    log_sigmas = model_output[:, WARMUP_STEPS:, :, 1]
    tgt = y[:, WARMUP_STEPS:, :]
    # Per-channel per-step NLL:
    nll_per_chan = log_sigmas + 0.5 * ((tgt - means) / jnp.exp(log_sigmas)) ** 2
    # Average over time AND channels → per-sample scalar loss (see §7.1).
    return jnp.mean(nll_per_chan, axis=(-1, -2))
```

Two things to notice, both discussed at length below:

- The mean over channels (`axis=-1`) is intentional and interacts with `param_penalty_weight` — see §7.
- The result shape is `(n_samples,)` — identical to v1, so `_optimize` and every downstream consumer of `loss_fn`'s output is unchanged.

### 3.3 `validate_step`

The v1 assertion `assert mean_arr.shape == ()` becomes:

```python
def validate_step(model_fn, default_params, program_code=""):
    # ... same prelude ...
    d = int(np.asarray(default_params["log_sigma_obs"]).size)  # channel count

    # Feed a shape-(d,) y_prev to catch shape mismatches early.
    y_prev = jnp.zeros((d,), dtype=jnp.float32) + 0.5
    new_state, mean = model_fn(init_state_j, y_prev, dyn_params_j)

    mean_arr = jnp.asarray(mean)
    assert mean_arr.shape == (d,), (
        f"model_fn must return mean of shape ({d},), got {mean_arr.shape}"
    )
    # ... rest of checks unchanged ...
```

Also add: check that `default_params["log_sigma_obs"]` is a length-d vector, and that every `s0_*` key whose corresponding state entry is a vector has the same length d (or is a scalar, if the LLM intentionally chose a scalar hidden state).

This is the *earliest* place we can catch "LLM wrote a scalar mean when we asked for a d-vector" — much better than surfacing as a broadcast error 500 lines deep in `lax.scan`.

### 3.4 `WARMUP_STEPS`

**Unchanged.** Still a module-level Python int. All the reasoning from §5.3 of the leakage-fix doc applies verbatim. Vector observations don't change the warmup story: 100 steps for hidden state to stabilise is a per-*model* property, not a per-channel property.

### 3.5 `_split_params_s0`

**Unchanged.** The prefix-strip logic is dict-of-key-value and doesn't care whether values are scalars or arrays.

There is one *convention* question: when the LLM declares `"s0_x": [0.0, 0.0, 0.0]`, does the parameter estimator return `{"s0_x": [0.0, 0.0, 0.0]}` (list) or `{"s0_x": jnp.array([0.0, 0.0, 0.0])}` (array)? The scoring pipeline already handles both — see the `jnp.stack([jnp.asarray(s[k]) for s in per_sample]) for k in per_sample[0]` at `edgar/scoring/scoring.py:130`. So no change. But the prompt for the param_estimator has to *say* this explicitly; see §5 below.

## 4. Fingerprint path

Recall v1's fingerprint layout: `apply_model` with `_fingerprint_only=True` returns `residuals = targets - means` of shape `(T-1,)`, per sample, giving `(n_samples, T-1)`. `island.py` calls `.flatten()` → `(n_samples * (T-1),)` and computes cosine similarity across programs.

Multivariate case: returning `(n_samples, T-1, d)` residuals is fine. `.flatten()` gives `(n_samples * (T-1) * d,)` — a length-d longer vector, same shape for any two programs with the same `d` (which they always have on the same project). The dedup code (`_are_duplicates`) already guards against shape mismatch with `if y_i.shape != y_j.shape: return False`, so mixed-d programs (which won't happen but who knows) would just be treated as non-duplicates.

**No engine change required.** This was the design goal — vector residuals are still an array, `.flatten()` doesn't care about shape, cosine similarity doesn't care about length.

### 4.1 Fingerprint discrimination is more delicate with multiple channels

There's a subtle risk. v1's fingerprint discrimination story (§5.7 of the leakage-fix doc) was that raw trajectories are strongly phase-locked to the driving signal, making cosine similarity ≥ 0.95 between *any* two programs — solved by using short (T=100) traces for `X_eval`. With d channels, the residuals concatenate: `[traj_1_ch_1, traj_1_ch_2, ..., traj_1_ch_d, traj_2_ch_1, ...]`. If two programs disagree on channel 1 but agree on channels 2..d, cosine similarity is *inflated* by the agreeing channels, hiding real differences.

Two mitigations:

- **Normalise per-channel before flattening.** Inside `apply_model`'s fingerprint path, divide each channel's residuals by that channel's std over time before returning. This makes each channel contribute equally to cosine similarity regardless of its scale.
- **Keep T_eval short.** Same as v1 — 100 steps rather than 2400. The rationale is unchanged: shorter trajectories reduce the phase-lock confound.

If in practice we see the fingerprint collapsing everything to one cluster (as it did in v1 before the T_eval fix), the natural next step is a PSD- or moment-based fingerprint rather than raw residuals. That's item 3 on the leakage-fix doc's follow-up list and is deferred here for the same reason.

## 5. Prompt changes

Three prompt schemas need updates: `model.code_guidelines`, `parameter_estimator.code_guidelines`, `jax_translator_model.code_guidelines`. All in the project's `prompts.yaml`.

### 5.1 model.code_guidelines — the key changes

The current univariate guideline (fhn_excitable) has three critical assertions that need multivariate versions:

Old:
> `state` is a dict of scalar floats (the current belief); `y_prev` is a scalar (the observation from the previous timestep); `params` is a dict of learnable scalars.
>
> OUTPUT: return the tuple `(new_state, mean)` where `new_state` is a dict with EXACTLY the same keys as `state`, and `mean` is a scalar
New:
> `state` is a dict whose values may be scalars OR vectors of shape (d,) or matrices of shape (d, d); `y_prev` is a JAX vector of shape (d,) (the previous observation); `params` is a dict of learnable scalars, vectors, or matrices.
>
> OUTPUT: return the tuple `(new_state, mean)` where `new_state` is a dict with EXACTLY the same keys and shapes as `state`, and `mean` is a JAX vector of shape (d,).
>
> The observation dimensionality d is fixed by the dataset. For this project, d = {d}. Every `mean` you return must have exactly d components; every observation `y_prev` has exactly d components. **Do not index y_prev with an integer that is not compile-time known** — use `y_prev[0]`, `y_prev[1]`, or vector arithmetic (`y_prev + state["x"]`), never `y_prev[t]` (t is not in scope here).
### 5.2 Worked example — 2-D coupled Kalman-lite

The v1 doc provides "persistence" and "FHN Kalman-lite" as worked examples. Here are the direct multivariate analogues:

**Vector persistence (baseline seed):**
```python
def model(state, y_prev, params):
    new_state = {"y_last": y_prev}  # shape (d,)
    mean = new_state["y_last"]  # shape (d,)
    return new_state, mean


model.DEFAULT_PARAMS = {
    "log_sigma_obs": [-1.5, -1.5, -1.5],  # d = 3 for Lorenz
    "s0_y_last": [0.0, 0.0, 0.0],
}
```

**2-D coupled Kalman-lite (for coupled VdP or 2-cell FHN):**
```python
def model(state, y_prev, params):
    x = state["x"]  # (2,) — observable position
    v = state["v"]  # (2,) — hidden velocity
    dt = params["dt"]
    mu = params["mu"]  # (2,) — per-cell VdP parameter
    k_coupling = params["k_coupling"]  # scalar — cross-cell coupling
    k_gain = params["k_gain"]  # (2,) — per-channel innovation gain

    innovation = y_prev - x  # (2,)
    x_corr = x + k_gain * innovation  # (2,)

    # Coupled VdP: each cell has its own nonlinear damping,
    # plus a linear coupling term to its neighbour.
    x_other = jnp.array([x_corr[1], x_corr[0]])  # swap for 2-cell coupling
    dv = mu * (1.0 - x_corr**2) * v - x_corr + k_coupling * (x_other - x_corr)
    v_new = v + dt * dv
    x_new = x_corr + dt * v_new

    new_state = {"x": x_new, "v": v_new}
    mean = x_new  # (2,)
    return new_state, mean


model.DEFAULT_PARAMS = {
    "dt": 0.05,
    "mu": [2.0, 2.0],
    "k_coupling": 0.1,
    "k_gain": [0.3, 0.3],
    "log_sigma_obs": [-1.5, -1.5],
    "s0_x": [2.0, -2.0],  # antiphase initial condition
    "s0_v": [0.0, 0.0],
}
```

Two features worth calling out to the LLM in the prompt (and worth being explicit in the seed docstring):

- **Vector arithmetic is the natural idiom.** `y_prev - x` computes a length-d innovation; `k_gain * innovation` does per-channel gain if `k_gain` is a vector, uniform gain if `k_gain` is a scalar. The LLM should be told that both forms are valid and that broadcasting Just Works.
- **Cross-channel coupling requires explicit index moves or matrix multiplies.** The 2-cell example above uses `jnp.array([x_corr[1], x_corr[0]])` to swap; for general d, it would use a coupling matrix `params["W"] @ x_corr` (shape (d, d) @ (d,) → (d,)). The prompt should give both patterns and mention that "using a coupling matrix `W` of shape (d, d) is often the cleanest way to express a network of interacting cells".

### 5.3 parameter_estimator.code_guidelines

The old guideline said "Values must be plain Python floats." The new guideline needs to say:

> Values must be plain Python floats OR plain Python lists of floats. Vector-valued keys (`log_sigma_obs`, `s0_x`, `s0_v`, `k_gain`, `mu`) must be lists of the correct length — for this project, d = {d}. Do NOT return numpy arrays; the framework converts to JAX arrays internally.
And: sensible-defaults advice for vector keys:

> - `log_sigma_obs`: `[float(np.log(np.diff(data["y"][:, i]).std())) for i in range(d)]` — per-channel log-std of first-difference residuals.
> - `s0_x` (observable-state initial value): `list(map(float, data["y"][0]))` — the first observation.
> - `s0_v` / hidden-state initial values: `[0.0] * d` — a small constant. Gradient descent will refine.
### 5.4 jax_translator_model.code_guidelines

Old:
> `state` and `new_state` are Python dicts with the SAME keys throughout the scan.
New — add:
> Every value in `state` has a fixed shape (scalar or (d,) or (d, d)) that must be invariant across scan iterations. **Do NOT reshape state values inside the model — a reshape from (d,) to (d, 1) would change the pytree shape and break the scan.**
>
> `y_prev` is a JAX array of shape (d,). Do not use `float(y_prev)` or `int(y_prev)` — these will fail on traced arrays. Use vector arithmetic throughout.
## 6. Coupled cells vs. independent channels — which is v1?

The user asked us to distinguish two cases explicitly:

### Case A — Independent channels

Each dimension is its own scalar univariate model with no cross-coupling. Formally, `dx_i/dt = f_i(x_i, params_i)` with no dependence on `x_{j≠i}`.

**Verdict**: this case is *trivially* the current univariate DSL run d times in parallel — vmap over channels. It doesn't require a multivariate DSL at all. We would not build a separate project for it; we would just run the univariate `fhn_excitable` on `d` datasets independently and average the losses.

**But wait**: the LLM might not *know* whether the channels are independent, and part of the scientific goal is to *discover* that they are or aren't. In that framing, "independent channels" is a *hypothesis* the LLM should be able to express (and be rewarded for if correct) inside the multivariate DSL, not a special case we hard-code. The multivariate DSL as designed above supports this: an LLM that writes `mean[i] = f_i(y_prev[i], state_i)` with no cross-channel state is expressing exactly the independent-channels hypothesis. If the true dynamics are independent, this program wins; if they're coupled, a coupled program wins.

### Case B — Coupled cells

Dimensions interact: `dx/dt = f(x, y)`, `dy/dt = g(x, y)`. Lorenz-63 (`dx/dt = σ(y-x)`, `dy/dt = x(ρ-z) - y`, `dz/dt = xy - βz`) is the paradigm.

**Verdict**: this is the interesting case and the *only reason* to build a multivariate DSL. If we only cared about independent channels, we would run univariate d times.

**Recommendation**: ship v1 supporting Case B natively. Case A falls out as a special case that the LLM can express inside the same contract.

### 6.1 On Appendix A's `model(cell_state, y_prev_self, coupling_input, params)` API

The design plan (`docs/plans/state_space_dsl.md`, referenced by the leakage-fix doc) sketches in an Appendix A a multi-cell extension where each *cell* has its own `model(cell_state, y_prev_self, coupling_input, params)` and the framework aggregates. That's a different design goal than this plan:

- **Appendix A's design**: model *one cell* per program; framework glues them into a network. Suited to homogeneous populations (100 neurons that all obey the same dynamics with different parameters). Requires a `coupling_fn` in the framework.
- **This plan's design**: model *the whole population* per program; each program returns a d-vector mean directly. Suited to small, heterogeneous coupled systems (Lorenz, coupled VdP, 2-cell FHN). No framework changes.

These are complementary, not competing. This plan targets the small-d, heterogeneous case first because it requires no engine changes; the homogeneous-population case (Appendix A) is a separate v3 project.

## 7. Loss-function scaling and `param_penalty_weight`

This is where I want to be most explicit, because §7 of the leakage-fix doc flagged this as the *single* place the DSL cost engineering effort.

### 7.1 Per-channel NLL: sum or mean?

The `loss_fn` I proposed uses `jnp.mean(nll_per_chan, axis=(-1, -2))` — average over time *and* channels. There are two defensible choices:

**Mean over channels (RECOMMENDED)**: `loss = mean_over_time(mean_over_channels(nll))`. At `d=1` this reduces exactly to v1's loss. The loss stays on the "nat per bin per channel" scale, comparable across projects with different d.

**Sum over channels**: `loss = mean_over_time(sum_over_channels(nll))`. This is the "total information" interpretation (joint likelihood of all channels). At `d=1` it's the same as v1. At `d=3` it makes the loss 3× larger, which inflates gradients by 3× and makes `param_penalty_weight` need retuning per-d.

I recommend mean. The evidence:

- **Adam is scale-sensitive** through the learning rate. Losses that scale with d push you toward retuning `learning_rate` per-project. Mean avoids this.
- **`param_penalty_weight` should express "how much I care about parsimony relative to fit per bin"** — a scale-invariant concept. Sum breaks that.
- **The rendered loss is directly interpretable** as an average per-channel NLL, which is what we want to compare across projects.

### 7.2 Does `param_penalty_weight` need to be retuned for multivariate?

Yes, but for a different reason than in the univariate DSL.

Recall §6.3 of the leakage-fix doc: the `s0_` convention added `|state|` scalar parameters, inflating `n_params` and forcing a 10× drop in `param_penalty_weight` (0.01 → 0.001). This effect *compounds* in multivariate:

- A univariate FHN model has `n_params ≈ 7` (params + `s0_V` + `s0_w`).
- A multivariate d=3 coupled model has `n_params ≈ 7*d = 21` (per-channel s0s, per-channel log_sigma, per-channel VdP mus). That's a 3× inflation on top of v1's inflation.

**Recommendation**: for a d-dim project, set `param_penalty_weight ≈ 0.001 / d`. For d=3 Lorenz, that's ~0.0003. For d=2 coupled VdP, ~0.0005.

**Caveat**: this heuristic assumes params scale linearly with d, which is roughly right for the vector `log_sigma_obs`, `s0_*`, and per-channel dynamics params. It's *wrong* for a full-Σ observation model (which is `O(d²)`) or for a full coupling matrix `W` of shape (d, d) (also `O(d²)`). If a project uses either, drop `param_penalty_weight` further (e.g. `0.001 / d²`).

We should NOT try to auto-scale this in the engine. It's a project-level knob and belongs in `config.yaml`.

### 7.3 An honest tradeoff

The `mean-over-channels` recommendation has one real cost. Suppose the LLM writes a program that predicts channel 1 well (near-oracle) and channels 2..d catastrophically badly. Under mean-over-channels, the good channel's contribution is diluted. Under sum-over-channels, the bad channels dominate. Which is "right"?

Sum-over-channels is more scientifically honest — a model that misses 2 of 3 channels is a bad model. Mean-over-channels is more forgiving and can reward "partial discovery" (getting one channel right is progress).

For an *evolutionary* system that needs a smooth fitness landscape, forgiving is better. Programs that discover *one* channel correctly should be rewarded and bred; sum-over-channels would kill them for still being bad on the other channels. This is the same argument v1 made for using WARMUP_STEPS (don't punish a program for the transient it can't help). I'm confident mean is right for v1.

If in practice we see programs that "cheat" by predicting one channel exactly and ignoring the others, revisit — but I'd expect this not to happen because the persistence baseline predicts all channels equally well and would already dominate a channel-cheater.

## 8. Engine changes required

Here I want to justify each proposed engine change and, wherever possible, argue against it (backwards-compatibility being the priority).

### 8.1 Files I initially thought needed changes but do not

- `edgar/scoring/scoring.py::_optimize` — `ravel_pytree` handles arbitrary pytrees. Vector params ravel to a longer flat vector, gradients flow normally. No change.
- `edgar/scoring/utils.py::eval_fingerprint` (imported into `scoring.py`) — internally the fingerprint call is `apply_model_fn(model_fn, X_eval, params_matched)`; the return value is passed through unchanged and stored on the program. No shape assumption. No change.
- `edgar/evolution/island.py::_are_duplicates` — calls `.flatten()` and cosine-similarity. Dimension-agnostic. No change.
- `edgar/evolution/program.py::default_params.setter` — uses `np.asarray(v).size`, which correctly sums vector-param sizes into `n_params`. No change.
- `edgar/io/task_spec.py::_extract_default_params` — reads `model.DEFAULT_PARAMS` verbatim. No change.
- `edgar/llm/response_schema.py::ModelSchema.default_params` — already declared as `dict` with values "numeric scalars or plain lists". No change.

This is the ideal outcome: a real DSL extension with zero engine changes. **Confidence: high — the subagent read every one of these files and each already handles vector-valued leaves correctly.**

### 8.2 Files that do need changes

None. That's the point of the design.

### 8.3 What we might *want* to change but shouldn't in v1

- **Response schema field for observation dimensionality.** We could add `n_channels: int` to `ModelSchema` so the LLM has to declare it. Rejected because (a) the LLM already declares it implicitly through the length of `log_sigma_obs` in `default_params`, and (b) adding a schema field is exactly the "five-file engine change" the leakage-fix doc §5.1 warned against. Keep `d` implicit; validate it in `validate_step`.
- **Config-level `n_channels`.** Same argument. Put it in `project_params` if we want it settable; do not add it to `edgar/io/config.py`.
- **Per-channel loss reporting on the dashboard.** Would be nice for debugging (which channel is a program getting right?). This is dashboard-only, not scoring, so it can be added later without touching the scoring loop.

## 9. Testbeds

Two concrete testbeds. Both are small enough to fit in the 3-5 day rollout.

### 9.1 Lorenz-63 (d=3, chaotic, everyone knows it)

```
dx/dt = σ(y - x)                            (σ = 10)
dy/dt = x(ρ - z) - y                        (ρ = 28)
dz/dt = xy - βz                             (β = 8/3)
```

Integrate with Euler at dt=0.01 (Lorenz is stiff-ish; dt=0.05 diverges). Add process noise `σ_p * dW` to each equation, add observation noise. Observe all three components.

**Why it's a good testbed**:
- Ground truth is known; oracle NLL is the one-step Euler prediction from true (x,y,z).
- Deterministic dynamics are chaotic → one-step-ahead prediction requires the *correct* nonlinear coupling; a linear (Kalman) filter cannot represent the `xy` and `xz` products.
- Small d, small state — fits in the compile-cost budget.
- Solvable oracle: `E[x[t+1]] = x[t] + dt * σ * (y[t] - x[t])` etc. Compute residual std per component after warmup; NLL floor is `(sum over d of log σ_d) + 0.5 * d`.

**Oracle-NLL calculation sketch**:
```
V_next_raw[i, t] = V[i, t-1] + dt * f_i(V[t-1])       # true Euler step
resid[i] = y[i, WARMUP:] - normalise(V_next_raw[i])
σ_MLE[i] = resid[i].std()
NLL_oracle = mean(log(σ_MLE)) + 0.5           # if using mean-over-channels
```

**Discovery budget**: on a preliminary sim (process_noise=0.3, obs_noise=0.1, dt=0.01, T=2400), rough expectation:
- Oracle: NLL ≈ -1.9 (with per-channel σ_MLE ≈ 0.15)
- Persistence: NLL ≈ -0.5 (chaotic → persistence is *bad*)
- Best linear seed: NLL ≈ -1.0 (Kalman on the true trajectory but no `xy` product)
- Discovery budget: ~1.4 nat — plenty of headroom.

(These numbers are estimates from experience with Lorenz; the actual oracle calculation would need to be run.)

### 9.2 Two coupled Van der Pol oscillators (d=2, quasi-periodic, gentler)

```
dx_1/dt = u_1
du_1/dt = μ(1 - x_1²)u_1 - x_1 + k(x_2 - x_1) + noise
dx_2/dt = u_2
du_2/dt = μ(1 - x_2²)u_2 - x_2 + k(x_1 - x_2) + noise
```

Observe x_1 and x_2. Hidden u_1, u_2. k controls coupling strength (k=0 → two independent VdP, k>0 → phase-locking).

**Why it's a good testbed**:
- Directly extends the existing `vdp_relaxation` project — reuse most of the data-synthesis code.
- Coupling k is a *scalar* parameter; the LLM has to discover cross-cell interaction, not multi-parameter magic.
- Ground truth: extend the existing 1-cell oracle NLL calc to 2 cells sharing coupling.
- Choice of k lets us set the "difficulty knob" — k=0 makes independent-channels the correct answer; k=0.5 makes coupling essential.

**Discovery budget** (estimate at k=0.3):
- Oracle: NLL ≈ -1.6 (matches vdp_relaxation's single-channel oracle at ~-1.7).
- Persistence: NLL ≈ -0.9.
- Best independent-channels seed: NLL ≈ -1.3 (each channel runs a VdP that ignores the other).
- Coupled model: NLL ≈ -1.55.
- Discovery budget between independent-and-coupled: ~0.25 nat. The distinctive feature is the *gap between best independent seed and best coupled model* — that gap is what tells us evolution has discovered the coupling.

### 9.3 Why these two and not something bigger

- Lorenz-63 is the smallest well-known chaotic system that requires *nonlinear* multivariate structure. Anything bigger (Lorenz-96, 100-neuron HH) becomes the Appendix-A homogeneous-population problem and needs a different DSL.
- Coupled VdP is the smallest system where the *coupling itself* is the discovery target. Great for validating that evolution finds cross-channel structure without being told to.
- Together they cover the two failure modes we most care about: nonlinear-product coupling (Lorenz's `xy`) and linear-additive coupling (VdP's `k*(x_j - x_i)`).

## 10. Risks, mitigations, and things that could go wrong that aren't obvious

### 10.1 JIT compile time explosion

Every new state shape → fresh JAX compile. In v1 that's ~5 seconds per program. In multivariate, state shapes get more varied (scalar vs (d,) vs (d,d) for coupling matrices), so the cache-hit rate across programs may be *lower* than in v1.

**Expected impact**: not order-of-magnitude worse. State shape is coarsely characterised by (n_scalar_state_keys, n_vector_state_keys, has_matrix_state), so a population of 32 programs might see 8-12 distinct trace signatures rather than 3-5 in v1. Compile budget rises from ~3 min to ~5-8 min per generation.

**Mitigation**: none for v1. If it becomes a bottleneck, consider requiring the LLM to use a canonical state form (e.g. always a single dict key `state["hidden"]` of shape (n_hidden,)) — but that would compromise expressiveness. Better to eat the compile cost.

### 10.2 Fingerprint collapse in high-d

If cosine similarity always exceeds 0.95 on vector residuals (because most of a residual vector's mass comes from the shared unpredictable component of the observations), dedup will over-cluster. This is a real risk — we saw it in v1 with long trajectories.

**Mitigation**: normalise residuals per-channel before flattening (§4.1). If that's not enough, drop T_eval further (from 100 to 50) or switch to a different fingerprint. Instrument dedup rate in the first few runs and adjust.

### 10.3 LLM writes scalar arithmetic when we asked for vector

The most likely LLM failure mode: `mean = state["V"] + params["dt"] * dV` where `V` is scalar and `dV` is scalar, returning a scalar mean of shape `()`. `validate_step` catches this ("expected mean shape (d,), got ()") but only during the eager check — during a run in the wild, if the LLM writes correct-looking code that happens to reduce to scalar, we get a shape error deep in the scan.

**Mitigation**: `validate_step` should be *explicitly* called on the LLM's model at load time in the scoring path, not just in the leakage-check script. This is a small addition — one call in `_worker` before `_optimize`. But it *does* touch the engine and I've said we shouldn't. Alternative: make the prompt aggressively clear ("mean has shape (d,); do not return a scalar"), rely on `inf` loss from a broken program to kill it in evolution, and accept that a program with `AssertionError` inside `_worker`'s catch-all handler already ends up dead-lettered.

The framework's existing behaviour on a shape mismatch is `inf` loss via `_worker`'s try/except. This is arguably fine — evolution kills it — but debug is annoying. I lean toward *not* adding an engine hook and instead making the prompt loud enough that this failure is rare.

### 10.4 The LLM confuses vector indexing with time indexing

The v1 doc's §3.1 explains why prompt discipline was inadequate for the leakage problem. Multivariate reintroduces a related risk: the LLM writes `y_prev[t]` thinking `t` is a time index, when `t` is not in scope (we're inside a scan step) and `y_prev` is a vector indexed by channel. This would raise a `NameError`, so it's a scope error, not a leakage; evolution kills it. But it *will* happen with some frequency and add to the noise floor.

**Mitigation**: the prompt should include a "common pitfall" section: `y_prev[i]` for integer `i` in `[0, d)` is a channel access; `y_prev[t]` will raise. Include a worked example that indexes both `y_prev[0]` and `state["x"][1]` correctly.

### 10.5 Initial conditions matter more with cross-cell coupling

For Lorenz, if all three components' initial `s0_*` are zero, the model has no gradient signal to learn the coupling (`dx/dt = σ(0-0) = 0`). The optimizer will get stuck. This isn't a bug — it's a common gradient-descent-on-coupled-systems failure mode.

**Mitigation**: prompt the LLM to initialise `s0_x`, `s0_y`, `s0_z` from the first observation (not zeros). The parameter estimator's default should be `s0_x = list(map(float, data["y"][0]))`. This propagates cleanly.

### 10.6 A subtle correctness trap: broadcasting bugs

In `apply_model`:
```python
log_sigma_vec = dyn_params["log_sigma_obs"]  # (d,)
log_sigma = jnp.broadcast_to(log_sigma_vec, means.shape)  # means: (T-1, d)
```

`jnp.broadcast_to((d,), (T-1, d))` broadcasts correctly *only* if the trailing axes align. This does — because the last axis of `means` is d — but if a project author writes `means` as `(d, T-1)` (channel-first) instead, the broadcast silently produces wrong shapes.

**Mitigation**: pick channel-*last* consistently in the framework; document it in the DSL contract; add an assertion in `apply_model` that `means.shape[-1] == log_sigma_vec.shape[0]`.

### 10.7 A subtle correctness trap: fingerprint returns wrong shape

If the fingerprint path returns residuals of shape `(T-1, d)` per sample, and a different program (buggy) returns `(T-1,)`, the shape check in `_are_duplicates` returns "not a duplicate" and both are kept. This is a *bug tolerance*, not a bug — but it means broken programs accumulate in the population without being deduped against each other.

**Mitigation**: `validate_step` should call `apply_model` in fingerprint-only mode and check the residual shape is `(T-1, d)`. This is a project-local check, no engine hit.

## 11. Rollout order and v1/v2 split

### 11.1 v1 (3-5 days)

Ships:
- Two new projects: `projects/lorenz_63/` and `projects/coupled_vdp/`.
- Each with: `config.yaml`, `prompts.yaml` (multivariate variants of the FHN/VdP prompts), `data_loader/load_data.py` (with multivariate `apply_model`, `loss_fn`, `validate_step`), a 4-program seed ladder (persistence, independent-channels, uncoupled dynamics, correctly-coupled dynamics), `leakage_check.py`, `scripts/lorenz_oracle_nll.py` and equivalent.
- No engine changes.

Ordering:
1. **Day 1**: `lorenz_63/` skeleton — data synthesis, `apply_model`, `loss_fn`, `validate_step`, oracle-NLL script. Verify end-to-end on hand-written seed.
2. **Day 2**: `lorenz_63/` seed ladder (4 programs). Verify leakage-check passes. Verify seed losses are > oracle.
3. **Day 3**: `lorenz_63/` prompts + smoke run (1 generation, 2 islands, 2 batch). Iterate on prompts until LLM produces syntactically valid multivariate programs.
4. **Day 4**: `coupled_vdp/` — mostly copy-paste from `lorenz_63/` with different dynamics and different prompt wording. Should be fast given day 1-3 gave us the pattern.
5. **Day 5**: full runs on both (~10-hour each in wall clock, mostly LLM latency), analysis, write-up.

### 11.2 v2 (deferred)

- **Full-Σ observation noise.** Only if a project needs it. Adds a Cholesky parameterisation to `params` and a full-Σ NLL to `loss_fn`. Project-local.
- **Multi-cell homogeneous populations** (Appendix A of the design plan `docs/plans/state_space_dsl.md`, not currently in this repo). Different DSL contract, different engine hook. Separate project (`projects/hh_population/`), separate design doc.
- **Per-step heteroscedastic noise.** Requires engine change to `apply_model → loss_fn` handshake. Deferred until we have a project that needs it.
- **Free-running rollout metric — no longer deferred.** `wilson_cowan` already evaluates on a free
  rollout (feeds its own `(E, I)` prediction back, teacher-forcing only the stimulus; commit
  c12736c). Still worth generalizing to the chaotic multivariate testbeds (Lorenz), where multi-step
  rollout is even more valuable than for oscillators, and promoting from a per-project `apply_model`
  choice to an engine-level metric.

### 11.3 What can slip out of v1 and be v1.5

- The `coupled_vdp/` project. `lorenz_63/` alone is enough to validate the multivariate DSL — and
  in fact `wilson_cowan` (a coupled 2-channel state-space project) already validates that
  multivariate works on the current base, so `lorenz_63`'s role is now more a *known-ground-truth
  chaotic* stress test than a first proof of the DSL. Coupled VdP is nice-to-have.
- Per-channel loss reporting on the dashboard.
- Fingerprint residual normalisation. Ship without; add if we see dedup collapse.

## 12. Files touched (v1)

*Engine*: zero.

*Projects* (all new, self-contained):

- `projects/lorenz_63/config.yaml`
- `projects/lorenz_63/prompts.yaml`
- `projects/lorenz_63/data_loader/load_data.py`
- `projects/lorenz_63/seed_programs/model{1..4}.py`, `param_est{1..4}.py`
- `projects/lorenz_63/leakage_check.py`
- `projects/lorenz_63/scripts/lorenz_oracle_nll.py`
- `projects/coupled_vdp/…` (same layout)

## 13. What could kill this plan

Two things, in order of concern:

1. **The LLM cannot reliably write vector-arithmetic JAX.** If, after prompt iteration, a substantial fraction (say >50%) of programs error out with shape mismatches, the discovery signal collapses under the noise. Mitigation is more prompt work — worked examples of good and bad patterns, explicit "d = 3 for this project", ban on `float()` and `int()` on JAX values. If this happens in practice, we may need to add a "shape hint" to the prompt template (auto-substituted from `default_params["log_sigma_obs"].size`) rather than hard-coding d in every prompt.
2. **Fingerprint dedup collapses to one cluster.** If cosine similarity between all programs' residuals is uniformly high, we lose the population-diversity mechanism and evolution converges early to a local minimum. Mitigation is per-channel normalisation (§4.1) and/or a shorter T_eval. If that isn't enough, we're in the same territory as the leakage-fix doc's follow-up item 3 — need a proper moment- or PSD-based fingerprint. That's a real engine change, defer to v2.

Neither is catastrophic. Both are things we can iterate on in place.

## 14. Explicit checklist for the implementer

Before starting:
- [ ] Re-read `docs/state_space_dsl_leakage_fix.md` end to end.
- [ ] Read `projects/fhn_excitable/data_loader/load_data.py` — this is the template.
- [ ] Read `projects/fhn_excitable/leakage_check.py` — the equivalent for multivariate is the acceptance test.

During implementation:
- [ ] Multivariate `apply_model` returns shape `(n_samples, T-1, d, 2)` in normal path, `(n_samples, T-1, d)` in fingerprint path.
- [ ] `loss_fn` uses mean-over-channels *and* mean-over-time.
- [ ] `validate_step` explicitly checks `mean.shape == (d,)`.
- [ ] `param_penalty_weight` in the project's `config.yaml` is `0.001 / d`.
- [ ] The prompts explicitly state `d = X for this project`.
- [ ] Seed ladder includes both an "independent channels" seed and a "coupled" seed so evolution can discover which the true dynamics are.
- [ ] Oracle NLL script exists and matches the loss_fn's channel-averaging convention.
- [ ] `leakage_check.py` has all four v1 invariants (isolation, shape, NLL sanity, seed-vs-oracle gap) *plus* a fifth: coupled seed beats best independent seed by the expected margin.

Acceptance:
- [ ] `leakage_check.py` passes on both new projects.
- [ ] A 1-generation smoke run produces at least one program with finite loss.
- [ ] A 4-generation full run produces at least one program within 25% of the discovery budget on the discover split.
- [ ] Post-run leakage inspection confirms zero programs read `y[t]` when predicting `y[t]` — vector indexing has not accidentally leaked.

---

*Plan ends. This is a specification for building; the empirical report should follow implementation in a separate document.*