---
name: data-loader-helper
description: Interactive helper for designing and writing a new EDGAR project's data_loader/load_data.py. Use at the preliminary stage of a new equation-discovery project to interview the user about their data and intended equation, work out the right (sample, trial) mapping and train/test/discover/validate splits, then write the loader. Triggers — "new project", "set up a data loader", "how should I structure my data for EDGAR", "what is one sample here", "help me write load_data".
---

# data-loader-helper

You are running an interview-and-design session with a scientist who is about to start a
new EDGAR equation-discovery project. Your job is **not** to immediately write code. It is
to first understand their system well enough — and make *them* understand EDGAR's fitting
architecture well enough — that the resulting `data_loader/load_data.py` encodes the
scientific claim they actually want to test. Only then do you write the loader.

The single most consequential decision in an EDGAR project is **"what is one sample, and
what is one trial?"** Everything else (splits, loss, seed models) follows from it, and
getting it wrong produces a loader that optimises and validates cleanly while testing a
*different* claim than intended, with no error raised. Treat that mapping as the spine of
the whole conversation.

You must hold the architecture below in your head precisely, then run the interview, then
produce the loader. Read `journal/data_structure_guidance.md` and (if relevant to their
domain) `journal/2026-06-16_eom_discovery_poc_spec.md` at the start — they are the
canonical worked discussion this skill operationalises.

---

## Part 1 — The architecture you must internalise before asking anything

**The fitting model (this is the part users get wrong).**
EDGAR fits **one parameter set per sample, completely independently**. Concretely, in
`edgar/scoring/scoring.py`, both the model and the params are `jax.vmap`'d over the leading
axis: `jax.vmap(model_fn, in_axes=(0, 0))(data, params)`. Each sample gets its own fitted
`params` dict; **no parameter value is ever shared across samples**. The reported loss is
the mean over samples of a per-sample loss.

Direct consequences you must reason from:

- **A constant that must be the *same value* across several entities is only enforced as
  shared if those entities live inside the *same sample*.** If they are separate samples,
  each gets its own independently-fit copy and nothing in the objective makes them agree —
  agreement becomes a post-hoc eyeball check, not something the loss enforces.
- **A quantity allowed to *vary* per entity should be its own sample** (it gets its own
  params for free).
- **`params` has no trial axis.** Whatever you flatten into the trial axis is explained by
  *one shared parameter set per sample*. So the trial axis is "the things one parameter set
  must simultaneously explain."

**The two splits, and what each actually tests.**
- **Train/test split is *within* a sample, along the *trials* axis.** Params are fit on
  train trials by gradient descent; loss is reported on held-out test trials. **The trials
  axis is therefore the only axis your validation probes generalisation across.** Whatever
  you call "trials" is the axis you are claiming the equation generalises over; every other
  axis is just "more data," not "more evidence of generalisation."
- **Discover/validate split is *across* samples (disjoint sets of samples).** `X_discover`
  is seen by the LLM discovery loop; `X_validate` is never seen during discovery and is the
  final held-out check that the discovered *form* transfers to fresh samples.

**Pointwise vs integrative (decides whether the time/trial axis can be shuffled).**
- **Pointwise / static map:** the target at each trial is computed from that trial's
  features alone (`f(state_t) -> y_t`), and the loss reduces over trials independently.
  Trials are exchangeable — you may shuffle, interleave, or drop them freely. Train/test
  splitting is pure index selection; no contiguity, no discontinuity markers, no NaN masks.
- **Integrative / autoregressive:** the target depends on *neighbouring* trials (velocity
  by finite-differencing positions, an ODE rollout scored against a trajectory, any
  recurrence `state_{t+1} = g(state_t)`). Now contiguity is load-bearing: a split that
  removes interior frames creates real breaks, so each split must stay contiguous runs or
  the loss must skip gaps (NaN-masking + NaN-aware loss).
- **The trap:** "I'm finding a differential equation" does *not* imply integrative. If the
  target is an analytic instantaneous derivative computed per frame (as in `particle_eom`,
  where velocity is the instantaneous `dx/dt` and the loss is plain per-frame MSE), it is
  **pointwise** and frames are fully exchangeable. Don't over-engineer discontinuity
  machinery for a problem that's actually pointwise. Probe this explicitly.

**Split granularity for a time/trial axis** (only relevant once pointwise is confirmed):
choose from how autocorrelated the frames are and how much the dynamics drift, not from a
default fraction. Block split → test may be an extrapolation to a later regime (conflates
wrong-form with different-regime). Per-frame interleave → adjacent train/test frames are
near-duplicates (leakage, test loss too optimistic). **Chunked interleave** (split into
contiguous blocks, alternate whole blocks to train/test) is the usual middle ground.

---

## Part 2 — The `load_data.py` contract (what you will ultimately write)

`data_loader/load_data.py` must define two callables. See the README "Setting Up a New
Project" section and `projects/particle_eom/data_loader/load_data.py` for a complete,
working reference; the latter shows session-as-sample with a (cell, time) trial flattening.

**`load_data(data_path, **kwargs) -> (X_discover, X_validate, X_eval)`**
- `X_discover = (X_disc_train, X_disc_test)` — seen by the LLM discovery loop.
- `X_validate = (X_val_train, X_val_test)` — never seen during discovery.
- `X_eval` (dict) — small subset of `X_disc_train`'s samples, used for model fingerprints.
  Same feature/response keys as the other splits, **plus `_sample_indices`** (a NumPy array
  of integer positions into `X_disc_train`'s sample axis selecting the included samples).
- All four split dicts share the same keys; values are **JAX arrays** whose **leading
  axis is samples** in that split. *That is the only axis the engine fixes.* Everything
  after it is free: the engine `jax.vmap`s the model/loss over axis 0 only and passes the
  entire remaining shape through to `model()`/`loss_fn()` untouched — it never inspects,
  reshapes, or assumes the rank of the trailing axes. So a feature may be

  `(n_samples, n_trials)`, `(n_samples, n_trials, n_neighbors)`, or keep several axes
  distinct, e.g. `(n_samples, n_cell, n_time, n_repeat)`. Whether to keep `(cell, time,
  repeat)` as separate axes or collapse some/all into one flattened "trial" axis
  (`(n_samples, n_cell*n_time*n_repeat)`) is **itself a design decision this skill helps
  make** — a layout/readability choice, not a constraint the engine imposes. The one hard
  rule is the reduction contract on `loss_fn` below: whatever trailing axes you keep,
  `loss_fn` must reduce *all* of them to one value per sample.
- `X_disc_*` and `X_val_*` hold **disjoint sets of samples** (discover/validate split).
  `X_disc_train`/`X_disc_test` (and the val pair) hold the **same samples**, split along
  **trials**.
- `kwargs` come from `project_params:` in `config.yaml`, so any knob (noise level, counts,
  cutoff, RNG seed) should be a named argument with a default.

**`loss_fn(model_output, data) -> JAX array of shape (n_samples,)`**
- Per-sample loss, and **this `(n_samples,)` return is the real contract** (the leading axis
  is the only thing the engine fixes). It is called on the already-`vmap`'d batch, so it must
  reduce *every* non-sample axis and keep only the sample axis; `scoring.py` then mean-reduces
  over samples itself. With a single flattened trial axis that reduction is `axis=-1`; if you
  kept several trailing axes (e.g. `(n_samples, n_cell, n_time, n_repeat)`), reduce all of them
  with `axis=tuple(range(1, output.ndim))` rather than a hard-coded `axis=-1`.
- `model_output` matches the model's per-sample output broadcast over samples (e.g.
  `(n_samples, n_trials)`, or `(n_samples, n_cell, n_time, n_repeat)` if that's the layout);
  `data` is the split dict (use e.g. `data['velocity']` as the target).

**Seed models** (`seed_programs/model*.py`) operate on **one sample**: `model(data, params)`
where `data['key']` has shape `(n_trials, ...)` (no sample axis — vmap removed it), returns
`(n_trials,)`, and carries `model.DEFAULT_PARAMS`. Parameter estimators
(`param_est*.py`, `parameter_estimator(data)`) return the same keys; keep them simple
(closed-form / heuristic, no `scipy.optimize`). You don't have to write these in this skill,
but the (sample, trial) decision constrains them, so discuss them.

---

## Part 2b — The running design log (maintain this throughout)

This interview gets long, and the failure mode it guards against — a loader that validates
cleanly while testing the *wrong* claim — is exactly what an unanchored conversation
produces: a decision made early drifts by the time you write code. So you keep a **living
design log** as a real file: your anti-drift anchor, the thing you re-read to re-ground after
a long exchange, and the end-of-process verification checklist. The fillable template
(decisions table + invariants checklist) lives next to this file as `design_log_template.md`.

**Lifecycle.**
1. **At the start of the interview**, copy `design_log_template.md` to
   `loader_design_scratch.md` in the repo root. (Scratch, because the project isn't
   scaffolded yet — there's no `projects/<name>/` to write into.)
2. **After each decision locks**, update the relevant row: fill `Decision` + `Rationale` and
   flip `Status` `proposed` → `confirmed`. Re-read the whole log before proposing the
   (sample, trial) mapping (step 6) — that step depends on every earlier row.
3. **At delivery**, after scaffolding, fold the log into `projects/<name>/DESIGN.md` (its
   design rationale), tick the invariants checklist against the loader you actually wrote,
   then delete `loader_design_scratch.md`. Don't commit the scratch file.

---

## Part 2c — Where you're starting from (ask this first)

You can't know the project's state on your own, so **open with two questions**:
1. **What is the name of the project?**
2. **Have you already created the project — is there a running or stub `load_data`, or
   nothing yet?**

Their answer puts you in one of two situations, which begin differently:

**Situation A — there's already a loader to react to** (a `load_data.py` with real content,
not just the `init-project` stub).
1. Read `projects/<name>/data_loader/load_data.py` plus the project's `config.yaml`
   `project_params`, seed models, and any `loss_fn`. Run it if you can — `uv run python` the
   loader, or `uv run edgar validate <name>` — to see actual shapes and catch errors.
2. Reverse-engineer the design *it encodes* and fill `loader_design_scratch.md` from it: what
   it treats as sample vs trial, the trailing-axis layout, pointwise-vs-integrative implied by
   the loss, the splits. Set each row's `Status` to `confirmed` only where the code clearly
   satisfies the Part 2b invariant; set it to **`broken`** where it violates a contract or the
   choice doesn't hold up (record what's wrong in `Rationale`).
3. Then run the interview (Part 3) **as a review**: walk the user through what their loader
   actually claims vs. what they intended, focusing on `broken` rows and any invariant the
   checklist fails. Resolve each to `confirmed`, then fix the loader.

**Situation B — nothing written yet** (no project folder, or only a stub loader). You don't
have enough to design anything, so start with the cold-start questionnaire in
`questionaire.md` (data shape/fields/description + the target equation as pseudo-code or
LaTeX). Work through it conversationally, recording answers into the design log, then proceed
into the deeper design questions in Part 3.

---

## Part 3 — Run the interview

Work conversationally and **one or two questions at a time** — this is a Socratic
back-and-forth, not a form to dump. Use the actual `AskUserQuestion` tool when a decision
has a small set of discrete options; use plain prose for open-ended elicitation. Lead with
your recommendation when you have one. Adapt order to what they volunteer; don't interrogate
mechanically. The goal is to fill in the design log from Part 2b — open the scratch file
first (or, in Situation A, the one you reverse-engineered), then update a row each time a
decision locks.

Things you must extract (and the reasoning each one drives):

1. **The data.** What was measured, what are the raw axes and their sizes, how is it
   stored, what's the noise story. Get concrete shapes.

2. **The target equation / hypothesis.** What relationship are they trying to discover, in
   words? What's the input (features) and what's the output (regression target)? Is there a
   known ground truth (synthetic) or not?

3. **What must be shared vs. what may vary** — *the* pivotal question. "Which quantities in
   your hypothesis must take the *same value* across multiple entities (a global constant, a
   population-level rule), and which are allowed to differ from one entity to the next?" The
   shared thing's scope defines a sample. Walk them through the consequence: a shared
   constant is only enforced if its scope = one sample.

4. **The generalisation claim.** "Across which axis do you want to *prove* the equation
   generalises?" That axis becomes trials (train/test runs along it). Make explicit that
   every other axis is just more data, not tested transfer.

5. **Pointwise vs integrative.** Probe how the target is computed. Is each trial's target a
   function of that trial alone, or does it depend on neighbouring trials (rollout, finite
   difference, recurrence)? Resolve the "it's an ODE so it must be integrative" trap
   directly. This decides whether trials are exchangeable and whether you need contiguity /
   NaN-masking.

6. **From 3–5, propose the (sample, trial) mapping** and *replay the consequences back*:
   "With this choice, train→test tests <X>, and constant <C> is/ isn't enforced as shared.
   Is that the claim you want?" Offer the rejected alternatives and why they're weaker (the
   `data_structure_guidance.md` table is the template). Iterate until they agree.

7. **Split details.** Discover/validate sample counts; train/test trial split granularity
   (block vs chunked-interleave vs per-frame) justified by autocorrelation/drift; `X_eval`
   subset size. For synthetic data, the generator parameters and how ground truth is
   persisted (it's usually just the `project_params`, already saved in `task_spec.yaml`).

8. **Seed sanity.** Briefly: what one or two ingredient seed models look like and whether
   the engine can express the needed primitive (e.g. a sum-over-neighbours). Flag early if
   the capability under test isn't actually expressible.

---

## Part 3b — Visualise the split before trusting it

Once you have a runnable `load_data`, **always render the partition and look at it** before
launching anything — a split that silently tests the wrong claim usually *looks* wrong the
moment you plot it. Use the bundled helper (next to this file):

```python
from plot_split import plot_split
out = load_data(**project_params)            # (X_discover, X_validate, X_eval)
# If the loader reduces arrays per split (different column counts), pass the in-sample
# index arrays so the held-out positions show as white masks and block-vs-interleave is
# visible. Omit them if the loader already returns full-width NaN-masked arrays.
plot_split(out, key="<your_target_key>",
           within_sample_index=(train_idx, test_idx),
           save_path="split.png")
```

It draws a 2x2 grid — discover/validate × fit(train)/eval(test) — of the **actual data
values** with each panel's non-member region masked white, plus shape/mean/std per panel.
Read it for: (a) the fit/eval split falling where you intended along the in-sample axis
(contiguous block vs interleaved chunks); (b) discover and validate being disjoint along the
sample axis; (c) no panel that is accidentally empty, constant, or unnormalised. Run
`uv run python plot_split.py` once to see a reference example. Show the figure to the user
and confirm it depicts the claim you agreed on.

## Part 4 — Deliver

Once the design is agreed:

1. Restate the final design compactly: sample = …, trial = …, features/keys + shapes,
   target, loss, splits, pointwise/integrative, and the one-line statement of what
   train→test and discover→validate each prove. This is just the Part 2b decisions table
   read back in prose — if any row is still `proposed`, resolve it before writing code.
2. Confirm the approach in 2–3 sentences before writing (per repo convention for non-trivial
   changes), then scaffold if needed (`uv run edgar init-project <name>`) and write
   `data_loader/load_data.py` + `loss_fn`, following the `particle_eom` loader's structure
   and docstring depth. Make every tunable a `project_params` kwarg with a default.
3. Validate it: `uv run edgar validate <name>`, and where feasible a `uv run edgar test
   projects/<name>/config.yaml` smoke run. Don't claim it works if you haven't run it; say
   what's untested (GPU/real-data/UI limits).
4. Fold the Part 2b design log into `projects/<name>/DESIGN.md`, tick the invariant
   checklist against the loader you actually wrote (any unticked box is a blocker, not a
   footnote), then delete `loader_design_scratch.md`.
5. Offer to capture the design rationale in the day's journal entry, and (if the
   sample/trial reasoning was non-obvious) suggest a memory note.

Keep responses short and match the user's depth. You are a skeptical collaborator: if their
intended mapping would silently test the wrong claim, say so plainly before writing
anything.
