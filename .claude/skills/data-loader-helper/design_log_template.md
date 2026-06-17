# Loader design log — <project name>

Working copy of the data-loader-helper design log, living at `projects/<name>/design_log.md`.
Seeded from this template by `setup_design_log.sh` once the project name is decided; filled in
as decisions lock. See the skill's `SKILL.md` ("Keep a running design log") for the lifecycle.
This is the anti-drift anchor — re-read it before proposing the (sample, trial) mapping, and
verify Section 2 against the loader you actually wrote before claiming done.

## Section 1 — Decisions table

Fill `Decision` + `Rationale` as each is settled. `Status` is one of:
🟡 (proposed — a candidate, not yet agreed) → ✅ (confirmed — agreed with the user) ·
❌ (broken — an *existing* loader makes this choice but it violates a contract / doesn't
satisfy the conditions — record what's wrong in `Rationale`, then resolve to ✅).
One or two lines per row — a ledger, not prose.

| Item | Decision | Rationale | Status |
|---|---|---|---|
| Raw data axes + sizes | (tbd) | | 🟡 |
| Target equation / hypothesis | (tbd) | | 🟡 |
| **Sample** = | (tbd) | what must be *shared* lives in one sample | 🟡 |
| **Trial** = | (tbd) | the axis train/test proves generalisation over | 🟡 |
| Trailing-axis layout | (tbd) | flat `(n_trials,)` vs distinct `(cell,time,repeat)` | 🟡 |
| Pointwise vs integrative | (tbd) | decides trial exchangeability / contiguity needs | 🟡 |
| Train/test split granularity | (tbd) | block / chunked-interleave / per-frame, by autocorr & drift | 🟡 |
| Discover/validate sample counts | (tbd) | | 🟡 |
| X_eval subset size | (tbd) | | 🟡 |

## Section 2 — Loader invariants checklist

Verify against the *written* loader before claiming done (a couple can be checked mid-way).
Any unticked box is a blocker, not a footnote.

- [ ] All four split dicts share the same keys; values are JAX arrays, leading axis = samples
- [ ] `X_disc_*` and `X_val_*` hold **disjoint** sample sets
- [ ] train/test hold the **same** samples, split along the trials axis
- [ ] `X_eval` carries `_sample_indices` indexing into `X_disc_train`'s sample axis
- [ ] `loss_fn` returns shape `(n_samples,)`, reducing *all* non-sample axes
- [ ] every `project_params` knob is a named `load_data` kwarg with a default
- [ ] split figure rendered ("Visualise the split") and confirmed by the user to depict the agreed claim
