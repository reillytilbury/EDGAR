# Loader design log — <project name>

Working copy of the data-loader-helper design log. Created by copying this template to
`loader_design_scratch.md` at the start of the interview; filled in as decisions lock; folded
into `projects/<name>/DESIGN.md` at delivery. See the skill's `SKILL.md` (Part 2b) for the
lifecycle. This is the anti-drift anchor — re-read it before proposing the (sample, trial)
mapping, and verify Section 2 against the loader you actually wrote before claiming done.

## Section 1 — Decisions table

Fill `Decision` + `Rationale` as each is settled. `Status` is one of:
`proposed` (a candidate, not yet agreed) → `confirmed` (agreed with the user) ·
`broken` (an *existing* loader makes this choice but it violates a contract / doesn't
satisfy the conditions — record what's wrong in `Rationale`, then resolve to `confirmed`).
One or two lines per row — a ledger, not prose.

| Item | Decision | Rationale | Status |
|---|---|---|---|
| Raw data axes + sizes | (tbd) | | proposed |
| Target equation / hypothesis | (tbd) | | proposed |
| **Sample** = | (tbd) | what must be *shared* lives in one sample | proposed |
| **Trial** = | (tbd) | the axis train/test proves generalisation over | proposed |
| Trailing-axis layout | (tbd) | flat `(n_trials,)` vs distinct `(cell,time,repeat)` | proposed |
| Pointwise vs integrative | (tbd) | decides trial exchangeability / contiguity needs | proposed |
| Train/test split granularity | (tbd) | block / chunked-interleave / per-frame, by autocorr & drift | proposed |
| Discover/validate sample counts | (tbd) | | proposed |
| X_eval subset size | (tbd) | | proposed |

## Section 2 — Loader invariants checklist

Verify against the *written* loader before claiming done (a couple can be checked mid-way).
Any unticked box is a blocker, not a footnote.

- [ ] All four split dicts share the same keys; values are JAX arrays, leading axis = samples
- [ ] `X_disc_*` and `X_val_*` hold **disjoint** sample sets
- [ ] train/test hold the **same** samples, split along the trials axis
- [ ] `X_eval` carries `_sample_indices` indexing into `X_disc_train`'s sample axis
- [ ] `loss_fn` returns shape `(n_samples,)`, reducing *all* non-sample axes
- [ ] every `project_params` knob is a named `load_data` kwarg with a default
- [ ] split figure rendered (Part 3b) and confirmed by the user to depict the agreed claim
