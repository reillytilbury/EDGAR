# Roadmap

Issues are disabled on this repository, so later phases are tracked here. Only Phase 0 is being built.

## Phase 0: like-for-like EDGAR replication (current)

Run EDGAR's unchanged discovery loop with OpenCode Go models in place of Gemini. See [PHASE_0.md](PHASE_0.md).

- [x] Free-model run on `hy4-flash-free` (no spend): 1-generation smoke run and a full 12-generation run, with the winner rescored on locked test cells.
- [ ] Paid run: DeepSeek V4.1 Flash writes all code; Muse Spark 1.3 Contributor reads diagnostic images only. Total budget $10.

## Phase 1: off-the-shelf agentic harness (idea)

Question: does EDGAR's evolutionary scaffold add anything over a coding agent that has the same scorer and budget?

- **Main idea: the harness as the scientist.** One `opencode run` agent session gets:
  - the discovery cells;
  - the frozen scorer (`hypothesis_engine.objective` + `loss_functions.quadratic_loss`, penalty 0.01) as a CLI tool that returns the penalized discovery loss and a diagnostic PNG;
  - a fixed budget.

  The agent writes, scores and refines `neuron_model` and `parameter_estimator` itself.
- **Secondary idea: the harness as the LLM call.** Each proposal, translation or estimator request becomes an `opencode run` subprocess. This only adds overhead unless Go limits automated traffic to its own harness.
- **Fairness rules:**
  - the same model;
  - the same budget, in scorer calls (576) or in list-equivalent USD;
  - the same `strict-holdout-v1` cells;
  - the same `rescore` selection rule, with locked cells used once.
- **Report:** best-so-far discovery loss against scorer calls, plotted against the Phase 0 curve; the locked-test Δ against the double Gaussian; and whether a peaky exponent (p < 2) appears.
- Running through the OpenCode harness is also the most clearly permitted way to use OpenCode Go.

## Phase 2: custom equation-discovery harness (idea)

- Tools: scorer, plotter, per-cell residual summaries, a parameter-ablation helper.
- Memory: the program database as a tool, with scores and lineage, rather than as prompt text.
- Optional parallel sub-agents, as a replacement for islands.
- Same fairness rules as Phase 1, compared against Phases 0 and 1.
