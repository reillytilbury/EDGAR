# Dynamic progress monitoring - project plan 
EDGAR is an equation discovery engine that uses an evolutionary algorithm involving LLMs to come up with equations that best describe a given dataset. 

Currently, evolutionary history is generated at the end of each run by creating a family tree. 
The family tree is generated as an HTML file based on a JSON file that is created and updated continuously. 

We now want to create another monitoring tool which allows us to track the progress made by EDGAR dynamically. 
In the monitoring tool, we want to see 
1. Number of iterations processed so far 
2. Progress of training loss so far (no test loss available until the end of the process)
3. Effect of gradient descent on model training loss vs model training loss with just parameter estimator 

I believe all of this information is already involved in the JSON file, so we just need to create a function like : 
    def create_dynamic_progress_update(JSON_FILE, OUTPUT_DIR): 
        if JSON_FILE is empty : 
            return None
        else : 
            # 1. Parse JSON file to extract all available information. Find out how far we are in the progress (n_iter / total_iter)
            # - extract the following detail of each program (this is very similar to family tree)
            # -- n_iter, island, batch, training loss, complexity penalty, exploration_expoitation_mode, llm name, learning rate, model code string, parameter estimator, parent program identity, prompt string used to generate this model, image prompt used to generate the program, model visualisation  
            # 2. Save 2 different HTML files - one showing the prgress of training loss, another showing the effect of gradient descent 

            # save files 
        return None 

---

## Step 1 — COMPLETED

### What was specified
0. Ensure complexity penalty is stored separately in the JSON file (not just the summed score).
1. Rename `src/family_tree` → `src/progress_report`; family tree building remains an available function.
2. Investigate what information is available dynamically in the JSONL file, how it is appended, and whether it is safe to read mid-run.
3. Modularise the JSONL reader from `family_tree` without changing family tree behaviour.
4. Write `create_dynamic_progress_update(json_file, output_dir)`.

### What was implemented

**Package layout (`src/progress_report/`)**
- `io.py` — `load_generation_log(path)`: line-by-line JSONL reader; skips empty/malformed lines; safe to call mid-run because each record is written atomically as a single `json.dumps() + '\n'` call.
- `family_tree.py` — moved from `src/family_tree`, unchanged in behaviour.
- `progress_monitor.py` — `create_dynamic_progress_update(json_file, output_dir)`: generates two standalone HTML files.

**JSONL record schema (per generated program)**
Key fields used by the monitor:
```
iteration_number, birth_island, batch_index,
train_loss, initial_loss, complexity_penalty, n_params,
parent1_id, parent2_id,           ← list [iter, island, batch]
model_code_numpy, param_est_code,
model_prompt, model_llm_response,
param_est_prompt, param_est_llm_response,
llm_name, temperature, mode
```
Seed programs currently have `iteration_number = -1`, `birth_island = -1`, `parent1/2_id = None`, but are **NOT** written to the JSONL (they live only in the in-memory DataFrame). This is addressed in Step 2.

**HTML outputs**

`progress_loss.html`
- x-axis: iteration number, jittered per island via `(island_idx - (n_islands-1)/2) * 0.05`
- y-axis: `-train_loss` (higher = better)
- Colour-coded scatter by island; marker size 14
- Initial Plotly view covers all iterations (explicit range computed from data in Python)
- Left-side control panel:
  - Penalty toggle: show `-train_loss` or `-(train_loss - complexity_penalty)`
  - Per-island checkboxes (any combination)
- Click any dot → slide-in sidebar with full program details (code, prompts, LLM responses)

`progress_gd_effect.html`
- x-axis: initial loss (parameter estimator only); y-axis: train loss after GD
- Dashed y=x reference line; points above diagonal = GD made things worse
- Same click-to-sidebar interaction; marker size 14

---

## Step 2 — COMPLETED

### What was implemented

**`src/hypothesis_engine.py`**
After the seed evaluation loop (`initial_programs` fully populated), a new block iterates over `initial_programs` and calls `_append_generation_record` for each seed with `iteration_number=-1`, `birth_island=-1`, `mode="seed"`, and all loss/code fields. Seeds are now included in the JSONL from the start of every run.

**`src/progress_report/progress_monitor.py`**

*Step 2.0 — Seeds in both plots*
- `create_dynamic_progress_update` separates `seed_records` (iter==-1) and `prog_records`, then builds `all_records = seed_records + prog_records` for consistent `rec_idx` indexing.
- Both HTML generators add a "Seeds" trace with a **star marker** at x=-1.
- `_build_sidebar_data` is called on `all_records` so seeds get their own sidebar entries.

*Step 2.1 — Relative perplexity y-axis*
- Added `_perplexity(L, L_0) = exp(-(L - L_0))`; `L_0 = min(seed train_losses)` (falls back to min of all prog losses if no seeds present).
- Loss progress HTML y-axis: `Relative perplexity P(L) = exp(−(L − L₀))` — higher is better, seeds sit at P≈1.
- Penalty toggle switches between `P(train_loss)` and `P(train_loss − penalty)`, each with its own `L_0`/`L_0_raw`.
- GD effect HTML axes: `Relative perplexity P(initial_loss)` vs `Relative perplexity P(train_loss)`; points above y=x diagonal mean GD improved the loss.

*Step 2.2 — Lineage edges in loss progress HTML*
- `_build_edge_structures` builds `edge_segs_penalty`, `edge_segs_raw` (lists of `{x:[child_x,parent_x], y:[...]}` per segment), `parent_map` ({rec_idx: [p1_idx, p2_idx]}), and `edge_map` ({child_rec_idx: [seg_indices]}).
- Node positions use the same jitter formula as scatter points; seeds are fixed at x=-1.
- Loss progress HTML carries two edge traces: **edges_dim** (grey, opacity 0.3, all edges always visible) and **edges_highlight** (orange, initially empty).
- Penalty toggle restyles both edge traces to the appropriate coordinate set.

*Step 2.3 — Hover-to-highlight lineage*
- `parentMap` and `edgeMap` injected as JS constants.
- `getAncestors(rec_idx)` BFS-walks `parentMap` to collect the full ancestor chain.
- `plotly_hover`: walks ancestors, collects their edge segments from `edgeMap`, fills `edges_highlight`; `plotly_unhover` clears it.
- `currentHoveredIdx` persists so highlight is reapplied correctly after a penalty-toggle.
