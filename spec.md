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

## Step 2

### Step 2.0 — Include seed models in both plots

**Why:** Seeds are the reference baseline and the root of all lineages. Their absence makes the lineage graph disconnected and prevents computing perplexity in 2.1.

**Changes in `src/hypothesis_engine.py`**
After the seed evaluation loop (after `initial_programs` is populated, around line 1432), call `_append_generation_record` for each seed:
```python
for seed_idx, row in initial_programs.iterrows():
    seed_n_params = int(row['params'][0].shape[1])
    _append_generation_record(generation_log_path, {
        "iteration_number": -1,
        "birth_island": -1,
        "batch_index": int(seed_idx),
        "parent1_id": None,
        "parent2_id": None,
        "train_loss": float(row['train_loss']),
        "initial_loss": float(row['initial_loss']),
        "n_params": seed_n_params,
        "complexity_penalty": float(param_penalty_weight * seed_n_params),
        "model_code_numpy": row['program_code_string'],
        "param_est_code": row['parameter_estimator_code_string'],
        "model_prompt": None,
        "model_llm_response": None,
        "param_est_prompt": None,
        "param_est_llm_response": None,
        "llm_name": None,
        "temperature": None,
        "mode": "seed",
    })
```

**Changes in `src/progress_report/progress_monitor.py`**
- Remove the filter `iteration_number != -1`; instead, separate `seed_records` and `prog_records`.
- In `_generate_loss_progress_html`: add seeds as a distinct trace at x = -1 with a special marker style (e.g. star shape, black outline, larger).
- In `_generate_gd_effect_html`: include seeds the same way.
- Update sidebar data to include seeds (they get their own indices).
- In `create_dynamic_progress_update`: pass both `seed_records` and `prog_records` to the two HTML generators.

### Step 2.1 — Relative perplexity metric

**Definition:** `P(L) = exp(-(L - L_0))` where `L_0 = min(seed train losses)`.
- P > 1: better than the best seed; P = 1: equal to seed; P < 1: worse.
- When penalty toggle is on: use `L = train_loss`; when off: use `L = train_loss - complexity_penalty`.

**Changes in `_generate_loss_progress_html`**
- Compute `L_0` in Python from `seed_records`: `L_0 = min(r['train_loss'] for r in seed_records)`.
- Replace `y_train_loss = -train_loss` with `y_train_loss = exp(-(train_loss - L_0))`.
- Same for `y_raw_loss` (without penalty).
- Update y-axis label: `'Relative perplexity P(L) = exp(-(L − L₀))'`.
- Update range computation accordingly (`y_range` is now always positive).
- Pass `L_0` into the HTML as a JS constant for tooltip display.
- Seeds will plot at P = 1 by definition (useful visual reference).

**Changes in `_generate_gd_effect_html`**
- Both axes change: x = P(initial_loss), y = P(train_loss).
- The y=x diagonal remains the "no improvement" reference.
- Update axis labels.

### Step 2.2 — Connect nodes to parents with lineage edges

**Data preparation in Python (before building HTML)**
- Build position map: `node_pos = {(iter, island, batch): (x_jittered, y_perplexity)}` for every record.
- Build an edge list: for each `prog_record` with valid `parent1_id` / `parent2_id`, add `(child_key, parent_key)` pairs.
- Aggregate into two arrays `edge_x` and `edge_y` (use `None` separators between line segments — Plotly's convention for disconnected lines in a single trace).
- Pass `edge_x`, `edge_y` to the HTML as a single dedicated line trace rendered before the scatter points, with low opacity (e.g. 0.3), thin line (width 1), grey colour, `hoverinfo: 'none'`, `showlegend: false`.

**Note on jitter consistency:** Use the same formula `(island_idx - (n_islands-1)/2) * 0.05` for parents; the parent island is `parent_id[1]`.

### Step 2.3 — Hover to highlight lineage

**Data in HTML**
- Inject a `parentMap` JS object: `{rec_idx: [parent1_rec_idx_or_null, parent2_rec_idx_or_null], ...}`.
- Inject an `edgeMap` JS object: `{rec_idx: [edge_segment_idx, ...]}` mapping each node to the indices of edge segments it is a child of.

**JS behaviour**
- On `plotly_hover`:
  1. Walk `parentMap` upward from the hovered `rec_idx` to collect all ancestor indices.
  2. Find all edge segment indices connecting this lineage.
  3. `Plotly.restyle` the edge trace to draw the ancestors' edges in a highlight colour (e.g. orange, width 2, opacity 1) while dimming others.
  4. Optionally increase marker size for ancestor scatter points.
- On `plotly_unhover`: restore original styling.

**Implementation note:** Rather than restyling a single combined edge trace, it may be cleaner to maintain two edge traces — `edges_dim` (always dim, all edges) and `edges_highlight` (initially empty, filled on hover) — to avoid complex index arithmetic.
