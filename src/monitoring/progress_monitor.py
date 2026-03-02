"""Dynamic progress monitoring for in-progress EDGAR runs.

Generates standalone HTML files from a partially or fully written
JSONL generation log, visualising training loss and gradient-descent
effectiveness per island.
"""

import json
import logging
import math
import os

from .io import load_generation_log, escape, resolve_image_path, build_record_entry, record_key, parse_parent_key

# Distinct colour palette for up to 8 islands
_ISLAND_COLOURS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
]


def _island_colour(island_idx: int) -> str:
    """Return a hex colour for island_idx, cycling if > 8 islands."""
    return _ISLAND_COLOURS[island_idx % len(_ISLAND_COLOURS)]


def _perplexity(L, L_0) -> float:
    """Compute relative perplexity P(L) = exp(-(L - L_0)).

    P > 1 means better than reference (L_0); P = 1 means equal; P < 1 worse.
    Returns 0.0 for overflow (very bad loss).
    """
    if L is None or L_0 is None:
        return None
    try:
        return math.exp(-(L - L_0))
    except OverflowError:
        return 0.0


def _build_sidebar_data(records: list[dict]) -> dict:
    """Build a JSON-serializable dict of node data keyed by record index.

    Args:
        records: All generation log records (seeds + programs).

    Returns:
        Dict mapping string index to sidebar-ready record dict.
    """
    sidebar = {}
    for idx, rec in enumerate(records):
        entry = build_record_entry(rec)
        entry["idx"] = idx
        sidebar[str(idx)] = entry
    return sidebar


def _build_edge_structures(
    all_records: list[dict],
    node_pos_penalty: dict,
    node_pos_raw: dict,
) -> tuple[list[dict], list[dict], dict, dict]:
    """Build lineage edge data for both penalty and raw modes.

    Args:
        all_records: Combined seed + program records (seeds first).
        node_pos_penalty: rec_idx -> (x, y) using penalty perplexity.
        node_pos_raw: rec_idx -> (x, y) using raw perplexity.

    Returns:
        Tuple of (edge_segs_penalty, edge_segs_raw, parent_map, edge_map).
        edge_segs_*: list of {'x': [x_child, x_parent], 'y': [y_child, y_parent]}
        parent_map: {rec_idx: [p1_idx_or_null, p2_idx_or_null]}
        edge_map: {rec_idx: [segment_index, ...]}  — child -> its edge segment indices
    """
    # Build key -> rec_idx lookup
    key_to_idx = {}
    for rec_idx, rec in enumerate(all_records):
        key_to_idx[record_key(rec)] = rec_idx

    # Build parent map
    parent_map = {}
    for rec_idx, rec in enumerate(all_records):
        p1_key = parse_parent_key(rec.get("parent1_id"))
        p2_key = parse_parent_key(rec.get("parent2_id"))
        p1_idx = key_to_idx.get(p1_key) if p1_key is not None else None
        p2_idx = key_to_idx.get(p2_key) if p2_key is not None else None
        if p1_idx is not None or p2_idx is not None:
            parent_map[rec_idx] = [p1_idx, p2_idx]

    # Build edge segments concurrently with edge_map
    edge_segs_penalty = []
    edge_segs_raw = []
    edge_map = {}

    for child_idx, parents in parent_map.items():
        edge_map[child_idx] = []
        for parent_idx in parents:
            if parent_idx is None:
                continue
            if child_idx not in node_pos_penalty or parent_idx not in node_pos_penalty:
                continue

            seg_idx = len(edge_segs_penalty)
            cx_p, cy_p = node_pos_penalty[child_idx]
            px_p, py_p = node_pos_penalty[parent_idx]
            cx_r, cy_r = node_pos_raw[child_idx]
            px_r, py_r = node_pos_raw[parent_idx]

            edge_segs_penalty.append({"x": [cx_p, px_p], "y": [cy_p, py_p]})
            edge_segs_raw.append({"x": [cx_r, px_r], "y": [cy_r, py_r]})
            edge_map[child_idx].append(seg_idx)

    return edge_segs_penalty, edge_segs_raw, parent_map, edge_map


_SIDEBAR_CSS = """
  body { margin: 0; font-family: 'Segoe UI', Tahoma, sans-serif; display: flex; height: 100vh; overflow: hidden; }
  #graph-container { flex: 1; position: relative; overflow: hidden; }
  #sidebar { width: 0; min-width: 0; overflow-y: auto; overflow-x: hidden; background: #f8f9fa;
             border-left: 2px solid #dee2e6;
             transition: width 0.3s ease, min-width 0.3s ease, padding 0.3s ease;
             padding: 0; box-sizing: border-box; }
  #sidebar.open { width: 33.33vw; min-width: 33.33vw; padding: 20px; }
  #sidebar h2 { margin-top: 0; color: #333; font-size: 18px; }
  #sidebar .field { margin-bottom: 12px; }
  #sidebar .field-label { font-weight: 600; color: #555; font-size: 13px; margin-bottom: 2px; }
  #sidebar .field-value { font-size: 13px; color: #222; }
  #sidebar pre { background: #272822; color: #f8f8f2; padding: 10px; border-radius: 4px;
                 font-size: 12px; overflow-x: auto; white-space: pre-wrap;
                 word-wrap: break-word; max-height: 300px; overflow-y: auto; }
  #sidebar details { margin-bottom: 8px; }
  #sidebar summary { cursor: pointer; font-weight: 600; color: #555; font-size: 13px; padding: 4px 0; }
  #sidebar summary:hover { color: #007bff; }
  #close-btn { position: absolute; top: 8px; right: 8px; background: none; border: none;
               font-size: 20px; cursor: pointer; color: #666; z-index: 10; }
  #close-btn:hover { color: #333; }
  .controls { position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.92);
              border: 1px solid #ccc; border-radius: 6px; padding: 10px 14px;
              font-size: 13px; z-index: 5; max-width: 260px; }
  .controls label { display: block; margin-bottom: 4px; cursor: pointer; }
  .controls .section-title { font-weight: 600; color: #444; margin-bottom: 6px; }
  .controls hr { margin: 8px 0; border: none; border-top: 1px solid #ddd; }
"""

_SIDEBAR_JS = """
function escapeHtml(s) {
  if (s === null || s === undefined) return '<em>N/A</em>';
  const div = document.createElement('div');
  div.textContent = String(s);
  return div.innerHTML;
}

function formatNum(v, digits) {
  if (v === null || v === undefined) return '<em>N/A</em>';
  return Number(v).toFixed(digits !== undefined ? digits : 4);
}

function showSidebar(idx) {
  const d = sidebarData[String(idx)];
  if (!d) return;
  const sb = document.getElementById('sidebar');
  const sc = document.getElementById('sidebar-content');

  let h = '<h2>Program ' + escapeHtml(d.iteration) + '_' + escapeHtml(d.island) + '_' + escapeHtml(d.batch) + '</h2>';
  h += '<div class="field"><span class="field-label">Iteration:</span> <span class="field-value">' + escapeHtml(d.iteration) + '</span></div>';
  h += '<div class="field"><span class="field-label">Island:</span> <span class="field-value">' + escapeHtml(d.island) + '</span></div>';
  h += '<div class="field"><span class="field-label">Batch:</span> <span class="field-value">' + escapeHtml(d.batch) + '</span></div>';
  h += '<div class="field"><span class="field-label">Train Loss:</span> <span class="field-value">' + formatNum(d.train_loss) + '</span></div>';
  h += '<div class="field"><span class="field-label">Initial Loss:</span> <span class="field-value">' + formatNum(d.initial_loss) + '</span></div>';
  h += '<div class="field"><span class="field-label">Complexity Penalty:</span> <span class="field-value">' + formatNum(d.complexity_penalty) + '</span></div>';
  h += '<div class="field"><span class="field-label">N Params:</span> <span class="field-value">' + escapeHtml(d.n_params) + '</span></div>';
  h += '<div class="field"><span class="field-label">Mode:</span> <span class="field-value">' + escapeHtml(d.mode) + '</span></div>';
  h += '<div class="field"><span class="field-label">LLM:</span> <span class="field-value">' + escapeHtml(d.llm_name) + '</span></div>';
  h += '<div class="field"><span class="field-label">Temperature:</span> <span class="field-value">' + escapeHtml(d.temperature) + '</span></div>';

  if (d.removal_reason) {
    h += '<div class="field" style="background:#fff3cd;padding:6px;border-radius:4px;margin-bottom:8px;">';
    h += '<span class="field-label" style="color:#856404;">Removed:</span> ';
    h += '<span class="field-value">' + escapeHtml(d.removal_reason.event_type) + ' (' + escapeHtml(d.removal_reason.rule) + ')</span>';
    h += '</div>';
  }

  if (d.model_code) {
    h += '<details open><summary>Model Code</summary><pre>' + escapeHtml(d.model_code) + '</pre></details>';
  }
  if (d.param_est_code) {
    h += '<details><summary>Parameter Estimator Code</summary><pre>' + escapeHtml(d.param_est_code) + '</pre></details>';
  }
  if (d.model_prompt) {
    h += '<details><summary>Model Prompt</summary><pre>' + escapeHtml(d.model_prompt) + '</pre></details>';
  }
  if (d.model_llm_response) {
    h += '<details><summary>LLM Response</summary><pre>' + escapeHtml(d.model_llm_response) + '</pre></details>';
  }
  if (d.param_est_prompt) {
    h += '<details><summary>Param Estimator Prompt</summary><pre>' + escapeHtml(d.param_est_prompt) + '</pre></details>';
  }
  if (d.param_est_llm_response) {
    h += '<details><summary>Param Estimator LLM Response</summary><pre>' + escapeHtml(d.param_est_llm_response) + '</pre></details>';
  }
  if (d.image_prompt_path) {
    h += '<details><summary>Prompt Image (Parents)</summary><img src="' + d.image_prompt_path + '" style="max-width:100%;margin-top:8px;"></details>';
  }
  if (d.train_fit_image_path) {
    h += '<details><summary>Train Fit</summary><img src="' + d.train_fit_image_path + '" style="max-width:100%;margin-top:8px;"></details>';
  }
  if (d.test_fit_image_path) {
    h += '<details><summary>Test Fit</summary><img src="' + d.test_fit_image_path + '" style="max-width:100%;margin-top:8px;"></details>';
  }

  sc.innerHTML = h;
  sb.classList.add('open');
  setTimeout(function() { Plotly.Plots.resize(document.getElementById('graph')); }, 350);
}

function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  setTimeout(function() { Plotly.Plots.resize(document.getElementById('graph')); }, 350);
}
"""

_COLOUR_MODE_JS = """
const DEAD_COLOUR = '#b0b0b0';
const CATEGORY_COLOURS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
  '#9467bd', '#8c564b', '#e377c2', '#17becf',
  '#bcbd22', '#7f7f7f'
];

let currentColourMode = 'dead_llm';

const islandColourByIdx = {};
allTraces.forEach(function(t) {
  islandColourByIdx[String(t.island_idx)] = t.colour;
});
// Seeds are drawn in a dedicated trace with island=-1.
islandColourByIdx['-1'] = '#222';

function _categoryKey(value, fallback) {
  if (value === null || value === undefined) return fallback;
  const s = String(value).trim();
  return s ? s : fallback;
}

function buildAliveCategoryColourMap(fieldName, fallbackLabel) {
  const keys = [];
  Object.keys(sidebarData).forEach(function(idx) {
    const d = sidebarData[idx];
    if (!d || d.is_dead) return;
    keys.push(_categoryKey(d[fieldName], fallbackLabel));
  });
  const unique = Array.from(new Set(keys)).sort();
  const out = {};
  unique.forEach(function(k, i) {
    out[k] = CATEGORY_COLOURS[i % CATEGORY_COLOURS.length];
  });
  return out;
}

const llmColourMap = buildAliveCategoryColourMap('llm_name', 'Unknown LLM');
const modeColourMap = buildAliveCategoryColourMap('mode', 'Unknown Mode');

function getNodeColour(recIdx, colourMode) {
  const d = sidebarData[String(recIdx)];
  if (!d) return '#444';

  if (colourMode === 'island') {
    return islandColourByIdx[String(d.island)] || '#444';
  }
  if (d.is_dead) {
    return DEAD_COLOUR;
  }
  if (colourMode === 'dead_llm') {
    const k = _categoryKey(d.llm_name, 'Unknown LLM');
    return llmColourMap[k] || '#1f77b4';
  }
  const k = _categoryKey(d.mode, 'Unknown Mode');
  return modeColourMap[k] || '#1f77b4';
}

function applyColourMode(colourMode) {
  currentColourMode = colourMode;
  const islandColours = allTraces.map(function(t) {
    return t.custom.map(function(idx) { return getNodeColour(idx, colourMode); });
  });
  const islandIndices = allTraces.map(function(_, i) { return i + TRACE_OFFSET; });
  if (islandIndices.length > 0) {
    Plotly.restyle(graphDiv, {'marker.color': islandColours}, islandIndices);
  }

  const seedTraceIdx = allTraces.length + TRACE_OFFSET;
  const seedColours = seedCustom.map(function(idx) { return getNodeColour(idx, colourMode); });
  Plotly.restyle(graphDiv, {'marker.color': [seedColours]}, [seedTraceIdx]);
}

function onColourModeToggle() {
  const selected = document.querySelector('input[name="colour-mode"]:checked');
  if (!selected) return;
  applyColourMode(selected.value);
}
"""


def _generate_loss_progress_html(
    all_records: list[dict],
    seed_records: list[dict],
    prog_records: list[dict],
    sidebar_data: dict,
    islands: list[int],
    title: str,
    L_0: float,
    L_0_raw: float,
    edge_segs_penalty: list[dict],
    edge_segs_raw: list[dict],
    parent_map: dict,
    edge_map: dict,
) -> str:
    """Generate the training loss progress HTML.

    Args:
        all_records: Seeds followed by program records (consistent rec_idx).
        seed_records: Seed-only records (iteration_number == -1).
        prog_records: Non-seed records.
        sidebar_data: Pre-built sidebar data dict keyed by rec_idx.
        islands: Sorted list of unique island indices (from prog_records only).
        title: HTML page title.
        L_0: Reference loss for penalty mode (min seed train_loss).
        L_0_raw: Reference loss for raw mode (min seed raw loss).
        edge_segs_penalty: Edge segments for penalty perplexity positions.
        edge_segs_raw: Edge segments for raw perplexity positions.
        parent_map: {rec_idx: [p1_idx_or_null, p2_idx_or_null]}.
        edge_map: {rec_idx: [segment_index, ...]} for lineage highlighting.

    Returns:
        Standalone HTML string.
    """
    n_islands = len(islands)
    seed_offset = len(seed_records)  # prog_records start at this index in all_records

    # Build one Plotly trace per island (prog records only)
    traces = []
    for island_idx in islands:
        island_recs = [
            (seed_offset + prog_idx, rec)
            for prog_idx, rec in enumerate(prog_records)
            if rec.get("birth_island") == island_idx
        ]
        if not island_recs:
            continue

        x_vals = []
        y_with_penalty = []
        y_without_penalty = []
        custom = []
        hover = []

        for rec_idx, rec in island_recs:
            iteration = rec.get("iteration_number", 0)
            jitter = (island_idx - (n_islands - 1) / 2.0) * 0.05
            x_vals.append(iteration + jitter)

            train_loss = rec.get("train_loss")
            complexity_penalty = rec.get("complexity_penalty", 0.0) or 0.0

            p_penalty = _perplexity(train_loss, L_0)
            raw_loss = (train_loss - complexity_penalty) if train_loss is not None else None
            p_raw = _perplexity(raw_loss, L_0_raw)

            y_with_penalty.append(p_penalty)
            y_without_penalty.append(p_raw)
            custom.append(rec_idx)

            hover_parts = [
                f"<b>i{iteration}_isl{island_idx}_b{rec.get('batch_index', '?')}</b>",
                f"P(L): {p_penalty:.4f}" if p_penalty is not None else "P(L): N/A",
                f"train loss: {train_loss:.4f}" if train_loss is not None else "train loss: N/A",
                f"penalty: {complexity_penalty:.4f}",
                f"mode: {rec.get('mode', '')}",
            ]
            hover.append("<br>".join(hover_parts))

        colour = _island_colour(island_idx)
        traces.append({
            "island_idx": island_idx,
            "x": x_vals,
            "y_with_penalty": y_with_penalty,
            "y_without_penalty": y_without_penalty,
            "custom": custom,
            "hover": hover,
            "colour": colour,
        })

    # Build seed trace data
    seed_x = []
    seed_y_penalty = []
    seed_y_raw = []
    seed_custom = []
    seed_hover = []
    for rec_idx, rec in enumerate(seed_records):
        train_loss = rec.get("train_loss")
        complexity_penalty = rec.get("complexity_penalty", 0.0) or 0.0
        p_penalty = _perplexity(train_loss, L_0)
        raw_loss = (train_loss - complexity_penalty) if train_loss is not None else None
        p_raw = _perplexity(raw_loss, L_0_raw)

        seed_x.append(-1)
        seed_y_penalty.append(p_penalty)
        seed_y_raw.append(p_raw)
        seed_custom.append(rec_idx)
        hover_parts = [
            f"<b>seed_{rec.get('batch_index', rec_idx)}</b>",
            f"P(L): {p_penalty:.4f}" if p_penalty is not None else "P(L): N/A",
            f"train loss: {train_loss:.4f}" if train_loss is not None else "train loss: N/A",
            f"penalty: {complexity_penalty:.4f}",
        ]
        seed_hover.append("<br>".join(hover_parts))

    # Compute data ranges for the initial view
    all_x = [xv for t in traces for xv in t["x"]] + seed_x
    all_y = [
        yv for t in traces for yv in t["y_with_penalty"] if yv is not None
    ] + [yv for yv in seed_y_penalty if yv is not None]
    if all_x:
        x_pad = max(0.5, (max(all_x) - min(all_x)) * 0.04)
        x_range = json.dumps([min(all_x) - x_pad, max(all_x) + x_pad])
    else:
        x_range = "null"
    if all_y:
        y_pad = max(0.05, (max(all_y) - min(all_y)) * 0.05)
        y_range = json.dumps([max(0.0, min(all_y) - y_pad), max(all_y) + y_pad])
    else:
        y_range = "null"

    traces_json = json.dumps(traces)
    sidebar_json = json.dumps(sidebar_data, default=str)
    islands_json = json.dumps(islands)
    seed_x_json = json.dumps(seed_x)
    seed_y_penalty_json = json.dumps(seed_y_penalty)
    seed_y_raw_json = json.dumps(seed_y_raw)
    seed_custom_json = json.dumps(seed_custom)
    seed_hover_json = json.dumps(seed_hover)
    edge_segs_penalty_json = json.dumps(edge_segs_penalty)
    edge_segs_raw_json = json.dumps(edge_segs_raw)
    parent_map_json = json.dumps({str(k): v for k, v in parent_map.items()})
    edge_map_json = json.dumps({str(k): v for k, v in edge_map.items()})
    title_esc = escape(title)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title_esc}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
{_SIDEBAR_CSS}
</style>
</head>
<body>
<div id="graph-container">
  <div class="controls" id="controls">
    <div class="section-title">Penalty</div>
    <label>
      <input type="checkbox" id="penalty-toggle" checked onchange="onPenaltyToggle()">
      Include complexity penalty
    </label>
    <hr>
    <div class="section-title">Colouring</div>
    <label><input type="radio" name="colour-mode" value="island" onchange="onColourModeToggle()"> By island</label>
    <label><input type="radio" name="colour-mode" value="dead_llm" checked onchange="onColourModeToggle()"> Dead vs alive (alive by LLM)</label>
    <label><input type="radio" name="colour-mode" value="dead_mode" onchange="onColourModeToggle()"> Dead vs alive (alive by mode)</label>
    <hr>
    <div class="section-title">Islands</div>
    <div id="island-checkboxes"></div>
  </div>
  <div id="graph" style="width:100%;height:100vh;"></div>
</div>
<div id="sidebar">
  <button id="close-btn" onclick="closeSidebar()">&times;</button>
  <div id="sidebar-content"></div>
</div>

<script>
const sidebarData = {sidebar_json};
const allTraces = {traces_json};
const islands = {islands_json};
const seedX = {seed_x_json};
const seedYPenalty = {seed_y_penalty_json};
const seedYRaw = {seed_y_raw_json};
const seedCustom = {seed_custom_json};
const seedHover = {seed_hover_json};
const edgeSegs = {{penalty: {edge_segs_penalty_json}, raw: {edge_segs_raw_json}}};
const parentMap = {parent_map_json};
const edgeMap = {edge_map_json};

// Trace layout:
//   0: edges_dim  (all edges, grey, dim)
//   1: edges_highlight  (ancestor edges, orange, on hover)
//   2 .. 2+n-1: island scatter traces
//   2+n: seed scatter trace
const TRACE_OFFSET = 2;
{_COLOUR_MODE_JS}

let showWithPenalty = true;
let currentHoveredIdx = null;

function buildEdgeArrays(segs) {{
  const ex = [], ey = [];
  segs.forEach(function(s) {{
    ex.push(s.x[0], s.x[1], null);
    ey.push(s.y[0], s.y[1], null);
  }});
  return {{x: ex, y: ey}};
}}

function getAncestors(rec_idx) {{
  const result = new Set();
  const queue = [rec_idx];
  while (queue.length > 0) {{
    const cur = queue.shift();
    if (result.has(cur)) continue;
    result.add(cur);
    const parents = parentMap[String(cur)];
    if (parents) {{
      parents.forEach(function(p) {{
        if (p !== null && p !== undefined && !result.has(p)) {{
          queue.push(p);
        }}
      }});
    }}
  }}
  return result;
}}

function updateHighlight(rec_idx, mode) {{
  const ancestors = getAncestors(rec_idx);
  const segs = edgeSegs[mode];
  const ex = [], ey = [];
  ancestors.forEach(function(a) {{
    const segList = edgeMap[String(a)];
    if (segList) {{
      segList.forEach(function(si) {{
        const s = segs[si];
        if (s) {{ ex.push(s.x[0], s.x[1], null); ey.push(s.y[0], s.y[1], null); }}
      }});
    }}
  }});
  Plotly.restyle(graphDiv, {{x: [ex], y: [ey]}}, [1]);
}}

// Build initial Plotly traces
function buildInitialTraces(withPenalty) {{
  const mode = withPenalty ? 'penalty' : 'raw';
  const edgeData = buildEdgeArrays(edgeSegs[mode]);

  const edgesDim = {{
    x: edgeData.x, y: edgeData.y,
    mode: 'lines', hoverinfo: 'none', showlegend: false,
    line: {{color: 'rgba(150,150,150,0.3)', width: 1}},
    type: 'scatter'
  }};

  const edgesHighlight = {{
    x: [], y: [],
    mode: 'lines', hoverinfo: 'none', showlegend: false,
    line: {{color: 'rgba(255,140,0,0.9)', width: 2}},
    type: 'scatter'
  }};

  const islandTraces = allTraces.map(function(t) {{
    return {{
      x: t.x,
      y: withPenalty ? t.y_with_penalty : t.y_without_penalty,
      customdata: t.custom,
      hovertext: t.hover,
      mode: 'markers',
      hoverinfo: 'text',
      name: 'Island ' + t.island_idx,
      marker: {{
        color: t.custom.map(function(idx) {{ return getNodeColour(idx, currentColourMode); }}),
        size: 14,
        line: {{width: 1, color: '#333'}}
      }},
      type: 'scatter'
    }};
  }});

  const seedTrace = {{
    x: seedX,
    y: withPenalty ? seedYPenalty : seedYRaw,
    customdata: seedCustom,
    hovertext: seedHover,
    mode: 'markers',
    hoverinfo: 'text',
    name: 'Seeds',
    marker: {{
      symbol: 'square',
      color: seedCustom.map(function(idx) {{ return getNodeColour(idx, currentColourMode); }}),
      size: 18,
      line: {{width: 1.5, color: '#fff'}}
    }},
    type: 'scatter'
  }};

  return [edgesDim, edgesHighlight].concat(islandTraces).concat([seedTrace]);
}}

const layout = {{
  title: {json.dumps(title)},
  showlegend: true,
  hovermode: 'closest',
  xaxis: {{
    title: 'Iteration (jittered by island)',
    showgrid: true,
    range: {x_range},
    autorange: {x_range} === null
  }},
  yaxis: {{
    title: 'Relative perplexity P(L) = exp(\u2212(L \u2212 L\u2080))',
    showgrid: true,
    range: {y_range},
    autorange: {y_range} === null
  }},
  margin: {{l: 60, r: 20, t: 60, b: 50}},
  plot_bgcolor: '#fff',
  paper_bgcolor: '#fff'
}};

const graphDiv = document.getElementById('graph');
Plotly.newPlot(graphDiv, buildInitialTraces(true), layout, {{responsive: true}});

// Click handler
graphDiv.on('plotly_click', function(data) {{
  if (data.points.length > 0) {{
    const pt = data.points[data.points.length - 1];
    if (pt.customdata !== undefined && pt.customdata !== null) {{
      showSidebar(pt.customdata);
    }}
  }}
}});

// Hover / unhover for lineage highlighting
graphDiv.on('plotly_hover', function(data) {{
  if (data.points.length > 0) {{
    const pt = data.points[data.points.length - 1];
    if (pt.customdata !== undefined && pt.customdata !== null) {{
      currentHoveredIdx = pt.customdata;
      updateHighlight(currentHoveredIdx, showWithPenalty ? 'penalty' : 'raw');
    }}
  }}
}});

graphDiv.on('plotly_unhover', function() {{
  currentHoveredIdx = null;
  Plotly.restyle(graphDiv, {{x: [[]], y: [[]]}}, [1]);
}});

// Penalty toggle
function onPenaltyToggle() {{
  showWithPenalty = document.getElementById('penalty-toggle').checked;
  const mode = showWithPenalty ? 'penalty' : 'raw';

  // Update edges_dim (trace 0)
  const edgeData = buildEdgeArrays(edgeSegs[mode]);
  Plotly.restyle(graphDiv, {{x: [edgeData.x], y: [edgeData.y]}}, [0]);

  // Update island traces
  const newY = allTraces.map(function(t) {{
    return showWithPenalty ? t.y_with_penalty : t.y_without_penalty;
  }});
  const islandIndices = allTraces.map(function(_, i) {{ return i + TRACE_OFFSET; }});
  Plotly.restyle(graphDiv, {{y: newY}}, islandIndices);

  // Update seed trace
  const seedTraceIdx = allTraces.length + TRACE_OFFSET;
  Plotly.restyle(graphDiv, {{y: [showWithPenalty ? seedYPenalty : seedYRaw]}}, [seedTraceIdx]);

  // Update axis title
  const newTitle = showWithPenalty
    ? 'Relative perplexity P(L) = exp(\u2212(L \u2212 L\u2080))'
    : 'Relative perplexity P(L\u2212penalty) = exp(\u2212(L\u2212penalty \u2212 L\u2080))';
  Plotly.relayout(graphDiv, {{'yaxis.title.text': newTitle}});

  // Re-apply highlight if hovering
  if (currentHoveredIdx !== null) {{
    updateHighlight(currentHoveredIdx, mode);
  }}
}}

// Island checkboxes
const cbContainer = document.getElementById('island-checkboxes');
islands.forEach(function(islandIdx) {{
  const label = document.createElement('label');
  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.checked = true;
  cb.onchange = function() {{
    const traceIdx = allTraces.findIndex(function(t) {{ return t.island_idx === islandIdx; }});
    if (traceIdx >= 0) {{
      Plotly.restyle(graphDiv, {{visible: cb.checked ? true : 'legendonly'}}, [traceIdx + TRACE_OFFSET]);
    }}
  }};
  label.appendChild(cb);
  label.appendChild(document.createTextNode(' Island ' + islandIdx));
  cbContainer.appendChild(label);
}});

{_SIDEBAR_JS}
</script>
</body>
</html>"""


def _generate_gd_effect_html(
    all_records: list[dict],
    seed_records: list[dict],
    prog_records: list[dict],
    sidebar_data: dict,
    islands: list[int],
    title: str,
    L_0: float,
    L_0_raw: float,
) -> str:
    """Generate the gradient-descent effect HTML.

    Axes show relative perplexity of initial_loss (x) vs train_loss (y).
    Points above the diagonal y=x means GD improved the loss (P(train) > P(initial)).

    Args:
        all_records: Seeds followed by program records.
        seed_records: Seed-only records.
        prog_records: Non-seed records.
        sidebar_data: Pre-built sidebar data dict.
        islands: Sorted list of unique island indices.
        title: HTML page title.
        L_0: Reference loss for perplexity computation.
        L_0_raw: Unused here; kept for consistent signature.

    Returns:
        Standalone HTML string.
    """
    seed_offset = len(seed_records)
    traces = []
    all_perp = []

    for island_idx in islands:
        island_recs = [
            (seed_offset + prog_idx, rec)
            for prog_idx, rec in enumerate(prog_records)
            if rec.get("birth_island") == island_idx
            and rec.get("initial_loss") is not None
            and rec.get("train_loss") is not None
        ]
        if not island_recs:
            continue

        x_vals = []
        y_vals = []
        custom = []
        hover = []

        for rec_idx, rec in island_recs:
            p_initial = _perplexity(rec["initial_loss"], L_0)
            p_train = _perplexity(rec["train_loss"], L_0)
            delta = rec["initial_loss"] - rec["train_loss"]

            x_vals.append(p_initial)
            y_vals.append(p_train)
            custom.append(rec_idx)
            hover.append(
                f"<b>i{rec.get('iteration_number')}_isl{island_idx}_b{rec.get('batch_index','?')}</b>"
                f"<br>P(initial): {p_initial:.4f}"
                f"<br>P(after GD): {p_train:.4f}"
                f"<br>raw improvement: {delta:.4f}"
                f"<br>mode: {rec.get('mode', '')}"
            )
            if p_initial is not None:
                all_perp.append(p_initial)
            if p_train is not None:
                all_perp.append(p_train)

        colour = _island_colour(island_idx)
        traces.append({
            "island_idx": island_idx,
            "x": x_vals,
            "y": y_vals,
            "custom": custom,
            "hover": hover,
            "colour": colour,
        })

    # Seed trace
    seed_x = []
    seed_y = []
    seed_custom = []
    seed_hover = []
    for rec_idx, rec in enumerate(seed_records):
        if rec.get("initial_loss") is None or rec.get("train_loss") is None:
            continue
        p_initial = _perplexity(rec["initial_loss"], L_0)
        p_train = _perplexity(rec["train_loss"], L_0)
        seed_x.append(p_initial)
        seed_y.append(p_train)
        seed_custom.append(rec_idx)
        delta = rec["initial_loss"] - rec["train_loss"]
        seed_hover.append(
            f"<b>seed_{rec.get('batch_index', rec_idx)}</b>"
            f"<br>P(initial): {p_initial:.4f}"
            f"<br>P(after GD): {p_train:.4f}"
            f"<br>raw improvement: {delta:.4f}"
        )
        if p_initial is not None:
            all_perp.append(p_initial)
        if p_train is not None:
            all_perp.append(p_train)

    # Diagonal reference line range
    if all_perp:
        diag_min = min(all_perp)
        diag_max = max(all_perp)
    else:
        diag_min, diag_max = 0.0, 2.0

    traces_json = json.dumps(traces)
    sidebar_json = json.dumps(sidebar_data, default=str)
    seed_x_json = json.dumps(seed_x)
    seed_y_json = json.dumps(seed_y)
    seed_custom_json = json.dumps(seed_custom)
    seed_hover_json = json.dumps(seed_hover)
    diag_json = json.dumps([diag_min, diag_max])
    title_esc = escape(title)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title_esc}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
{_SIDEBAR_CSS}
</style>
</head>
<body>
<div id="graph-container">
  <div class="controls" id="controls">
    <div class="section-title">Colouring</div>
    <label><input type="radio" name="colour-mode" value="island" onchange="onColourModeToggle()"> By island</label>
    <label><input type="radio" name="colour-mode" value="dead_llm" checked onchange="onColourModeToggle()"> Dead vs alive (alive by LLM)</label>
    <label><input type="radio" name="colour-mode" value="dead_mode" onchange="onColourModeToggle()"> Dead vs alive (alive by mode)</label>
  </div>
  <div id="graph" style="width:100%;height:100vh;"></div>
</div>
<div id="sidebar">
  <button id="close-btn" onclick="closeSidebar()">&times;</button>
  <div id="sidebar-content"></div>
</div>

<script>
const sidebarData = {sidebar_json};
const allTraces = {traces_json};
const seedX = {seed_x_json};
const seedY = {seed_y_json};
const seedCustom = {seed_custom_json};
const seedHover = {seed_hover_json};
const diagRange = {diag_json};
const TRACE_OFFSET = 1;
{_COLOUR_MODE_JS}

// Diagonal y=x reference (no improvement line)
// Points above diagonal: P(train) > P(initial) => GD improved the loss
const diagTrace = {{
  x: diagRange,
  y: diagRange,
  mode: 'lines',
  line: {{color: '#999', width: 1.5, dash: 'dash'}},
  hoverinfo: 'none',
  name: 'y = x (no improvement)',
  showlegend: true,
  type: 'scatter'
}};

const seedTrace = {{
  x: seedX,
  y: seedY,
  customdata: seedCustom,
  hovertext: seedHover,
  mode: 'markers',
  hoverinfo: 'text',
  name: 'Seeds',
  marker: {{
    symbol: 'square',
    color: seedCustom.map(function(idx) {{ return getNodeColour(idx, currentColourMode); }}),
    size: 18,
    line: {{width: 1.5, color: '#fff'}}
  }},
  type: 'scatter'
}};

const plotlyTraces = [diagTrace].concat(allTraces.map(function(t) {{
  return {{
    x: t.x,
    y: t.y,
    customdata: t.custom,
    hovertext: t.hover,
    mode: 'markers',
    hoverinfo: 'text',
    name: 'Island ' + t.island_idx,
    marker: {{
      color: t.custom.map(function(idx) {{ return getNodeColour(idx, currentColourMode); }}),
      size: 14,
      line: {{width: 1, color: '#333'}}
    }},
    type: 'scatter'
  }};
}})).concat([seedTrace]);

const layout = {{
  title: {json.dumps(title)},
  showlegend: true,
  hovermode: 'closest',
  xaxis: {{
    title: 'Relative perplexity P(initial loss) — param estimator only',
    showgrid: true
  }},
  yaxis: {{
    title: 'Relative perplexity P(train loss) — after GD  [above diagonal = GD improved]',
    showgrid: true
  }},
  margin: {{l: 60, r: 20, t: 60, b: 50}},
  plot_bgcolor: '#fff',
  paper_bgcolor: '#fff'
}};

const graphDiv = document.getElementById('graph');
Plotly.newPlot(graphDiv, plotlyTraces, layout, {{responsive: true}});

// Click handler (trace 0 is diagonal — skip it; seeds have customdata too)
graphDiv.on('plotly_click', function(data) {{
  if (data.points.length > 0) {{
    const pt = data.points[data.points.length - 1];
    if (pt.customdata !== undefined && pt.customdata !== null) {{
      showSidebar(pt.customdata);
    }}
  }}
}});

{_SIDEBAR_JS}
</script>
</body>
</html>"""


def create_dynamic_progress_update(json_file: str, output_dir: str) -> None:
    """Generate dynamic progress HTML files from an in-progress JSONL log.

    Safe to call mid-run: partial or corrupt JSONL lines are skipped.
    Writes two standalone HTML files into output_dir:
      - progress_loss.html  — training loss per iteration
      - progress_gd_effect.html — initial vs final loss showing GD effect

    Args:
        json_file: Path to the JSONL generation log (may be partially written).
        output_dir: Directory in which to write the HTML files.
    """
    if not os.path.isfile(json_file):
        logging.warning(
            "[progress_monitor] No generation log found at %s, skipping.", json_file
        )
        return

    records = load_generation_log(json_file)
    if not records:
        logging.warning("[progress_monitor] Generation log is empty, skipping.")
        return

    seed_records = [r for r in records if r.get("iteration_number", 0) == -1]
    prog_records = [r for r in records if r.get("iteration_number", 0) != -1]

    if not prog_records and not seed_records:
        logging.warning("[progress_monitor] No records found, skipping HTML generation.")
        return

    # Compute reference loss L_0 for perplexity
    seed_losses = [
        r["train_loss"] for r in seed_records if r.get("train_loss") is not None
    ]
    seed_raw_losses = [
        r["train_loss"] - (r.get("complexity_penalty") or 0.0)
        for r in seed_records
        if r.get("train_loss") is not None
    ]
    if seed_losses:
        L_0 = min(seed_losses)
        L_0_raw = min(seed_raw_losses) if seed_raw_losses else L_0
    else:
        # Fall back to min across all prog records
        all_losses = [r["train_loss"] for r in prog_records if r.get("train_loss") is not None]
        L_0 = min(all_losses) if all_losses else 0.0
        all_raw = [
            r["train_loss"] - (r.get("complexity_penalty") or 0.0)
            for r in prog_records
            if r.get("train_loss") is not None
        ]
        L_0_raw = min(all_raw) if all_raw else L_0

    # Unified record list: seeds first, then programs (gives consistent rec_idx)
    all_records = seed_records + prog_records
    sidebar_data = _build_sidebar_data(all_records)
    image_base_dir = os.path.dirname(json_file)
    rel_base = output_dir or image_base_dir or "."
    for entry in sidebar_data.values():
        for key in ("image_prompt_path", "train_fit_image_path", "test_fit_image_path"):
            abs_path = resolve_image_path(entry.get(key), image_base_dir)
            if abs_path:
                entry[key] = os.path.relpath(abs_path, rel_base)
            else:
                entry[key] = None

    islands = sorted({r.get("birth_island", 0) for r in prog_records})

    # Build node positions for edge rendering
    n_islands = len(islands)
    node_pos_penalty: dict[int, tuple[float, float]] = {}
    node_pos_raw: dict[int, tuple[float, float]] = {}
    for rec_idx, rec in enumerate(all_records):
        iter_num = rec.get("iteration_number", -1)
        island_idx = rec.get("birth_island", -1)
        train_loss = rec.get("train_loss")
        complexity_penalty = rec.get("complexity_penalty") or 0.0

        if iter_num == -1 or island_idx == -1:
            x = -1.0
        else:
            jitter = (island_idx - (n_islands - 1) / 2.0) * 0.05 if n_islands > 0 else 0.0
            x = iter_num + jitter

        y_p = _perplexity(train_loss, L_0) if train_loss is not None else 0.0
        raw_loss = (train_loss - complexity_penalty) if train_loss is not None else None
        y_r = _perplexity(raw_loss, L_0_raw) if raw_loss is not None else 0.0

        node_pos_penalty[rec_idx] = (x, y_p)
        node_pos_raw[rec_idx] = (x, y_r)

    edge_segs_penalty, edge_segs_raw, parent_map, edge_map = _build_edge_structures(
        all_records, node_pos_penalty, node_pos_raw
    )

    # Dead node = program explicitly removed via deduplication or pruning.
    for idx_str, entry in sidebar_data.items():
        entry["is_dead"] = entry.get("removal_reason") is not None

    os.makedirs(output_dir, exist_ok=True)

    # HTML 1: training loss progress
    loss_html = _generate_loss_progress_html(
        all_records=all_records,
        seed_records=seed_records,
        prog_records=prog_records,
        sidebar_data=sidebar_data,
        islands=islands,
        title="EDGAR — Training Loss Progress",
        L_0=L_0,
        L_0_raw=L_0_raw,
        edge_segs_penalty=edge_segs_penalty,
        edge_segs_raw=edge_segs_raw,
        parent_map=parent_map,
        edge_map=edge_map,
    )
    loss_path = os.path.join(output_dir, "progress_loss.html")
    with open(loss_path, "w") as f:
        f.write(loss_html)
    logging.info("[progress_monitor] Wrote %s", loss_path)

    # HTML 2: gradient descent effect
    gd_html = _generate_gd_effect_html(
        all_records=all_records,
        seed_records=seed_records,
        prog_records=prog_records,
        sidebar_data=sidebar_data,
        islands=islands,
        title="EDGAR — Gradient Descent Effect",
        L_0=L_0,
        L_0_raw=L_0_raw,
    )
    gd_path = os.path.join(output_dir, "progress_gd_effect.html")
    with open(gd_path, "w") as f:
        f.write(gd_html)
    logging.info("[progress_monitor] Wrote %s", gd_path)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Generate dynamic progress HTML files from an EDGAR generation log."
    )
    parser.add_argument(
        "--jsonl", required=True,
        help="Path to program_generation_log.jsonl (or directory containing it)",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory to write HTML files into (defaults to same directory as the JSONL file)",
    )
    args = parser.parse_args()

    jsonl_path = args.jsonl
    if os.path.isdir(jsonl_path):
        jsonl_path = os.path.join(jsonl_path, "program_generation_log.jsonl")

    if not os.path.isfile(jsonl_path):
        print(f"Error: JSONL file not found at {jsonl_path}")
        raise SystemExit(1)

    output_dir = args.output_dir or os.path.dirname(jsonl_path)
    create_dynamic_progress_update(jsonl_path, output_dir)
