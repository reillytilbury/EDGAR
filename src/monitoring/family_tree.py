"""Family tree visualization for EDGAR program evolution.

Generates interactive HTML files showing parent-child relationships
between programs across iterations, with click-to-inspect details.
"""

import json
import math
import os
from collections import defaultdict

import networkx as nx

from .io import (
    assign_display_labels,
    build_record_entry,
    escape,
    island_label,
    load_generation_log,
    make_program_id,
    parse_parent_key,
    record_key,
    resolve_image_path,
)



def _parse_parent_id(parent_id):
    """Parse a parent ID list [iteration, island, batch] into a node ID string.

    Returns None if parent_id is None or not a valid 3-element list.
    """
    key = parse_parent_key(parent_id)
    return make_program_id(*key) if key is not None else None


def _build_seed_nodes():
    """Create two synthetic seed nodes for iteration=-1."""
    seeds = []
    for batch_idx in range(2):
        seeds.append({
            "iteration_number": -1,
            "birth_island": -1,
            "batch_index": batch_idx,
            "train_loss": None,
            "train_fit_loss": None,
            "initial_loss": None,
            "n_params": None,
            "complexity_penalty": None,
            "model_code_numpy": None,
            "model_code_jax": None,
            "param_est_code": None,
            "model_prompt": None,
            "model_llm_response": None,
            "param_est_prompt": None,
            "param_est_llm_response": None,
            "llm_name": None,
            "temperature": None,
            "mode": "seed",
            "parent1_id": None,
            "parent2_id": None,
            "image_prompt_path": None,
            "train_fit_image_path": None,
            "test_fit_image_path": None,
            "test_fit_loss": None,
            "is_seed": True,
            "is_winner": False,
        })
    return seeds


def _display_train_loss(rec):
    """Return train-fit loss if available, otherwise fall back to objective train loss."""
    if not rec:
        return None
    train_fit = rec.get("train_fit_loss")
    if train_fit is not None:
        return train_fit
    return rec.get("train_loss")


def _display_test_loss(rec):
    """Return test-fit loss if available, otherwise fall back to objective test loss."""
    if not rec:
        return None
    test_fit = rec.get("test_fit_loss")
    if test_fit is not None:
        return test_fit
    return rec.get("test_loss")


def _merge_duplicate_records(records):
    """Merge duplicate records by program id, preserving later non-null fields."""
    merged_by_id = {}
    record_order = []

    for rec in records:
        node_id = make_program_id(*record_key(rec))
        if node_id is None:
            continue
        if node_id not in merged_by_id:
            merged_by_id[node_id] = dict(rec)
            record_order.append(node_id)
            continue

        merged = merged_by_id[node_id]
        for key, value in rec.items():
            if value is not None or key not in merged:
                merged[key] = value

    return [merged_by_id[node_id] for node_id in record_order]


def _compute_hierarchical_layout(G, records_by_id, island_gap=1.5):
    """Compute a hierarchical layout with iteration as Y-level.

    Within each level, nodes are grouped by birth island with a small extra
    gap between islands so they are visually distinct without hard dividers.
    """
    pos = {}
    levels = defaultdict(list)
    for node in G.nodes():
        rec = records_by_id.get(node)
        if rec:
            levels[rec["iteration_number"]].append(node)
        else:
            levels[-1].append(node)

    for level, nodes in levels.items():
        # Sort by island first, then deterministically within island
        nodes.sort(key=lambda n: (
            records_by_id.get(n, {}).get("birth_island", -1),
            records_by_id.get(n, {}).get("batch_index", 0),
        ))

        # Assign x positions: unit spacing within an island, extra gap between islands
        x_positions = []
        prev_island = None
        x = 0.0
        for i, node in enumerate(nodes):
            island = records_by_id.get(node, {}).get("birth_island", -1)
            if i > 0:
                x += 1.0
                if island != prev_island:
                    x += island_gap
            x_positions.append(x)
            prev_island = island

        # Center the row
        mid = (x_positions[0] + x_positions[-1]) / 2 if x_positions else 0.0
        y = -level
        for node, xp in zip(nodes, x_positions):
            pos[node] = (xp - mid, y)

    return pos


def _loss_to_color(loss, min_loss, max_loss):
    """Map a loss value to an RGB color string (green=good, red=bad).

    Uses log-scale normalisation so that the green-to-red gradient is spread
    across the full range of surviving node losses rather than being compressed
    near the minimum by right-skewed loss distributions.
    """
    if loss is None or min_loss is None or max_loss is None:
        return "rgb(180,180,180)"
    if max_loss == min_loss or min_loss <= 0 or max_loss <= 0 or loss <= 0:
        return "rgb(50,180,50)"
    log_min = math.log(min_loss)
    log_max = math.log(max_loss)
    if log_max == log_min:
        return "rgb(50,180,50)"
    t = min(1.0, max(0.0, (math.log(loss) - log_min) / (log_max - log_min)))
    # Interpolate green (good) to red (bad)
    r = int(50 + 205 * t)
    g = int(180 - 130 * t)
    b = int(50)
    return f"rgb({r},{g},{b})"


def _build_sidebar_data(records_by_id):
    """Build a JSON-serializable dict of node data for the sidebar."""
    label_map = {
        node_id: rec.get("display_label")
        for node_id, rec in records_by_id.items()
        if rec.get("display_label")
    }
    sidebar = {}
    for node_id, rec in records_by_id.items():
        entry = build_record_entry(rec, label_map)
        entry.update({
            "id": node_id,
            "display_label": rec.get("display_label"),
            "island_label": island_label(rec.get("birth_island", -1)),
            "test_loss": rec.get("test_loss"),
            "train_fit_loss": rec.get("train_fit_loss"),
            "test_fit_loss": rec.get("test_fit_loss"),
            "initial_loss": rec.get("initial_loss"),
            "mode": rec.get("mode"),
            "temperature": rec.get("temperature"),
            "llm_name": rec.get("llm_name"),
            "parent1_id": _parse_parent_id(rec.get("parent1_id")),
            "parent2_id": _parse_parent_id(rec.get("parent2_id")),
            "is_migrant": rec.get("is_migrant", False),
            "migrant_from_id": rec.get("migrant_from_id"),
            "migrant_from_label": rec.get("migrant_from_label"),
            "is_extinct": rec.get("is_extinct", False),
            "model_code": rec.get("model_code_numpy"),
            "model_code_jax": rec.get("model_code_jax"),
            "param_est_code": rec.get("param_est_code"),
            "model_prompt": rec.get("model_prompt"),
            "model_llm_response": rec.get("model_llm_response"),
            "param_est_prompt": rec.get("param_est_prompt", rec.get("param_est_prompt")),
            "param_est_llm_response": rec.get("param_est_llm_response"),
            "image_prompt_path": rec.get("image_prompt_path"),
            "train_fit_image_path": rec.get("train_fit_image_path"),
            "test_fit_image_path": rec.get("test_fit_image_path"),
            "is_seed": rec.get("is_seed", False),
        })
        sidebar[node_id] = entry
    return sidebar


def _resolve_image_path(raw_path, image_base_dir):
    """Resolve an image path: make absolute if relative, return None if missing."""
    if not raw_path:
        return None
    if not os.path.isabs(raw_path) and image_base_dir:
        raw_path = os.path.join(image_base_dir, raw_path)
    if os.path.isfile(raw_path):
        return raw_path
    return None


def _generate_html(G, pos, records_by_id, title, image_base_dir=None, html_output_dir=None, island_dividers=None):
    """Generate a standalone interactive HTML string for the family tree."""
    # Prepare edge traces
    edge_x = []
    edge_y = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Prepare node data
    node_x = []
    node_y = []
    node_ids = []
    node_labels = []
    node_hover = []
    node_colors = []
    node_symbols = []
    node_sizes = []

    # Find the winning node (marked with is_winner flag)
    best_node = None
    for node in G.nodes():
        rec = records_by_id.get(node, {})
        if rec.get("is_winner", False):
            best_node = node
            break

    # Compute loss range for coloring (based on test loss)
    losses = [(_display_test_loss(r)) for r in records_by_id.values()
              if _display_test_loss(r) is not None]

    min_loss = min(losses) if losses else None
    max_loss = max(losses) if losses else None
    # Use 95th percentile as max to avoid outlier skew
    if losses and len(losses) > 2:
        sorted_losses = sorted(losses)
        p95_idx = int(len(sorted_losses) * 0.95)
        max_loss = sorted_losses[min(p95_idx, len(sorted_losses) - 1)]

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_ids.append(node)

        rec = records_by_id.get(node, {})
        is_seed = rec.get("is_seed", False)
        iteration = rec.get("iteration_number", "?")
        batch = rec.get("batch_index", "?")
        loss = _display_train_loss(rec)
        test_loss = _display_test_loss(rec)
        mode = rec.get("mode", "")
        llm = rec.get("llm_name", "")

        label = rec.get("display_label")
        if not label:
            if is_seed:
                label = f"S{int(batch) + 1}" if batch != "?" else "S?"
            else:
                label = f"i{iteration}_b{batch}"
        node_labels.append(label)

        hover_parts = [f"<b>{label}</b>"]
        if loss is not None:
            hover_parts.append(f"primary train metric loss: {loss:.4f}")
        if test_loss is not None:
            hover_parts.append(f"primary test metric loss: {test_loss:.4f}")
        hover_parts.append(f"mode: {mode}")
        if llm:
            hover_parts.append(f"LLM: {llm}")
        node_hover.append("<br>".join(hover_parts))

        if is_seed:
            node_colors.append("rgb(180,180,180)")
        else:
            node_colors.append(_loss_to_color(test_loss, min_loss, max_loss))

        if node == best_node:
            node_symbols.append("star")
            node_sizes.append(48)
        else:
            node_symbols.append("circle")
            node_sizes.append(36)

    # Build sidebar data as JSON for embedding
    sidebar_data = _build_sidebar_data(records_by_id)

    # Resolve image paths to relative paths from the HTML output directory
    rel_base = html_output_dir or image_base_dir or "."
    for nid, entry in sidebar_data.items():
        for key in ("image_prompt_path", "train_fit_image_path", "test_fit_image_path"):
            abs_path = resolve_image_path(entry.get(key), image_base_dir)
            if abs_path:
                entry[key] = os.path.relpath(abs_path, rel_base)
            else:
                entry[key] = None

    # Build parent map: nodeId -> [parentId, ...] (only parents present in graph)
    parent_map = {}
    for nid, rec in records_by_id.items():
        parents = []
        for pk in ("parent1_id", "parent2_id"):
            pid = _parse_parent_id(rec.get(pk))
            if pid and pid in records_by_id:
                parents.append(pid)
        if parents:
            parent_map[nid] = parents

    # Build pos map: nodeId -> [x, y]
    pos_map = {nid: list(xy) for nid, xy in pos.items()}

    sidebar_json = json.dumps(sidebar_data, default=str)
    parent_map_json = json.dumps(parent_map)
    pos_map_json = json.dumps(pos_map)
    edge_x_json = json.dumps(edge_x)
    edge_y_json = json.dumps(edge_y)
    node_x_json = json.dumps(node_x)
    node_y_json = json.dumps(node_y)
    node_ids_json = json.dumps(node_ids)
    node_labels_json = json.dumps(node_labels)
    node_hover_json = json.dumps(node_hover)
    node_colors_json = json.dumps(node_colors)
    node_symbols_json = json.dumps(node_symbols)
    node_sizes_json = json.dumps(node_sizes)
    title_json = json.dumps(title)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ margin: 0; font-family: 'Segoe UI', Tahoma, sans-serif; display: flex; height: 100vh; overflow: hidden; }}
  #graph-container {{ flex: 1; position: relative; }}
  #sidebar {{ width: 0; min-width: 0; overflow-y: auto; overflow-x: hidden; background: #f8f9fa; border-left: 2px solid #dee2e6;
              transition: width 0.3s ease, min-width 0.3s ease, padding 0.3s ease; padding: 0; box-sizing: border-box; }}
  #sidebar.open {{ width: 33.33vw; min-width: 33.33vw; padding: 20px; }}
  #sidebar h2 {{ margin-top: 0; color: #333; font-size: 18px; }}
  #sidebar .field {{ margin-bottom: 12px; }}
  #sidebar .field-label {{ font-weight: 600; color: #555; font-size: 13px; margin-bottom: 2px; }}
  #sidebar .field-value {{ font-size: 13px; color: #222; }}
  #sidebar pre {{ background: #272822; color: #f8f8f2; padding: 10px; border-radius: 4px;
                  font-size: 12px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; max-height: 300px; overflow-y: auto; }}
  #sidebar details {{ margin-bottom: 8px; }}
  #sidebar summary {{ cursor: pointer; font-weight: 600; color: #555; font-size: 13px; padding: 4px 0; }}
  #sidebar summary:hover {{ color: #007bff; }}
  #close-btn {{ position: absolute; top: 8px; right: 8px; background: none; border: none; font-size: 20px;
                cursor: pointer; color: #666; z-index: 10; }}
  #close-btn:hover {{ color: #333; }}
  .loss-good {{ color: #28a745; font-weight: 600; }}
  .loss-bad {{ color: #dc3545; font-weight: 600; }}
</style>
</head>
<body>
<div id="graph-container">
  <div id="graph" style="width:100%;height:100%;"></div>
</div>
<div id="sidebar">
  <button id="close-btn" onclick="closeSidebar()">&times;</button>
  <div id="sidebar-content"></div>
</div>

<script>
const sidebarData = {sidebar_json};
const parentMap = {parent_map_json};
const posMap = {pos_map_json};

const edgeTrace = {{
  x: {edge_x_json},
  y: {edge_y_json},
  mode: 'lines',
  line: {{ color: '#aaa', width: 1.5 }},
  hoverinfo: 'none',
  type: 'scatter'
}};

const nodeTrace = {{
  x: {node_x_json},
  y: {node_y_json},
  customdata: {node_ids_json},
  text: {node_labels_json},
  hovertext: {node_hover_json},
  mode: 'markers+text',
  textposition: 'middle center',
  textfont: {{ size: 12, color: '#fff' }},
  hoverinfo: 'text',
  marker: {{
    color: {node_colors_json},
    symbol: {node_symbols_json},
    size: {node_sizes_json},
    line: {{ width: 1.5, color: '#333' }}
  }},
  type: 'scatter'
}};

const highlightTrace = {{
  x: [],
  y: [],
  mode: 'lines',
  line: {{ color: '#e8a020', width: 3 }},
  hoverinfo: 'none',
  type: 'scatter'
}};

const layout = {{
  title: {title_json},
  showlegend: false,
  hovermode: 'closest',
  xaxis: {{ showgrid: false, zeroline: false, showticklabels: false }},
  yaxis: {{ showgrid: false, zeroline: false, showticklabels: false }},
  margin: {{ l: 20, r: 20, t: 50, b: 20 }},
  plot_bgcolor: '#fff',
  paper_bgcolor: '#fff'
}};

const graphDiv = document.getElementById('graph');
Plotly.newPlot(graphDiv, [edgeTrace, highlightTrace, nodeTrace], layout, {{responsive: true}});

function getAncestorEdgeCoords(nodeId) {{
  const hx = [], hy = [];
  const visited = new Set();
  const queue = [nodeId];
  while (queue.length > 0) {{
    const current = queue.shift();
    if (visited.has(current)) continue;
    visited.add(current);
    const parents = parentMap[current] || [];
    for (const parent of parents) {{
      if (posMap[parent] && posMap[current]) {{
        hx.push(posMap[parent][0], posMap[current][0], null);
        hy.push(posMap[parent][1], posMap[current][1], null);
        queue.push(parent);
      }}
    }}
  }}
  return {{ hx, hy }};
}}

graphDiv.on('plotly_hover', function(data) {{
  if (data.points.length > 0) {{
    const pt = data.points[data.points.length - 1];
    if (pt.customdata) {{
      const {{ hx, hy }} = getAncestorEdgeCoords(pt.customdata);
      Plotly.restyle(graphDiv, {{ x: [hx], y: [hy] }}, [1]);
    }}
  }}
}});

graphDiv.on('plotly_unhover', function() {{
  Plotly.restyle(graphDiv, {{ x: [[]], y: [[]] }}, [1]);
}});

graphDiv.on('plotly_click', function(data) {{
  if (data.points.length > 0) {{
    const pt = data.points[data.points.length - 1];
    if (pt.customdata) {{
      showSidebar(pt.customdata);
    }}
  }}
}});

function escapeHtml(s) {{
  if (s === null || s === undefined) return '<em>N/A</em>';
  const div = document.createElement('div');
  div.textContent = String(s);
  return div.innerHTML;
}}

function formatLoss(v) {{
  if (v === null || v === undefined) return '<em>N/A</em>';
  const cls = v < 10 ? 'loss-good' : 'loss-bad';
  return '<span class="' + cls + '">' + Number(v).toFixed(4) + '</span>';
}}

function formatTemp(v) {{
  if (v === null || v === undefined) return '<em>N/A</em>';
  const num = Number(v);
  if (Number.isNaN(num)) return escapeHtml(v);
  return num.toFixed(2);
}}

function formatRemovalDetailValue(value) {{
  if (value === null || value === undefined) return '<em>N/A</em>';
  if (Array.isArray(value) || (typeof value === 'object' && value !== null)) {{
    return '<code>' + escapeHtml(JSON.stringify(value)) + '</code>';
  }}
  if (typeof value === 'number') {{
    return Number.isInteger(value) ? String(value) : Number(value).toFixed(4);
  }}
  return escapeHtml(value);
}}

function formatRemovalDetails(details) {{
  if (!details || typeof details !== 'object' || Object.keys(details).length === 0) return '';
  let html = '<div class="field" style="margin-top:6px;">';
  html += '<span class="field-label" style="color:#856404;">Details:</span>';
  html += '<div class="field-value" style="margin-top:4px;">';
  Object.entries(details).forEach(function([key, value]) {{
    html += '<div><strong>' + escapeHtml(key) + ':</strong> ' + formatRemovalDetailValue(value) + '</div>';
  }});
  html += '</div></div>';
  return html;
}}

function labelFor(id) {{
  if (!id) return '<em>N/A</em>';
  const rec = sidebarData[id];
  if (rec && rec.display_label) return escapeHtml(rec.display_label);
  return escapeHtml(id);
}}

function showSidebar(nodeId) {{
  const d = sidebarData[nodeId];
  if (!d) return;
  const sb = document.getElementById('sidebar');
  const sc = document.getElementById('sidebar-content');

  const displayLabel = d.display_label || nodeId;
  let h = '<h2>' + escapeHtml(d.is_seed ? ('Seed ' + displayLabel) : ('Program ' + displayLabel)) + '</h2>';

  // Basic info
  h += '<div class="field"><span class="field-label">ID:</span> <span class="field-value">' + escapeHtml(d.program_id) + '</span></div>';
  h += '<div class="field"><span class="field-label">Generation:</span> <span class="field-value">' + escapeHtml(d.iteration) + '</span></div>';
  h += '<div class="field"><span class="field-label">Island:</span> <span class="field-value">' + escapeHtml(d.island_label || d.island) + '</span></div>';
  if (d.is_migrant) {{
    h += '<div class="field"><span class="field-label">Migrant From:</span> <span class="field-value">' + escapeHtml(d.migrant_from_label || d.migrant_from_id) + '</span></div>';
  }}
  if (d.is_extinct) {{
    h += '<div class="field"><span class="field-label">Status:</span> <span class="field-value">Extinct</span></div>';
  }}
  h += '<div class="field"><span class="field-label">Initial Loss:</span> <span class="field-value">' + formatLoss(d.initial_loss) + '</span></div>';
  h += '<div class="field"><span class="field-label">Train Loss:</span> <span class="field-value">' + formatLoss(d.train_loss) + '</span></div>';
  h += '<div class="field"><span class="field-label">Test Loss:</span> <span class="field-value">' + formatLoss(d.test_loss) + '</span></div>';
  h += '<div class="field"><span class="field-label">Complexity Penalty:</span> <span class="field-value">' + escapeHtml(d.complexity_penalty !== null && d.complexity_penalty !== undefined ? Number(d.complexity_penalty).toFixed(4) : null) + '</span></div>';
  h += '<div class="field"><span class="field-label">N Params:</span> <span class="field-value">' + escapeHtml(d.n_params) + '</span></div>';
  h += '<div class="field"><span class="field-label">Mode:</span> <span class="field-value">' + escapeHtml(d.mode) + '</span></div>';
  h += '<div class="field"><span class="field-label">Temperature:</span> <span class="field-value">' + formatTemp(d.temperature) + '</span></div>';
  h += '<div class="field"><span class="field-label">LLM:</span> <span class="field-value">' + escapeHtml(d.llm_name) + '</span></div>';
  h += '<div class="field"><span class="field-label">Parents:</span> <span class="field-value">' + labelFor(d.parent1_id) + ', ' + labelFor(d.parent2_id) + '</span></div>';

  if (d.removal_reason) {{
    h += '<div class="field" style="background:#fff3cd;padding:6px;border-radius:4px;margin-bottom:8px;">';
    h += '<span class="field-label" style="color:#856404;">Removed:</span> ';
    h += '<span class="field-value">' + escapeHtml(d.removal_reason.event_type) + ' (' + escapeHtml(d.removal_reason.rule) + ')</span>';
    h += formatRemovalDetails(d.removal_reason.details);
    h += '</div>';
  }}

  // Model code
  if (d.model_code) {{
    h += '<details><summary>Model Code</summary><pre>' + escapeHtml(d.model_code) + '</pre></details>';
  }}
  if (d.model_code_jax) {{
    h += '<details><summary>Model Code (JAX)</summary><pre>' + escapeHtml(d.model_code_jax) + '</pre></details>';
  }}

  // Param estimator code
  if (d.param_est_code) {{
    h += '<details><summary>Parameter Estimator Code</summary><pre>' + escapeHtml(d.param_est_code) + '</pre></details>';
  }}

  // Model prompt (collapsible)
  if (d.model_prompt) {{
    h += '<details><summary>Model Prompt</summary><pre>' + escapeHtml(d.model_prompt) + '</pre></details>';
  }}

  // LLM response (collapsible)
  if (d.model_llm_response) {{
    h += '<details><summary>LLM Response</summary><pre>' + escapeHtml(d.model_llm_response) + '</pre></details>';
  }}

  // Param est prompt (collapsible)
  if (d.param_est_prompt) {{
    h += '<details><summary>Param Estimator Prompt</summary><pre>' + escapeHtml(d.param_est_prompt) + '</pre></details>';
  }}

  // Param est LLM response (collapsible)
  if (d.param_est_llm_response) {{
    h += '<details><summary>Param Estimator LLM Response</summary><pre>' + escapeHtml(d.param_est_llm_response) + '</pre></details>';
  }}

  // Images
  if (d.image_prompt_path) {{
    h += '<details><summary>Prompt Image (Parents)</summary><img src="' + d.image_prompt_path + '" style="max-width:100%;margin-top:8px;"></details>';
  }}
  if (d.train_fit_image_path) {{
    h += '<details><summary>Train Fit</summary><img src="' + d.train_fit_image_path + '" style="max-width:100%;margin-top:8px;"></details>';
  }}
  if (d.test_fit_image_path) {{
    h += '<details><summary>Test Fit</summary><img src="' + d.test_fit_image_path + '" style="max-width:100%;margin-top:8px;"></details>';
  }}

  sc.innerHTML = h;
  sb.classList.add('open');
  resizeGraph();
}}

function resizeGraph() {{
  setTimeout(function() {{ Plotly.Plots.resize(graphDiv); }}, 350);
}}

function closeSidebar() {{
  document.getElementById('sidebar').classList.remove('open');
  resizeGraph();
}}
</script>
</body>
</html>"""
    return html_content


def create_family_tree(generation_log_path, output_dir, n_islands):
    """Create interactive family tree HTML files from the generation log.

    Generates one HTML file per island plus one combined view.

    Args:
        generation_log_path: Path to program_generation_log.jsonl
        output_dir: Directory to write HTML files into
        n_islands: Number of islands in the experiment
    """
    if not os.path.isfile(generation_log_path):
        print(f"[family_tree] No generation log found at {generation_log_path}, skipping.")
        return

    records = _merge_duplicate_records(load_generation_log(generation_log_path))
    if not records:
        print("[family_tree] Generation log is empty, skipping.")
        return

    has_seed_records = any(
        rec.get("is_seed") or rec.get("iteration_number") == -1 for rec in records
    )
    if has_seed_records:
        all_records = records
    else:
        seed_nodes = _build_seed_nodes()
        all_records = seed_nodes + records

    # Build lookup by node ID
    records_by_id = {}
    for rec in all_records:
        nid = make_program_id(*record_key(rec))
        records_by_id[nid] = rec

    # Assign shared compact labels so the tree and progress monitor stay aligned.
    assign_display_labels(all_records)

    image_base_dir = os.path.dirname(generation_log_path)

    # Combined view (all islands)
    G_all = nx.DiGraph()
    for nid in records_by_id:
        G_all.add_node(nid)
    for nid, rec in records_by_id.items():
        for pid_key in ("parent1_id", "parent2_id"):
            pid = _parse_parent_id(rec.get(pid_key))
            if pid and pid in records_by_id:
                G_all.add_edge(pid, nid)

    pos_all = _compute_hierarchical_layout(G_all, records_by_id)
    html_all = _generate_html(G_all, pos_all, records_by_id,
                              "Family Tree",
                              image_base_dir=image_base_dir,
                              html_output_dir=output_dir)
    out_path_all = os.path.join(output_dir, "family_tree.html")
    with open(out_path_all, "w") as f:
        f.write(html_all)
    print(f"[family_tree] Wrote {out_path_all}")


def _infer_n_islands(records):
    """Infer the number of islands from the generation log records."""
    islands = {r["birth_island"] for r in records if r.get("birth_island", -1) >= 0}
    return max(islands) + 1 if islands else 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate interactive family tree HTML files from an EDGAR generation log."
    )
    parser.add_argument(
        "--jsonl", required=True,
        help="Path to program_generation_log.jsonl (or directory containing it)",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory to write HTML files into (defaults to same directory as the JSONL file)",
    )
    parser.add_argument(
        "--n_islands", type=int, default=None,
        help="Number of islands (auto-detected from log if not provided)",
    )
    args = parser.parse_args()

    # Resolve JSONL path: accept either the file or its parent directory
    jsonl_path = args.jsonl
    if os.path.isdir(jsonl_path):
        jsonl_path = os.path.join(jsonl_path, "program_generation_log.jsonl")

    if not os.path.isfile(jsonl_path):
        print(f"Error: JSONL file not found at {jsonl_path}")
        raise SystemExit(1)

    output_dir = args.output_dir or os.path.dirname(jsonl_path)

    # Infer n_islands if not provided
    n_islands = args.n_islands
    if n_islands is None:
        records = load_generation_log(jsonl_path)
        n_islands = _infer_n_islands(records)
        print(f"[family_tree] Auto-detected {n_islands} islands from log.")

    create_family_tree(jsonl_path, output_dir, n_islands)
