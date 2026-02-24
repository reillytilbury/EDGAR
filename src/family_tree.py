"""Family tree visualization for EDGAR program evolution.

Generates interactive HTML files showing parent-child relationships
between programs across iterations, with click-to-inspect details.
"""

import json
import os
import html as html_module
from collections import defaultdict

import networkx as nx


def _load_generation_log(log_path):
    """Load JSONL generation log into a list of dicts."""
    records = []
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _make_node_id(iteration, island, batch):
    """Create a unique string node ID from (iteration, island, batch)."""
    return f"{iteration}_{island}_{batch}"


def _parse_parent_id(parent_id):
    """Parse a parent ID tuple [iteration, island, batch] into a node ID string.

    Returns None if parent_id is None or not a valid 3-element list.
    """
    if parent_id is None:
        return None
    if isinstance(parent_id, (list, tuple)) and len(parent_id) == 3:
        return _make_node_id(int(parent_id[0]), int(parent_id[1]), int(parent_id[2]))
    return None


def _build_seed_nodes():
    """Create two synthetic seed nodes for iteration=-1."""
    seeds = []
    for batch_idx in range(2):
        seeds.append({
            "iteration_number": -1,
            "birth_island": -1,
            "batch_index": batch_idx,
            "train_loss": None,
            "initial_loss": None,
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
            "is_seed": True,
            "is_winner": False,
        })
    return seeds


def _compute_hierarchical_layout(G, records_by_id):
    """Compute a hierarchical layout with iteration as Y-level."""
    pos = {}
    # Group nodes by iteration
    levels = defaultdict(list)
    for node in G.nodes():
        rec = records_by_id.get(node)
        if rec:
            levels[rec["iteration_number"]].append(node)
        else:
            levels[-1].append(node)

    for level, nodes in levels.items():
        nodes.sort()  # deterministic ordering
        n = len(nodes)
        for idx, node in enumerate(nodes):
            x = (idx - (n - 1) / 2) if n > 1 else 0
            # Negate iteration so seeds (iter=-1) are at top (y=1 -> highest)
            y = -level
            pos[node] = (x, y)
    return pos


def _island_letter(island_idx: int) -> str:
    """Map island index to a letter label (0->A, 1->B, ..., 25->Z, 26->AA, ...)."""
    if island_idx < 0:
        return "S"
    letters = []
    idx = island_idx
    while True:
        idx, rem = divmod(idx, 26)
        letters.append(chr(ord('A') + rem))
        if idx == 0:
            break
        idx -= 1
    return "".join(reversed(letters))


def _assign_node_labels(records):
    """Assign compact labels like A12 per island and S1/S2 for seeds."""
    label_map = {}
    # Seed labels
    for rec in records:
        if rec.get("is_seed", False):
            nid = _make_node_id(rec["iteration_number"], rec["birth_island"], rec["batch_index"])
            label_map[nid] = f"S{int(rec.get('batch_index', 0)) + 1}"

    # Per-island labels (birth order)
    islands = sorted({rec.get("birth_island") for rec in records if rec.get("birth_island", -1) >= 0})
    for island_idx in islands:
        island_records = [
            rec for rec in records
            if rec.get("birth_island") == island_idx and rec.get("iteration_number", -1) >= 0
        ]
        island_records.sort(key=lambda r: (r.get("iteration_number", 0), r.get("batch_index", 0)))
        for i, rec in enumerate(island_records, start=1):
            nid = _make_node_id(rec["iteration_number"], rec["birth_island"], rec["batch_index"])
            label_map[nid] = f"{_island_letter(island_idx)}{i}"
    return label_map


def _compute_island_layout(
    node_ids,
    records_by_id,
    island_centers,
    intra_spacing=0.7,
    seed_center_x=0.0,
    seed_offset=0.5,
):
    """Compute layout with islands separated into vertical columns."""
    pos = {}
    groups = defaultdict(list)
    for nid in node_ids:
        rec = records_by_id.get(nid, {})
        if rec.get("is_seed", False):
            continue
        island = rec.get("birth_island", -1)
        iteration = rec.get("iteration_number", -1)
        groups[(island, iteration)].append(nid)

    for (island, iteration), nodes in groups.items():
        nodes.sort()
        n = len(nodes)
        if n == 1:
            offsets = [0.0]
        else:
            offsets = [(i - (n - 1) / 2) * intra_spacing for i in range(n)]
        center_x = island_centers.get(island, 0.0)
        y = -iteration
        for nid, off in zip(nodes, offsets):
            pos[nid] = (center_x + off, y)

    # Place seeds centered above all islands
    seed_nodes = [nid for nid in node_ids if records_by_id.get(nid, {}).get("is_seed", False)]
    seed_nodes.sort()
    for idx, nid in enumerate(seed_nodes):
        rec = records_by_id.get(nid, {})
        iteration = rec.get("iteration_number", -1)
        y = -iteration
        x = seed_center_x + (idx - (len(seed_nodes) - 1) / 2) * seed_offset
        pos[nid] = (x, y)
    return pos


def _loss_to_color(loss, min_loss, max_loss):
    """Map a loss value to an RGB color string (green=good, red=bad)."""
    if loss is None or min_loss is None or max_loss is None:
        return "rgb(180,180,180)"  # gray for seeds/unknown
    if max_loss == min_loss:
        return "rgb(50,180,50)"
    # Clamp and normalize
    t = min(1.0, max(0.0, (loss - min_loss) / (max_loss - min_loss)))
    # Interpolate green (good) to red (bad)
    r = int(50 + 205 * t)
    g = int(180 - 130 * t)
    b = int(50)
    return f"rgb({r},{g},{b})"


def _escape(text):
    """HTML-escape a string, handling None."""
    if text is None:
        return "<em>N/A</em>"
    return html_module.escape(str(text))


def _build_sidebar_data(records_by_id):
    """Build a JSON-serializable dict of node data for the sidebar."""
    sidebar = {}
    for node_id, rec in records_by_id.items():
        entry = {
            "id": node_id,
            "display_label": rec.get("display_label"),
            "iteration": rec.get("iteration_number"),
            "island": rec.get("birth_island"),
            "island_label": _island_letter(rec.get("birth_island", -1)),
            "batch": rec.get("batch_index"),
            "train_loss": rec.get("train_loss"),
            "test_loss": rec.get("test_loss"),
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
            "param_est_code": rec.get("param_est_code"),
            "model_prompt": rec.get("model_prompt"),
            "model_llm_response": rec.get("model_llm_response"),
            "param_est_prompt": rec.get("param_est_prompt", rec.get("param_est_prompt")),
            "param_est_llm_response": rec.get("param_est_llm_response"),
            "image_prompt_path": rec.get("image_prompt_path"),
            "train_fit_image_path": rec.get("train_fit_image_path"),
            "test_fit_image_path": rec.get("test_fit_image_path"),
            "is_seed": rec.get("is_seed", False),
        }
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

    # Compute loss range for coloring
    losses = [r.get("train_loss") for r in records_by_id.values()
              if r.get("train_loss") is not None]
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
        is_migrant = rec.get("is_migrant", False)
        is_extinct = rec.get("is_extinct", False)
        iteration = rec.get("iteration_number", "?")
        batch = rec.get("batch_index", "?")
        loss = rec.get("train_loss")
        test_loss = rec.get("test_loss")
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
        if is_migrant and rec.get("migrant_from_id"):
            hover_parts.append(f"migrant from: {rec.get('migrant_from_label', rec.get('migrant_from_id'))}")
        if loss is not None:
            hover_parts.append(f"train loss: {loss:.4f}")
        if test_loss is not None:
            hover_parts.append(f"test loss: {test_loss:.4f}")
        hover_parts.append(f"mode: {mode}")
        if llm:
            hover_parts.append(f"LLM: {llm}")
        if is_extinct:
            hover_parts.append("status: extinct")
        node_hover.append("<br>".join(hover_parts))

        if is_seed:
            node_colors.append("rgb(180,180,180)")
        elif is_extinct:
            node_colors.append("rgb(200,30,30)")
        elif is_migrant:
            node_colors.append("rgb(128,0,200)")
        else:
            node_colors.append(_loss_to_color(loss, min_loss, max_loss))

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
            abs_path = _resolve_image_path(entry.get(key), image_base_dir)
            if abs_path:
                entry[key] = os.path.relpath(abs_path, rel_base)
            else:
                entry[key] = None

    sidebar_json = json.dumps(sidebar_data, default=str)
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

    divider_shapes = []
    if island_dividers:
        for div in island_dividers:
            x = div.get("x") if isinstance(div, dict) else div
            y0 = div.get("y0") if isinstance(div, dict) else None
            y1 = div.get("y1") if isinstance(div, dict) else None
            shape = {
                "type": "line",
                "xref": "x",
                "x0": x,
                "x1": x,
                "line": {"color": "#e0e0e0", "width": 1}
            }
            if y0 is not None and y1 is not None:
                shape["yref"] = "y"
                shape["y0"] = y0
                shape["y1"] = y1
            else:
                shape["yref"] = "paper"
                shape["y0"] = 0
                shape["y1"] = 1
            divider_shapes.append(shape)
    divider_shapes_json = json.dumps(divider_shapes)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{_escape(title)}</title>
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

const layout = {{
  title: {title_json},
  showlegend: false,
  hovermode: 'closest',
  xaxis: {{ showgrid: false, zeroline: false, showticklabels: false }},
  yaxis: {{ showgrid: false, zeroline: false, showticklabels: false }},
  margin: {{ l: 20, r: 20, t: 50, b: 20 }},
  plot_bgcolor: '#fff',
  paper_bgcolor: '#fff',
  shapes: {divider_shapes_json}
}};

const graphDiv = document.getElementById('graph');
Plotly.newPlot(graphDiv, [edgeTrace, nodeTrace], layout, {{responsive: true}});

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
  h += '<div class="field"><span class="field-label">Mode:</span> <span class="field-value">' + escapeHtml(d.mode) + '</span></div>';
  h += '<div class="field"><span class="field-label">Temperature:</span> <span class="field-value">' + formatTemp(d.temperature) + '</span></div>';
  h += '<div class="field"><span class="field-label">LLM:</span> <span class="field-value">' + escapeHtml(d.llm_name) + '</span></div>';
  h += '<div class="field"><span class="field-label">Parents:</span> <span class="field-value">' + labelFor(d.parent1_id) + ', ' + labelFor(d.parent2_id) + '</span></div>';

  // Model code
  if (d.model_code) {{
    h += '<details><summary>Model Code</summary><pre>' + escapeHtml(d.model_code) + '</pre></details>';
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

    records = _load_generation_log(generation_log_path)
    if not records:
        print("[family_tree] Generation log is empty, skipping.")
        return

    # Add seed nodes
    seed_nodes = _build_seed_nodes()
    all_records = seed_nodes + records

    # Build lookup by node ID
    records_by_id = {}
    for rec in all_records:
        nid = _make_node_id(rec["iteration_number"], rec["birth_island"], rec["batch_index"])
        records_by_id[nid] = rec

    # Assign compact labels
    label_map = _assign_node_labels(all_records)
    for nid, rec in records_by_id.items():
        rec["display_label"] = label_map.get(nid)

    # Compute extinction based on whether ever used as parent (ignore last iteration)
    used_as_parent = set()
    for rec in records:
        for pid_key in ("parent1_id", "parent2_id"):
            pid = _parse_parent_id(rec.get(pid_key))
            if pid:
                used_as_parent.add(pid)
    max_iter = max((rec.get("iteration_number", -1) for rec in records), default=-1)
    for nid, rec in records_by_id.items():
        if rec.get("is_seed", False):
            rec["is_extinct"] = False
            continue
        if rec.get("iteration_number", -1) >= max_iter:
            rec["is_extinct"] = False
            continue
        rec["is_extinct"] = nid not in used_as_parent

    seed_ids = {_make_node_id(-1, -1, i) for i in range(2)}
    image_base_dir = os.path.dirname(generation_log_path)

    # Combined view (all islands), with migrant copies to keep parent lines within islands
    combined_records = {nid: dict(rec) for nid, rec in records_by_id.items()}
    G_all = nx.DiGraph()
    for nid in combined_records:
        G_all.add_node(nid)

    migrant_copies = {}
    for nid, rec in records_by_id.items():
        child_island = rec.get("birth_island")
        for pid_key in ("parent1_id", "parent2_id"):
            pid = _parse_parent_id(rec.get(pid_key))
            if not pid or pid not in records_by_id:
                continue
            parent_rec = records_by_id[pid]
            parent_island = parent_rec.get("birth_island")
            if parent_rec.get("is_seed", False):
                G_all.add_edge(pid, nid)
                continue
            if parent_island == child_island:
                G_all.add_edge(pid, nid)
                continue
            key = (pid, child_island)
            if key not in migrant_copies:
                migrant_id = f"{pid}__m{child_island}"
                migrant_rec = dict(parent_rec)
                migrant_rec["birth_island"] = child_island
                migrant_rec["is_migrant"] = True
                migrant_rec["migrant_from_id"] = pid
                migrant_rec["migrant_from_label"] = label_map.get(pid)
                migrant_rec["display_label"] = label_map.get(pid)
                migrant_rec["is_extinct"] = False
                migrant_rec["is_seed"] = False
                combined_records[migrant_id] = migrant_rec
                G_all.add_node(migrant_id)
                migrant_copies[key] = migrant_id
            G_all.add_edge(migrant_copies[key], nid)

    # Compute island centers and dividers
    island_spacing = 6.0
    island_centers = {}
    for island_idx in range(n_islands):
        island_centers[island_idx] = island_idx * island_spacing
    divider_positions = []
    for island_idx in range(0, n_islands - 1):
        left = island_centers.get(island_idx, 0.0)
        right = island_centers.get(island_idx + 1, 0.0)
        divider_positions.append((left + right) / 2)

    seed_center_x = (island_centers[0] + island_centers[n_islands - 1]) / 2 if n_islands > 0 else 0.0
    pos_all = _compute_island_layout(
        G_all.nodes(),
        combined_records,
        island_centers,
        seed_center_x=seed_center_x,
        seed_offset=0.8,
    )

    if pos_all:
        min_y = min(y for _, y in pos_all.values())
        max_y = max(y for _, y in pos_all.values())
    else:
        min_y = -1.0
        max_y = 1.0
    divider_shapes = [{"x": x, "y0": max_y - 0.4, "y1": min_y - 0.2} for x in divider_positions]
    html_all = _generate_html(
        G_all,
        pos_all,
        combined_records,
        "Family Tree — All Islands",
        image_base_dir=image_base_dir,
        html_output_dir=output_dir,
        island_dividers=divider_shapes,
    )
    out_path_all = os.path.join(output_dir, "genealogy.html")
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
        records = _load_generation_log(jsonl_path)
        n_islands = _infer_n_islands(records)
        print(f"[family_tree] Auto-detected {n_islands} islands from log.")

    create_family_tree(jsonl_path, output_dir, n_islands)
