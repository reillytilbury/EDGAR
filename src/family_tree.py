"""Family tree visualization for EDGAR program evolution.

Generates interactive HTML files showing parent-child relationships
between programs across iterations, with click-to-inspect details.
"""

import json
import os
import base64
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
            "is_seed": True,
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
            "iteration": rec.get("iteration_number"),
            "island": rec.get("birth_island"),
            "batch": rec.get("batch_index"),
            "train_loss": rec.get("train_loss"),
            "initial_loss": rec.get("initial_loss"),
            "mode": rec.get("mode"),
            "temperature": rec.get("temperature"),
            "llm_name": rec.get("llm_name"),
            "parent1_id": _parse_parent_id(rec.get("parent1_id")),
            "parent2_id": _parse_parent_id(rec.get("parent2_id")),
            "model_code": rec.get("model_code_numpy"),
            "param_est_code": rec.get("param_est_code"),
            "model_prompt": rec.get("model_prompt"),
            "model_llm_response": rec.get("model_llm_response"),
            "param_est_prompt": rec.get("param_est_prompt", rec.get("param_est_prompt")),
            "param_est_llm_response": rec.get("param_est_llm_response"),
            "image_path": rec.get("image_prompt_path"),
            "is_seed": rec.get("is_seed", False),
        }
        sidebar[node_id] = entry
    return sidebar


def _try_embed_image(image_path):
    """Return base64-encoded img tag if file exists, else a text note."""
    if not image_path or not os.path.isfile(image_path):
        return ""
    try:
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower().lstrip(".")
        if ext == "jpg":
            ext = "jpeg"
        return f'<img src="data:image/{ext};base64,{data}" style="max-width:100%;margin-top:8px;">'
    except Exception:
        return f"<p><em>Image: {_escape(image_path)}</em></p>"


def _generate_html(G, pos, records_by_id, title, image_base_dir=None):
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
        iteration = rec.get("iteration_number", "?")
        batch = rec.get("batch_index", "?")
        loss = rec.get("train_loss")
        mode = rec.get("mode", "")
        llm = rec.get("llm_name", "")

        if is_seed:
            label = f"seed_{batch}"
        else:
            label = f"i{iteration}_b{batch}"
        node_labels.append(label)

        hover_parts = [f"<b>{label}</b>"]
        if loss is not None:
            hover_parts.append(f"loss: {loss:.4f}")
        hover_parts.append(f"mode: {mode}")
        if llm:
            hover_parts.append(f"LLM: {llm}")
        node_hover.append("<br>".join(hover_parts))

        node_colors.append(_loss_to_color(loss, min_loss, max_loss))

    # Build sidebar data as JSON for embedding
    sidebar_data = _build_sidebar_data(records_by_id)

    # Try to embed images into sidebar data
    for nid, entry in sidebar_data.items():
        img_path = entry.get("image_path")
        if img_path and image_base_dir and not os.path.isabs(img_path):
            img_path = os.path.join(image_base_dir, img_path)
        entry["image_html"] = _try_embed_image(img_path)

    sidebar_json = json.dumps(sidebar_data, default=str)
    edge_x_json = json.dumps(edge_x)
    edge_y_json = json.dumps(edge_y)
    node_x_json = json.dumps(node_x)
    node_y_json = json.dumps(node_y)
    node_ids_json = json.dumps(node_ids)
    node_labels_json = json.dumps(node_labels)
    node_hover_json = json.dumps(node_hover)
    node_colors_json = json.dumps(node_colors)
    title_json = json.dumps(title)

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
  textposition: 'top center',
  textfont: {{ size: 10 }},
  hoverinfo: 'text',
  marker: {{
    color: {node_colors_json},
    size: 14,
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
  paper_bgcolor: '#fff'
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

function showSidebar(nodeId) {{
  const d = sidebarData[nodeId];
  if (!d) return;
  const sb = document.getElementById('sidebar');
  const sc = document.getElementById('sidebar-content');

  let h = '<h2>' + escapeHtml(d.is_seed ? 'Seed ' + d.batch : 'Program ' + nodeId) + '</h2>';

  // Basic info
  h += '<div class="field"><span class="field-label">Iteration:</span> <span class="field-value">' + escapeHtml(d.iteration) + '</span></div>';
  h += '<div class="field"><span class="field-label">Island:</span> <span class="field-value">' + escapeHtml(d.island) + '</span></div>';
  h += '<div class="field"><span class="field-label">Batch:</span> <span class="field-value">' + escapeHtml(d.batch) + '</span></div>';
  h += '<div class="field"><span class="field-label">Train Loss:</span> <span class="field-value">' + formatLoss(d.train_loss) + '</span></div>';
  h += '<div class="field"><span class="field-label">Initial Loss:</span> <span class="field-value">' + formatLoss(d.initial_loss) + '</span></div>';
  h += '<div class="field"><span class="field-label">Mode:</span> <span class="field-value">' + escapeHtml(d.mode) + '</span></div>';
  h += '<div class="field"><span class="field-label">Temperature:</span> <span class="field-value">' + escapeHtml(d.temperature) + '</span></div>';
  h += '<div class="field"><span class="field-label">LLM:</span> <span class="field-value">' + escapeHtml(d.llm_name) + '</span></div>';
  h += '<div class="field"><span class="field-label">Parents:</span> <span class="field-value">' + escapeHtml(d.parent1_id) + ', ' + escapeHtml(d.parent2_id) + '</span></div>';

  // Model code
  if (d.model_code) {{
    h += '<details open><summary>Model Code</summary><pre>' + escapeHtml(d.model_code) + '</pre></details>';
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

  // Diagnostic image
  if (d.image_html) {{
    h += '<details open><summary>Diagnostic Image</summary>' + d.image_html + '</details>';
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

    seed_ids = {_make_node_id(-1, -1, i) for i in range(2)}
    image_base_dir = os.path.dirname(generation_log_path)

    # Per-island trees
    for island_idx in range(n_islands):
        # Collect nodes born on this island
        island_nodes = set()
        for nid, rec in records_by_id.items():
            if rec.get("birth_island") == island_idx:
                island_nodes.add(nid)

        # Also include parents of those nodes (seeds, migrants from other islands)
        parent_nodes = set()
        for nid in island_nodes:
            rec = records_by_id[nid]
            for pid_key in ("parent1_id", "parent2_id"):
                pid = _parse_parent_id(rec.get(pid_key))
                if pid and pid in records_by_id:
                    parent_nodes.add(pid)
        island_nodes |= parent_nodes

        if not island_nodes:
            continue

        # Build graph
        G = nx.DiGraph()
        island_records = {nid: records_by_id[nid] for nid in island_nodes}
        for nid in island_nodes:
            G.add_node(nid)
        for nid in island_nodes:
            rec = records_by_id[nid]
            for pid_key in ("parent1_id", "parent2_id"):
                pid = _parse_parent_id(rec.get(pid_key))
                if pid and pid in island_nodes:
                    G.add_edge(pid, nid)

        pos = _compute_hierarchical_layout(G, island_records)
        html = _generate_html(G, pos, island_records,
                              f"Family Tree — Island {island_idx}",
                              image_base_dir=image_base_dir)
        out_path = os.path.join(output_dir, f"family_tree_island_{island_idx}.html")
        with open(out_path, "w") as f:
            f.write(html)
        print(f"[family_tree] Wrote {out_path}")

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
                              "Family Tree — All Islands",
                              image_base_dir=image_base_dir)
    out_path_all = os.path.join(output_dir, "family_tree_all.html")
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
