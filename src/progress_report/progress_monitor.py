"""Dynamic progress monitoring for in-progress EDGAR runs.

Generates standalone HTML files from a partially or fully written
JSONL generation log, visualising training loss and gradient-descent
effectiveness per island.
"""

import html as html_module
import json
import logging
import os

from .io import load_generation_log

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


def _escape(text) -> str:
    """HTML-escape a value, returning N/A for None."""
    if text is None:
        return "<em>N/A</em>"
    return html_module.escape(str(text))


def _build_sidebar_data(records: list[dict]) -> dict:
    """Build a JSON-serializable dict of node data keyed by record index.

    Args:
        records: List of generation log records (seeds already excluded).

    Returns:
        Dict mapping string index to sidebar-ready record dict.
    """
    sidebar = {}
    for idx, rec in enumerate(records):
        sidebar[str(idx)] = {
            "idx": idx,
            "iteration": rec.get("iteration_number"),
            "island": rec.get("birth_island"),
            "batch": rec.get("batch_index"),
            "train_loss": rec.get("train_loss"),
            "initial_loss": rec.get("initial_loss"),
            "n_params": rec.get("n_params"),
            "complexity_penalty": rec.get("complexity_penalty"),
            "mode": rec.get("mode"),
            "llm_name": rec.get("llm_name"),
            "temperature": rec.get("temperature"),
            "model_code": rec.get("model_code_numpy"),
            "param_est_code": rec.get("param_est_code"),
            "model_prompt": rec.get("model_prompt"),
            "model_llm_response": rec.get("model_llm_response"),
            "param_est_prompt": rec.get("param_est_prompt"),
            "param_est_llm_response": rec.get("param_est_llm_response"),
        }
    return sidebar


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

  sc.innerHTML = h;
  sb.classList.add('open');
  setTimeout(function() { Plotly.Plots.resize(document.getElementById('graph')); }, 350);
}

function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  setTimeout(function() { Plotly.Plots.resize(document.getElementById('graph')); }, 350);
}
"""


def _generate_loss_progress_html(
    records: list[dict],
    sidebar_data: dict,
    islands: list[int],
    title: str,
) -> str:
    """Generate the training loss progress HTML.

    Args:
        records: Non-seed generation records.
        sidebar_data: Pre-built sidebar data dict.
        islands: Sorted list of unique island indices.
        title: HTML page title.

    Returns:
        Standalone HTML string.
    """
    # Build one Plotly trace per island
    traces = []
    for island_idx in islands:
        island_recs = [
            (rec_idx, rec)
            for rec_idx, rec in enumerate(records)
            if rec.get("birth_island") == island_idx
        ]
        if not island_recs:
            continue

        x_vals = []
        y_train_loss = []      # -(train_loss)  — with penalty
        y_raw_loss = []        # -(train_loss - complexity_penalty)  — without penalty
        custom = []
        hover = []

        for rec_idx, rec in island_recs:
            iteration = rec.get("iteration_number", 0)
            jitter = island_idx * 0.15
            x_vals.append(iteration + jitter)

            train_loss = rec.get("train_loss")
            complexity_penalty = rec.get("complexity_penalty", 0.0) or 0.0

            y_train_loss.append(-train_loss if train_loss is not None else None)
            raw = (train_loss - complexity_penalty) if train_loss is not None else None
            y_raw_loss.append(-raw if raw is not None else None)

            custom.append(rec_idx)
            hover_parts = [
                f"<b>i{iteration}_isl{island_idx}_b{rec.get('batch_index', '?')}</b>",
                f"train loss: {train_loss:.4f}" if train_loss is not None else "train loss: N/A",
                f"penalty: {complexity_penalty:.4f}",
                f"mode: {rec.get('mode', '')}",
            ]
            hover.append("<br>".join(hover_parts))

        colour = _island_colour(island_idx)
        traces.append({
            "island_idx": island_idx,
            "x": x_vals,
            "y_with_penalty": y_train_loss,
            "y_without_penalty": y_raw_loss,
            "custom": custom,
            "hover": hover,
            "colour": colour,
        })

    traces_json = json.dumps(traces)
    sidebar_json = json.dumps(sidebar_data, default=str)
    islands_json = json.dumps(islands)
    title_esc = _escape(title)

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

// showWithPenalty tracks current toggle state
let showWithPenalty = true;

// Build Plotly traces
function buildPlotlyTraces(withPenalty) {{
  return allTraces.map(function(t) {{
    return {{
      x: t.x,
      y: withPenalty ? t.y_with_penalty : t.y_without_penalty,
      customdata: t.custom,
      hovertext: t.hover,
      mode: 'markers',
      hoverinfo: 'text',
      name: 'Island ' + t.island_idx,
      marker: {{
        color: t.colour,
        size: 10,
        line: {{ width: 1, color: '#333' }}
      }},
      type: 'scatter'
    }};
  }});
}}

const layout = {{
  title: {json.dumps(title)},
  showlegend: true,
  hovermode: 'closest',
  xaxis: {{ title: 'Iteration (jittered by island)', showgrid: true }},
  yaxis: {{ title: '-Train Loss (higher = better)', showgrid: true }},
  margin: {{ l: 60, r: 20, t: 60, b: 50 }},
  plot_bgcolor: '#fff',
  paper_bgcolor: '#fff'
}};

const graphDiv = document.getElementById('graph');
Plotly.newPlot(graphDiv, buildPlotlyTraces(true), layout, {{responsive: true}});

// Click handler
graphDiv.on('plotly_click', function(data) {{
  if (data.points.length > 0) {{
    const pt = data.points[data.points.length - 1];
    if (pt.customdata !== undefined && pt.customdata !== null) {{
      showSidebar(pt.customdata);
    }}
  }}
}});

// Penalty toggle
function onPenaltyToggle() {{
  showWithPenalty = document.getElementById('penalty-toggle').checked;
  const newY = allTraces.map(function(t) {{
    return showWithPenalty ? t.y_with_penalty : t.y_without_penalty;
  }});
  Plotly.restyle(graphDiv, {{ y: newY }});
  graphDiv.layout.yaxis.title.text = showWithPenalty
    ? '-Train Loss (higher = better)'
    : '-Raw Loss without penalty (higher = better)';
  Plotly.relayout(graphDiv, {{'yaxis.title': graphDiv.layout.yaxis.title.text}});
}}

// Island checkboxes
const cbContainer = document.getElementById('island-checkboxes');
islands.forEach(function(islandIdx) {{
  const label = document.createElement('label');
  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.checked = true;
  cb.onchange = function() {{
    // Find trace index for this island
    const traceIdx = allTraces.findIndex(function(t) {{ return t.island_idx === islandIdx; }});
    if (traceIdx >= 0) {{
      Plotly.restyle(graphDiv, {{ visible: cb.checked ? true : 'legendonly' }}, [traceIdx]);
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
    records: list[dict],
    sidebar_data: dict,
    islands: list[int],
    title: str,
) -> str:
    """Generate the gradient-descent effect HTML.

    Args:
        records: Non-seed generation records.
        sidebar_data: Pre-built sidebar data dict.
        islands: Sorted list of unique island indices.
        title: HTML page title.

    Returns:
        Standalone HTML string.
    """
    traces = []
    all_losses = []

    for island_idx in islands:
        island_recs = [
            (rec_idx, rec)
            for rec_idx, rec in enumerate(records)
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
            initial_loss = rec["initial_loss"]
            train_loss = rec["train_loss"]
            delta = initial_loss - train_loss

            x_vals.append(initial_loss)
            y_vals.append(train_loss)
            custom.append(rec_idx)
            hover.append(
                f"<b>i{rec.get('iteration_number')}_isl{island_idx}_b{rec.get('batch_index','?')}</b>"
                f"<br>initial: {initial_loss:.4f}"
                f"<br>after GD: {train_loss:.4f}"
                f"<br>improvement: {delta:.4f}"
                f"<br>mode: {rec.get('mode', '')}"
            )
            all_losses.extend([initial_loss, train_loss])

        colour = _island_colour(island_idx)
        traces.append({
            "island_idx": island_idx,
            "x": x_vals,
            "y": y_vals,
            "custom": custom,
            "hover": hover,
            "colour": colour,
        })

    # Diagonal reference line range
    if all_losses:
        diag_min = min(all_losses)
        diag_max = max(all_losses)
    else:
        diag_min, diag_max = 0.0, 1.0

    traces_json = json.dumps(traces)
    sidebar_json = json.dumps(sidebar_data, default=str)
    diag_json = json.dumps([diag_min, diag_max])
    title_esc = _escape(title)

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
  <div id="graph" style="width:100%;height:100vh;"></div>
</div>
<div id="sidebar">
  <button id="close-btn" onclick="closeSidebar()">&times;</button>
  <div id="sidebar-content"></div>
</div>

<script>
const sidebarData = {sidebar_json};
const allTraces = {traces_json};
const diagRange = {diag_json};

// Diagonal y=x reference line
const diagTrace = {{
  x: diagRange,
  y: diagRange,
  mode: 'lines',
  line: {{ color: '#999', width: 1.5, dash: 'dash' }},
  hoverinfo: 'none',
  name: 'y = x (no improvement)',
  showlegend: true,
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
      color: t.colour,
      size: 10,
      line: {{ width: 1, color: '#333' }}
    }},
    type: 'scatter'
  }};
}}));

const layout = {{
  title: {json.dumps(title)},
  showlegend: true,
  hovermode: 'closest',
  xaxis: {{ title: 'Initial Loss (param estimator only)', showgrid: true }},
  yaxis: {{ title: 'Train Loss (after GD)', showgrid: true }},
  margin: {{ l: 60, r: 20, t: 60, b: 50 }},
  plot_bgcolor: '#fff',
  paper_bgcolor: '#fff'
}};

const graphDiv = document.getElementById('graph');
Plotly.newPlot(graphDiv, plotlyTraces, layout, {{responsive: true}});

// Click handler (trace index 0 is diagonal — skip it)
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
      - progress_loss.html  — training loss per iteration coloured by island
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

    # Exclude seed records (iteration_number == -1)
    prog_records = [r for r in records if r.get("iteration_number", -1) != -1]
    if not prog_records:
        logging.warning(
            "[progress_monitor] No non-seed records found yet, skipping HTML generation."
        )
        return

    islands = sorted({r.get("birth_island", 0) for r in prog_records})
    sidebar_data = _build_sidebar_data(prog_records)

    os.makedirs(output_dir, exist_ok=True)

    # HTML 1: training loss progress
    loss_html = _generate_loss_progress_html(
        prog_records,
        sidebar_data,
        islands,
        title="EDGAR — Training Loss Progress",
    )
    loss_path = os.path.join(output_dir, "progress_loss.html")
    with open(loss_path, "w") as f:
        f.write(loss_html)
    logging.info("[progress_monitor] Wrote %s", loss_path)

    # HTML 2: gradient descent effect
    gd_html = _generate_gd_effect_html(
        prog_records,
        sidebar_data,
        islands,
        title="EDGAR — Gradient Descent Effect",
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
