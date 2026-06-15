// EDGAR dashboard front-end. One Alpine component (`dashboard()`) drives both views.

function dashboard() {
  return {
    // ── state ──
    view: 'live',
    runs: [],
    runId: '',
    roots: [],
    summary: {},
    state: {},
    programs: [],
    familyTreeData: null,
    loading: true,
    autoPoll: true,
    pollIntervalMs: 2500,
    autoScrollLog: true,
    _pollTimer: null,

    // sort
    sort: { key: 'rank', dir: 'asc' },

    // panel
    panelOpen: false,
    panelTab: 'code',
    codeTab: 'model',
    programDetail: {},
    latexLoading: false,
    latexError: '',
    latexResult: {},

    // ── lifecycle ──
    async init() {
      await this.bootstrapConfig();
      await this.reloadRuns();
      if (this.runs.length) {
        // pick from URL hash or default to most recent
        const fromHash = this.parseHash();
        const wanted = fromHash.runId || this.runId;
        const exists = this.runs.find(r => r.run_id === wanted);
        this.runId = exists ? wanted : this.runs[0].run_id;
        this.view = fromHash.view || this.view;
      }
      this.loading = false;
      if (this.runId) await this.refreshAll();
      this.startPolling();
      // re-render charts on view change
      this.$watch('view', async (v) => {
        if (v === 'family_tree' && !this.familyTreeData) {
          await this.fetchFamilyTree();
        }
        this.renderCharts();
      });
      this.$watch('state', () => this.renderCharts());
      // sync hash
      this.$watch('runId', v => this.updateHash());
      this.$watch('view', v => this.updateHash());
      // route to inspect when run finishes? leave to user.
    },

    parseHash() {
      const h = window.location.hash || '';
      const m = h.match(/^#\/(live|inspect|family_tree)(?:\?run=([^&]+))?/);
      if (!m) return {};
      return { view: m[1], runId: m[2] ? decodeURIComponent(m[2]) : null };
    },
    updateHash() {
      if (!this.runId) return;
      const newHash = `#/${this.view}?run=${encodeURIComponent(this.runId)}`;
      if (window.location.hash !== newHash) {
        window.history.replaceState(null, '', newHash);
      }
    },

    async bootstrapConfig() {
      try {
        const r = await fetch('/api/config');
        const j = await r.json();
        this.roots = j.roots || [];
        if (j.default_run_id) this.runId = j.default_run_id;
      } catch (e) {
        console.warn('config fetch failed', e);
      }
    },

    async reloadRuns() {
      try {
        const r = await fetch('/api/runs');
        this.runs = await r.json();
      } catch (e) {
        console.warn('runs fetch failed', e);
        this.runs = [];
      }
    },

    async onRunChange() {
      this.closePanel();
      await this.refreshAll();
    },

    async refreshAll() {
      if (!this.runId) return;
      await Promise.all([
        this.fetchSummary(),
        this.fetchState(),
        this.fetchPrograms(),
        this.view === 'family_tree' ? this.fetchFamilyTree() : Promise.resolve(),
      ]);
      this.renderCharts();
    },

    startPolling() {
      if (this._pollTimer) clearInterval(this._pollTimer);
      this._pollTimer = setInterval(async () => {
        if (!this.autoPoll || !this.runId) return;
        // Always poll state (cheap) so the live view updates.
        await this.fetchState();
        // When the run is live, refresh whichever view-specific data we need.
        if (this.state.status === 'running' || this.state.status === 'starting') {
          if (this.view === 'inspect') {
            await this.fetchPrograms();
            await this.fetchSummary();
          }
          if (this.view === 'family_tree') {
            await this.fetchFamilyTree();
          }
          this.renderCharts();
        }
      }, this.pollIntervalMs);
    },

    // ── fetchers ──
    async fetchSummary() {
      const r = await fetch(`/api/runs/${this.runId}/summary`);
      if (r.ok) this.summary = await r.json();
    },
    async fetchState() {
      const r = await fetch(`/api/runs/${this.runId}/state`);
      if (r.ok) this.state = await r.json();
    },
    async fetchPrograms() {
      const r = await fetch(`/api/runs/${this.runId}/programs`);
      if (r.ok) this.programs = await r.json();
    },
    async fetchFamilyTree() {
      const r = await fetch(`/api/runs/${this.runId}/family_tree`);
      if (r.ok) this.familyTreeData = await r.json();
    },

    // ── helpers ──
    fmt(v, p = 3) {
      if (v === null || v === undefined || Number.isNaN(v)) return '-';
      if (typeof v !== 'number') return String(v);
      if (Math.abs(v) >= 1000) return v.toFixed(0);
      return v.toFixed(p);
    },
    fmtDuration(s) {
      if (s === null || s === undefined) return '-';
      s = Math.max(0, Math.round(s));
      const h = Math.floor(s / 3600); s -= h * 3600;
      const m = Math.floor(s / 60); s -= m * 60;
      if (h) return `${h}h ${m}m`;
      if (m) return `${m}m ${s}s`;
      return `${s}s`;
    },
    fmtTokens(n) {
      if (n === null || n === undefined) return '-';
      if (n < 1000) return String(n);
      if (n < 1e6) return `${(n / 1000).toFixed(1)}k`;
      return `${(n / 1e6).toFixed(2)}M`;
    },
    fmtPct(a, b) {
      if (!a || !b) return '0%';
      return `${Math.round((a / b) * 100)}%`;
    },
    scrollLogTail() {
      if (!this.autoScrollLog) return;
      const el = document.getElementById('run-log-tail');
      if (el) el.scrollTop = el.scrollHeight;
    },
    rateStr(r) {
      if (r === null || r === undefined) return '-';
      return `${Math.round(r * 100)}%`;
    },
    rateClass(r) {
      if (r === null || r === undefined) return 'text-zinc-400';
      if (r >= 0.9) return 'text-emerald-400';
      if (r >= 0.5) return 'text-amber-400';
      return 'text-rose-400';
    },
    statusPillClass(s) {
      const base = 'bg-zinc-800 text-zinc-300';
      if (s === 'running' || s === 'starting') return 'bg-emerald-700/40 text-emerald-300';
      if (s === 'complete') return 'bg-zinc-700 text-zinc-200';
      if (s === 'failed') return 'bg-rose-700/50 text-rose-200';
      if (s === 'stalled') return 'bg-amber-700/40 text-amber-200';
      return base;
    },
    statusLabel(state) {
      // Surface the more specific 'stalled' label when a run is dead but no
      // explicit failure was recorded.
      if (this.state?.is_stale) return 'stalled';
      return state || 'unknown';
    },
    fmtStage(s) {
      if (!s) return '\u00a0';
      const labels = {
        translate_programs: 'translate',
        translate_seeds: 'translate',
        score: 'score',
        score_seeds: 'score',
        score_validate: 'score',
      };
      return labels[s] || s;
    },
    lossClass(v) {
      if (v === null || v === undefined) return 'text-zinc-500';
      // green at low loss, red at high. anchor at orientation-tuning scale.
      if (v < 30) return 'text-emerald-400';
      if (v < 50) return 'text-amber-300';
      return 'text-rose-400';
    },
    paramsPretty(params) {
      if (!params) return '';
      try {
        return JSON.stringify(params, (k, v) => {
          if (Array.isArray(v) && v.length > 6) {
            return `[${v.slice(0, 6).map(x => typeof x === 'number' ? x.toFixed(3) : x).join(', ')} … (+${v.length - 6})]`;
          }
          if (typeof v === 'number') return Number(v.toFixed(4));
          return v;
        }, 2);
      } catch {
        return String(params);
      }
    },

    // ── table sort ──
    sortBy(key) {
      if (this.sort.key === key) this.sort.dir = this.sort.dir === 'asc' ? 'desc' : 'asc';
      else { this.sort.key = key; this.sort.dir = 'asc'; }
    },
    get sortedPrograms() {
      const k = this.sort.key, dir = this.sort.dir === 'asc' ? 1 : -1;
      const big = Number.POSITIVE_INFINITY;
      const get = p => {
        const v = p[k];
        if (v === null || v === undefined) return big;
        return v;
      };
      return [...this.programs].sort((a, b) => {
        const va = get(a), vb = get(b);
        if (typeof va === 'string') return va.localeCompare(vb) * dir;
        return (va - vb) * dir;
      });
    },

    // ── charts ──
    renderCharts() {
      // delay so x-show transitions don't yield zero-width container
      requestAnimationFrame(() => {
        this.renderSwimlanes();
        this.renderSpark();
        this.renderStageChart();
        this.renderFamilyTree();
      });
    },

    renderSwimlanes() {
      const el = document.getElementById('swimlanes-chart');
      if (!el || !this.state.islands) return;
      const islands = this.state.islands;
      const traces = [];
      const lossOf = p => p.loss_discover ?? p.loss_validate;
      const colors = islands.map((_, i) => islandColor(i));
      let maxGen = 0;
      islands.forEach((row, ri) => {
        const xs = [], ys = [], texts = [], custom = [], markers = [], sizes = [];
        row.programs.forEach(p => {
          xs.push(p.gen);
          ys.push(ri);
          custom.push(p.idx);
          maxGen = Math.max(maxGen, p.gen);
          const loss = lossOf(p);
          texts.push(
            `<b>${escapeHtml(p.name)}</b><br>#${p.idx} · gen ${p.gen} · island ${p.island}<br>` +
            `mode: ${p.mode || '-'} · llm: ${p.llm || '-'}<br>` +
            `discover: ${fmtN(p.loss_discover)} · validate: ${fmtN(p.loss_validate)}<br>` +
            `parents: ${(p.parents || []).map(x => '#' + x).join(', ') || '-'}` +
            (p.rank ? `<br>rank: ${p.rank}` : '') +
            (p.alive ? '' : '<br>(pruned)')
          );
          markers.push(nodeColor(loss));
          sizes.push(p.rank === 1 ? 20 : (p.alive ? 13 : 9));
        });
        traces.push({
          x: xs, y: ys.map(y => y + jitterFromIdx(custom)), customdata: custom,
          text: texts, hovertemplate: '%{text}<extra></extra>',
          mode: 'markers', type: 'scatter',
          marker: {
            color: markers, size: sizes,
            line: { color: colors[ri], width: 1.2 },
            symbol: 'circle',
          },
          name: `island ${row.idx}`,
        });
      });
      // Give the plot an explicit width that scales with the number of gens so
      // it can overflow horizontally. The outer wrapper div has overflow-x: auto.
      const totalGens = Math.max(maxGen + 1, this.state.n_gens || 1, 1);
      const width = Math.max(800, totalGens * 90);
      el.style.width = `${width}px`;
      const layout = {
        width,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#d4d4d8', family: 'ui-sans-serif' },
        showlegend: false,
        margin: { l: 84, r: 16, t: 8, b: 36 },
        xaxis: {
          title: { text: 'generation', font: { size: 11 } },
          gridcolor: '#27272a', zeroline: false,
          dtick: 1,
          range: [-0.5, totalGens - 0.5],
        },
        yaxis: {
          tickmode: 'array',
          tickvals: islands.map((_, i) => i),
          ticktext: islands.map(r => `island ${r.idx} (${r.size_alive} alive)`),
          gridcolor: '#27272a', zeroline: false,
          autorange: 'reversed',
        },
        hoverlabel: { bgcolor: '#0a0a0a', bordercolor: '#3f3f46', font: { color: '#fafafa' } },
      };
      Plotly.react(el, traces, layout, { displayModeBar: false, responsive: false })
        .then(() => {
          el.removeAllListeners?.('plotly_click');
          el.on('plotly_click', evt => {
            const idx = evt.points?.[0]?.customdata;
            if (idx != null) this.openProgram(idx);
          });
        });
    },

    renderSpark() {
      const el = document.getElementById('spark-chart');
      if (!el || !this.state.best_per_gen) return;
      const xs = this.state.best_per_gen.map(p => p.gen);
      const ys = this.state.best_per_gen.map(p => p.loss);
      Plotly.react(el, [{
        x: xs, y: ys, mode: 'lines+markers', type: 'scatter',
        line: { color: '#34d399', width: 2 },
        marker: { color: '#34d399', size: 6 },
        hovertemplate: 'gen %{x}<br>best loss %{y:.4f}<extra></extra>',
      }], {
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#a1a1aa', size: 10 },
        margin: { l: 36, r: 8, t: 4, b: 20 },
        xaxis: { gridcolor: '#27272a', dtick: 1 },
        yaxis: { gridcolor: '#27272a' },
        showlegend: false,
      }, { displayModeBar: false, responsive: true });
    },

    renderStageChart() {
      const el = document.getElementById('stage-chart');
      if (!el) return;
      const rows = this.state.metrics || [];
      // We collapse seed/validate (gen < 0 / gen >= n_gens) into negative xs
      // for visibility but drop them if there are no real generations yet.
      const stagePalette = {
        spawn: '#a1a1aa',
        generate_models: '#60a5fa',
        generate_param_ests: '#a78bfa',
        translate_programs: '#22d3ee',
        translate_seeds: '#22d3ee',
        score: '#34d399',
        score_seeds: '#34d399',
        score_validate: '#34d399',
        deduplicate: '#fbbf24',
        prune: '#fb7185',
        migrate: '#facc15',
      };
      const stageLabels = {
        translate_programs: 'translate',
        translate_seeds: 'translate',
        score: 'score',
        score_seeds: 'score',
        score_validate: 'score',
      };
      const stageOrder = [
        'spawn', 'generate_models', 'generate_param_ests',
        'translate_programs', 'translate_seeds',
        'score', 'score_seeds', 'score_validate',
        'deduplicate', 'prune', 'migrate',
      ];
      const xs = rows.map(r => r.gen);
      const traces = [];
      const seenLabels = new Set();

      for (const stage of stageOrder) {
        const ys = rows.map(r => (r.stage_times && r.stage_times[stage]) || 0);
        if (ys.every(v => !v)) continue;

        const label = stageLabels[stage] || stage;
        traces.push({
          x: xs, y: ys, type: 'bar', name: label,
          marker: { color: stagePalette[stage] || '#71717a' },
          hovertemplate: `${stage}: %{y:.1f}s<br>gen %{x}<extra></extra>`,
          legendgroup: label,
          showlegend: !seenLabels.has(label),
        });
        seenLabels.add(label);
      }
      Plotly.react(el, traces, {
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#a1a1aa', size: 10 },
        margin: { l: 40, r: 8, t: 4, b: 20 },
        barmode: 'stack',
        xaxis: { gridcolor: '#27272a', dtick: 1, title: { text: '' } },
        yaxis: { gridcolor: '#27272a', title: { text: '' } },
        showlegend: true,
        legend: { font: { size: 9 }, orientation: 'h', y: -0.25 },
      }, { displayModeBar: false, responsive: true });
    },

    renderFamilyTree() {
      const el = document.getElementById('family-tree-chart');
      if (!el || this.view !== 'family_tree' || !this.familyTreeData) return;

      const d = this.familyTreeData;

      const edgeTrace = {
        x: d.edge_x,
        y: d.edge_y,
        mode: 'lines',
        line: { color: '#ccc', width: 1 },
        hoverinfo: 'none',
        type: 'scatter'
      };

      const highlightTrace = {
        x: [],
        y: [],
        mode: 'lines',
        line: { color: '#fbbf24', width: 3 },
        hoverinfo: 'none',
        type: 'scatter'
      };

      const nodeTrace = {
        x: d.node_x,
        y: d.node_y,
        customdata: d.node_ids,
        text: d.node_labels,
        hovertext: d.node_hover,
        mode: 'markers+text',
        textposition: 'bottom center',
        textfont: { size: 9, color: '#fff' },
        hoverinfo: 'text',
        marker: {
          color: d.node_colors,
          symbol: d.node_symbols,
          size: d.node_sizes,
          line: { width: 1, color: '#333' }
        },
        type: 'scatter'
      };

      const layout = {
        showlegend: false,
        hovermode: 'closest',
        xaxis: { showgrid: false, zeroline: false, showticklabels: false },
        yaxis: { showgrid: false, zeroline: false, showticklabels: false },
        margin: { l: 40, r: 40, t: 40, b: 40 },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#d4d4d8', family: 'ui-sans-serif' },
        hoverlabel: { bgcolor: '#0a0a0a', bordercolor: '#3f3f46', font: { color: '#fafafa' } },
      };

      Plotly.react(el, [edgeTrace, highlightTrace, nodeTrace], layout, { responsive: true, displayModeBar: false });

      const getAncestorEdgeCoords = (nodeId) => {
        const hx = [], hy = [];
        const visited = new Set();
        const queue = [String(nodeId)];
        while (queue.length > 0) {
          const current = queue.shift();
          if (visited.has(current)) continue;
          visited.add(current);
          const parents = d.parent_map[current] || [];
          for (const parent of parents) {
            const pStr = String(parent);
            if (d.pos_map[pStr] && d.pos_map[current]) {
              hx.push(d.pos_map[pStr][0], d.pos_map[current][0], null);
              hy.push(d.pos_map[pStr][1], d.pos_map[current][1], null);
              queue.push(pStr);
            }
          }
        }
        return { hx, hy };
      };

      el.removeAllListeners?.('plotly_hover');
      el.on('plotly_hover', (data) => {
        if (data.points.length > 0) {
          const pt = data.points[0];
          if (pt.customdata != null) {
            const { hx, hy } = getAncestorEdgeCoords(pt.customdata);
            Plotly.restyle(el, { x: [hx], y: [hy] }, [1]);
          }
        }
      });

      el.removeAllListeners?.('plotly_unhover');
      el.on('plotly_unhover', () => {
        Plotly.restyle(el, { x: [[]], y: [[]] }, [1]);
      });

      el.removeAllListeners?.('plotly_click');
      el.on('plotly_click', (data) => {
        if (data.points.length > 0) {
          const pt = data.points[0];
          if (pt.customdata != null) {
            this.openProgram(pt.customdata);
          }
        }
      });
    },

    // ── program panel ──
    async openProgram(idx) {
      this.programDetail = {};
      this.latexResult = {};
      this.latexError = '';
      this.panelOpen = true;
      this.panelTab = 'code';
      this.codeTab = 'model';
      try {
        const r = await fetch(`/api/runs/${this.runId}/programs/${idx}`);
        if (r.ok) this.programDetail = await r.json();
      } catch (e) {
        console.warn('program detail failed', e);
      }
    },
    closePanel() {
      this.panelOpen = false;
    },

    async loadLatex(force = false) {
      if (this.programDetail.idx == null) return;
      this.latexError = '';
      this.latexLoading = true;
      try {
        const r = await fetch(
          `/api/runs/${this.runId}/programs/${this.programDetail.idx}/latex?force=${force ? 'true' : 'false'}`,
          { method: 'POST' }
        );
        const j = await r.json();
        if (!r.ok) {
          this.latexError = j.detail || 'LLM request failed';
          this.latexResult = {};
        } else {
          this.latexResult = j;
        }
      } catch (e) {
        this.latexError = String(e);
        this.latexResult = {};
      } finally {
        this.latexLoading = false;
      }
    },

    latexHtml(s) {
      if (!s) return '';
      // Render the LLM output via marked so display equations stay $$...$$
      // and KaTeX auto-render picks them up afterwards.
      try {
        return marked.parse(s);
      } catch {
        return `<pre>${escapeHtml(s)}</pre>`;
      }
    },
    renderMath() {
      const el = document.getElementById('latex-render');
      if (!el || !window.renderMathInElement) return;
      try {
        renderMathInElement(el, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\(', right: '\\)', display: false },
            { left: '\\[', right: '\\]', display: true },
          ],
          throwOnError: false,
        });
      } catch (e) { console.warn('KaTeX render failed', e); }
    },
    renderPromptMarkdown(s) {
      if (!s) return '<span class="text-zinc-500 italic">no prompt recorded</span>';
      try { return marked.parse(s); } catch { return `<pre>${escapeHtml(s)}</pre>`; }
    },
    hljsRender(el) {
      try { window.hljs?.highlightElement(el); } catch {}
    },
  };
}

// ── utility globals ──

function escapeHtml(s) {
  if (s === null || s === undefined) return '';
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[c]);
}

function fmtN(v) {
  if (v === null || v === undefined) return '-';
  if (typeof v !== 'number') return String(v);
  return v.toFixed(3);
}

const ISLAND_PALETTE = [
  '#60a5fa', '#f472b6', '#fbbf24', '#34d399', '#a78bfa',
  '#fb7185', '#22d3ee', '#facc15', '#4ade80', '#c084fc',
];
function islandColor(i) {
  return ISLAND_PALETTE[i % ISLAND_PALETTE.length];
}

function nodeColor(loss) {
  if (loss === null || loss === undefined) return '#52525b';
  if (loss < 30) return '#34d399';
  if (loss < 50) return '#fbbf24';
  return '#fb7185';
}

// Stable per-point jitter (so points don't overlap on the same gen)
function jitterFromIdx(custom) {
  return 0;
}
