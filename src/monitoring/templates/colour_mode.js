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
