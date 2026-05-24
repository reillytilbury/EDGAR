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

function formatRemovalDetailValue(value) {
  if (value === null || value === undefined) return '<em>N/A</em>';
  if (Array.isArray(value) || (typeof value === 'object' && value !== null)) {
    return '<code>' + escapeHtml(JSON.stringify(value)) + '</code>';
  }
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : Number(value).toFixed(4);
  }
  return escapeHtml(value);
}

function formatRemovalDetails(details) {
  if (!details || typeof details !== 'object' || Object.keys(details).length === 0) return '';
  let html = '<div class="field" style="margin-top:6px;">';
  html += '<span class="field-label" style="color:#856404;">Details:</span>';
  html += '<div class="field-value" style="margin-top:4px;">';
  Object.entries(details).forEach(function([key, value]) {
    html += '<div><strong>' + escapeHtml(key) + ':</strong> ' + formatRemovalDetailValue(value) + '</div>';
  });
  html += '</div></div>';
  return html;
}

function showSidebar(idx) {
  const d = sidebarData[String(idx)];
  if (!d) return;
  const sb = document.getElementById('sidebar');
  const sc = document.getElementById('sidebar-content');

  const displayLabel = d.display_label || d.program_id || (String(d.iteration) + '_' + String(d.island) + '_' + String(d.batch));
  let h = '<h2>' + escapeHtml(d.iteration === -1 ? ('Seed ' + displayLabel) : ('Program ' + displayLabel)) + '</h2>';
  h += '<div class="field"><span class="field-label">ID:</span> <span class="field-value">' + escapeHtml(d.program_id) + '</span></div>';
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
    h += formatRemovalDetails(d.removal_reason.details);
    h += '</div>';
  }

  if (d.model_code) {
    h += '<details open><summary>Model Code</summary><pre>' + escapeHtml(d.model_code) + '</pre></details>';
  }
  if (d.model_code_jax) {
    h += '<details><summary>Model Code (JAX)</summary><pre>' + escapeHtml(d.model_code_jax) + '</pre></details>';
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
