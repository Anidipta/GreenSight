const API = '/api/v1';

const state = {
  map:           null,
  drawControl:   null,
  drawnItems:    null,
  chipLayers:    [],
  resultLayers:  [],
  currentAOI:    null,
  drawing:       false,
  busy:          false,
  eventSource:   null,
};

const el = id => document.getElementById(id);

function $(sel) { return document.querySelector(sel); }


function setStatus(label, mode) {
  el('statusLabel').textContent = label;
  const dot = el('statusDot');
  dot.className = 'status-dot';
  if (mode) dot.classList.add(mode);
}

function showCard(id)  { el(id).classList.remove('hidden'); }
function hideCard(id)  { el(id).classList.add('hidden'); }
function glowCard(id)  { el(id).classList.add('glow'); setTimeout(() => el(id).classList.remove('glow'), 1800); }

function setProgress(value, message) {
  const pct = Math.round(value * 100);
  el('progressFill').style.width = pct + '%';
  el('progressPct').textContent  = pct + '%';
  if (message) el('progressMsg').textContent = message;
}

function showError(message) {
  showCard('errorCard');
  el('errorBanner').textContent = '⚠ ' + message;
  hideCard('progressCard');
  setStatus('Error', 'error');
}

function clearError() {
  hideCard('errorCard');
  el('errorBanner').textContent = '';
}


function initMap() {
  state.map = L.map('map', { zoomControl: true, attributionControl: true });

  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap © CartoDB',
    subdomains:  'abcd',
    maxZoom:     19,
  }).addTo(state.map);

  state.drawnItems = new L.FeatureGroup().addTo(state.map);

  state.drawControl = new L.Control.Draw({
    draw: {
      rectangle:  { shapeOptions: { color: '#3b82f6', weight: 2, fillOpacity: 0.08 } },
      polyline:   false,
      polygon:    false,
      circle:     false,
      circlemarker: false,
      marker:     false,
    },
    edit: { featureGroup: state.drawnItems, edit: false, remove: false },
  });

  state.map.on(L.Draw.Event.CREATED, onRectangleDrawn);
  state.map.on(L.Draw.Event.DRAWSTART, () => {
    el('mapHint').classList.add('hidden');
    state.drawing = true;
  });
  state.map.on(L.Draw.Event.DRAWSTOP, () => { state.drawing = false; });

  state.map.setView([40.0, -95.0], 5);
}


function onRectangleDrawn(e) {
  state.drawnItems.clearLayers();
  state.drawnItems.addLayer(e.layer);

  const b = e.layer.getBounds();
  state.currentAOI = {
    west:  b.getWest(),
    south: b.getSouth(),
    east:  b.getEast(),
    north: b.getNorth(),
  };

  el('aW').textContent = state.currentAOI.west.toFixed(5);
  el('aS').textContent = state.currentAOI.south.toFixed(5);
  el('aE').textContent = state.currentAOI.east.toFixed(5);
  el('aN').textContent = state.currentAOI.north.toFixed(5);

  showCard('aoiInfo');
  el('btnClear').disabled = false;
  el('btnRun').disabled   = false;
  el('btnRun').classList.add('glowing');
  clearError();
}


function enableDraw() {
  if (state.busy) return;
  state.map.addControl(state.drawControl);
  new L.Draw.Rectangle(state.map, state.drawControl.options.draw.rectangle).enable();
}

function clearAOI() {
  state.drawnItems.clearLayers();
  clearResultLayers();
  state.currentAOI = null;
  hideCard('aoiInfo');
  hideCard('resultsCard');
  clearError();
  el('btnClear').disabled = true;
  el('btnRun').disabled   = true;
  el('btnRun').classList.remove('glowing');
  el('mapHint').classList.remove('hidden');
}

function clearResultLayers() {
  state.resultLayers.forEach(l => state.map.removeLayer(l));
  state.resultLayers = [];
}


async function loadChips() {
  setStatus('Loading chips …', null);
  try {
    const resp = await fetch(`${API}/chips`);
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();

    if (!data.chips || data.chips.length === 0) {
      setStatus('No chips found — run temp.py', 'error');
      return;
    }

    const bounds = [];
    data.chips.forEach(chip => addChipOverlay(chip, bounds));

    if (bounds.length) {
      const all = L.latLngBounds(bounds.map(b => [
        [b.south, b.west], [b.north, b.east],
      ]).flat());
      state.map.fitBounds(all, { padding: [40, 40] });
    }

    setStatus(`${data.chips.length} chips loaded`, 'ready');
  } catch (err) {
    setStatus('Chip load failed', 'error');
    showError('Could not load chips: ' + err.message + '. Did you run temp.py?');
  }
}

function addChipOverlay(chip, boundsAccum) {
  const b = chip.bounds;
  boundsAccum.push(b);

  const rect = L.rectangle(
    [[b.south, b.west], [b.north, b.east]],
    {
      color:       '#22c55e',
      weight:      1.5,
      fillColor:   '#22c55e',
      fillOpacity: 0.10,
      dashArray:   '5,4',
    }
  ).addTo(state.map);

  const popHtml = `
    <div class="popup-inner">
      <div class="popup-title">${chip.id}</div>
      <div class="popup-row"><span class="pk">Size</span><span class="pv">${chip.width_px} × ${chip.height_px} px</span></div>
      <div class="popup-row"><span class="pk">Bands</span><span class="pv">${chip.n_bands}</span></div>
      <div class="popup-row"><span class="pk">Center</span><span class="pv">${chip.center.lat.toFixed(3)}, ${chip.center.lon.toFixed(3)}</span></div>
    </div>`;
  rect.bindPopup(popHtml, { className: 'result-overlay-popup' });
  rect.on('mouseover', () => rect.setStyle({ fillOpacity: 0.22, color: '#16a34a' }));
  rect.on('mouseout',  () => rect.setStyle({ fillOpacity: 0.10, color: '#22c55e' }));

  state.chipLayers.push(rect);
}


async function runInference() {
  if (!state.currentAOI || state.busy) return;

  state.busy = true;
  clearResultLayers();
  clearError();
  hideCard('resultsCard');
  showCard('progressCard');
  setProgress(0, 'Sending request …');
  setStatus('Processing …', 'working');

  el('btnRun').disabled = true;
  el('btnRun').classList.remove('glowing');
  el('btnDraw').disabled  = true;
  el('btnClear').disabled = true;

  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }

  try {
    const resp = await fetch(`${API}/analyze/aoi`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(state.currentAOI),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ message: resp.statusText }));
      throw new Error(err.message || resp.statusText);
    }

    const { job_id } = await resp.json();
    subscribeSSE(job_id);

  } catch (err) {
    showError(err.message);
    resetControls();
  }
}


function subscribeSSE(job_id) {
  const src = new EventSource(`${API}/stream/${job_id}`);
  state.eventSource = src;

  src.onmessage = e => {
    let event;
    try { event = JSON.parse(e.data); } catch { return; }
    handleSSEEvent(event, src);
  };

  src.onerror = () => {
    src.close();
    if (state.busy) {
      showError('Connection to server lost during processing.');
      resetControls();
    }
  };
}


function handleSSEEvent(event, src) {
  switch (event.type) {

    case 'status':
      el('progressMsg').textContent = event.message;
      break;

    case 'progress':
      setProgress(event.value, event.message);
      break;

    case 'warning':
      console.warn('[SSE warning]', event.message);
      break;

    case 'result':
      src.close();
      state.eventSource = null;
      renderResult(event);
      resetControls();
      break;

    case 'error':
      src.close();
      state.eventSource = null;
      showError(event.message);
      resetControls();
      break;

    case 'done':
      src.close();
      state.eventSource = null;
      hideCard('progressCard');
      setStatus('Ready', 'ready');
      resetControls();
      break;
  }
}


function renderResult(event) {
  const m     = event.metrics;
  const sev   = m.stress_severity;
  const color = event.severity_color;

  hideCard('progressCard');
  showCard('resultsCard');
  glowCard('resultsCard');

  el('resultChipId').textContent = event.chip_id || '';

  const banner = el('severityBanner');
  banner.className = 'severity-banner ' + sev.toLowerCase();
  el('severityIcon').textContent = sev === 'MILD' ? '✓' : sev === 'MODERATE' ? '⚠' : '✗';
  el('severityText').textContent = sev + ' stress';

  const fmt1 = v => v != null ? Number(v).toFixed(1) : '—';
  const fmt2 = v => v != null ? Number(v).toFixed(2) : '—';

  el('mVeg').textContent       = fmt1(m.vegetation_coverage_pct);
  el('mChlStress').textContent = fmt1(m.chlorophyll_stress_pct);
  el('mChl').textContent       = fmt2(m.chlorophyll_ug_cm2);
  el('mN').textContent         = fmt2(m.n_concentration_pct);
  el('mBio').textContent       = fmt2(m.biomass_agb_mgha);
  el('mLoss').textContent      = fmt2(m.biomass_loss_mgha);

  renderDetails(event);
  renderMapOverlay(event, color, sev);
  setStatus('Done', 'ready');
}


function renderDetails(event) {
  const m   = event.metrics;
  const gt  = event.gt_proxies  || {};
  const ec  = event.error_chlorophyll || {};
  const eb  = event.error_biomass     || {};

  const rows = [
    ['Chl GT (proxy)',   gt.chl_gt_ug_cm2 != null ? gt.chl_gt_ug_cm2.toFixed(2) + ' μg/cm²' : '—'],
    ['Chl MAE',         ec.mae != null ? ec.mae.toFixed(4) : '—'],
    ['Chl r',           ec.r   != null ? ec.r.toFixed(3)   : '—'],
    ['AGB GT (proxy)',  gt.agb_gt_mgha != null ? gt.agb_gt_mgha.toFixed(1) + ' Mg/ha' : '—'],
    ['AGB MAE',         eb.mae != null ? eb.mae.toFixed(2) : '—'],
    ['AGB r',           eb.r   != null ? eb.r.toFixed(3)   : '—'],
    ['Biomass % max',   m.biomass_pct_of_max + '%'],
    ['N normalized',    m.n_normalized_pct + '%'],
    ['Veg. stress %',   m.stressed_area_pct + '%'],
  ];

  el('detailsInner').innerHTML = rows.map(([k, v]) =>
    `<div class="detail-row"><span class="dk">${k}</span><span class="dv">${v}</span></div>`
  ).join('');
}


function renderMapOverlay(event, color, sev) {
  const b = event.bbox;
  if (!b) return;

  const fillOpacity = sev === 'SEVERE' ? 0.35 : sev === 'MODERATE' ? 0.25 : 0.18;

  const rect = L.rectangle(
    [[b.south, b.west], [b.north, b.east]],
    {
      color,
      weight:      2.5,
      fillColor:   color,
      fillOpacity,
      dashArray:   null,
    }
  ).addTo(state.map);

  const m = event.metrics;
  const popHtml = `
    <div class="popup-inner">
      <div class="popup-title" style="color:${color}">${sev} Stress · ${event.chip_id}</div>
      <div class="popup-row"><span class="pk">Veg cover</span><span class="pv">${m.vegetation_coverage_pct}%</span></div>
      <div class="popup-row"><span class="pk">Chl stress</span><span class="pv">${m.chlorophyll_stress_pct}%</span></div>
      <div class="popup-row"><span class="pk">Chlorophyll</span><span class="pv">${Number(m.chlorophyll_ug_cm2).toFixed(2)} μg/cm²</span></div>
      <div class="popup-row"><span class="pk">Biomass</span><span class="pv">${Number(m.biomass_agb_mgha).toFixed(1)} Mg/ha</span></div>
      <div class="popup-row"><span class="pk">N conc.</span><span class="pv">${Number(m.n_concentration_pct).toFixed(3)}%</span></div>
    </div>`;

  rect.bindPopup(popHtml, { className: 'result-overlay-popup' }).openPopup();
  state.resultLayers.push(rect);
  state.map.fitBounds([[b.south, b.west], [b.north, b.east]], { padding: [60, 60] });
}


function resetControls() {
  state.busy = false;
  el('btnDraw').disabled  = false;
  el('btnClear').disabled = !state.currentAOI;
  el('btnRun').disabled   = !state.currentAOI;
  if (state.currentAOI) el('btnRun').classList.add('glowing');
}


async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    if (d.model_ready) {
      setStatus('Model ready', 'ready');
    } else {
      setStatus('Model loading …', null);
      setTimeout(checkHealth, 3000);
    }
  } catch {
    setStatus('Server offline', 'error');
    setTimeout(checkHealth, 5000);
  }
}


function bindButtons() {
  el('btnDraw').addEventListener('click', enableDraw);
  el('btnClear').addEventListener('click', clearAOI);
  el('btnRun').addEventListener('click', runInference);
}


async function boot() {
  initMap();
  bindButtons();
  await checkHealth();
  await loadChips();
}

document.addEventListener('DOMContentLoaded', boot);
