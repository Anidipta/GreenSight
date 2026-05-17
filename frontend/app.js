const API = '/api/v1';

const state = {
  map:           null,
  drawControl:   null,
  drawnItems:    null,
  chipLayers:    [],
  resultLayers:  [],
  chips:         [],
  currentChipIdx: 0,
  currentAOI:    null,
  currentProgress: 0,
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
  state.currentProgress = value;
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

  // Satellite/Geo imagery layer
  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: '© Esri, DigitalGlobe, Earthstar Geographics',
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
    state.drawing = true;
  });
  state.map.on(L.Draw.Event.DRAWSTOP, () => { state.drawing = false; });

  state.map.setView([40.0, -95.0], 5);
}

function getCurrentChip() {
  return state.chips[state.currentChipIdx] || null;
}

function chipIntersectsAOI(chip, aoi) {
  return !(aoi.east < chip.bounds.west || aoi.west > chip.bounds.east ||
           aoi.north < chip.bounds.south || aoi.south > chip.bounds.north);
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

  // Check if AOI intersects with any chip
  const hitsChips = state.chips.some(chip => chipIntersectsAOI(chip, state.currentAOI));

  if (!hitsChips) {
    showError('AOI must overlap a green chip area. Draw within a chip.');
    state.currentAOI = null;
    state.drawnItems.clearLayers();
    el('btnRun').disabled = true;
    return;
  }

  el('aW').textContent = state.currentAOI.west.toFixed(5);
  el('aS').textContent = state.currentAOI.south.toFixed(5);
  el('aE').textContent = state.currentAOI.east.toFixed(5);
  el('aN').textContent = state.currentAOI.north.toFixed(5);

  showCard('aoiInfo');
  el('btnRun').disabled = false;
  el('btnRun').classList.add('glowing');
  clearError();
}

function clearResultLayers() {
  state.resultLayers.forEach(l => state.map.removeLayer(l));
  state.resultLayers = [];
}

function updateChipDisplay() {
  const chip = getCurrentChip();
  if (!chip) return;

  // Fit map to this chip
  const b = chip.bounds;
  state.map.fitBounds([[b.south, b.west], [b.north, b.east]], { padding: [40, 40] });

  clearResultLayers();
  hideCard('resultsCard');
  hideCard('aoiInfo');
  state.drawnItems.clearLayers();
  state.currentAOI = null;
  el('btnRun').disabled = true;
  el('btnRun').classList.remove('glowing');
  clearError();
}

function selectChip(chipIdx) {
  if (chipIdx >= 0 && chipIdx < state.chips.length) {
    state.currentChipIdx = chipIdx;
    el('chipSelect').value = chipIdx;
    updateChipDisplay();
  }
}

function prevChip() {
  if (state.currentChipIdx > 0) {
    selectChip(state.currentChipIdx - 1);
  }
}

function nextChip() {
  if (state.currentChipIdx < state.chips.length - 1) {
    selectChip(state.currentChipIdx + 1);
  }
}

function enableDraw() {
  if (state.busy) return;
  state.map.addControl(state.drawControl);
  new L.Draw.Rectangle(state.map, state.drawControl.options.draw.rectangle).enable();
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

    state.chips = data.chips;
    state.currentChipIdx = 0;

    // Populate dropdown
    const select = el('chipSelect');
    select.innerHTML = '';
    data.chips.forEach((chip, idx) => {
      const option = document.createElement('option');
      option.value = idx;
      option.textContent = chip.id;
      select.appendChild(option);
    });
    select.value = 0;

    // Draw all chips on map with orange dotted border outline
    const bounds = [];
    data.chips.forEach(chip => {
      const b = chip.bounds;
      bounds.push(b);

      const rect = L.rectangle(
        [[b.south, b.west], [b.north, b.east]],
        {
          color:       '#f97316',
          weight:      3,
          fillColor:   'transparent',
          fillOpacity: 0,
          dashArray:   '5,5',
        }
      ).addTo(state.map);

      state.chipLayers.push(rect);
    });

    setStatus(`${data.chips.length} chips loaded`, 'ready');
    updateChipDisplay();

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
  setProgress(0, 'Sending request to server …');
  setStatus('Processing …', 'working');

  el('btnDraw').disabled = true;
  el('btnRun').disabled = true;
  el('chipSelect').disabled = true;

  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }

  try {
    console.log('Sending AOI:', state.currentAOI);
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

    const resp = await fetch(`${API}/analyze/aoi`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(state.currentAOI),
      signal:  controller.signal,
    });

    clearTimeout(timeoutId);
    
    console.log('Response received:', resp.status);

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ message: resp.statusText }));
      throw new Error(err.message || resp.statusText);
    }

    const data = await resp.json();
    console.log('Job ID:', data.job_id);
    
    if (!data.job_id) {
      throw new Error('No job_id returned from server');
    }

    setProgress(5, 'Connected. Processing tiles …');
    subscribeSSE(data.job_id);

  } catch (err) {
    console.error('Fetch error:', err);
    if (err.name === 'AbortError') {
      showError('Request timeout. Server is taking too long. Check if model is loading.');
    } else {
      showError('Request failed: ' + err.message);
    }
    resetControls();
  }
}


function subscribeSSE(job_id) {
  const src = new EventSource(`${API}/stream/${job_id}`);
  state.eventSource = src;
  
  let lastEventTime = Date.now();
  const noProgressTimeout = setInterval(() => {
    if (state.busy && Date.now() - lastEventTime > 15000) {
      console.warn('No progress for 15s');
      setProgress(state.currentProgress || 0, 'Processing tiles (no update for 15s)…');
    }
  }, 5000);

  src.onmessage = e => {
    lastEventTime = Date.now();
    let event;
    try { event = JSON.parse(e.data); } catch { return; }
    handleSSEEvent(event, src, noProgressTimeout);
  };

  src.onerror = () => {
    clearInterval(noProgressTimeout);
    src.close();
    if (state.busy) {
      console.error('SSE connection error');
      showError('Connection to server lost during processing.');
      resetControls();
    }
  };
}


function handleSSEEvent(event, src, noProgressTimeout) {
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
      clearInterval(noProgressTimeout);
      src.close();
      state.eventSource = null;
      renderResult(event);
      resetControls();
      break;

    case 'error':
      clearInterval(noProgressTimeout);
      src.close();
      state.eventSource = null;
      showError(event.message);
      resetControls();
      break;

    case 'done':
      clearInterval(noProgressTimeout);
      src.close();
      state.eventSource = null;
      hideCard('progressCard');
      setStatus('Ready', 'ready');
      resetControls();
      break;
  }
}


function getHealthColor(metric, value) {
  // Returns 'green', 'orange', or 'red' based on metric value
  if (value == null || value === '—') return 'neutral';
  
  const v = Number(value);
  switch(metric) {
    case 'vegetation':
      return v >= 70 ? 'green' : v >= 40 ? 'orange' : 'red';
    case 'chl_stress':
      return v < 20 ? 'green' : v < 50 ? 'orange' : 'red';
    case 'chlorophyll':
      return v >= 20 ? 'green' : v >= 10 ? 'orange' : 'red';
    case 'nitrogen':
      return v >= 2.0 ? 'green' : v >= 1.5 ? 'orange' : 'red';
    case 'biomass':
      return v >= 5000 ? 'green' : v >= 2000 ? 'orange' : 'red';
    case 'bio_loss':
      return v < 500 ? 'green' : v < 2000 ? 'orange' : 'red';
    default:
      return 'neutral';
  }
}

function renderResult(event) {
  const m     = event.metrics;
  const gt    = event.gt_proxies || {};

  hideCard('progressCard');
  showCard('resultsCard');
  glowCard('resultsCard');

  el('resultChipId').textContent = event.chip_id || '';

  // Hide severity banner
  el('severityBanner').style.display = 'none';

  const fmt1 = v => v != null ? Number(v).toFixed(1) : '—';
  const fmt2 = v => v != null ? Number(v).toFixed(2) : '—';

  const vegVal = fmt1(gt.vegetation_pct != null ? gt.vegetation_pct : m.vegetation_coverage_pct);
  const chlStressVal = fmt1(m.chlorophyll_stress_pct);
  const chlVal = fmt2(m.chlorophyll_ug_cm2);
  const nVal = fmt2(m.n_concentration_pct);
  const bioVal = fmt2(m.biomass_agb_mgha);
  const lossVal = fmt2(m.biomass_loss_mgha);

  el('mVeg').textContent = vegVal;
  el('mChlStress').textContent = chlStressVal;
  el('mChl').textContent = chlVal;
  el('mN').textContent = nVal;
  el('mBio').textContent = bioVal;
  el('mLoss').textContent = lossVal;

  // Apply health-based colors to metric cards
  const metricMap = {
    'mVeg': ['vegetation', vegVal],
    'mChlStress': ['chl_stress', chlStressVal],
    'mChl': ['chlorophyll', chlVal],
    'mN': ['nitrogen', nVal],
    'mBio': ['biomass', bioVal],
    'mLoss': ['bio_loss', lossVal],
  };

  Object.entries(metricMap).forEach(([elId, [metricName, value]]) => {
    const el_item = document.querySelector(`[id="${elId}"]`);
    if (el_item) {
      const parentCard = el_item.closest('.metric-item');
      if (parentCard) {
        const color = getHealthColor(metricName, value);
        parentCard.className = 'metric-item metric-' + color;
      }
    }
  });

  renderDetails(event);
  renderMapOverlay(event);
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


function renderMapOverlay(event) {
  const b = event.bbox;
  if (!b) return;

  const rect = L.rectangle(
    [[b.south, b.west], [b.north, b.east]],
    {
      color:       '#3b82f6',
      weight:      2.5,
      fillColor:   '#3b82f6',
      fillOpacity: 0.15,
      dashArray:   null,
    }
  ).addTo(state.map);

  const m = event.metrics;
  const popHtml = `
    <div class="popup-inner">
      <div class="popup-title">${event.chip_id}</div>
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
  el('btnDraw').disabled = false;
  el('btnRun').disabled = !state.currentAOI;
  el('chipSelect').disabled = false;
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
  el('chipSelect').addEventListener('change', (e) => {
    const idx = parseInt(e.target.value);
    if (!isNaN(idx)) selectChip(idx);
  });
  el('btnDraw').addEventListener('click', enableDraw);
  el('btnRun').addEventListener('click', runInference);
  
  // Add help icon event listeners for mobile
  const helpIcons = document.querySelectorAll('.help-icon');
  helpIcons.forEach(icon => {
    icon.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
    });
    icon.addEventListener('keypress', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
      }
    });
  });
}


async function boot() {
  initMap();
  bindButtons();
  await checkHealth();
  await loadChips();
}

document.addEventListener('DOMContentLoaded', boot);
