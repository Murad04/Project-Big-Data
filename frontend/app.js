// ── DOM refs ──────────────────────────────────────────────────────────────────
const form          = document.getElementById('prediction-form');
const modelBadge    = document.getElementById('model-badge');
const submitBtn     = document.getElementById('submit-btn');
const btnText       = document.getElementById('btn-text');
const resultEmpty   = document.getElementById('result-empty');
const loadingState  = document.getElementById('loading-state');
const resultContent = document.getElementById('result-content');
const errorMsg      = document.getElementById('error-msg');
const historyList   = document.getElementById('history-list');
const clearHistBtn  = document.getElementById('clear-history');

// ── Severity tiers ────────────────────────────────────────────────────────────
const SEVERITY = [
  { max: 0,        label: 'On Time',           cls: 'sev-ontime',      color: '#16a34a' },
  { max: 15,       label: 'Minor Delay',        cls: 'sev-minor',       color: '#65a30d' },
  { max: 30,       label: 'Moderate Delay',     cls: 'sev-moderate',    color: '#ca8a04' },
  { max: 60,       label: 'Significant Delay',  cls: 'sev-significant', color: '#ea580c' },
  { max: Infinity, label: 'Major Delay',        cls: 'sev-major',       color: '#dc2626' },
];

function getSeverity(minutes) {
  return SEVERITY.find(s => minutes <= s.max) ?? SEVERITY[SEVERITY.length - 1];
}

// ── Prediction history ────────────────────────────────────────────────────────
const history = [];
function addHistory(entry) {
  history.unshift(entry);
  if (history.length > 8) history.pop();
  renderHistory();
}
function renderHistory() {
  if (history.length === 0) {
    historyList.innerHTML = '<li class="history-empty">No predictions yet.</li>';
    clearHistBtn.classList.add('hidden');
    return;
  }
  clearHistBtn.classList.remove('hidden');
  historyList.innerHTML = history.map(h => {
    const sev = getSeverity(h.delay);
    return `<li class="history-item">
      <span class="hist-route">${h.route}</span>
      <span class="hist-delay" style="color:${sev.color}">${h.delay.toFixed(1)} min</span>
      <span class="hist-sev" style="background:${sev.color}22;color:${sev.color}">${sev.label}</span>
    </li>`;
  }).join('');
}
clearHistBtn?.addEventListener('click', () => { history.length = 0; renderHistory(); });

// ── Sliders ───────────────────────────────────────────────────────────────────
document.getElementById('weather_severity')?.addEventListener('input', e =>
  document.getElementById('ws-val').textContent = Number(e.target.value).toFixed(1));
document.getElementById('airport_congestion')?.addEventListener('input', e =>
  document.getElementById('ac-val').textContent = Number(e.target.value).toFixed(1));

// ── Date helpers ──────────────────────────────────────────────────────────────
const DAYS   = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
const MONTHS = ['January','February','March','April','May','June','July','August','September','October','November','December'];

function updateDateFields() {
  const val = document.getElementById('departure_time')?.value;
  if (!val) return;
  const d = new Date(val);
  if (isNaN(d.getTime())) return;
  const jsDay  = d.getDay();
  const isoDow = jsDay === 0 ? 7 : jsDay;
  const month  = d.getMonth() + 1;
  document.getElementById('day_of_week_field').value = isoDow;
  document.getElementById('month_field').value        = month;
  document.getElementById('schedule-hint').textContent =
    `${DAYS[jsDay]}, ${MONTHS[month - 1]} ${d.getFullYear()} · Month ${month}, Day of week ${isoDow}`;
}
document.getElementById('departure_time')?.addEventListener('change', updateDateFields);
updateDateFields();

// ── Model info ────────────────────────────────────────────────────────────────
async function loadModelInfo() {
  try {
    const res  = await fetch('/model-info');
    if (!res.ok) throw new Error(res.status);
    const info = await res.json();
    modelBadge.textContent = `${info.model_name} (${info.is_trained_model ? 'trained' : 'baseline'})`;
    modelBadge.classList.add(info.is_trained_model ? 'badge-trained' : 'badge-baseline');
  } catch { modelBadge.textContent = 'Model: unavailable'; }
}

// ── Metrics ───────────────────────────────────────────────────────────────────
async function loadMetrics() {
  try {
    const res = await fetch('/metrics');
    if (!res.ok) throw new Error(res.status);
    const m = await res.json();
    document.getElementById('m-mae').textContent       = m.mae?.toFixed(2)                    ?? '--';
    document.getElementById('m-rmse').textContent      = m.rmse?.toFixed(2)                   ?? '--';
    document.getElementById('m-r2').textContent        = m.r2?.toFixed(3)                     ?? '--';
    document.getElementById('m-f1').textContent        = m.f1_at_threshold?.toFixed(3)        ?? '--';
    document.getElementById('m-precision').textContent = m.precision_at_threshold?.toFixed(3) ?? '--';
    document.getElementById('m-recall').textContent    = m.recall_at_threshold?.toFixed(3)    ?? '--';
    document.getElementById('metrics-note').textContent =
      `Trained on ${(m.sample_count ?? 0).toLocaleString()} NYC flight records (nycflights13) · CatBoost Regressor`;
  } catch {
    document.getElementById('metrics-note').textContent = 'Metrics unavailable.';
  }
}

// ── Feature Importance Chart ──────────────────────────────────────────────────
let importanceChart = null;
const CHART_COLORS = [
  'rgba(37,99,235,0.82)', 'rgba(37,99,235,0.74)', 'rgba(37,99,235,0.66)',
  'rgba(37,99,235,0.58)', 'rgba(37,99,235,0.50)', 'rgba(37,99,235,0.44)',
  'rgba(37,99,235,0.38)', 'rgba(37,99,235,0.32)', 'rgba(37,99,235,0.26)',
  'rgba(37,99,235,0.20)',
];

const FEATURE_LABELS = {
  weather_severity:   'Weather Severity',
  airport_congestion: 'Airport Congestion',
  departure_hour:     'Departure Hour',
  day_of_week:        'Day of Week',
  month:              'Month',
  airline_code:       'Airline',
  origin:             'Origin Airport',
  destination:        'Destination Airport',
  route_avg_delay:    'Route Avg Delay',
  carrier_avg_delay:  'Carrier Avg Delay',
};

async function loadFeatureImportance() {
  const note = document.getElementById('importance-note');
  try {
    const res = await fetch('/feature-importance');
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();

    const labels = data.map(d => FEATURE_LABELS[d.feature] ?? d.feature);
    const values = data.map(d => d.importance);

    const ctx = document.getElementById('importance-chart')?.getContext('2d');
    if (!ctx) return;
    if (importanceChart) importanceChart.destroy();

    importanceChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Importance (%)',
          data: values,
          backgroundColor: CHART_COLORS.slice(0, labels.length),
          borderRadius: 6,
          borderSkipped: false,
        }],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: { label: ctx => ` ${ctx.parsed.x.toFixed(2)}%` },
          },
        },
        scales: {
          x: {
            beginAtZero: true,
            title: { display: true, text: 'Importance (%)', color: '#94a3b8', font: { size: 11 } },
            grid: { color: 'rgba(148,163,184,0.15)' },
            ticks: { color: '#64748b' },
          },
          y: {
            grid: { display: false },
            ticks: { color: '#334155', font: { weight: '600' } },
          },
        },
      },
    });
    note.textContent = 'CatBoost PredictionValuesChange importance — higher % = stronger influence on predicted delay.';
  } catch {
    note.textContent = 'Feature importance unavailable.';
  }
}

// ── Single prediction form ────────────────────────────────────────────────────
form?.addEventListener('submit', async e => {
  e.preventDefault();
  resultEmpty.classList.add('hidden');
  resultContent.classList.add('hidden');
  errorMsg.classList.add('hidden');
  loadingState.classList.remove('hidden');
  submitBtn.disabled = true;
  btnText.textContent = 'Predicting…';

  const data = new FormData(form);
  const payload = {
    flight_number:      String(data.get('flight_number')      || ''),
    airline_code:       String(data.get('airline_code')       || ''),
    origin:             String(data.get('origin')             || ''),
    destination:        String(data.get('destination')        || ''),
    departure_time:     String(data.get('departure_time')     || ''),
    weather_severity:   Number(data.get('weather_severity')   || 0),
    airport_congestion: Number(data.get('airport_congestion') || 0),
    day_of_week:        Number(data.get('day_of_week')        || 1),
    month:              Number(data.get('month')              || 1),
  };

  try {
    const res = await fetch('/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`Server error ${res.status}: ${await res.text()}`);
    const result = await res.json();
    displayResult(result, payload);
    addHistory({ route: `${payload.origin} → ${payload.destination}`, delay: result.predicted_delay_minutes });
  } catch (err) {
    loadingState.classList.add('hidden');
    resultEmpty.classList.remove('hidden');
    errorMsg.textContent = err.message || 'Prediction failed.';
    errorMsg.classList.remove('hidden');
  } finally {
    submitBtn.disabled  = false;
    btnText.textContent = 'Predict Delay';
  }
});

function displayResult(result, payload) {
  loadingState.classList.add('hidden');
  const delay = result.predicted_delay_minutes;
  const sev   = getSeverity(delay);
  document.getElementById('delay-minutes').textContent = delay.toFixed(1);
  const badge = document.getElementById('severity-badge');
  badge.textContent = sev.label;
  badge.className   = `severity-badge ${sev.cls}`;
  document.getElementById('delay-ring').style.setProperty('--sev-color', sev.color);
  document.getElementById('result-route').textContent      = `${payload.origin} → ${payload.destination}`;
  document.getElementById('result-flight').textContent     = `${payload.airline_code} ${payload.flight_number}`;
  document.getElementById('result-conditions').textContent =
    `Weather ${payload.weather_severity.toFixed(1)} / Congestion ${payload.airport_congestion.toFixed(1)}`;
  document.getElementById('result-model').textContent = result.model_name;
  resultContent.classList.remove('hidden');
}

// ── Batch prediction ──────────────────────────────────────────────────────────
const BATCH_FLIGHTS = [
  { flight_number:'AA101', airline_code:'AA', origin:'EWR', destination:'LAX', departure_time:'2026-01-05T10:00', weather_severity:1.0, airport_congestion:1.0, day_of_week:1, month:1 },
  { flight_number:'DL202', airline_code:'DL', origin:'JFK', destination:'ATL', departure_time:'2026-12-18T17:00', weather_severity:9.0, airport_congestion:8.5, day_of_week:5, month:12 },
  { flight_number:'UA303', airline_code:'UA', origin:'LGA', destination:'ORD', departure_time:'2026-06-10T12:00', weather_severity:4.5, airport_congestion:5.0, day_of_week:3, month:6 },
  { flight_number:'B6404', airline_code:'B6', origin:'JFK', destination:'BQN', departure_time:'2026-02-14T08:00', weather_severity:2.0, airport_congestion:3.0, day_of_week:6, month:2 },
  { flight_number:'EV505', airline_code:'EV', origin:'EWR', destination:'MIA', departure_time:'2026-03-10T06:00', weather_severity:0.5, airport_congestion:0.5, day_of_week:2, month:3 },
  { flight_number:'MQ606', airline_code:'MQ', origin:'LGA', destination:'DFW', departure_time:'2026-11-25T16:00', weather_severity:8.0, airport_congestion:9.0, day_of_week:4, month:11 },
];

function renderBatchTable(predictions) {
  const tbody = document.getElementById('batch-tbody');
  tbody.innerHTML = BATCH_FLIGHTS.map((f, i) => {
    const p   = predictions?.[i];
    const delay = p ? p.predicted_delay_minutes : null;
    const sev   = delay !== null ? getSeverity(delay) : null;
    const hour  = f.departure_time.split('T')[1] ?? '??:??';
    return `<tr>
      <td class="td-mono">${f.flight_number}</td>
      <td>${f.origin} → ${f.destination}</td>
      <td>${f.airline_code}</td>
      <td class="td-mono">${hour}</td>
      <td>${f.weather_severity.toFixed(1)}</td>
      <td>${f.airport_congestion.toFixed(1)}</td>
      <td class="td-delay">${delay !== null ? `<strong>${delay.toFixed(1)} min</strong>` : '<span class="td-pending">—</span>'}</td>
      <td>${sev ? `<span class="hist-sev" style="background:${sev.color}22;color:${sev.color}">${sev.label}</span>` : '<span class="td-pending">—</span>'}</td>
    </tr>`;
  }).join('');
}

// Pre-populate table with no predictions yet
renderBatchTable(null);

document.getElementById('run-batch-btn')?.addEventListener('click', async () => {
  const btn     = document.getElementById('run-batch-btn');
  const btnTxt  = document.getElementById('batch-btn-text');
  const errEl   = document.getElementById('batch-error');
  btn.disabled  = true;
  btnTxt.textContent = 'Running…';
  errEl.classList.add('hidden');

  try {
    const res = await fetch('/predict/batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ flights: BATCH_FLIGHTS }),
    });
    if (!res.ok) throw new Error(`Server error ${res.status}: ${await res.text()}`);
    const data = await res.json();
    renderBatchTable(data.predictions);
  } catch (err) {
    errEl.textContent = err.message || 'Batch prediction failed.';
    errEl.classList.remove('hidden');
  } finally {
    btn.disabled = false;
    btnTxt.textContent = 'Run Batch Test';
  }
});

// ── Init ──────────────────────────────────────────────────────────────────────
loadModelInfo();
loadMetrics();
loadFeatureImportance();