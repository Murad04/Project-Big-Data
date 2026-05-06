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

// ── Prediction history (in-memory) ───────────────────────────────────────────
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

clearHistBtn?.addEventListener('click', () => {
  history.length = 0;
  renderHistory();
});

// ── Sliders ───────────────────────────────────────────────────────────────────
document.getElementById('weather_severity')?.addEventListener('input', e => {
  document.getElementById('ws-val').textContent = Number(e.target.value).toFixed(1);
});
document.getElementById('airport_congestion')?.addEventListener('input', e => {
  document.getElementById('ac-val').textContent = Number(e.target.value).toFixed(1);
});

// ── Departure time → day_of_week + month ─────────────────────────────────────
const DAYS   = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
const MONTHS = ['January','February','March','April','May','June','July','August','September','October','November','December'];

function updateDateFields() {
  const val = document.getElementById('departure_time')?.value;
  if (!val) return;
  const d = new Date(val);
  if (isNaN(d.getTime())) return;
  const jsDay  = d.getDay();
  const isoDow = jsDay === 0 ? 7 : jsDay;   // 1=Mon … 7=Sun
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
    const tag  = info.is_trained_model ? 'trained' : 'baseline';
    modelBadge.textContent = `${info.model_name} (${tag})`;
    modelBadge.classList.add(info.is_trained_model ? 'badge-trained' : 'badge-baseline');
  } catch {
    modelBadge.textContent = 'Model: unavailable';
  }
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
    document.getElementById('metrics-note').textContent = 'Metrics unavailable — make sure the API server is running.';
  }
}

// ── Prediction form ───────────────────────────────────────────────────────────
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
    flight_number:     String(data.get('flight_number')     || ''),
    airline_code:      String(data.get('airline_code')      || ''),
    origin:            String(data.get('origin')            || ''),
    destination:       String(data.get('destination')       || ''),
    departure_time:    String(data.get('departure_time')    || ''),
    weather_severity:  Number(data.get('weather_severity')  || 0),
    airport_congestion: Number(data.get('airport_congestion') || 0),
    day_of_week:       Number(data.get('day_of_week')       || 1),
    month:             Number(data.get('month')             || 1),
  };

  try {
    const res = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`Server error ${res.status}: ${detail}`);
    }
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

  const ring = document.getElementById('delay-ring');
  ring.style.setProperty('--sev-color', sev.color);

  document.getElementById('result-route').textContent      = `${payload.origin} → ${payload.destination}`;
  document.getElementById('result-flight').textContent     = `${payload.airline_code} ${payload.flight_number}`;
  document.getElementById('result-conditions').textContent =
    `Weather ${payload.weather_severity.toFixed(1)} / Congestion ${payload.airport_congestion.toFixed(1)}`;
  document.getElementById('result-model').textContent = result.model_name;

  resultContent.classList.remove('hidden');
}

// ── Init ──────────────────────────────────────────────────────────────────────
loadModelInfo();
loadMetrics();
