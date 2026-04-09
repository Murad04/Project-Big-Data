const form = document.getElementById('prediction-form');
const status = document.getElementById('status');
const resultState = document.getElementById('result-state');
const submitButton = document.getElementById('submit-button');
const activeModelBadge = document.getElementById('active-model');

async function loadModelInfo() {
  if (!activeModelBadge) {
    return;
  }

  try {
    const response = await fetch('/model-info');
    if (!response.ok) {
      throw new Error(`status ${response.status}`);
    }
    const info = await response.json();
    const flavor = info.is_trained_model ? 'trained' : 'baseline';
    activeModelBadge.textContent = `Model: ${info.model_name} (${flavor})`;
  } catch (_error) {
    activeModelBadge.textContent = 'Model: unavailable';
  }
}

loadModelInfo();

function readFormValues() {
  const data = new FormData(form);
  return {
    flight_number: String(data.get('flight_number') || ''),
    origin: String(data.get('origin') || ''),
    destination: String(data.get('destination') || ''),
    departure_time: String(data.get('departure_time') || ''),
    weather_severity: Number(data.get('weather_severity') || 0),
    airport_congestion: Number(data.get('airport_congestion') || 0),
    day_of_week: Number(data.get('day_of_week') || 1),
    month: Number(data.get('month') || 1),
    airline_code: String(data.get('airline_code') || ''),
  };
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  status.textContent = 'Running prediction...';
  submitButton.disabled = true;
  resultState.className = 'empty-state';
  resultState.innerHTML = '<p>Loading prediction result...</p>';

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(readFormValues()),
    });

    if (!response.ok) {
      const details = await response.text();
      throw new Error(`Prediction request failed (${response.status}): ${details}`);
    }

    const result = await response.json();
    status.textContent = 'Prediction complete.';
    resultState.className = 'result-card';
    resultState.innerHTML = `
      <strong>${result.predicted_delay_minutes} minutes</strong>
      <span>Model: ${result.model_name}</span>
    `;
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : 'Failed to call backend.';
    resultState.className = 'empty-state';
    resultState.innerHTML = '<p>No prediction available.</p>';
  } finally {
    submitButton.disabled = false;
  }
});
