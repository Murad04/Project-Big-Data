# CLAUDE.md — Flight Delay Prediction Platform

GWU · Big Data & Cloud Computing · Final Project  
Stack: CatBoost · Dask · FastAPI · Open-Meteo · Chart.js

---

## What This Project Does

Predicts flight departure delay in minutes given route, schedule, and weather conditions.  
Training data: nycflights13 (336 K NYC flight records, 2013).  
Live weather is auto-fetched from Open-Meteo when the user picks an origin airport.

---

## Project Layout

```
src/flight_delay_platform/
  api/
    app.py          ← FastAPI app — all endpoints defined here
    schemas.py      ← Pydantic request/response models
  ml/
    train.py        ← CatBoost training, evaluation, analysis — single source of truth
  pipelines/
    preprocess.py       ← Pandas preprocessing (default path)
    dask_preprocess.py  ← Dask preprocessing (--use-dask flag)
    train_catboost.py   ← CLI entry point for training
  services/
    model_registry.py   ← Loads model, resolves lookup features at inference time
    kafka_consumer.py   ← STUB only — not wired up
    cassandra_store.py  ← STUB only — not wired up

frontend/
  index.html    ← Single-page app
  app.js        ← All frontend logic (weather fetch, charts, prediction, history)
  styles.css    ← Styling

data/raw/           ← nyc_flights.csv + nyc_weather.csv (gitignored)
data/processed/     ← Dask Parquet output (gitignored)
models/             ← catboost_delay_model.cbm (gitignored)
artifacts/          ← catboost_metrics.json, delay_lookups.json, training_analysis.json (gitignored)
catboost_info/      ← CatBoost training logs (gitignored)

architecture.html   ← Animated architecture diagrams (open in browser)
training_logs.html  ← Training curves + analysis (served at /training-logs-view)
```

---

## How to Run

```powershell
# Start server (model must be trained first)
$env:PYTHONPATH = "src"
.\.venv\Scripts\uvicorn.exe flight_delay_platform.api.app:app --app-dir src --port 8000 --reload

# Train model (Dask pipeline + CatBoost, ~4 min)
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m flight_delay_platform.pipelines.train_catboost --use-dask --max-rows 150000

# Run tests
.\.venv\Scripts\pytest.exe tests/ -v
```

---

## API Endpoints

All defined in `src/flight_delay_platform/api/app.py`.

| Method | Path | Returns |
|--------|------|---------|
| GET | `/` | Frontend HTML |
| GET | `/training-logs-view` | Training logs HTML page |
| POST | `/predict` | Single flight delay prediction |
| POST | `/predict/batch` | Up to 200 flights per call |
| GET | `/feature-importance` | CatBoost importance scores (sorted) |
| GET | `/metrics` | MAE, RMSE, R², F1, Precision, Recall |
| GET | `/training-logs` | RMSE per iteration (train + val) |
| GET | `/training-analysis` | Confusion matrix, per-airline/month/hour MAE |
| GET | `/model-info` | Model name and source path |
| GET | `/health` | `{"status": "ok"}` |

---

## The 11 Model Features

Defined in `FEATURE_COLUMNS` inside `src/flight_delay_platform/ml/train.py`.  
Feature engineering lives in `pipelines/preprocess.py` and `pipelines/dask_preprocess.py`.

| Feature | Type | Source |
|---------|------|--------|
| `weather_severity` | float 0–10 | Computed from wind, gust, precip, temp, humid, pressure |
| `airport_congestion` | float 0–10 | Flight count per origin/hour ÷ 5, clipped |
| `departure_hour` | int 0–23 | flights.hour |
| `day_of_week` | int 1–7 | ISO: Mon=1, Sun=7 |
| `month` | int 1–12 | flights.month |
| `airline_code` | categorical | flights.carrier (AA, UA, DL, …) |
| `origin` | categorical | flights.origin (EWR, JFK, LGA, …) |
| `destination` | categorical | flights.dest |
| `route_avg_delay` | float ≥ 0 | Mean dep_delay per origin-dest pair |
| `carrier_avg_delay` | float ≥ 0 | Mean dep_delay per carrier |
| `distance` | float miles | flights.distance |

At inference time, `route_avg_delay`, `carrier_avg_delay`, and `distance` are resolved  
from `artifacts/delay_lookups.json` using the origin+destination key.

---

## Inference Flow

```
POST /predict  →  PredictionRequest (schemas.py)
    ↓
load_active_model()  (model_registry.py)
    ↓ looks up route_avg, carrier_avg, distance from delay_lookups.json
    ↓ extracts departure_hour from ISO datetime string
    ↓ builds 11-feature DataFrame row
    ↓
CatBoostRegressor.predict()
    ↓ clip to >= 0
    ↓
PredictionResponse { predicted_delay_minutes, model_name, inputs }
```

---

## Training Flow

```
train_catboost.py --use-dask
    ↓
dask_preprocess.py  →  data/processed/features.parquet
    ↓
pd.read_parquet()
    ↓
train_catboost_model()  (ml/train.py)
  80/20 split → CatBoostRegressor(1000 iter, depth=8, lr=0.05, early_stop=50)
    ↓
Save: models/catboost_delay_model.cbm
      artifacts/catboost_metrics.json
      artifacts/delay_lookups.json
      artifacts/training_analysis.json
```

---

## Architecture Rules

**1. Feature columns are the single source of truth.**  
`FEATURE_COLUMNS` in `ml/train.py` is the authoritative list. If you add a feature,  
update it there AND in `model_registry.py`'s `predict()` method AND in both preprocessors.

**2. Training and inference must use the same feature scales.**  
`weather_severity` is 0–10. `airport_congestion` is 0–10 (raw count ÷ 5).  
The Open-Meteo JS formula in `frontend/app.js` must match `_compute_weather_severity()`  
in both `preprocess.py` and `dask_preprocess.py`.

**3. Lookup tables connect training to inference.**  
`delay_lookups.json` is regenerated every retrain. It stores `route_avg`, `carrier_avg`,  
`overall_avg`, `route_distance`, `overall_distance`. Always fall back to `overall_avg`  
and `overall_distance` for unknown routes.

**4. Never add a feature at training time without handling it at inference time.**  
`model_registry.py → DelayModel.predict()` must always build a row with exactly  
the same features as `FEATURE_COLUMNS`. CatBoost will error on column count mismatch.

**5. Kafka and Cassandra are intentional stubs.**  
`services/kafka_consumer.py` and `services/cassandra_store.py` are scaffolding.  
Do not wire them into the main prediction path without a running broker/cluster.

**6. The frontend calls Open-Meteo directly — no proxy.**  
`fetchLiveWeather()` in `app.js` calls `api.open-meteo.com` from the browser.  
No API key required. No backend involvement. Keep it this way.

**7. The model hot-reloads on file change.**  
`model_registry.py` checks `catboost_delay_model.cbm` mtime on every request.  
After retraining, the running server picks up the new model automatically.

**8. All endpoints are defined in one file.**  
`api/app.py` is the only place endpoints are registered. Do not create new routers  
or split endpoints across files without good reason.

---

## What NOT to Do

- Do not add `dist` feature from `arr_delay` — it is data leakage (unknown before departure).
- Do not import `ml/features.py`, `ml/inference.py`, or `ml/evaluate.py` — they were  
  removed as unused. The equivalent logic lives in `train.py` and `model_registry.py`.
- Do not import `config/settings.py` — it was removed as unused. Configuration is  
  handled directly via environment variables in each module.
- Do not run `frontend/src/` or `frontend/public/` — they were removed (empty scaffolding).
- Do not commit `catboost_info/`, `artifacts/`, `models/`, or `data/` — they are gitignored.

---

## Model Performance (last training run)

| Metric | Value |
|--------|-------|
| MAE | 18.94 min |
| RMSE | 36.57 min |
| R² | 0.130 |
| F1 @15min | 0.479 |
| Precision @15min | 0.381 |
| Recall @15min | 0.643 |
| Best iteration | 988 / 1000 |

R² ~0.13 is near the ceiling for nycflights13 features.  
The dataset lacks tail-delay tracking and live ATC data, which are the primary predictors in production systems.