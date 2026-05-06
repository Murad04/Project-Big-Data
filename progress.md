# Project Progress

GWU · Big Data & Cloud Computing · Final Project  
Last updated: 2026-05-06

---

## Status: Complete

All core components are built, tested, and running.

---

## What Is Done

### Data Pipeline
- [x] Raw data: `nyc_flights.csv` (336K rows) + `nyc_weather.csv` from nycflights13
- [x] Pandas preprocessing pipeline (`pipelines/preprocess.py`)
- [x] Dask parallel preprocessing pipeline (`pipelines/dask_preprocess.py`) — no Java required
- [x] 11 features engineered: departure_hour, weather_severity (enhanced), airport_congestion, day_of_week, month, airline_code, origin, destination, route_avg_delay, carrier_avg_delay, distance
- [x] Fixed feature scale mismatch: weather_severity 0–10, airport_congestion 0–10
- [x] Fixed sampling: random sample instead of head() — covers all seasons
- [x] Enhanced weather_severity formula using wind_gust, temp, humid, pressure, precip

### ML Model
- [x] CatBoost Regressor — 3000 iterations, depth=6, lr=0.05, early stopping=100
- [x] 80/20 train-validation split
- [x] Fixed target-encoding leakage: route/carrier averages now computed from training split only
- [x] Regularisation improved: l2_leaf_reg=5, min_data_in_leaf=20, subsample=0.8, colsample_bylevel=0.8
- [x] max_iterations saved to catboost_metrics.json and displayed dynamically on training logs page
- [x] Full evaluation: MAE 18.93, RMSE 36.58, Median AE 10.26, R² 0.130, F1 0.477 @15min
- [x] Route/carrier average delay lookup tables saved at training time
- [x] Route distance lookup table saved at training time
- [x] Comprehensive training analysis saved: confusion matrix, error histogram, per-airline/month/hour MAE, scatter sample, PR by threshold

### API — FastAPI (:8000)
- [x] `POST /predict` — single flight prediction with severity label
- [x] `POST /predict/batch` — up to 200 flights in one call
- [x] `GET /feature-importance` — CatBoost importance scores
- [x] `GET /metrics` — all evaluation metrics
- [x] `GET /training-logs` — RMSE per iteration (train + val)
- [x] `GET /training-analysis` — full breakdown analysis
- [x] `GET /model-info` — loaded model name and source
- [x] `GET /health` — service health check
- [x] `GET /training-logs-view` — serves training logs HTML page
- [x] Model hot-reload on file change (no restart needed after retraining)

### Web Frontend (http://localhost:8000)
- [x] Flight prediction form with airline/airport dropdowns
- [x] Live weather auto-fetch from Open-Meteo API (no API key, browser-side)
- [x] Weather severity auto-computed and fills slider using same formula as training
- [x] Severity ring: On Time / Minor / Moderate / Significant / Major (color-coded)
- [x] Prediction history (last 8 predictions)
- [x] Batch prediction table (6 pre-loaded test scenarios)
- [x] Feature importance horizontal bar chart (Chart.js)
- [x] Model performance metrics panel (6 cards)

### Training Logs Page (http://localhost:8000/training-logs-view)
- [x] 10 metric cards: MAE, RMSE, Median AE, R², MAPE, F1, Precision, Recall, Best Iter, Train Time
- [x] RMSE per iteration chart (train + validation curves)
- [x] Overfitting gap chart (val − train RMSE)
- [x] Training speed chart
- [x] Prediction error distribution histogram
- [x] Predicted vs actual scatter plot (1000-sample)
- [x] Confusion matrix (2×2 color-coded)
- [x] Precision/Recall/F1 at 5/10/15/30/60-min thresholds
- [x] Per-airline MAE horizontal bar chart
- [x] Per-month MAE chart (seasonal analysis)
- [x] Per-hour MAE chart (time-of-day analysis)
- [x] Iteration log table (every 50th iteration, best row highlighted)

### Architecture & Documentation
- [x] `architecture.html` — animated 5-diagram architecture with flowing dots
- [x] `CLAUDE.md` — architecture rules, feature definitions, what not to do
- [x] `.claudeignore` — tells Claude to skip large/generated files
- [x] `README.md` — complete quickstart, API reference, feature table, metrics
- [x] `report.md` — detailed project report

### Infrastructure Stubs
- [x] `services/kafka_consumer.py` — Kafka consumer config stub (not wired)
- [x] `services/cassandra_store.py` — Cassandra store stub (not wired)

---

## Model Performance

| Metric | Value | Notes |
|--------|-------|-------|
| MAE | 18.94 min | Average error |
| RMSE | 36.57 min | Penalizes large errors |
| Median AE | 10.34 min | 50% of predictions within 10 min |
| R² | 0.130 | ~13% variance explained |
| F1 @15min | 0.479 | Binary delay classification |
| Precision @15min | 0.381 | 38% of predicted delays are real |
| Recall @15min | 0.643 | Catches 64% of real delays |
| Best iteration | 988 / 1000 | Early stopping fired at 988 |

**Note**: R² ~0.13 is near the ceiling for nycflights13 features. The dataset lacks tail-delay tracking (same aircraft's prior flight delay) and live ATC data, which are the primary predictors in production systems.

---

## Known Limitations

| Issue | Cause | Impact |
|-------|-------|--------|
| Low R² ceiling (~0.13) | nycflights13 lacks tail-delay tracking and ATC data | Fundamental ceiling, not a model quality issue |
| Poor large-delay prediction | Rare events, no tail-delay feature | Model underestimates 100+ min delays |
| Data from 2013 only | nycflights13 is a static historical dataset | Modern airline patterns not represented |

---

## How to Run

```powershell
# Train (Dask + CatBoost, ~4 min)
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m flight_delay_platform.pipelines.train_catboost --use-dask --max-rows 150000

# Start server
.\.venv\Scripts\uvicorn.exe flight_delay_platform.api.app:app --app-dir src --port 8000 --reload

# Tests
.\.venv\Scripts\pytest.exe tests/ -v
```

URLs:
- Website: http://localhost:8000
- Training logs: http://localhost:8000/training-logs-view
- API docs: http://localhost:8000/docs