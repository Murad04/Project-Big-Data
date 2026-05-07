# CLAUDE.md — Flight Delay Prediction Platform

GWU · Big Data & Cloud Computing · Final Project  
Stack: LightGBM · Dask · FastAPI · Chart.js

---

## What This Project Does

Predicts flight departure delay in minutes given route, schedule, and weather conditions.  
Training data: BTS On-Time Performance 2023 (6.76 M US flight records, all airports).

---

## Project Layout

```
src/flight_delay_platform/
  api/
    app.py          ← FastAPI app — all endpoints defined here
    schemas.py      ← Pydantic request/response models
  ml/
    train.py        ← LightGBM training, evaluation, analysis — single source of truth
  pipelines/
    preprocess.py       ← Pandas preprocessing (nycflights13 path)
    dask_preprocess.py  ← Dask preprocessing (--use-dask flag, nycflights13)
    bts_preprocess.py   ← BTS On-Time Performance preprocessor (production path)
    train_lgb.py   ← CLI entry point for training
  services/
    model_registry.py   ← Loads model, resolves lookup features at inference time

frontend/
  index.html    ← Single-page app
  app.js        ← All frontend logic (charts, prediction, history)
  styles.css    ← Styling

data/raw/           ← bts_flights.csv (gitignored)
data/processed/     ← bts_features.parquet (gitignored)
models/             ← lgb_delay_model.txt (gitignored)
artifacts/          ← lgb_metrics.json, delay_lookups.json, label_encoders.json,
                       lgb_training_logs.json, training_analysis.json,
                       feature_columns.json (gitignored)

architecture.html   ← Animated architecture diagrams (open in browser)
training_logs.html  ← Training curves + analysis (served at /training-logs-view)
```

---

## How to Run

```powershell
# Start server (model must be trained first)
$env:PYTHONPATH = "src"
.\.venv\Scripts\uvicorn.exe flight_delay_platform.api.app:app --app-dir src --port 8000 --reload

# Train model on BTS data (~10 min preprocess + ~60 min training)
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m flight_delay_platform.pipelines.bts_preprocess --input data/raw/bts_flights.csv --output data/processed/bts_features.parquet
.\.venv\Scripts\python.exe -m flight_delay_platform.pipelines.train_lgb --processed-data data/processed/bts_features.parquet

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
| GET | `/architecture-view` | Architecture diagrams HTML page |
| POST | `/predict` | Single flight delay prediction |
| POST | `/predict/batch` | Up to 200 flights per call |
| GET | `/feature-importance` | LightGBM gain importance scores (sorted) |
| GET | `/metrics` | MAE, RMSE, R², F1, Precision, Recall |
| GET | `/training-logs` | RMSE per iteration (train + val) |
| GET | `/training-analysis` | Confusion matrix, per-airline/month/hour MAE |
| GET | `/model-info` | Model name and source path |
| GET | `/health` | `{"status": "ok"}` |

---

## The 18 Model Features

Defined in `FEATURE_COLUMNS` + `BTS_FEATURE_COLUMNS` inside `src/flight_delay_platform/ml/train.py`.  
The active feature list for a given model run is saved to `artifacts/feature_columns.json`.

| Feature | Type | Source |
|---------|------|--------|
| `airport_congestion` | float 0–10 | Flight count per origin/hour ÷ 5, clipped |
| `departure_hour` | int 0–23 | CRSDepTime HHMM → hour |
| `day_of_week` | int 1–7 | ISO: Mon=1, Sun=7 |
| `month` | int 1–12 | FlightDate.month |
| `quarter` | int 1–4 | Derived from month: Q1=winter, Q2=spring, Q3=summer, Q4=fall |
| `is_weekend` | int 0/1 | day_of_week >= 6 |
| `is_peak_hour` | int 0/1 | departure_hour in {7,8,9,16,17,18,19} |
| `airline_code` | categorical | Reporting_Airline (AA, UA, DL, …) |
| `origin` | categorical | Origin airport code |
| `destination` | categorical | Dest airport code |
| `route_avg_delay` | float ≥ 0 | Mean dep_delay per origin-dest pair |
| `carrier_avg_delay` | float ≥ 0 | Mean dep_delay per carrier |
| `distance` | float miles | Distance |
| `prev_tail_delay` | float ≥ 0 | Previous flight delay for same tail number (key feature) |
| `late_aircraft_flag` | int 0/1 | prev_tail_delay > 15 min |
| `weather_delay` | float ≥ 0 | BTS-reported weather delay minutes (0 at inference) |
| `carrier_delay` | float ≥ 0 | BTS-reported carrier/mechanical delay (0 at inference) |
| `nas_delay` | float ≥ 0 | BTS-reported ATC/ground-stop delay (0 at inference) |

At inference time, `route_avg_delay`, `carrier_avg_delay`, and `distance` are resolved  
from `artifacts/delay_lookups.json`. BTS delay-cause features default to 0 (unknown before flight).

Categoricals (`airline_code`, `origin`, `destination`) are label-encoded to integers at  
training time. The mapping is saved to `artifacts/label_encoders.json` and applied at  
inference time in `model_registry.py`.

---

## Inference Flow

```
POST /predict  →  PredictionRequest (schemas.py)
    ↓
load_active_model()  (model_registry.py)
    ↓ looks up route_avg, carrier_avg, distance from delay_lookups.json
    ↓ extracts departure_hour from ISO datetime string
    ↓ derives quarter, is_weekend, is_peak_hour
    ↓ BTS delay-cause features default to 0 (unknown pre-departure)
    ↓ label-encodes airline_code, origin, destination via label_encoders.json
    ↓ filters to active feature_columns.json list → 18-feature DataFrame row
    ↓
lgb.Booster.predict()
    ↓ clip to >= 0
    ↓
PredictionResponse { predicted_delay_minutes, model_name, inputs }
```

---

## Training Flow

```
bts_preprocess.py --input data/raw/bts_flights.csv
    ↓ tail-delay sort + shift (prev_tail_delay)
    ↓ airport congestion, route/carrier averages
    ↓
data/processed/bts_features.parquet
    ↓
train_lgb.py --processed-data bts_features.parquet
    ↓
train_model()  (ml/train.py)
  auto-detects 18 BTS features
  80/20 stratified split → lgb.train(3000 rounds, num_leaves=63, lr=0.05, early_stop=100)
    ↓
Save: models/lgb_delay_model.txt
      artifacts/lgb_metrics.json
      artifacts/delay_lookups.json
      artifacts/label_encoders.json
      artifacts/lgb_training_logs.json
      artifacts/training_analysis.json
      artifacts/feature_columns.json
```

---

## Architecture Rules

**1. Feature columns are the single source of truth.**  
`FEATURE_COLUMNS` + `BTS_FEATURE_COLUMNS` in `ml/train.py` define all possible features.  
The active set for each run is saved to `artifacts/feature_columns.json`. If you add a feature,  
update `train.py`, `model_registry.py`'s `_build_row()`, and the relevant preprocessor.

**2. Active feature list must match the loaded model.**  
`model_registry.py` loads `feature_columns.json` alongside the model. They must always  
come from the same training run. Hot-reload reloads both together.

**3. Lookup tables connect training to inference.**  
`delay_lookups.json` is regenerated every retrain. It stores `route_avg`, `carrier_avg`,  
`overall_avg`, `route_distance`, `overall_distance`. Always fall back to `overall_avg`  
and `overall_distance` for unknown routes.

**4. Label encoders must stay in sync with the model.**  
`label_encoders.json` maps airline/airport string codes → integers for LightGBM.  
It is regenerated every retrain from training-split data. `model_registry.py` loads  
it alongside the model; they must always come from the same training run.

**5. BTS delay-cause features default to 0 at inference.**  
`weather_delay`, `carrier_delay`, `nas_delay` are only known post-flight.  
At inference time they are set to 0. The model compensates via `route_avg_delay`  
and `carrier_avg_delay` which embed historical delay patterns.

**6. All endpoints are defined in one file.**  
`api/app.py` is the only place endpoints are registered. Do not create new routers  
or split endpoints across files without good reason.

**7. The model hot-reloads on file change.**  
`model_registry.py` checks `lgb_delay_model.txt` mtime on every request.  
After retraining, the running server picks up the new model automatically.

---

## What NOT to Do

- Do not add `arr_delay` as a feature — it is data leakage (unknown before departure).
- Do not add current-flight `WEATHER_DELAY`, `NAS_DELAY`, or `CARRIER_DELAY` directly — leakage.
- Do not import `ml/features.py`, `ml/inference.py`, or `ml/evaluate.py` — they were  
  removed as unused. The equivalent logic lives in `train.py` and `model_registry.py`.
- Do not import `config/settings.py` — it was removed as unused. Configuration is  
  handled directly via environment variables in each module.
- Do not run `frontend/src/` or `frontend/public/` — they were removed (empty scaffolding).
- Do not commit `artifacts/`, `models/`, or `data/` — they are gitignored.

---

## Model Performance (last training run)

**Dataset: BTS On-Time Performance 2023 (transtats.bts.gov) — full year, all US airports**

| Metric | Value |
|--------|-------|
| MAE | 8.10 min |
| RMSE (validation) | 27.02 min |
| RMSE (train) | 25.14 min |
| Train/Val gap | 1.88 min |
| Median AE | 3.00 min |
| R² | 0.750 |
| F1 @15min | 0.786 |
| Precision @15min | 0.819 |
| Recall @15min | 0.756 |
| Best iteration | 2257 / 3000 |
| Training rows | 5,410,696 |
| Validation rows | 1,352,674 |

R² 0.75 is production-quality. The key upgrade over nycflights13 was `prev_tail_delay`
(previous flight delay for the same aircraft), which captures tail-delay propagation —
the single largest predictor of departure delays in real-world systems.
