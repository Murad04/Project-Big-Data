# Flight Delay Prediction Platform

GWU · Big Data & Cloud Computing · Final Project

A full-stack machine learning application that predicts flight departure delays using CatBoost, Dask, and a real-time weather API. Built on nycflights13 (336K NYC flight records, 2013).

---

## What It Does

- User submits flight details → **LightGBM** predicts delay in minutes with a severity label
- **Batch prediction**: test 6 pre-loaded scenarios in one API call
- **Feature importance chart**: see which inputs drive predictions most

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data preprocessing | Pandas · **Dask** (parallel, no Java) |
| ML model | **CatBoost Regressor** · 11 features · 1000 iterations |
| API | **FastAPI** · 6 endpoints |
| Frontend | HTML · CSS · JavaScript · **Chart.js** |

---

## Project Structure

```
src/flight_delay_platform/
  api/          FastAPI app + schemas
  ml/           CatBoost training + evaluation
  pipelines/    preprocess.py · dask_preprocess.py · train_catboost.py
  services/     model_registry.py
frontend/       index.html · app.js · styles.css
models/         catboost_delay_model.cbm
artifacts/      catboost_metrics.json · delay_lookups.json
data/raw/       nyc_flights.csv · nyc_weather.csv
architecture.html   Animated architecture diagrams
```

---

## Quickstart

```powershell
# 1. Install dependencies
.\.venv\Scripts\pip.exe install -r requirements.txt

# 2. Train the model (Dask pipeline + CatBoost, ~4 min)
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m flight_delay_platform.pipelines.train_catboost --use-dask --max-rows 150000

# 3. Start the server
.\.venv\Scripts\uvicorn.exe flight_delay_platform.api.app:app --app-dir src --port 8000 --reload

# 4. Open browser
start http://localhost:8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Single flight delay prediction |
| POST | `/predict/batch` | Up to 200 flights in one call |
| GET | `/feature-importance` | CatBoost feature importance scores |
| GET | `/metrics` | MAE, RMSE, R², F1, Precision, Recall |
| GET | `/model-info` | Loaded model name and source |
| GET | `/health` | Service health check |

Interactive docs: **http://localhost:8000/docs**

---

## Model Features (11)

| Feature | Source | How |
|---------|--------|-----|
| `departure_hour` | flights.hour | Extracted from departure_time |
| `day_of_week` | flights.day | ISO 1=Mon, 7=Sun |
| `month` | flights.month | 1–12 |
| `airline_code` | flights.carrier | Categorical (AA, UA, DL, ...) |
| `origin` | flights.origin | Categorical (EWR, JFK, LGA, ...) |
| `destination` | flights.dest | Categorical |
| `weather_severity` | weather (5 cols) | 0–10: wind, gusts, precip, temp, humidity, pressure |
| `airport_congestion` | flights count | Flights/hr at origin, normalised 0–10 |
| `route_avg_delay` | dep_delay mean | Per origin-destination pair |
| `carrier_avg_delay` | dep_delay mean | Per airline |
| `distance` | flights.distance | Miles, looked up from route at inference |

---

## Model Performance

| Metric | Value |
|--------|-------|
| MAE | 18.94 min |
| RMSE | 36.57 min |
| R² | 0.130 |
| F1 @15min | 0.479 |
| Precision @15min | 0.381 |
| Recall @15min | 0.643 |

---

## Run Tests

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\pytest.exe tests/ -v
```

---

## Architecture

Open `architecture.html` in a browser for animated system diagrams.