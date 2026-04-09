# Flight Delay Prediction Platform

This project is a big data and machine learning scaffold for predicting flight delays from large-scale aviation, weather, and operational data.

## Main Components
- Kafka-based ingestion for streaming events
- Spark-based preprocessing and feature engineering
- Cassandra-backed storage for scalable persistence
- ML pipeline for training and evaluation
- FastAPI service for prediction requests
- Static HTML/CSS/JS frontend for submitting flight details and viewing predictions

## Layout
- `src/flight_delay_platform/api/`: API layer
- `src/flight_delay_platform/pipelines/`: ingestion and preprocessing jobs
- `src/flight_delay_platform/ml/`: feature engineering, training, evaluation, inference
- `src/flight_delay_platform/services/`: infrastructure adapters
- `frontend/`: browser UI served by FastAPI
- `configs/`: environment and deployment settings
- `tests/`: smoke and unit tests

## Run Target
Use the FastAPI app in `src/flight_delay_platform/api/app.py` as the service entry point.

## Run Project
From the project root, start the backend and serve the frontend from the same process:

```cmd
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn flight_delay_platform.api.app:app --reload --app-dir src
```

## Open The App
- Backend health check: `http://localhost:8000/health`
- API docs: `http://localhost:8000/docs`
- Frontend: `http://localhost:8000/`
