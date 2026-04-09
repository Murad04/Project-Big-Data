# Copilot Architecture Guide

## Project Goal
Build a flight delay prediction platform for big data and machine learning. The design is driven by the project brief: large-scale aviation records, streaming ingestion, real-time weather signals, distributed processing, scalable storage, and predictive modeling.

## Core Architecture
- Ingestion layer: Apache Kafka for real-time event streams from flight, weather, airport, and operations sources.
- Processing layer: Apache Spark for cleaning, joining, feature generation, and batch/stream transformations.
- Storage layer: Apache Cassandra for high-volume, low-latency persistence of raw events, curated records, and predictions.
- ML layer: feature engineering, training, validation, and scoring for models such as Random Forest, XGBoost, Gradient Boosted Trees, and neural networks.
- API layer: FastAPI service for health checks and prediction requests.
- Frontend layer: static HTML/CSS/JS dashboard for submitting flight records and displaying prediction results.
- Orchestration layer: repeatable pipelines for ingestion, preprocessing, training, and evaluation.

## Repository Layout
- `src/flight_delay_platform/api/`: FastAPI app, request and response schemas.
- `src/flight_delay_platform/config/`: runtime settings and environment configuration.
- `src/flight_delay_platform/pipelines/`: ingestion and preprocessing pipeline entry points.
- `src/flight_delay_platform/ml/`: feature logic, training, evaluation, and inference helpers.
- `src/flight_delay_platform/services/`: Kafka, Cassandra, and model registry abstractions.
- `frontend/`: browser UI files served by FastAPI.
- `configs/`: deployment and environment configuration files.
- `tests/`: smoke tests and future unit tests.
- `data/raw/`, `data/processed/`, `models/`, `artifacts/`, `logs/`: generated outputs and runtime data.

## Data Flow
1. Flight, weather, and operational events land in Kafka topics.
2. Spark jobs validate and normalize the incoming records.
3. Curated records are written to Cassandra and optionally exported to processed datasets.
4. Feature engineering combines temporal, weather, airport congestion, and airline-specific signals.
5. Models are trained, evaluated, and persisted.
6. The API loads the active model and returns delay predictions.
7. The frontend submits user input to the API and renders the returned delay.

## Modeling Approach
- Start with interpretable baselines for validation.
- Train tree-based ensembles first because they handle heterogeneous feature sets well.
- Add neural models only after the feature pipeline is stable.
- Track metrics such as MAE, RMSE, AUC for delay classification, and latency for online scoring.

## Working Rules
- Keep generated data and local environment folders out of Copilot context.
- Prefer small, composable modules over a single monolithic pipeline.
- Keep interfaces stable between ingestion, feature engineering, and inference.
- Preserve the architecture described here when extending the codebase.
- Prefer Python-only startup for local development; no npm-based frontend build is required.
