import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .schemas import (
    BatchPredictionItem,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from ..services.model_registry import load_active_model, model_status

app = FastAPI(
    title="Flight Delay Prediction Platform",
    version="0.2.0",
    description="Predicts flight delay minutes from weather, congestion, and flight context.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FRONTEND_DIR = PROJECT_ROOT / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


def _classify_severity(minutes: float) -> str:
    if minutes <= 0:
        return "On Time"
    if minutes <= 15:
        return "Minor Delay"
    if minutes <= 30:
        return "Moderate Delay"
    if minutes <= 60:
        return "Significant Delay"
    return "Major Delay"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/training-logs-view")
def training_logs_view() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "training_logs.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    model = load_active_model()
    feats = request.to_features()
    return PredictionResponse(
        predicted_delay_minutes=model.predict(feats),
        model_name=model.name,
        inputs=feats,
        feature_contributions=model.explain(feats),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    if not request.flights:
        raise HTTPException(status_code=422, detail="flights list must not be empty")
    if len(request.flights) > 200:
        raise HTTPException(status_code=422, detail="Maximum 200 flights per batch request")

    model = load_active_model()
    predictions: list[BatchPredictionItem] = []
    for flight in request.flights:
        delay = model.predict(flight.to_features())
        predictions.append(
            BatchPredictionItem(
                flight_number=flight.flight_number,
                airline_code=flight.airline_code,
                origin=flight.origin,
                destination=flight.destination,
                predicted_delay_minutes=delay,
                severity=_classify_severity(delay),
            )
        )
    return BatchPredictionResponse(
        predictions=predictions,
        model_name=model.name,
        count=len(predictions),
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def get_model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**model_status())


@app.get("/metrics")
def get_metrics() -> dict:
    metrics_path = PROJECT_ROOT / "artifacts" / "catboost_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found")
    with open(metrics_path) as f:
        return json.load(f)


@app.get("/training-logs")
def get_training_logs() -> dict:
    import csv
    learn_path = PROJECT_ROOT / "catboost_info" / "learn_error.tsv"
    test_path  = PROJECT_ROOT / "catboost_info" / "test_error.tsv"
    time_path  = PROJECT_ROOT / "catboost_info" / "time_left.tsv"
    if not learn_path.exists():
        raise HTTPException(status_code=404, detail="Training logs not found — retrain the model first")

    def read_tsv(path, step=5):
        rows = []
        with open(path, newline="") as f:
            for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
                if i % step == 0 or i == 0:
                    rows.append({"iter": int(row["iter"]), "rmse": round(float(row["RMSE"]), 4)})
        return rows

    result = {"train": read_tsv(learn_path), "validation": read_tsv(test_path)}
    if time_path.exists():
        with open(time_path, newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
            if rows:
                result["total_seconds"] = round(float(rows[-1].get("Passed", 0)), 1)
    return result


@app.get("/training-analysis")
def get_training_analysis() -> dict:
    path = PROJECT_ROOT / "artifacts" / "training_analysis.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Training analysis not found — retrain the model first")
    with open(path) as f:
        return json.load(f)


@app.get("/optimal-threshold")
def get_optimal_threshold() -> dict:
    path = PROJECT_ROOT / "artifacts" / "training_analysis.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Training analysis not found — retrain the model first")
    with open(path) as f:
        analysis = json.load(f)
    pr_data = analysis.get("precision_recall_by_threshold", [])
    best = max(pr_data, key=lambda x: x["f1"], default=None)
    return {
        "optimal_threshold_minutes": best["threshold"] if best else 15,
        "best_f1": round(best["f1"], 3) if best else None,
        "data": pr_data,
    }


@app.get("/feature-importance")
def get_feature_importance() -> list[dict]:
    model = load_active_model()
    if model.trained_model is None:
        raise HTTPException(status_code=404, detail="No trained model is loaded")
    importances = model.trained_model.get_feature_importance()
    feature_names = model.trained_model.feature_names_
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return [{"feature": name, "importance": round(float(imp), 4)} for name, imp in pairs]