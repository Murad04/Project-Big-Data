from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .schemas import HealthResponse, PredictionRequest, PredictionResponse
from ..services.model_registry import load_active_model

app = FastAPI(
    title="Flight Delay Prediction Platform",
    version="0.1.0",
    description="API for predicting flight delay minutes from streaming aviation data.",
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


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    model = load_active_model()
    predicted_delay = model.predict(request.to_features())
    return PredictionResponse(
        predicted_delay_minutes=predicted_delay,
        model_name=model.name,
        inputs=request.to_features(),
    )
