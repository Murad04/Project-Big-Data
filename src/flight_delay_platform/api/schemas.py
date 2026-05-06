from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    service: str = Field(default="flight-delay-prediction-api")


class PredictionRequest(BaseModel):
    flight_number: str
    origin: str
    destination: str
    departure_time: str
    weather_severity: float = 0.0
    airport_congestion: float = 0.0
    day_of_week: int = Field(ge=1, le=7)
    month: int = Field(ge=1, le=12)
    airline_code: str

    def to_features(self) -> dict[str, Any]:
        return self.model_dump()


class PredictionResponse(BaseModel):
    predicted_delay_minutes: float
    model_name: str
    inputs: dict[str, Any]


class ModelInfoResponse(BaseModel):
    model_name: str
    is_trained_model: bool
    source: str


class BatchFlightInput(BaseModel):
    flight_number: str
    airline_code: str
    origin: str
    destination: str
    departure_time: str
    weather_severity: float = 0.0
    airport_congestion: float = 0.0
    day_of_week: int = Field(ge=1, le=7)
    month: int = Field(ge=1, le=12)

    def to_features(self) -> dict[str, Any]:
        return self.model_dump()


class BatchPredictionRequest(BaseModel):
    flights: list[BatchFlightInput]


class BatchPredictionItem(BaseModel):
    flight_number: str
    airline_code: str
    origin: str
    destination: str
    predicted_delay_minutes: float
    severity: str


class BatchPredictionResponse(BaseModel):
    predictions: list[BatchPredictionItem]
    model_name: str
    count: int