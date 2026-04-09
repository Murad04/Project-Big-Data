from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from catboost import CatBoostRegressor
except ModuleNotFoundError:  # pragma: no cover
    CatBoostRegressor = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DelayModel:
    name: str
    trained_model: Any | None = None

    def predict(self, features: dict[str, Any]) -> float:
        if self.trained_model is not None:
            row = {
                "weather_severity": float(features.get("weather_severity", 0.0)),
                "airport_congestion": float(features.get("airport_congestion", 0.0)),
                "day_of_week": int(features.get("day_of_week", 1)),
                "month": int(features.get("month", 1)),
                "airline_code": str(features.get("airline_code", "UNKNOWN")),
                "origin": str(features.get("origin", "UNK")),
                "destination": str(features.get("destination", "UNK")),
            }
            df = pd.DataFrame([row])
            prediction = self.trained_model.predict(df)[0]
            return round(float(max(prediction, 0.0)), 2)

        weather = float(features.get("weather_severity", 0.0))
        congestion = float(features.get("airport_congestion", 0.0))
        day_of_week = float(features.get("day_of_week", 1))
        month = float(features.get("month", 1))
        base_delay = weather * 12.0 + congestion * 18.0
        calendar_adjustment = (day_of_week % 7) + (month / 12.0)
        return round(base_delay + calendar_adjustment, 2)


def _load_trained_model() -> DelayModel | None:
    if CatBoostRegressor is None:
        return None

    model_path = Path(__file__).resolve().parents[3] / "models" / "catboost_delay_model.cbm"
    if not model_path.exists():
        return None

    trained_model = CatBoostRegressor()
    trained_model.load_model(model_path)
    return DelayModel(name="catboost-delay-regressor", trained_model=trained_model)


_ACTIVE_MODEL = _load_trained_model() or DelayModel(name="baseline-rule-model")


def load_active_model() -> DelayModel:
    return _ACTIVE_MODEL
