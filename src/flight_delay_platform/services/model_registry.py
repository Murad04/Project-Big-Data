from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from catboost import CatBoostRegressor
except ModuleNotFoundError:  # pragma: no cover
    CatBoostRegressor = None  # type: ignore[assignment]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_delay_lookups() -> dict[str, Any]:
    path = _project_root() / "artifacts" / "delay_lookups.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"route_avg": {}, "carrier_avg": {}, "overall_avg": 15.0}


def _extract_hour(departure_time: str) -> int:
    """Parse hour from ISO datetime strings like '2026-05-05T12:00' or '12:00'."""
    try:
        time_part = departure_time.split("T")[-1]
        return int(time_part.split(":")[0])
    except (ValueError, IndexError, AttributeError):
        return 12


@dataclass(frozen=True)
class DelayModel:
    name: str
    trained_model: Any | None = None
    source: str = "baseline"

    def predict(self, features: dict[str, Any]) -> float:
        if self.trained_model is not None:
            origin      = str(features.get("origin", "UNK"))
            destination = str(features.get("destination", "UNK"))
            airline     = str(features.get("airline_code", "UNKNOWN"))
            dep_time    = str(features.get("departure_time", "12:00"))

            route_key   = f"{origin}_{destination}"
            overall_avg = float(_DELAY_LOOKUPS.get("overall_avg", 15.0))
            route_avg   = float(_DELAY_LOOKUPS["route_avg"].get(route_key, overall_avg))
            carrier_avg = float(_DELAY_LOOKUPS["carrier_avg"].get(airline, overall_avg))

            row = {
                "weather_severity":  float(features.get("weather_severity", 0.0)),
                "airport_congestion": float(features.get("airport_congestion", 0.0)),
                "departure_hour":    _extract_hour(dep_time),
                "day_of_week":       int(features.get("day_of_week", 1)),
                "month":             int(features.get("month", 1)),
                "airline_code":      airline,
                "origin":            origin,
                "destination":       destination,
                "route_avg_delay":   route_avg,
                "carrier_avg_delay": carrier_avg,
            }
            df = pd.DataFrame([row])
            prediction = self.trained_model.predict(df)[0]
            return round(float(max(prediction, 0.0)), 2)

        # Baseline fallback (no trained model)
        weather    = float(features.get("weather_severity", 0.0))
        congestion = float(features.get("airport_congestion", 0.0))
        day_of_week = float(features.get("day_of_week", 1))
        month       = float(features.get("month", 1))
        base_delay  = weather * 1.2 + congestion * 1.8
        calendar_adjustment = (day_of_week % 7) + (month / 12.0)
        return round(base_delay + calendar_adjustment, 2)


def _load_trained_model() -> DelayModel | None:
    if CatBoostRegressor is None:
        return None

    model_path = _project_root() / "models" / "catboost_delay_model.cbm"
    if not model_path.exists():
        return None

    trained_model = CatBoostRegressor()
    trained_model.load_model(model_path)
    return DelayModel(
        name="catboost-delay-regressor",
        trained_model=trained_model,
        source=str(model_path),
    )


_DELAY_LOOKUPS: dict[str, Any] = _load_delay_lookups()
_ACTIVE_MODEL: DelayModel = _load_trained_model() or DelayModel(name="baseline-rule-model", source="baseline")
_ACTIVE_MODEL_MTIME: float | None = None


def _model_path() -> Path:
    return _project_root() / "models" / "catboost_delay_model.cbm"


def _needs_reload(path: Path) -> bool:
    global _ACTIVE_MODEL_MTIME

    if not path.exists():
        return False

    current_mtime = path.stat().st_mtime
    if _ACTIVE_MODEL_MTIME is None:
        _ACTIVE_MODEL_MTIME = current_mtime
        return _ACTIVE_MODEL.trained_model is None

    if current_mtime > _ACTIVE_MODEL_MTIME:
        _ACTIVE_MODEL_MTIME = current_mtime
        return True

    return _ACTIVE_MODEL.trained_model is None


def load_active_model() -> DelayModel:
    global _ACTIVE_MODEL, _DELAY_LOOKUPS

    trained_path = _model_path()
    if _needs_reload(trained_path):
        # Reload lookups too whenever the model is refreshed
        _DELAY_LOOKUPS = _load_delay_lookups()
        refreshed = _load_trained_model()
        if refreshed is not None:
            _ACTIVE_MODEL = refreshed

    return _ACTIVE_MODEL


def model_status() -> dict[str, Any]:
    model = load_active_model()
    return {
        "model_name": model.name,
        "is_trained_model": model.trained_model is not None,
        "source": model.source,
    }
