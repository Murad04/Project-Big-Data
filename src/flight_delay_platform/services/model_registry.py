from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import lightgbm as lgb
except ModuleNotFoundError:
    lgb = None  # type: ignore[assignment]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_delay_lookups() -> dict[str, Any]:
    path = _project_root() / "artifacts" / "delay_lookups.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"route_avg": {}, "carrier_avg": {}, "overall_avg": 15.0}


def _load_label_encoders() -> dict[str, dict[str, int]]:
    path = _project_root() / "artifacts" / "label_encoders.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _load_feature_columns() -> list[str]:
    path = _project_root() / "artifacts" / "feature_columns.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # fallback: base features if feature_columns.json not present
    return [
        "weather_severity", "airport_congestion", "departure_hour",
        "day_of_week", "month", "quarter", "is_weekend", "is_peak_hour",
        "airline_code", "origin", "destination",
        "route_avg_delay", "carrier_avg_delay", "distance",
    ]


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

    def _build_row(self, features: dict[str, Any]) -> pd.DataFrame:
        origin      = str(features.get("origin", "UNK"))
        destination = str(features.get("destination", "UNK"))
        airline     = str(features.get("airline_code", "UNKNOWN"))
        dep_time    = str(features.get("departure_time", "12:00"))
        dep_hour    = _extract_hour(dep_time)
        dow         = int(features.get("day_of_week", 1))

        route_key   = f"{origin}_{destination}"
        overall_avg = float(_DELAY_LOOKUPS.get("overall_avg", 15.0))
        route_avg   = float(_DELAY_LOOKUPS["route_avg"].get(route_key, overall_avg))
        carrier_avg = float(_DELAY_LOOKUPS["carrier_avg"].get(airline, overall_avg))
        distance    = float(_DELAY_LOOKUPS.get("route_distance", {}).get(
                          route_key, _DELAY_LOOKUPS.get("overall_distance", 1000.0)))

        month = int(features.get("month", 1))
        row = {
            # ── base features ────────────────────────────────────────────────
            "weather_severity":   float(features.get("weather_severity", 0.0)),
            "airport_congestion": float(features.get("airport_congestion", 0.0)),
            "departure_hour":     dep_hour,
            "day_of_week":        dow,
            "month":              month,
            "quarter":            (month - 1) // 3 + 1,
            "is_weekend":         int(dow >= 6),
            "is_peak_hour":       int(dep_hour in (7, 8, 9, 16, 17, 18, 19)),
            "airline_code":       airline,
            "origin":             origin,
            "destination":        destination,
            "route_avg_delay":    route_avg,
            "carrier_avg_delay":  carrier_avg,
            "distance":           distance,
            # ── BTS features (default 0 = no known previous delay / no cause data)
            "prev_tail_delay":    float(features.get("prev_tail_delay", 0.0)),
            "late_aircraft_flag": int(features.get("late_aircraft_flag", 0)),
            "weather_delay":      float(features.get("weather_delay", 0.0)),
            "carrier_delay":      float(features.get("carrier_delay", 0.0)),
            "nas_delay":          float(features.get("nas_delay", 0.0)),
        }

        df = pd.DataFrame([row])

        # Keep only columns the model was actually trained on
        df = df[[c for c in _FEATURE_COLUMNS if c in df.columns]]

        # Apply label encoding to match training-time integer codes
        for col, mapping in _LABEL_ENCODERS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(len(mapping)).astype(int)

        return df

    def predict(self, features: dict[str, Any]) -> float:
        if self.trained_model is not None:
            df = self._build_row(features)
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

    def explain(self, features: dict[str, Any]) -> dict[str, float]:
        """Return SHAP feature contributions (minutes) for a single prediction."""
        if self.trained_model is None:
            return {}
        try:
            df = self._build_row(features)
            # pred_contrib returns (n_samples, n_features + 1); last col is expected value
            contribs = self.trained_model.predict(df, pred_contrib=True)
            return {
                col: round(float(contribs[0][i]), 2)
                for i, col in enumerate(df.columns)
            }
        except Exception:
            return {}

    def feature_importance(self) -> list[tuple[str, float]]:
        if self.trained_model is None:
            return []
        names = self.trained_model.feature_name()
        scores = self.trained_model.feature_importance(importance_type="gain").tolist()
        return list(zip(names, [float(s) for s in scores]))


def _load_trained_model() -> DelayModel | None:
    if lgb is None:
        return None

    model_path = _project_root() / "models" / "lgb_delay_model.txt"
    if not model_path.exists():
        return None

    trained_model = lgb.Booster(model_file=str(model_path))
    return DelayModel(
        name="lgb-delay-regressor",
        trained_model=trained_model,
        source=str(model_path),
    )


_DELAY_LOOKUPS: dict[str, Any] = _load_delay_lookups()
_LABEL_ENCODERS: dict[str, dict[str, int]] = _load_label_encoders()
_FEATURE_COLUMNS: list[str] = _load_feature_columns()
_ACTIVE_MODEL: DelayModel = _load_trained_model() or DelayModel(name="baseline-rule-model", source="baseline")
_ACTIVE_MODEL_MTIME: float | None = None


def _model_path() -> Path:
    return _project_root() / "models" / "lgb_delay_model.txt"


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
    global _ACTIVE_MODEL, _DELAY_LOOKUPS, _LABEL_ENCODERS, _FEATURE_COLUMNS

    trained_path = _model_path()
    if _needs_reload(trained_path):
        _DELAY_LOOKUPS   = _load_delay_lookups()
        _LABEL_ENCODERS  = _load_label_encoders()
        _FEATURE_COLUMNS = _load_feature_columns()
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
