from __future__ import annotations

from typing import Any

from ..services.model_registry import load_active_model


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    model = load_active_model()
    prediction = model.predict(record)
    return {
        "model_name": model.name,
        "predicted_delay_minutes": prediction,
    }
