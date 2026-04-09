from __future__ import annotations

from typing import Iterable


def evaluate_model(predictions: Iterable[float], targets: Iterable[float]) -> dict[str, float]:
    prediction_list = list(predictions)
    target_list = list(targets)
    if len(prediction_list) != len(target_list):
        raise ValueError("Predictions and targets must have the same length.")
    if not prediction_list:
        return {"mae": 0.0, "rmse": 0.0}

    errors = [abs(pred - target) for pred, target in zip(prediction_list, target_list)]
    squared_errors = [(pred - target) ** 2 for pred, target in zip(prediction_list, target_list)]
    mae = sum(errors) / len(errors)
    rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5
    return {"mae": mae, "rmse": rmse}
