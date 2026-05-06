from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from catboost import CatBoostRegressor


@dataclass(frozen=True)
class TrainingResult:
    model_name: str
    metrics: dict[str, float]
    model_path: str
    metrics_path: str


FEATURE_COLUMNS = [
    "weather_severity",
    "airport_congestion",
    "departure_hour",
    "day_of_week",
    "month",
    "airline_code",
    "origin",
    "destination",
    "route_avg_delay",
    "carrier_avg_delay",
]
TARGET_COLUMN = "delay_minutes"
CATEGORICAL_COLUMNS = ["airline_code", "origin", "destination"]
DELAY_THRESHOLD = 15.0


def _build_delay_lookups(df: pd.DataFrame) -> dict[str, Any]:
    """Compute route and carrier average delays for use at inference time."""
    route_avg = (
        df.groupby(["origin", "destination"])["delay_minutes"]
        .mean()
        .clip(lower=0.0)
        .round(2)
    )
    carrier_avg = (
        df.groupby("airline_code")["delay_minutes"]
        .mean()
        .clip(lower=0.0)
        .round(2)
    )
    overall_avg = round(float(df["delay_minutes"].mean()), 2)

    route_dict = {f"{o}_{d}": float(v) for (o, d), v in route_avg.items()}
    carrier_dict = {k: float(v) for k, v in carrier_avg.items()}

    return {
        "route_avg": route_dict,
        "carrier_avg": carrier_dict,
        "overall_avg": overall_avg,
    }


def train_catboost_model(
    df: pd.DataFrame,
    models_dir: Path,
    artifacts_dir: Path,
    model_name: str = "catboost-delay-regressor",
) -> TrainingResult:
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required_columns - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Training dataframe is missing required columns: {missing_str}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(float)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    cat_features_idx = [X.columns.get_loc(col) for col in CATEGORICAL_COLUMNS]

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=1000,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
    )
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features_idx,
        eval_set=(X_valid, y_valid),
    )

    pred_valid = model.predict(X_valid)
    pred_valid = np.maximum(pred_valid, 0.0)

    y_pred_binary = (pred_valid >= DELAY_THRESHOLD).astype(int)
    y_true_binary = (y_valid.values >= DELAY_THRESHOLD).astype(int)

    nonzero = y_valid.values > 0
    mape = (
        float(np.mean(np.abs((y_valid.values[nonzero] - pred_valid[nonzero]) / y_valid.values[nonzero])) * 100)
        if nonzero.any()
        else 0.0
    )

    metrics: dict[str, Any] = {
        "sample_count": float(len(df)),
        "train_rows": float(len(X_train)),
        "valid_rows": float(len(X_valid)),
        "mae": float(mean_absolute_error(y_valid, pred_valid)),
        "rmse": float(root_mean_squared_error(y_valid, pred_valid)),
        "median_absolute_error": float(median_absolute_error(y_valid, pred_valid)),
        "r2": float(r2_score(y_valid, pred_valid)),
        "mape_percent": mape,
        "threshold_minutes": DELAY_THRESHOLD,
        "precision_at_threshold": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
        "f1_at_threshold": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
        "best_iteration": int(model.get_best_iteration() or model.tree_count_),
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "catboost_delay_model.cbm"
    metrics_path = artifacts_dir / "catboost_metrics.json"
    lookups_path = artifacts_dir / "delay_lookups.json"

    model.save_model(model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lookups = _build_delay_lookups(df)
    lookups_path.write_text(json.dumps(lookups, indent=2), encoding="utf-8")

    return TrainingResult(
        model_name=model_name,
        metrics={k: float(v) for k, v in metrics.items()},
        model_path=str(model_path),
        metrics_path=str(metrics_path),
    )
