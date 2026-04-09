from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
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
    "day_of_week",
    "month",
    "airline_code",
    "origin",
    "destination",
]
TARGET_COLUMN = "delay_minutes"
CATEGORICAL_COLUMNS = ["airline_code", "origin", "destination"]


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
        iterations=600,
        depth=8,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
    )
    model.fit(X_train, y_train, cat_features=cat_features_idx)

    pred_valid = model.predict(X_valid)
    metrics: dict[str, Any] = {
        "sample_count": float(len(df)),
        "train_rows": float(len(X_train)),
        "valid_rows": float(len(X_valid)),
        "mae": float(mean_absolute_error(y_valid, pred_valid)),
        "rmse": float(root_mean_squared_error(y_valid, pred_valid)),
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "catboost_delay_model.cbm"
    metrics_path = artifacts_dir / "catboost_metrics.json"
    model.save_model(model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return TrainingResult(
        model_name=model_name,
        metrics={k: float(v) for k, v in metrics.items()},
        model_path=str(model_path),
        metrics_path=str(metrics_path),
    )
