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

import lightgbm as lgb


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
    "quarter",
    "is_weekend",
    "is_peak_hour",
    "airline_code",
    "origin",
    "destination",
    "route_avg_delay",
    "carrier_avg_delay",
    "distance",
]
# BTS-only features — included automatically when the DataFrame contains them
BTS_FEATURE_COLUMNS = [
    "prev_tail_delay",    # previous flight delay for same aircraft (biggest R² boost)
    "late_aircraft_flag", # binary: was inbound plane > 15 min late?
    "weather_delay",      # BTS-reported weather delay minutes
    "carrier_delay",      # BTS-reported carrier/mechanical delay
    "nas_delay",          # BTS-reported ATC/ground-stop delay
]
TARGET_COLUMN = "delay_minutes"
CATEGORICAL_COLUMNS = ["airline_code", "origin", "destination"]
DELAY_THRESHOLD = 15.0
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
_N_ITERATIONS = 3000


def _build_delay_lookups(df: pd.DataFrame) -> dict[str, Any]:
    route_avg = (
        df.groupby(["origin", "destination"])["delay_minutes"]
        .mean().clip(lower=0.0).round(2)
    )
    carrier_avg = (
        df.groupby("airline_code")["delay_minutes"]
        .mean().clip(lower=0.0).round(2)
    )
    route_dist = df.groupby(["origin", "destination"])["distance"].mean().round(0)

    return {
        "route_avg":      {f"{o}_{d}": float(v) for (o, d), v in route_avg.items()},
        "carrier_avg":    {k: float(v) for k, v in carrier_avg.items()},
        "overall_avg":    round(float(df["delay_minutes"].mean()), 2),
        "route_distance": {f"{o}_{d}": float(v) for (o, d), v in route_dist.items()},
        "overall_distance": round(float(df["distance"].mean()), 0),
    }


def _compute_training_analysis(
    df_valid: pd.DataFrame,
    y_valid: np.ndarray,
    pred_valid: np.ndarray,
    artifacts_dir: Path,
) -> None:
    errors = np.abs(y_valid - pred_valid)
    df_v = df_valid.copy()
    df_v["_pred"]  = pred_valid
    df_v["_error"] = errors
    df_v["_actual"] = y_valid

    # ── Error histogram ───────────────────────────────────────────────────────
    bins = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 300]
    hist, edges = np.histogram(errors, bins=bins)
    error_histogram = [
        {"bin_label": f"{int(edges[i])}-{int(edges[i+1])}", "count": int(hist[i])}
        for i in range(len(hist))
    ]

    # ── Scatter sample (predicted vs actual) ─────────────────────────────────
    n = min(1000, len(y_valid))
    idx = np.random.default_rng(42).choice(len(y_valid), n, replace=False)
    scatter_sample = [
        {"actual": round(float(y_valid[i]), 1), "predicted": round(float(pred_valid[i]), 1)}
        for i in idx
    ]

    # ── Confusion matrix at 15-min threshold ─────────────────────────────────
    y_true_bin = (y_valid >= DELAY_THRESHOLD).astype(int)
    y_pred_bin = (pred_valid >= DELAY_THRESHOLD).astype(int)
    confusion_matrix = {
        "tp": int(np.sum((y_true_bin == 1) & (y_pred_bin == 1))),
        "fp": int(np.sum((y_true_bin == 0) & (y_pred_bin == 1))),
        "tn": int(np.sum((y_true_bin == 0) & (y_pred_bin == 0))),
        "fn": int(np.sum((y_true_bin == 1) & (y_pred_bin == 0))),
        "threshold": DELAY_THRESHOLD,
    }

    # ── Precision / Recall / F1 at multiple thresholds ───────────────────────
    pr_by_threshold = []
    for t in [5, 10, 15, 30, 60]:
        yt = (y_valid >= t).astype(int)
        yp = (pred_valid >= t).astype(int)
        prec = float(precision_score(yt, yp, zero_division=0))
        rec  = float(recall_score(yt, yp, zero_division=0))
        f1   = float(f1_score(yt, yp, zero_division=0))
        pr_by_threshold.append({"threshold": t, "precision": round(prec, 3),
                                  "recall": round(rec, 3), "f1": round(f1, 3)})

    # ── Per-airline MAE ───────────────────────────────────────────────────────
    per_airline = (
        df_v.groupby("airline_code")
        .agg(mae=("_error", "mean"), count=("_error", "count"))
        .round(2).reset_index()
        .query("count >= 30")
        .sort_values("mae")
        .rename(columns={"airline_code": "airline"})
        .to_dict(orient="records")
    )

    # ── Per-month MAE ─────────────────────────────────────────────────────────
    per_month = (
        df_v.groupby("month")
        .agg(mae=("_error", "mean"), count=("_error", "count"))
        .round(2).reset_index()
    )
    per_month["month_name"] = per_month["month"].apply(lambda m: MONTH_NAMES[int(m) - 1])
    per_month = per_month.to_dict(orient="records")

    # ── Per-hour MAE ──────────────────────────────────────────────────────────
    per_hour = (
        df_v.groupby("departure_hour")
        .agg(mae=("_error", "mean"), count=("_error", "count"))
        .round(2).reset_index()
        .rename(columns={"departure_hour": "hour"})
        .to_dict(orient="records")
    )

    analysis = {
        "error_histogram":          error_histogram,
        "scatter_sample":           scatter_sample,
        "confusion_matrix":         confusion_matrix,
        "precision_recall_by_threshold": pr_by_threshold,
        "per_airline":              per_airline,
        "per_month":                per_month,
        "per_hour":                 per_hour,
    }

    path = artifacts_dir / "training_analysis.json"
    path.write_text(json.dumps(analysis), encoding="utf-8")
    print(f"Training analysis saved -> {path}")


def train_model(
    df: pd.DataFrame,
    models_dir: Path,
    artifacts_dir: Path,
    model_name: str = "lgb-delay-regressor",
) -> TrainingResult:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    # Use base features that exist + any BTS extras present in the DataFrame
    active_features = (
        [c for c in FEATURE_COLUMNS if c in df.columns]
        + [c for c in BTS_FEATURE_COLUMNS if c in df.columns]
    )
    missing_base = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_base:
        print(f"  Note: base features not found (skipped): {missing_base}")
    bts_found = [c for c in BTS_FEATURE_COLUMNS if c in df.columns]
    if bts_found:
        print(f"  BTS features detected: {bts_found}")

    # Stratified split keeps delayed/on-time proportions equal across train and val
    stratify_labels = (df[TARGET_COLUMN] >= DELAY_THRESHOLD).astype(int)
    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=42, stratify=stratify_labels)
    X_train = df_train[active_features].copy()
    X_valid = df_valid[active_features].copy()
    y_train = df_train[TARGET_COLUMN].astype(float)
    y_valid = df_valid[TARGET_COLUMN].astype(float)

    # Fix target-encoding leakage: recompute route/carrier averages from training
    # split only so validation rows never see their own delays during training.
    route_avg_train = (
        df_train.groupby(["origin", "destination"])[TARGET_COLUMN]
        .mean().clip(lower=0.0).to_dict()
    )
    carrier_avg_train = (
        df_train.groupby("airline_code")[TARGET_COLUMN]
        .mean().clip(lower=0.0).to_dict()
    )
    overall_avg_train = float(df_train[TARGET_COLUMN].mean())

    for X in (X_train, X_valid):
        route_keys = list(zip(X["origin"], X["destination"]))
        X["route_avg_delay"] = [
            float(route_avg_train.get((o, d), overall_avg_train))
            for o, d in route_keys
        ]
        X["carrier_avg_delay"] = (
            X["airline_code"].map(carrier_avg_train).fillna(overall_avg_train)
        )

    # Label-encode categoricals: LightGBM requires non-negative integer codes
    active_cats = [c for c in CATEGORICAL_COLUMNS if c in active_features]
    label_encoders: dict[str, dict[str, int]] = {}
    for col in active_cats:
        cats = sorted(X_train[col].unique().tolist())
        label_encoders[col] = {c: i for i, c in enumerate(cats)}
        n = len(cats)
        X_train[col] = X_train[col].map(label_encoders[col]).astype(int)
        # Unknown categories at val time → n (out-of-range, treated as missing)
        X_valid[col] = X_valid[col].map(label_encoders[col]).fillna(n).astype(int)

    evals_result: dict = {}
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=active_cats)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)

    params = {
        "objective": "regression",    # RMSE loss
        "metric": "rmse",
        "num_leaves": 63,              # ~depth 6 equivalent (2^6 - 1)
        "learning_rate": 0.02,
        "reg_lambda": 5.0,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "cat_smooth": 10,
        "random_state": 42,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=_N_ITERATIONS,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100),
            lgb.record_evaluation(evals_result),
        ],
    )

    pred_valid = np.maximum(model.predict(X_valid), 0.0)
    y_true_bin = (y_valid.values >= DELAY_THRESHOLD).astype(int)
    y_pred_bin = (pred_valid >= DELAY_THRESHOLD).astype(int)

    nonzero = y_valid.values > 0
    mape = (
        float(np.mean(np.abs((y_valid.values[nonzero] - pred_valid[nonzero]) / y_valid.values[nonzero])) * 100)
        if nonzero.any() else 0.0
    )

    best_iter = int(model.best_iteration) if model.best_iteration > 0 else _N_ITERATIONS

    metrics: dict[str, Any] = {
        "sample_count":            float(len(df)),
        "train_rows":              float(len(X_train)),
        "valid_rows":              float(len(X_valid)),
        "mae":                     float(mean_absolute_error(y_valid, pred_valid)),
        "rmse":                    float(root_mean_squared_error(y_valid, pred_valid)),
        "median_absolute_error":   float(median_absolute_error(y_valid, pred_valid)),
        "r2":                      float(r2_score(y_valid, pred_valid)),
        "mape_percent":            mape,
        "threshold_minutes":       DELAY_THRESHOLD,
        "precision_at_threshold":  float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall_at_threshold":     float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1_at_threshold":         float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "best_iteration":          best_iter,
        "max_iterations":          _N_ITERATIONS,
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path    = models_dir    / "lgb_delay_model.txt"
    metrics_path  = artifacts_dir / "lgb_metrics.json"
    lookups_path  = artifacts_dir / "delay_lookups.json"
    encoders_path = artifacts_dir / "label_encoders.json"
    logs_path     = artifacts_dir / "lgb_training_logs.json"
    features_path = artifacts_dir / "feature_columns.json"

    model.save_model(str(model_path))
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lookups_path.write_text(json.dumps(_build_delay_lookups(df), indent=2), encoding="utf-8")
    encoders_path.write_text(json.dumps(label_encoders, indent=2), encoding="utf-8")
    logs_path.write_text(json.dumps(evals_result, indent=2), encoding="utf-8")
    features_path.write_text(json.dumps(active_features, indent=2), encoding="utf-8")

    _compute_training_analysis(df_valid, y_valid.values, pred_valid, artifacts_dir)

    return TrainingResult(
        model_name=model_name,
        metrics={k: float(v) for k, v in metrics.items()},
        model_path=str(model_path),
        metrics_path=str(metrics_path),
    )