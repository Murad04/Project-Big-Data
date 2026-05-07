"""
BTS On-Time Performance preprocessor.

Reads the raw BTS CSV (transtats.bts.gov or Kaggle 2015 dataset)
and produces a feature Parquet compatible with train_catboost.py.

Key upgrade over nycflights13:
  - prev_tail_delay  : previous flight delay of same aircraft (tail-delay propagation)
  - late_aircraft_flag: binary flag if inbound aircraft was late
  - weather_delay    : BTS-reported weather delay minutes (replaces computed severity)
  - nas_delay        : ATC/ground-stop delay
  - carrier_delay    : mechanical/crew delay
  - All US airports  (not just EWR/JFK/LGA)

Usage:
    python -m flight_delay_platform.pipelines.bts_preprocess \
        --input data/raw/bts_flights.csv \
        --output data/processed/bts_features.parquet

Then train:
    python -m flight_delay_platform.pipelines.train_catboost \
        --processed-data data/processed/bts_features.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Columns to read (subset keeps memory low on 5M+ row files)
_USECOLS = [
    "FL_DATE", "OP_CARRIER", "TAIL_NUM",
    "ORIGIN", "DEST", "CRS_DEP_TIME", "DEP_DELAY",
    "DISTANCE", "CANCELLED",
    "WEATHER_DELAY", "CARRIER_DELAY", "LATE_AIRCRAFT_DELAY", "NAS_DELAY",
]

_CONGESTION_NORM = 5.0


def _parse_hour(crs_dep_time: pd.Series) -> pd.Series:
    """CRS_DEP_TIME is HHMM integer (e.g. 835 = 08:35). Extract hour."""
    return (pd.to_numeric(crs_dep_time, errors="coerce").fillna(0).astype(int) // 100).clip(0, 23)


def load_and_prepare_bts(
    flights_path: Path,
    max_rows: int | None = None,
) -> pd.DataFrame:
    print(f"Reading BTS CSV: {flights_path}")
    df = pd.read_csv(
        flights_path,
        usecols=[c for c in _USECOLS if c != "CANCELLED"],  # some exports omit CANCELLED
        dtype=str,
        low_memory=False,
    )

    # ── Coerce numeric columns ────────────────────────────────────────────────
    for col in ["DEP_DELAY", "DISTANCE", "CRS_DEP_TIME",
                "WEATHER_DELAY", "CARRIER_DELAY", "LATE_AIRCRAFT_DELAY", "NAS_DELAY"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop cancelled / missing target
    df = df.dropna(subset=["DEP_DELAY"])
    df = df[df["DEP_DELAY"].notna()].copy()

    # Random sample for faster iteration (0 = all rows)
    if max_rows and max_rows > 0:
        df = df.sample(min(max_rows, len(df)), random_state=42).copy()

    print(f"  {len(df):,} usable rows after filtering")

    # ── Date features ─────────────────────────────────────────────────────────
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    df = df.dropna(subset=["FL_DATE"])
    df["month"]      = df["FL_DATE"].dt.month.astype(int)
    df["day_of_week"] = df["FL_DATE"].dt.dayofweek + 1   # Mon=1 … Sun=7
    df["quarter"]    = ((df["month"] - 1) // 3 + 1).astype(int)

    # ── Departure hour ────────────────────────────────────────────────────────
    df["departure_hour"] = _parse_hour(df["CRS_DEP_TIME"])
    df["is_weekend"]     = (df["day_of_week"] >= 6).astype(int)
    df["is_peak_hour"]   = df["departure_hour"].isin([7,8,9,16,17,18,19]).astype(int)

    # ── Target ───────────────────────────────────────────────────────────────
    df["delay_minutes"] = df["DEP_DELAY"].clip(lower=0.0)

    # ── Tail-delay propagation (the key new feature) ──────────────────────────
    # Sort each aircraft's flights chronologically, then shift dep_delay by 1.
    # If the previous flight of the same plane was late, this one likely will be too.
    print("  Computing tail-delay propagation…")
    df = df.sort_values(["TAIL_NUM", "FL_DATE", "CRS_DEP_TIME"]).reset_index(drop=True)
    df["prev_tail_delay"] = (
        df.groupby("TAIL_NUM")["DEP_DELAY"]
        .shift(1)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    df["late_aircraft_flag"] = (df["prev_tail_delay"] > 15).astype(int)

    # ── BTS delay-cause features (fill NaN = 0 for on-time flights) ──────────
    df["weather_delay"]       = df["WEATHER_DELAY"].fillna(0.0).clip(lower=0.0)
    df["carrier_delay"]       = df["CARRIER_DELAY"].fillna(0.0).clip(lower=0.0)
    df["nas_delay"]           = df["NAS_DELAY"].fillna(0.0).clip(lower=0.0)

    # ── Airport congestion ────────────────────────────────────────────────────
    df["_date_str"] = df["FL_DATE"].dt.date.astype(str)
    df["airport_congestion"] = (
        df.groupby(["ORIGIN", "_date_str", "departure_hour"])["DEP_DELAY"]
        .transform("count")
        / _CONGESTION_NORM
    ).clip(0.0, 10.0)

    # ── Route & carrier target encoding ──────────────────────────────────────
    df["route_avg_delay"] = (
        df.groupby(["ORIGIN", "DEST"])["DEP_DELAY"]
        .transform("mean").clip(lower=0.0)
    )
    df["carrier_avg_delay"] = (
        df.groupby("OP_CARRIER")["DEP_DELAY"]
        .transform("mean").clip(lower=0.0)
    )

    # ── Rename to match existing feature schema ───────────────────────────────
    df = df.rename(columns={
        "OP_CARRIER": "airline_code",
        "ORIGIN":     "origin",
        "DEST":       "destination",
        "DISTANCE":   "distance",
    })

    df["distance"] = pd.to_numeric(df["distance"], errors="coerce").fillna(0.0)

    # ── Final feature selection ───────────────────────────────────────────────
    feature_cols = [
        "departure_hour", "day_of_week", "month", "quarter",
        "is_weekend", "is_peak_hour",
        "airline_code", "origin", "destination",
        "route_avg_delay", "carrier_avg_delay", "distance",
        "airport_congestion",
        # BTS-only features
        "prev_tail_delay", "late_aircraft_flag",
        "weather_delay", "carrier_delay", "nas_delay",
        # target
        "delay_minutes",
    ]
    existing = [c for c in feature_cols if c in df.columns]
    prepared = df[existing].replace([np.inf, -np.inf], np.nan).dropna()

    print(f"  Final dataset: {len(prepared):,} rows · {len(existing)-1} features")
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description="BTS On-Time Performance preprocessor")
    parser.add_argument("--input",    required=True, help="Path to raw BTS CSV file")
    parser.add_argument("--output",   default="data/processed/bts_features.parquet")
    parser.add_argument("--max-rows", type=int, default=0, help="Row limit (0 = all)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_bts(Path(args.input), max_rows=args.max_rows)
    df.to_parquet(str(output_path), index=False)
    print(f"\nSaved → {output_path}")
    print("\nNext: train LightGBM from BTS features:")
    print(f"  python -m flight_delay_platform.pipelines.train_catboost --processed-data {args.output}")


if __name__ == "__main__":
    main()
