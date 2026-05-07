"""
BTS On-Time Performance preprocessor.

Reads the raw BTS CSV (transtats.bts.gov or Kaggle 2015 dataset)
and produces a feature Parquet compatible with train_lgb.py.

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
    python -m flight_delay_platform.pipelines.train_lgb \
        --processed-data data/processed/bts_features.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Actual BTS column names (mixed-case as downloaded from transtats.bts.gov)
_USECOLS = [
    "FlightDate", "Reporting_Airline", "Tail_Number",
    "Origin", "Dest", "CRSDepTime", "DepDelay",
    "Distance", "Cancelled",
    "WeatherDelay", "CarrierDelay", "LateAircraftDelay", "NASDelay",
]

_CONGESTION_NORM = 5.0


def _parse_hour(crs_dep_time: pd.Series) -> pd.Series:
    """CRSDepTime is HHMM integer (e.g. 835 = 08:35). Extract hour."""
    return (pd.to_numeric(crs_dep_time, errors="coerce").fillna(0).astype(int) // 100).clip(0, 23)


def load_and_prepare_bts(
    flights_path: Path,
    max_rows: int | None = None,
) -> pd.DataFrame:
    print(f"Reading BTS CSV: {flights_path}")
    # Read only the columns we need; skip optional ones gracefully
    df = pd.read_csv(
        flights_path,
        usecols=lambda c: c in _USECOLS,
        dtype=str,
        low_memory=False,
    )

    # ── Coerce numeric columns ────────────────────────────────────────────────
    for col in ["DepDelay", "Distance", "CRSDepTime",
                "WeatherDelay", "CarrierDelay", "LateAircraftDelay", "NASDelay"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop cancelled / missing target
    df = df.dropna(subset=["DepDelay"])
    df = df.copy()

    # Random sample for faster iteration (0 = all rows)
    if max_rows and max_rows > 0:
        df = df.sample(min(max_rows, len(df)), random_state=42).copy()

    print(f"  {len(df):,} usable rows after filtering")

    # ── Date features ─────────────────────────────────────────────────────────
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
    df = df.dropna(subset=["FlightDate"])
    df["month"]       = df["FlightDate"].dt.month.astype(int)
    df["day_of_week"] = df["FlightDate"].dt.dayofweek + 1   # Mon=1 … Sun=7
    df["quarter"]     = ((df["month"] - 1) // 3 + 1).astype(int)

    # ── Departure hour ────────────────────────────────────────────────────────
    df["departure_hour"] = _parse_hour(df["CRSDepTime"])
    df["is_weekend"]     = (df["day_of_week"] >= 6).astype(int)
    df["is_peak_hour"]   = df["departure_hour"].isin([7,8,9,16,17,18,19]).astype(int)

    # ── Target ───────────────────────────────────────────────────────────────
    df["delay_minutes"] = df["DepDelay"].clip(lower=0.0)

    # ── Tail-delay propagation (the key new feature) ──────────────────────────
    print("  Computing tail-delay propagation…")
    df = df.sort_values(["Tail_Number", "FlightDate", "CRSDepTime"]).reset_index(drop=True)
    df["prev_tail_delay"] = (
        df.groupby("Tail_Number")["DepDelay"]
        .shift(1)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    df["late_aircraft_flag"] = (df["prev_tail_delay"] > 15).astype(int)

    # ── BTS delay-cause features (fill NaN = 0 for on-time flights) ──────────
    df["weather_delay"] = df["WeatherDelay"].fillna(0.0).clip(lower=0.0)
    df["carrier_delay"] = df["CarrierDelay"].fillna(0.0).clip(lower=0.0)
    df["nas_delay"]     = df["NASDelay"].fillna(0.0).clip(lower=0.0)

    # ── Airport congestion ────────────────────────────────────────────────────
    df["_date_str"] = df["FlightDate"].dt.date.astype(str)
    df["airport_congestion"] = (
        df.groupby(["Origin", "_date_str", "departure_hour"])["DepDelay"]
        .transform("count")
        / _CONGESTION_NORM
    ).clip(0.0, 10.0)

    # ── Route & carrier target encoding ──────────────────────────────────────
    df["route_avg_delay"] = (
        df.groupby(["Origin", "Dest"])["DepDelay"]
        .transform("mean").clip(lower=0.0)
    )
    df["carrier_avg_delay"] = (
        df.groupby("Reporting_Airline")["DepDelay"]
        .transform("mean").clip(lower=0.0)
    )

    # ── Rename to match existing feature schema ───────────────────────────────
    df = df.rename(columns={
        "Reporting_Airline": "airline_code",
        "Origin":            "origin",
        "Dest":              "destination",
        "Distance":          "distance",
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
    print(f"  python -m flight_delay_platform.pipelines.train_lgb --processed-data {args.output}")


if __name__ == "__main__":
    main()
