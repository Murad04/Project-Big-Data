"""
Dask-based feature engineering pipeline for flight delay prediction.

Runs on any machine with Python — no Java required.
Uses Dask's lazy task graph for parallel, out-of-core processing.

Usage:
    python -m flight_delay_platform.pipelines.dask_preprocess
    python -m flight_delay_platform.pipelines.dask_preprocess --max-rows 150000

Then train from the output:
    python -m flight_delay_platform.pipelines.train_catboost --use-dask
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

try:
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

_CONGESTION_NORM = 5.0   # 50 flights/hr at origin ≈ congestion score 10


def _weather_severity(ddf: "dd.DataFrame") -> "dd.Series":
    """Composite weather severity on 0–10 scale (matches pandas pipeline)."""
    precip   = ddf["precip"].fillna(0.0).clip(0.0, 1.0)
    wind     = (ddf["wind_speed"].fillna(0.0) / 40.0).clip(0.0, 1.0)
    vis_pen  = ((10.0 - ddf["visib"].fillna(10.0)) / 10.0).clip(0.0, 1.0)
    gust     = (ddf["wind_gust"].fillna(ddf["wind_speed"].fillna(0.0) * 1.3) / 50.0).clip(0.0, 1.0)
    temp     = ddf["temp"].fillna(55.0)
    temp_pen = (((32.0 - temp) / 32.0).clip(0.0, 1.0) + ((temp - 85.0) / 20.0).clip(0.0, 1.0)).clip(0.0, 1.0)
    humid    = (ddf["humid"].fillna(50.0) / 100.0).clip(0.0, 1.0)
    press_pen = ((1013.0 - ddf["pressure"].fillna(1013.0)) / 30.0).clip(0.0, 1.0)
    raw = (0.24 * precip + 0.16 * wind + 0.14 * vis_pen + 0.16 * gust
           + 0.12 * temp_pen + 0.10 * humid * precip + 0.08 * press_pen)
    return (raw * 10.0).clip(0.0, 10.0)


def run_dask_pipeline(
    flights_path: Path,
    weather_path: Path,
    output_path: Path,
    max_rows: int | None = None,
) -> int:
    """
    Run the full Dask feature-engineering pipeline.
    Writes a Parquet dataset to output_path and returns the row count.
    """
    if not DASK_AVAILABLE:
        raise RuntimeError(
            "Dask is not installed. Run:  pip install 'dask[dataframe]' pyarrow"
        )

    # ── Read CSVs (lazy — no data loaded yet) ────────────────────────────────
    print("Reading CSVs (lazy Dask graph)…")
    flights_dtypes = {
        "year": "float64", "month": "float64", "day": "float64",
        "hour": "float64", "carrier": "object",
        "origin": "object", "dest": "object", "dep_delay": "float64",
        "distance": "float64",
    }
    weather_dtypes = {
        "origin": "object", "year": "float64", "month": "float64",
        "day": "float64",   "hour": "float64",
        "wind_speed": "float64", "wind_gust": "float64",
        "precip": "float64", "visib": "float64",
        "temp": "float64", "humid": "float64", "pressure": "float64",
    }

    flights = dd.read_csv(
        str(flights_path),
        usecols=list(flights_dtypes.keys()),
        dtype=flights_dtypes,
        assume_missing=True,
    )
    weather = dd.read_csv(
        str(weather_path),
        usecols=list(weather_dtypes.keys()),
        dtype=weather_dtypes,
        assume_missing=True,
    )

    # ── Clean ────────────────────────────────────────────────────────────────
    flights = flights.dropna(subset=["dep_delay", "hour", "origin", "dest", "carrier"])
    weather = weather.dropna(subset=["hour", "origin"])

    for col in ["year", "month", "day", "hour"]:
        flights[col] = flights[col].astype("int32")
        weather[col] = weather[col].astype("int32")

    # ── Random sample (triggers one count pass, then samples) ────────────────
    if max_rows:
        print(f"Counting total rows to compute sample fraction…")
        total = len(flights)          # one compute() pass to count
        frac  = min(max_rows / max(total, 1), 1.0)
        flights = flights.sample(frac=frac, random_state=42)
        print(f"  {total:,} total rows -> sampling {frac:.3f} (~{int(total*frac):,} rows)")

    # ── Day of week (ISO Mon=1 … Sun=7) ──────────────────────────────────────
    # pandas dt.dayofweek: Mon=0 … Sun=6  →  +1 → Mon=1 … Sun=7
    date_str = (
        flights["year"].astype(str) + "-"
        + flights["month"].astype(str).str.zfill(2) + "-"
        + flights["day"].astype(str).str.zfill(2)
    )
    flights["day_of_week"] = (
        dd.to_datetime(date_str, format="%Y-%m-%d", errors="coerce")
        .dt.dayofweek + 1
    ).astype("int32")

    # ── Weather severity ─────────────────────────────────────────────────────
    weather["weather_severity"] = _weather_severity(weather)

    # ── Airport congestion (flight count per origin/hour, norm to 0–10) ──────
    print("Computing airport congestion…")
    congestion = (
        flights.groupby(["origin", "year", "month", "day", "hour"])["dep_delay"]
        .count()
        .rename("raw_congestion")
        .reset_index()
    )
    flights = flights.merge(congestion, on=["origin", "year", "month", "day", "hour"], how="left")
    flights["airport_congestion"] = (flights["raw_congestion"] / _CONGESTION_NORM).clip(0.0, 10.0)

    # ── Route average delay (target encoding) ────────────────────────────────
    print("Computing route average delays…")
    route_avg = (
        flights.groupby(["origin", "dest"])["dep_delay"]
        .mean()
        .rename("route_avg_delay")
        .reset_index()
    )
    flights = flights.merge(route_avg, on=["origin", "dest"], how="left")
    flights["route_avg_delay"] = flights["route_avg_delay"].clip(lower=0.0)

    # ── Carrier average delay (target encoding) ──────────────────────────────
    print("Computing carrier average delays…")
    carrier_avg = (
        flights.groupby("carrier")["dep_delay"]
        .mean()
        .rename("carrier_avg_delay")
        .reset_index()
    )
    flights = flights.merge(carrier_avg, on="carrier", how="left")
    flights["carrier_avg_delay"] = flights["carrier_avg_delay"].clip(lower=0.0)

    # ── Join weather ─────────────────────────────────────────────────────────
    print("Joining weather data…")
    merged = flights.merge(
        weather[["origin", "year", "month", "day", "hour", "weather_severity"]],
        on=["origin", "year", "month", "day", "hour"],
        how="left",
    )
    merged["weather_severity"] = merged["weather_severity"].fillna(0.0)

    # ── Final feature selection ───────────────────────────────────────────────
    prepared = merged.assign(
        departure_hour=merged["hour"].astype("int32"),
        delay_minutes=merged["dep_delay"].clip(lower=0.0),
        airline_code=merged["carrier"],
        destination=merged["dest"],
        distance=merged["distance"].fillna(0.0),
        is_weekend=(merged["day_of_week"] >= 6).astype("int32"),
        is_peak_hour=merged["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype("int32"),
    )[
        [
            "weather_severity", "airport_congestion", "departure_hour",
            "day_of_week", "month", "is_weekend", "is_peak_hour",
            "airline_code", "origin", "destination",
            "route_avg_delay", "carrier_avg_delay", "distance", "delay_minutes",
        ]
    ].dropna()

    # ── Write Parquet (triggers full computation) ─────────────────────────────
    parquet_dir = output_path / "features.parquet"
    if parquet_dir.exists():
        shutil.rmtree(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExecuting Dask task graph -> writing Parquet to {parquet_dir}")
    print("(Progress bar shows partition completion)")
    with ProgressBar():
        prepared.to_parquet(str(parquet_dir), write_index=False)

    # Count rows from written Parquet
    row_count = len(pd.read_parquet(str(parquet_dir)))
    print(f"\nDask pipeline complete — {row_count:,} rows written.")
    return row_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dask feature-engineering pipeline for flight delay prediction."
    )
    parser.add_argument("--data-dir",   default="data/raw",       help="Directory with raw CSV files")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for Parquet")
    parser.add_argument("--max-rows",   type=int, default=150000, help="Max flight rows to sample")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    flights_path = data_dir / "nyc_flights.csv"
    weather_path = data_dir / "nyc_weather.csv"

    if not flights_path.exists() or not weather_path.exists():
        print("Raw CSV files not found — run the pandas pipeline once to download them:")
        print("  python -m flight_delay_platform.pipelines.train_catboost --max-rows 1")
        raise SystemExit(1)

    run_dask_pipeline(
        flights_path=flights_path,
        weather_path=weather_path,
        output_path=output_dir,
        max_rows=args.max_rows,
    )
    print("\nNext step - train CatBoost from Dask output:")
    print("  python -m flight_delay_platform.pipelines.train_catboost --processed-data data/processed/features.parquet")


if __name__ == "__main__":
    main()