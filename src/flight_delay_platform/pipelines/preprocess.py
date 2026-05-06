from __future__ import annotations

from pathlib import Path
import site
import sys
from typing import Any

import numpy as np
import pandas as pd


FALLBACK_DATASET_URLS: list[tuple[str, str]] = [
    (
        "https://raw.githubusercontent.com/tidyverse/nycflights13/master/data-raw/flights.csv",
        "https://raw.githubusercontent.com/tidyverse/nycflights13/master/data-raw/weather.csv",
    ),
    (
        "https://raw.githubusercontent.com/byuidatascience/data4python4ds/master/data-raw/flights/flights.csv",
        "https://raw.githubusercontent.com/tidyverse/nycflights13/master/data-raw/weather.csv",
    ),
]

# Peak hourly throughput at NYC airports (~50 flights/hr = score 10).
_CONGESTION_NORM = 5.0


def clean_flight_record(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(record)
    for key, value in list(cleaned.items()):
        if isinstance(value, str):
            cleaned[key] = value.strip()
    return cleaned


def build_feature_row(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = clean_flight_record(record)
    return {
        "weather_severity": float(cleaned.get("weather_severity", 0.0)),
        "airport_congestion": float(cleaned.get("airport_congestion", 0.0)),
        "departure_hour": int(cleaned.get("departure_hour", 12)),
        "day_of_week": int(cleaned.get("day_of_week", 1)),
        "month": int(cleaned.get("month", 1)),
        "airline_code": cleaned.get("airline_code", "UNKNOWN"),
        "route_avg_delay": float(cleaned.get("route_avg_delay", 0.0)),
        "carrier_avg_delay": float(cleaned.get("carrier_avg_delay", 0.0)),
    }


def _load_from_installed_package_data() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    candidate_data_dirs: list[Path] = []

    for package_root in site.getsitepackages():
        candidate_data_dirs.append(Path(package_root) / "nycflights13" / "data")

    candidate_data_dirs.append(Path(sys.prefix) / "Lib" / "site-packages" / "nycflights13" / "data")

    for data_dir in candidate_data_dirs:
        flights_zip = data_dir / "flights.csv.zip"
        weather_csv = data_dir / "weather.csv"
        if flights_zip.exists() and weather_csv.exists():
            flights = pd.read_csv(flights_zip)
            weather = pd.read_csv(weather_csv)
            return flights, weather

    return None


def download_public_datasets(data_dir: Path) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    flights_path = data_dir / "nyc_flights.csv"
    weather_path = data_dir / "nyc_weather.csv"

    if flights_path.exists() and weather_path.exists():
        return flights_path, weather_path

    package_data = _load_from_installed_package_data()
    if package_data is not None:
        flights_df, weather_df = package_data
        flights_df.to_csv(flights_path, index=False)
        weather_df.to_csv(weather_path, index=False)
        return flights_path, weather_path

    for flights_url, weather_url in FALLBACK_DATASET_URLS:
        try:
            pd.read_csv(flights_url).to_csv(flights_path, index=False)
            pd.read_csv(weather_url).to_csv(weather_path, index=False)
            return flights_path, weather_path
        except Exception:
            continue

    raise RuntimeError(
        "Unable to fetch dataset files automatically. "
        "Install nycflights13 in the active environment or place nyc_flights.csv and nyc_weather.csv in data/raw/."
    )

    return flights_path, weather_path


def _compute_weather_severity(weather: pd.DataFrame) -> pd.Series:
    precip    = weather["precip"].fillna(0.0).clip(0.0, 1.0)
    wind      = (weather["wind_speed"].fillna(0.0) / 40.0).clip(0.0, 1.0)
    vis_pen   = ((10.0 - weather["visib"].fillna(10.0)) / 10.0).clip(0.0, 1.0)
    # wind_gust: fill missing with wind_speed * 1.3 (typical gust ratio)
    gust      = (weather["wind_gust"].fillna(weather["wind_speed"].fillna(0.0) * 1.3) / 50.0).clip(0.0, 1.0)
    # temperature extremes: below 32F (freezing) or above 85F (heat) both cause delays
    temp      = weather["temp"].fillna(55.0)
    temp_pen  = ((32.0 - temp) / 32.0).clip(0.0, 1.0) + ((temp - 85.0) / 20.0).clip(0.0, 1.0)
    temp_pen  = temp_pen.clip(0.0, 1.0)
    # humidity amplifies precipitation impact
    humid     = (weather["humid"].fillna(50.0) / 100.0).clip(0.0, 1.0)
    # low pressure signals approaching storms (normal ~1013 hPa)
    press_pen = ((1013.0 - weather["pressure"].fillna(1013.0)) / 30.0).clip(0.0, 1.0)

    raw = (
        0.24 * precip
        + 0.16 * wind
        + 0.14 * vis_pen
        + 0.16 * gust
        + 0.12 * temp_pen
        + 0.10 * humid * precip   # humidity matters most when raining
        + 0.08 * press_pen
    )
    return (raw * 10.0).clip(0.0, 10.0)


def load_and_prepare_training_data(
    flights_path: Path,
    weather_path: Path,
    max_rows: int | None = None,
) -> pd.DataFrame:
    flights = pd.read_csv(flights_path)
    weather = pd.read_csv(weather_path)

    # Random sample so we cover all months/seasons, not just the first N rows.
    if max_rows is not None and max_rows > 0:
        flights = flights.sample(min(max_rows, len(flights)), random_state=42).copy()

    required_flight_cols = [
        "year",
        "month",
        "day",
        "hour",
        "carrier",
        "origin",
        "dest",
        "dep_delay",
        "distance",
    ]
    flights = flights[required_flight_cols].copy()
    flights["distance"] = pd.to_numeric(flights["distance"], errors="coerce").fillna(0.0)

    flights["dep_delay"] = pd.to_numeric(flights["dep_delay"], errors="coerce")
    flights["hour"] = pd.to_numeric(flights["hour"], errors="coerce")
    flights = flights.dropna(subset=["dep_delay", "hour", "origin", "dest", "carrier"])
    flights["hour"] = flights["hour"].astype(int)

    flights["event_time"] = pd.to_datetime(
        flights[["year", "month", "day"]],
        errors="coerce",
    )
    flights = flights.dropna(subset=["event_time"])
    flights["day_of_week"] = flights["event_time"].dt.dayofweek + 1

    weather_keep_cols = ["origin", "year", "month", "day", "hour",
                         "wind_speed", "wind_gust", "precip", "visib", "temp", "humid", "pressure"]
    weather = weather[weather_keep_cols].copy()
    weather["hour"] = pd.to_numeric(weather["hour"], errors="coerce")
    weather = weather.dropna(subset=["hour", "origin"])
    weather["hour"] = weather["hour"].astype(int)
    weather["weather_severity"] = _compute_weather_severity(weather)

    # Normalize flight count to 0–10 scale (50 flights/hr ≈ peak = 10.0)
    flights["airport_congestion"] = (
        flights.groupby(["origin", "year", "month", "day", "hour"])["dep_delay"]
        .transform("count")
        / _CONGESTION_NORM
    ).clip(0.0, 10.0)

    merged = flights.merge(
        weather[["origin", "year", "month", "day", "hour", "weather_severity"]],
        on=["origin", "year", "month", "day", "hour"],
        how="left",
    )
    merged["weather_severity"] = merged["weather_severity"].fillna(0.0)

    # Route-level and carrier-level average delays (target encoding within training set)
    route_avg = (
        merged.groupby(["origin", "dest"])["dep_delay"]
        .transform("mean")
        .clip(lower=0.0)
    )
    carrier_avg = (
        merged.groupby("carrier")["dep_delay"]
        .transform("mean")
        .clip(lower=0.0)
    )

    prepared = pd.DataFrame(
        {
            "weather_severity": merged["weather_severity"].astype(float),
            "airport_congestion": merged["airport_congestion"].astype(float),
            "departure_hour": merged["hour"].astype(int),
            "day_of_week": merged["day_of_week"].astype(int),
            "month": merged["month"].astype(int),
            "airline_code": merged["carrier"].astype(str),
            "origin": merged["origin"].astype(str),
            "destination": merged["dest"].astype(str),
            "route_avg_delay": route_avg.astype(float),
            "carrier_avg_delay": carrier_avg.astype(float),
            "distance": merged["distance"].astype(float),
            "delay_minutes": np.maximum(merged["dep_delay"].astype(float), 0.0),
        }
    )

    prepared = prepared.replace([np.inf, -np.inf], np.nan).dropna()
    return prepared
