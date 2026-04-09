from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FLIGHTS_DATASET_URL = "https://raw.githubusercontent.com/tidyverse/nycflights13/main/data-raw/flights.csv"
WEATHER_DATASET_URL = "https://raw.githubusercontent.com/tidyverse/nycflights13/main/data-raw/weather.csv"


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
        "day_of_week": int(cleaned.get("day_of_week", 1)),
        "month": int(cleaned.get("month", 1)),
        "airline_code": cleaned.get("airline_code", "UNKNOWN"),
    }


def download_public_datasets(data_dir: Path) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    flights_path = data_dir / "nyc_flights.csv"
    weather_path = data_dir / "nyc_weather.csv"

    if not flights_path.exists():
        pd.read_csv(FLIGHTS_DATASET_URL).to_csv(flights_path, index=False)
    if not weather_path.exists():
        pd.read_csv(WEATHER_DATASET_URL).to_csv(weather_path, index=False)

    return flights_path, weather_path


def _compute_weather_severity(weather: pd.DataFrame) -> pd.Series:
    precip = weather["precip"].fillna(0.0).clip(lower=0.0, upper=1.0)
    wind = (weather["wind_speed"].fillna(0.0) / 40.0).clip(lower=0.0, upper=1.0)
    visib_penalty = ((10.0 - weather["visib"].fillna(10.0)) / 10.0).clip(lower=0.0, upper=1.0)
    return 0.5 * precip + 0.3 * wind + 0.2 * visib_penalty


def load_and_prepare_training_data(
    flights_path: Path,
    weather_path: Path,
    max_rows: int | None = None,
) -> pd.DataFrame:
    flights = pd.read_csv(flights_path)
    weather = pd.read_csv(weather_path)

    if max_rows is not None and max_rows > 0:
        flights = flights.head(max_rows).copy()

    required_flight_cols = [
        "year",
        "month",
        "day",
        "hour",
        "carrier",
        "origin",
        "dest",
        "dep_delay",
    ]
    flights = flights[required_flight_cols].copy()

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

    weather_keep_cols = ["origin", "year", "month", "day", "hour", "wind_speed", "precip", "visib"]
    weather = weather[weather_keep_cols].copy()
    weather["hour"] = pd.to_numeric(weather["hour"], errors="coerce")
    weather = weather.dropna(subset=["hour", "origin"])
    weather["hour"] = weather["hour"].astype(int)
    weather["weather_severity"] = _compute_weather_severity(weather)

    flights["airport_congestion"] = flights.groupby(["origin", "year", "month", "day", "hour"])[
        "dep_delay"
    ].transform("count")

    merged = flights.merge(
        weather[["origin", "year", "month", "day", "hour", "weather_severity"]],
        on=["origin", "year", "month", "day", "hour"],
        how="left",
    )
    merged["weather_severity"] = merged["weather_severity"].fillna(0.0)

    prepared = pd.DataFrame(
        {
            "weather_severity": merged["weather_severity"].astype(float),
            "airport_congestion": merged["airport_congestion"].astype(float),
            "day_of_week": merged["day_of_week"].astype(int),
            "month": merged["month"].astype(int),
            "airline_code": merged["carrier"].astype(str),
            "origin": merged["origin"].astype(str),
            "destination": merged["dest"].astype(str),
            "delay_minutes": np.maximum(merged["dep_delay"].astype(float), 0.0),
        }
    )

    prepared = prepared.replace([np.inf, -np.inf], np.nan).dropna()
    return prepared
