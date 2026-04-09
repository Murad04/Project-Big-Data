from __future__ import annotations

from typing import Any


FEATURE_GROUPS = {
    "weather": ["weather_severity"],
    "airport": ["airport_congestion"],
    "temporal": ["day_of_week", "month"],
    "categorical": ["airline_code"],
}


def feature_spec() -> dict[str, list[str]]:
    return {group: list(features) for group, features in FEATURE_GROUPS.items()}


def select_features(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for group in FEATURE_GROUPS.values() for key in group}
