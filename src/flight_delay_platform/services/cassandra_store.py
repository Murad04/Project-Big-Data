from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CassandraStore:
    keyspace: str

    def write_prediction(self, row: dict[str, Any]) -> None:
        _ = row

    def write_feature_row(self, row: dict[str, Any]) -> None:
        _ = row
