from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import List


@dataclass(frozen=True)
class AppSettings:
    name: str = "flight-delay-prediction-platform"
    environment: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    kafka_bootstrap_servers: List[str] = field(
        default_factory=lambda: [os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")]
    )
    cassandra_keyspace: str = field(default_factory=lambda: os.getenv("CASSANDRA_KEYSPACE", "flight_delay"))
    model_path: str = field(default_factory=lambda: os.getenv("MODEL_PATH", "models/active_model.joblib"))


settings = AppSettings()
