from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class KafkaConsumerConfig:
    bootstrap_servers: tuple[str, ...]
    topic: str
    group_id: str


def build_consumer_config(bootstrap_servers: Iterable[str], topic: str, group_id: str) -> KafkaConsumerConfig:
    servers = tuple(server.strip() for server in bootstrap_servers if server.strip())
    if not servers:
        raise ValueError("Kafka bootstrap servers are required.")
    return KafkaConsumerConfig(bootstrap_servers=servers, topic=topic.strip(), group_id=group_id.strip())
