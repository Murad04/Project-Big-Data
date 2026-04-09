from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class IngestionPlan:
    sources: tuple[str, ...]
    output_topic: str


def build_ingestion_plan(sources: Iterable[str], output_topic: str) -> IngestionPlan:
    source_list = tuple(source.strip() for source in sources if source.strip())
    if not source_list:
        raise ValueError("At least one ingestion source is required.")
    if not output_topic.strip():
        raise ValueError("An output topic is required.")
    return IngestionPlan(sources=source_list, output_topic=output_topic.strip())
