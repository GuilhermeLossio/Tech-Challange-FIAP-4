from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class NewsSignal:
    symbol: str
    headline: str
    source: str
    published_at_utc: datetime | None = None
    summary: str | None = None
    url: str | None = None
    sentiment_score: float | None = None
    sentiment_label: str | None = None
    source_weight: float | None = None
