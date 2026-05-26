from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class NewsSignalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headline: str
    source: str
    published_at_utc: str | None = None
    summary: str | None = None
    url: str | None = None
    sentiment_score: float | None = None
    sentiment_label: str | None = None


class NewsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    target_date: str
    sentiment_score: float
    sentiment_label: str
    analyzer: str
    signals: list[NewsSignalResponse]
    sources_fetched: int
    sources_failed: int
    live_news_enabled: bool
    notes: list[str]
