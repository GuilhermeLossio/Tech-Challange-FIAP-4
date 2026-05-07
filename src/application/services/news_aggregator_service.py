from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Callable

# Allows direct execution from IDEs that run the file instead of the package module.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.domain.entities.news_signal import NewsSignal
from src.domain.exceptions.news_fetch_error import NewsFetchError
from src.domain.interfaces.i_news_repository import INewsRepository

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_WEIGHTS: dict[str, float] = {
    "reuters": 1.0,
    "alpha_vantage": 0.9,
    "alphavantage": 0.9,
    "newsapi": 0.7,
    "seekingalpha": 0.6,
    "seeking_alpha": 0.6,
}

SUPPORTED_AGGREGATION_METHODS = frozenset({"weighted_mean", "mean", "median"})

_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%Y%m%d")
_MIN_PUBLISHED_AT_UTC = datetime.min.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class AggregatedNewsResult:
    symbol: str
    target_date: str
    signals: tuple[NewsSignal, ...]
    sentiment_score: float
    sentiment_label: str
    sources_fetched: int
    sources_failed: int


class NewsAggregatorService:
    def __init__(
        self,
        sources: list[INewsRepository],
        dedup_window_hours: int = 6,
        source_weights: dict[str, float] | None = None,
        on_source_error: Callable[[str, Exception], None] | None = None,
    ) -> None:
        if not sources:
            raise ValueError("At least one news source must be provided.")
        if dedup_window_hours < 0:
            raise ValueError("dedup_window_hours must be non-negative.")

        self._sources = sources
        self._dedup_window = timedelta(hours=dedup_window_hours)
        self._source_weights = {
            key.lower(): value
            for key, value in (source_weights or DEFAULT_SOURCE_WEIGHTS).items()
        }
        self._on_source_error = on_source_error

    def fetch(
        self, symbol: str, target_date: date | str
    ) -> tuple[list[NewsSignal], int, int]:
        """Fetch and deduplicate signals from all sources.

        Returns:
            A tuple of (signals, sources_fetched, sources_failed).
        """
        resolved_date = self._normalize_date(target_date)
        raw_signals: list[NewsSignal] = []
        sources_fetched = 0
        sources_failed = 0

        for source in self._sources:
            source_name = type(source).__name__
            try:
                signals = source.fetch(symbol=symbol, date=resolved_date)
                raw_signals.extend(self._apply_default_weights(signals))
                sources_fetched += 1
                logger.debug(
                    "Fetched %d signal(s) from source '%s' for %s on %s.",
                    len(signals),
                    source_name,
                    symbol,
                    resolved_date,
                )
            except NewsFetchError as exc:
                sources_failed += 1
                logger.warning(
                    "Source '%s' failed for symbol=%s date=%s: %s",
                    source_name,
                    symbol,
                    resolved_date,
                    exc,
                )
                if self._on_source_error is not None:
                    self._on_source_error(source_name, exc)

        if not raw_signals:
            logger.warning(
                "No signals obtained for symbol=%s on %s "
                "(sources_fetched=%d, sources_failed=%d).",
                symbol,
                resolved_date,
                sources_fetched,
                sources_failed,
            )

        deduplicated = self._deduplicate(raw_signals)
        logger.debug(
            "Deduplication: %d → %d signal(s) for %s on %s.",
            len(raw_signals),
            len(deduplicated),
            symbol,
            resolved_date,
        )

        return deduplicated, sources_fetched, sources_failed

    def aggregate(
        self,
        symbol: str,
        target_date: date | str,
        method: str = "weighted_mean",
    ) -> AggregatedNewsResult:
        if method not in SUPPORTED_AGGREGATION_METHODS:
            raise ValueError(
                f"Unsupported aggregation method: {method!r}. "
                f"Choose one of: {sorted(SUPPORTED_AGGREGATION_METHODS)}"
            )

        signals, sources_fetched, sources_failed = self.fetch(
            symbol=symbol, target_date=target_date
        )
        sentiment_score = self.aggregate_daily_sentiment(signals, method=method)

        return AggregatedNewsResult(
            symbol=symbol.upper(),
            target_date=self._normalize_date(target_date).isoformat(),
            signals=tuple(signals),
            sentiment_score=sentiment_score,
            sentiment_label=self._label_from_score(sentiment_score),
            sources_fetched=sources_fetched,
            sources_failed=sources_failed,
        )

    def aggregate_daily_sentiment(
        self,
        signals: list[NewsSignal],
        method: str = "weighted_mean",
    ) -> float:
        scored_signals = [s for s in signals if s.sentiment_score is not None]
        if not scored_signals:
            return 0.0

        scores = [float(s.sentiment_score) for s in scored_signals]
        weights = [
            float(s.source_weight or self._default_weight_for(s.source))
            for s in scored_signals
        ]

        if method == "weighted_mean":
            total_weight = sum(weights)
            if total_weight == 0:
                return 0.0
            return (
                sum(score * w for score, w in zip(scores, weights)) / total_weight
            )

        if method == "mean":
            return sum(scores) / len(scores)

        if method == "median":
            sorted_scores = sorted(scores)
            mid = len(sorted_scores) // 2
            if len(sorted_scores) % 2 == 0:
                return (sorted_scores[mid - 1] + sorted_scores[mid]) / 2
            return sorted_scores[mid]

        # Should never reach here given the guard in `aggregate`, but kept
        # so the method is safe when called directly.
        raise ValueError(f"Unsupported aggregation method: {method!r}")


    def _apply_default_weights(self, signals: list[NewsSignal]) -> list[NewsSignal]:
        result: list[NewsSignal] = []
        for signal in signals:
            if signal.source_weight is None:
                result.append(
                    replace(
                        signal,
                        source_weight=self._default_weight_for(signal.source),
                    )
                )
            else:
                result.append(signal)
        return result

    def _default_weight_for(self, source_name: str) -> float:
        normalized = self._normalize_source_name(source_name)
        weight = self._source_weights.get(normalized, 0.5)
        logger.debug("Resolved weight %.2f for source '%s'.", weight, source_name)
        return weight

    def _deduplicate(self, signals: list[NewsSignal]) -> list[NewsSignal]:
        deduped: list[NewsSignal] = []
        seen_keys: set[tuple[str, str, str | None]] = set()

        for signal in sorted(
            signals,
            key=lambda item: self._published_sort_key(item.published_at_utc),
        ):
            dedup_key = (
                signal.symbol.upper(),
                self._headline_key(signal.headline),
                self._published_bucket(signal.published_at_utc),
            )
            if dedup_key in seen_keys:
                logger.debug(
                    "Duplicate signal skipped: symbol=%s headline='%s'.",
                    signal.symbol,
                    signal.headline[:60],
                )
                continue
            seen_keys.add(dedup_key)
            deduped.append(signal)

        return deduped

    def _headline_key(self, headline: str) -> str:
        """Normalise headline for dedup comparison.

        Strips punctuation and casing so minor editorial differences
        between syndicated copies of the same article are collapsed.
        """
        import re

        normalised = re.sub(r"[^\w\s]", "", headline.lower().strip())
        normalised = " ".join(normalised.split())
        return normalised[:80]

    @staticmethod
    def _normalize_published_at_utc(
        published_at_utc: datetime | None,
    ) -> datetime | None:
        if published_at_utc is None:
            return None
        if published_at_utc.tzinfo is None:
            return published_at_utc.replace(tzinfo=timezone.utc)
        return published_at_utc.astimezone(timezone.utc)

    @classmethod
    def _published_sort_key(cls, published_at_utc: datetime | None) -> datetime:
        return cls._normalize_published_at_utc(published_at_utc) or _MIN_PUBLISHED_AT_UTC

    def _published_bucket(self, published_at_utc: datetime | None) -> str | None:
        normalized = self._normalize_published_at_utc(published_at_utc)
        if normalized is None:
            return None

        bucket_seconds = int(self._dedup_window.total_seconds())
        if bucket_seconds <= 0:
            return normalized.replace(microsecond=0).isoformat()

        timestamp = int(normalized.timestamp())
        bucket_start = timestamp - (timestamp % bucket_seconds)
        return datetime.fromtimestamp(bucket_start, tz=timezone.utc).isoformat()

    @staticmethod
    def _normalize_date(value: date | str) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        raise ValueError(
            f"Cannot parse date {value!r}. "
            f"Accepted formats: {', '.join(_DATE_FORMATS)}"
        )

    @staticmethod
    def _normalize_source_name(source_name: str) -> str:
        return source_name.strip().lower().replace(" ", "_")

    @staticmethod
    def _label_from_score(score: float) -> str:
        if score > 0.05:
            return "positive"
        if score < -0.05:
            return "negative"
        return "neutral"


def main() -> int:
    print(
        "NewsAggregatorService is a library module. "
        "Instantiate it from the application layer and inject INewsRepository implementations."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
