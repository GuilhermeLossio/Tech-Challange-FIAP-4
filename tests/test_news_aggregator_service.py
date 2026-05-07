from __future__ import annotations

from datetime import datetime, timezone
import unittest

from src.application.services.news_aggregator_service import NewsAggregatorService
from src.domain.entities.news_signal import NewsSignal


class _DummyNewsRepository:
    def fetch(self, symbol: str, date: object) -> list[NewsSignal]:
        return []


class NewsAggregatorServiceTests(unittest.TestCase):
    def test_deduplicate_handles_missing_and_timezone_aware_timestamps(self) -> None:
        service = NewsAggregatorService([_DummyNewsRepository()])
        published_at = datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc)

        signals = [
            NewsSignal(
                symbol="NVDA",
                headline="Same article",
                source="reuters",
                published_at_utc=published_at,
            ),
            NewsSignal(
                symbol="NVDA",
                headline="Same article",
                source="newsapi",
                published_at_utc=None,
            ),
        ]

        deduplicated = service._deduplicate(signals)

        self.assertEqual(len(deduplicated), 2)
        self.assertIsNone(deduplicated[0].published_at_utc)
        self.assertEqual(deduplicated[1].published_at_utc, published_at)

    def test_published_bucket_normalizes_naive_timestamp_as_utc(self) -> None:
        service = NewsAggregatorService([_DummyNewsRepository()])
        naive_timestamp = datetime(2026, 5, 5, 12, 34, 56)

        bucket = service._published_bucket(naive_timestamp)

        self.assertEqual(bucket, "2026-05-05T12:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
