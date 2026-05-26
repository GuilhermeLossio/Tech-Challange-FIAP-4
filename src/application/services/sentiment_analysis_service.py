from __future__ import annotations

from dataclasses import dataclass


POSITIVE_TERMS = frozenset(
    {
        "beat",
        "beats",
        "bullish",
        "demand",
        "gain",
        "gains",
        "growth",
        "guidance",
        "outperform",
        "raise",
        "raises",
        "rally",
        "record",
        "strong",
        "upgrade",
        "upside",
    }
)

NEGATIVE_TERMS = frozenset(
    {
        "bearish",
        "cut",
        "cuts",
        "delay",
        "downgrade",
        "fall",
        "falls",
        "loss",
        "miss",
        "misses",
        "pressure",
        "recall",
        "risk",
        "shortage",
        "slump",
        "weak",
        "warning",
    }
)


@dataclass(frozen=True)
class SentimentAnalysisResult:
    score: float
    label: str
    headline_count: int
    analyzer: str


class KeywordSentimentAnalyzer:
    """Small deterministic fallback for demos when live NLP/news adapters are absent."""

    analyzer_name = "keyword_fallback_v1"

    def analyze_headlines(self, headlines: list[str]) -> SentimentAnalysisResult:
        normalized_headlines = [headline.strip() for headline in headlines if headline.strip()]
        if not normalized_headlines:
            return SentimentAnalysisResult(
                score=0.0,
                label="neutral",
                headline_count=0,
                analyzer=self.analyzer_name,
            )

        total_score = 0.0
        for headline in normalized_headlines:
            tokens = self._tokenize(headline)
            positive_hits = sum(1 for token in tokens if token in POSITIVE_TERMS)
            negative_hits = sum(1 for token in tokens if token in NEGATIVE_TERMS)
            total_score += self._clamp((positive_hits - negative_hits) / 3.0)

        score = self._clamp(total_score / len(normalized_headlines))
        return SentimentAnalysisResult(
            score=score,
            label=self.label_from_score(score),
            headline_count=len(normalized_headlines),
            analyzer=self.analyzer_name,
        )

    @staticmethod
    def label_from_score(score: float) -> str:
        if score >= 0.15:
            return "positive"
        if score <= -0.15:
            return "negative"
        return "neutral"

    @staticmethod
    def _tokenize(value: str) -> list[str]:
        import re

        return re.findall(r"[a-z]+", value.lower())

    @staticmethod
    def _clamp(value: float) -> float:
        return max(-1.0, min(1.0, float(value)))
