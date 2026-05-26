from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends

from src.api.dependencies import get_metrics_service, get_sentiment_analyzer
from src.api.schemas.news_response import NewsResponse
from src.application.services.api_metrics_service import ApiMetricsService
from src.application.services.sentiment_analysis_service import KeywordSentimentAnalyzer


router = APIRouter(tags=["news"])


@router.get("/news/{symbol}", response_model=NewsResponse)
def get_news(
    symbol: str,
    target_date: date | None = None,
    metrics_service: ApiMetricsService = Depends(get_metrics_service),
    sentiment_analyzer: KeywordSentimentAnalyzer = Depends(get_sentiment_analyzer),
) -> NewsResponse:
    route_name = "news"
    metrics_service.record_request(route_name)
    resolved_date = target_date or date.today()
    sentiment = sentiment_analyzer.analyze_headlines([])
    return NewsResponse(
        symbol=symbol.strip().upper(),
        target_date=resolved_date.isoformat(),
        sentiment_score=sentiment.score,
        sentiment_label=sentiment.label,
        analyzer=sentiment.analyzer,
        signals=[],
        sources_fetched=0,
        sources_failed=0,
        live_news_enabled=False,
        notes=[
            "Live news adapters are not configured in this repository snapshot.",
            "The endpoint returns a neutral offline fallback instead of raising 501.",
            "Provide headlines to POST /predict/enriched to use the local keyword sentiment fallback.",
        ],
    )
