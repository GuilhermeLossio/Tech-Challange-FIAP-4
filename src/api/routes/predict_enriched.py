from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import (
    get_metrics_service,
    get_predict_closing_price_use_case,
    get_sentiment_analyzer,
)
from src.api.schemas.enriched_predict_request import EnrichedPredictRequest
from src.api.schemas.enriched_predict_response import EnrichedPredictResponse
from src.application.services.api_metrics_service import ApiMetricsService
from src.application.services.sentiment_analysis_service import KeywordSentimentAnalyzer
from src.application.use_cases.predict_closing_price import (
    PredictClosingPriceRequest,
    PredictClosingPriceUseCase,
)


router = APIRouter(tags=["predict"])


@router.post("/predict/enriched", response_model=EnrichedPredictResponse)
def predict_enriched(
    request: EnrichedPredictRequest,
    metrics_service: ApiMetricsService = Depends(get_metrics_service),
    predict_use_case: PredictClosingPriceUseCase = Depends(
        get_predict_closing_price_use_case
    ),
    sentiment_analyzer: KeywordSentimentAnalyzer = Depends(get_sentiment_analyzer),
) -> EnrichedPredictResponse:
    route_name = "predict_enriched"
    metrics_service.record_request(route_name)
    try:
        prediction = predict_use_case.execute(
            PredictClosingPriceRequest(
                symbol=request.symbol,
                prices=request.prices,
                extraction_date=request.extraction_date,
                reference_date=request.reference_date,
            )
        )
        sentiment = sentiment_analyzer.analyze_headlines(request.news_headlines)
    except ValueError as exc:
        metrics_service.record_error(route_name)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        metrics_service.record_error(route_name)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    enrichment_applied = sentiment.headline_count > 0
    sentiment_adjusted_close = float(prediction.predicted_close)
    notes = [
        "Live news retrieval and FinBERT-style NLP are roadmap items for this repository snapshot.",
        "This endpoint uses a deterministic local keyword sentiment fallback when headlines are supplied.",
    ]
    if not enrichment_applied:
        notes.append(
            "No headlines were supplied, so the prediction is the standard classical output with neutral sentiment."
        )

    return EnrichedPredictResponse(
        symbol=prediction.symbol,
        predicted_close=prediction.predicted_close,
        sentiment_adjusted_close=sentiment_adjusted_close,
        lower_bound=prediction.lower_bound,
        upper_bound=prediction.upper_bound,
        confidence=prediction.confidence,
        currency=prediction.currency,
        model=prediction.model,
        timestamp=prediction.timestamp,
        extraction_date=prediction.extraction_date,
        predict_type="enriched_fallback",
        target_column=prediction.target_column,
        input_mode=prediction.input_mode,
        prices_provided_count=prediction.prices_provided_count,
        requested_reference_date=prediction.requested_reference_date,
        resolved_window_start_date=prediction.resolved_window_start_date,
        resolved_window_end_date=prediction.resolved_window_end_date,
        sentiment_score=sentiment.score,
        sentiment_label=sentiment.label,
        sentiment_analyzer=sentiment.analyzer,
        sentiment_headline_count=sentiment.headline_count,
        enrichment_applied=enrichment_applied,
        enrichment_mode="keyword_fallback",
        notes=notes,
    )
