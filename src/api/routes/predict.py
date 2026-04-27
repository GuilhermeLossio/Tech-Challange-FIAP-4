from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_metrics_service, get_predict_closing_price_use_case
from src.api.schemas.predict_request import PredictRequest
from src.api.schemas.predict_response import PredictResponse
from src.application.services.api_metrics_service import ApiMetricsService
from src.application.use_cases.predict_closing_price import (
    PredictClosingPriceRequest,
    PredictClosingPriceUseCase,
)


router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    use_case: PredictClosingPriceUseCase = Depends(get_predict_closing_price_use_case),
    metrics_service: ApiMetricsService = Depends(get_metrics_service),
) -> PredictResponse:
    route_name = "predict"
    metrics_service.record_request(route_name)
    try:
        result = use_case.execute(
            PredictClosingPriceRequest(
                symbol=payload.symbol,
                prices=payload.prices,
                extraction_date=payload.extraction_date,
                reference_date=payload.reference_date,
            )
        )
    except ValueError as exc:
        metrics_service.record_error(route_name)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        metrics_service.record_error(route_name)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictResponse(
        symbol=result.symbol,
        predicted_close=result.predicted_close,
        lower_bound=result.lower_bound,
        upper_bound=result.upper_bound,
        confidence=result.confidence,
        currency=result.currency,
        model=result.model,
        timestamp=result.timestamp,
        extraction_date=result.extraction_date,
        predict_type=result.predict_type,
        target_column=result.target_column,
        input_mode=result.input_mode,
        prices_provided_count=result.prices_provided_count,
        requested_reference_date=result.requested_reference_date,
        resolved_window_start_date=result.resolved_window_start_date,
        resolved_window_end_date=result.resolved_window_end_date,
    )
