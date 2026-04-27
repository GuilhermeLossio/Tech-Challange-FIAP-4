from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_future_predictions_use_case, get_metrics_service
from src.api.schemas.forecast_response import ForecastItemResponse, ForecastResponse
from src.application.services.api_metrics_service import ApiMetricsService
from src.application.use_cases.get_future_predictions import (
    GetFuturePredictionsRequest,
    GetFuturePredictionsUseCase,
)


router = APIRouter(tags=["forecasts"])


@router.get("/forecasts/{symbol}", response_model=ForecastResponse)
def get_forecasts(
    symbol: str,
    extraction_date: date | None = None,
    predict_type: Literal["all", "normal", "quant"] = "all",
    forecast_date_from: date | None = Query(default=None),
    forecast_date_to: date | None = Query(default=None),
    lookback: int = Query(default=60, ge=1),
    horizon_days: int = Query(default=30, ge=1),
    limit: int | None = Query(default=None, ge=1),
    use_case: GetFuturePredictionsUseCase = Depends(get_future_predictions_use_case),
    metrics_service: ApiMetricsService = Depends(get_metrics_service),
) -> ForecastResponse:
    route_name = "forecasts"
    metrics_service.record_request(route_name)
    try:
        result = use_case.execute(
            GetFuturePredictionsRequest(
                symbol=symbol,
                extraction_date=extraction_date,
                predict_type=predict_type,
                forecast_date_from=forecast_date_from,
                forecast_date_to=forecast_date_to,
                lookback=lookback,
                horizon_days=horizon_days,
                limit=limit,
            )
        )
    except ValueError as exc:
        metrics_service.record_error(route_name)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        metrics_service.record_error(route_name)
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ForecastResponse(
        symbol=result.symbol,
        source=result.source,
        extraction_date=result.extraction_date,
        generated_at=result.generated_at,
        generated_at_utc=result.generated_at_utc,
        lookback=result.lookback,
        horizon_days=result.horizon_days,
        available_predict_types=list(result.available_predict_types),
        available_forecast_start_date=result.available_forecast_start_date,
        available_forecast_end_date=result.available_forecast_end_date,
        returned_forecast_start_date=result.returned_forecast_start_date,
        returned_forecast_end_date=result.returned_forecast_end_date,
        last_observed_date=result.last_observed_date,
        last_observed_close=result.last_observed_close,
        online_quantum_inference_enabled=False,
        row_count=result.row_count,
        items=[
            ForecastItemResponse(
                forecast_step=int(row["forecast_step"]),
                forecast_date=str(row["forecast_date"]),
                predict_type=str(row["predict_type"]),
                model_family=str(row["model_family"]),
                model_name=str(row["model_name"]),
                predicted_close=float(row["predicted_close"]),
                predicted_direction=int(row["predicted_direction"]),
                predicted_direction_label=str(row["predicted_direction_label"]),
                is_price_proxy=bool(row["is_price_proxy"]),
                price_proxy_method=(
                    str(row["price_proxy_method"])
                    if row["price_proxy_method"] is not None
                    else None
                ),
            )
            for row in result.rows
        ],
    )
