from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import API_VERSION, get_future_prediction_service, get_metrics_service
from src.api.dependencies import get_standard_predictor_service
from src.api.schemas.health_response import HealthResponse
from src.application.services.api_metrics_service import ApiMetricsService
from src.application.services.future_prediction_service import FuturePredictionService
from src.application.services.predictor_service import StandardPredictorService


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def get_health(
    metrics_service: ApiMetricsService = Depends(get_metrics_service),
    predictor_service: StandardPredictorService = Depends(get_standard_predictor_service),
    future_prediction_service: FuturePredictionService = Depends(get_future_prediction_service),
) -> HealthResponse:
    route_name = "health"
    metrics_service.record_request(route_name)
    try:
        registry = predictor_service.describe_registry()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        metrics_service.record_error(route_name)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return HealthResponse(
        status="ok",
        model=registry["default_model_name"],
        version=API_VERSION,
        uptime_seconds=metrics_service.get_uptime_seconds(),
        supported_symbols=registry["supported_symbols"],
        latest_extraction_date=registry["latest_extraction_date"],
        online_quantum_inference_enabled=False,
        future_predict_ready=future_prediction_service.has_materialized_forecasts(),
        materialized_forecast_symbols=list(future_prediction_service.list_symbols()),
    )
