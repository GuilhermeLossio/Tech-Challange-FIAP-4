from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_future_prediction_service, get_metrics_service
from src.api.dependencies import get_standard_predictor_service
from src.api.serving_defaults import resolve_latest_api_extraction_date
from src.api.schemas.data_usage_response import DataUsageResponse
from src.api.schemas.method_response import MethodCatalogItemResponse, MethodCatalogResponse
from src.application.services.api_metrics_service import ApiMetricsService
from src.application.services.future_prediction_service import FuturePredictionService
from src.application.services.predictor_service import StandardPredictorService


router = APIRouter(tags=["metadata"])


@router.get("/methods", response_model=MethodCatalogResponse)
def get_methods(
    metrics_service: ApiMetricsService = Depends(get_metrics_service),
    predictor_service: StandardPredictorService = Depends(get_standard_predictor_service),
    future_prediction_service: FuturePredictionService = Depends(get_future_prediction_service),
) -> MethodCatalogResponse:
    route_name = "methods"
    metrics_service.record_request(route_name)
    try:
        registry = predictor_service.describe_registry()
        extraction_dates = resolve_latest_api_extraction_date(
            predictor_service=predictor_service,
            future_prediction_service=future_prediction_service,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        metrics_service.record_error(route_name)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return MethodCatalogResponse(
        online_quantum_inference_enabled=False,
        methods=[
            MethodCatalogItemResponse(
                method_id="normal_online",
                availability="online",
                predict_type="normal",
                title="Classical LSTM next-day prediction",
                summary=(
                    "Uses the trained Keras LSTM model with a 60-day closing-price "
                    "window supplied by the caller."
                ),
                inputs=["symbol", "60 closing prices"],
                outputs=["predicted_close", "lower_bound", "upper_bound"],
                limitations=[
                    "Supports only the classical model in live requests.",
                    "Depends on the latest trained scaler and Keras artifact available locally.",
                ],
            ),
            MethodCatalogItemResponse(
                method_id="quant_batch_only",
                availability="batch_only",
                predict_type="quant",
                title="Quantum future prediction serving path",
                summary=(
                    "Serves precomputed quantum direction predictions from the "
                    "`future_predict` dataset only."
                ),
                inputs=["symbol", "materialized future_predict parquet rows"],
                outputs=["predicted_close", "predicted_direction", "predict_type=quant"],
                limitations=[
                    "The API never triggers live quantum inference.",
                    "Quantum outputs are batch-generated offline and may expose a price proxy instead of a direct regressed price.",
                ],
            ),
        ],
        supported_symbols=registry["supported_symbols"],
        latest_extraction_date=extraction_dates.latest_extraction_date,
        latest_trained_extraction_date=extraction_dates.latest_trained_extraction_date,
    )


@router.get("/data-usage", response_model=DataUsageResponse)
def get_data_usage(
    metrics_service: ApiMetricsService = Depends(get_metrics_service),
    predictor_service: StandardPredictorService = Depends(get_standard_predictor_service),
    future_prediction_service: FuturePredictionService = Depends(get_future_prediction_service),
) -> DataUsageResponse:
    route_name = "data_usage"
    metrics_service.record_request(route_name)
    try:
        registry = predictor_service.describe_registry()
        extraction_dates = resolve_latest_api_extraction_date(
            predictor_service=predictor_service,
            future_prediction_service=future_prediction_service,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        metrics_service.record_error(route_name)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return DataUsageResponse(
        raw_market_source="Yahoo Finance via yfinance",
        training_target="Next-day closing price (D+1)",
        target_column="close",
        lookback=60,
        forecast_horizon_days=30,
        supported_symbols=registry["supported_symbols"],
        latest_extraction_date=extraction_dates.latest_extraction_date,
        latest_trained_extraction_date=extraction_dates.latest_trained_extraction_date,
        processed_forecast_dataset="data/processed/future_predict",
        online_quantum_inference_enabled=False,
        materialized_forecasts_available=future_prediction_service.has_materialized_forecasts(),
        notes=[
            "latest_extraction_date is the newest partition currently aligned between online prediction serving and materialized forecast serving.",
            "latest_trained_extraction_date is the newest training partition discovered locally, even when it is not yet exposed as the API default.",
            "POST /predict uses only the classical LSTM model.",
            "GET /forecasts/{symbol} can return both `normal` and `quant` rows when they were materialized offline.",
            "Quantum predictions served by the API come from stored parquet data and never consume IBM Quantum tokens at request time.",
        ],
    )
