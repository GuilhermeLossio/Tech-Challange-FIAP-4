from __future__ import annotations

from functools import lru_cache

from src.application.services.api_metrics_service import ApiMetricsService
from src.application.services.future_prediction_service import FuturePredictionService
from src.application.services.predictor_service import StandardPredictorService
from src.application.use_cases.get_future_predictions import GetFuturePredictionsUseCase
from src.application.use_cases.predict_closing_price import PredictClosingPriceUseCase
from src.infrastructure.config.settings import ForecastPipelineSettings


API_TITLE = "Tech Challenge Phase 4 API"
API_VERSION = "1.0.0"


@lru_cache(maxsize=1)
def get_forecast_settings() -> ForecastPipelineSettings:
    return ForecastPipelineSettings.from_env()


@lru_cache(maxsize=1)
def get_metrics_service() -> ApiMetricsService:
    return ApiMetricsService()


@lru_cache(maxsize=1)
def get_standard_predictor_service() -> StandardPredictorService:
    settings = get_forecast_settings()
    return StandardPredictorService(
        raw_root_dir=settings.local_raw_dir,
        processed_root_dir=settings.local_processed_dir,
        models_root_dir=settings.local_models_dir,
    )


@lru_cache(maxsize=1)
def get_future_prediction_service() -> FuturePredictionService:
    settings = get_forecast_settings()
    return FuturePredictionService(
        processed_root_dir=settings.local_processed_dir,
    )


@lru_cache(maxsize=1)
def get_predict_closing_price_use_case() -> PredictClosingPriceUseCase:
    return PredictClosingPriceUseCase(get_standard_predictor_service())


@lru_cache(maxsize=1)
def get_future_predictions_use_case() -> GetFuturePredictionsUseCase:
    return GetFuturePredictionsUseCase(get_future_prediction_service())
