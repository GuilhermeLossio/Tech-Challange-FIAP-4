from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.application.services.future_prediction_service import (
    FuturePredictionQueryResult,
    FuturePredictionService,
)


@dataclass(frozen=True)
class GetFuturePredictionsRequest:
    symbol: str
    extraction_date: date | None = None
    predict_type: str = "all"
    forecast_date_from: date | None = None
    forecast_date_to: date | None = None
    lookback: int | None = None
    horizon_days: int | None = None
    limit: int | None = None


class GetFuturePredictionsUseCase:
    def __init__(self, future_prediction_service: FuturePredictionService) -> None:
        self._future_prediction_service = future_prediction_service

    def execute(
        self,
        request: GetFuturePredictionsRequest,
    ) -> FuturePredictionQueryResult:
        return self._future_prediction_service.load_forecasts(
            symbol=request.symbol,
            extraction_date=request.extraction_date,
            predict_type=request.predict_type,
            forecast_date_from=request.forecast_date_from,
            forecast_date_to=request.forecast_date_to,
            lookback=request.lookback,
            horizon_days=request.horizon_days,
            limit=request.limit,
        )
