from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.application.services.predictor_service import (
    StandardPredictionResult,
    StandardPredictorService,
)


@dataclass(frozen=True)
class PredictClosingPriceRequest:
    symbol: str
    prices: list[float] | None = None
    extraction_date: date | None = None
    reference_date: date | None = None


class PredictClosingPriceUseCase:
    def __init__(self, predictor_service: StandardPredictorService) -> None:
        self._predictor_service = predictor_service

    def execute(
        self,
        request: PredictClosingPriceRequest,
    ) -> StandardPredictionResult:
        return self._predictor_service.predict(
            symbol=request.symbol,
            prices=request.prices,
            extraction_date=request.extraction_date,
            reference_date=request.reference_date,
        )
