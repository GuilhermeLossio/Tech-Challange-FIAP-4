from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.application.services.future_prediction_service import FuturePredictionService
from src.application.services.predictor_service import StandardPredictorService


DEFAULT_LOOKBACK = 60
DEFAULT_HORIZON_DAYS = 30


@dataclass(frozen=True)
class ExtractionDateMetadata:
    latest_extraction_date: str
    latest_trained_extraction_date: str


def resolve_latest_api_extraction_date(
    *,
    predictor_service: StandardPredictorService,
    future_prediction_service: FuturePredictionService,
    lookback: int = DEFAULT_LOOKBACK,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> ExtractionDateMetadata:
    latest_trained_extraction_date = predictor_service.get_latest_extraction_date()
    supported_symbols = predictor_service.get_supported_symbols()
    forecast_symbols = set(future_prediction_service.list_symbols())
    shared_symbols = tuple(
        symbol for symbol in supported_symbols if symbol in forecast_symbols
    )

    common_candidates: set[date] | None = None
    for symbol in shared_symbols:
        symbol_candidates = set(
            list_aligned_symbol_extraction_dates(
                symbol=symbol,
                predictor_service=predictor_service,
                future_prediction_service=future_prediction_service,
                lookback=lookback,
                horizon_days=horizon_days,
            )
        )
        if not symbol_candidates:
            continue
        common_candidates = (
            symbol_candidates
            if common_candidates is None
            else common_candidates & symbol_candidates
        )

    if common_candidates:
        latest_extraction_date = max(common_candidates)
    else:
        default_symbol = "NVDA" if "NVDA" in supported_symbols else supported_symbols[0]
        latest_extraction_date = resolve_symbol_default_forecast_extraction_date(
            symbol=default_symbol,
            predictor_service=predictor_service,
            future_prediction_service=future_prediction_service,
            lookback=lookback,
            horizon_days=horizon_days,
        )

    return ExtractionDateMetadata(
        latest_extraction_date=latest_extraction_date.isoformat(),
        latest_trained_extraction_date=latest_trained_extraction_date.isoformat(),
    )


def resolve_symbol_default_forecast_extraction_date(
    *,
    symbol: str,
    predictor_service: StandardPredictorService,
    future_prediction_service: FuturePredictionService,
    lookback: int = DEFAULT_LOOKBACK,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> date:
    aligned_dates = list_aligned_symbol_extraction_dates(
        symbol=symbol,
        predictor_service=predictor_service,
        future_prediction_service=future_prediction_service,
        lookback=lookback,
        horizon_days=horizon_days,
    )
    if aligned_dates:
        return aligned_dates[-1]

    predictor_default_date = predictor_service.resolve_serving_extraction_date(
        symbol=symbol,
    )
    return future_prediction_service.resolve_effective_extraction_date(
        symbol=symbol,
        requested_extraction_date=predictor_default_date,
        lookback=lookback,
        horizon_days=horizon_days,
    )


def list_aligned_symbol_extraction_dates(
    *,
    symbol: str,
    predictor_service: StandardPredictorService,
    future_prediction_service: FuturePredictionService,
    lookback: int = DEFAULT_LOOKBACK,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> tuple[date, ...]:
    aligned_dates: list[date] = []
    for forecast_date in future_prediction_service.list_available_extraction_dates(
        symbol=symbol,
        lookback=lookback,
        horizon_days=horizon_days,
    ):
        try:
            resolved_predict_date = predictor_service.resolve_serving_extraction_date(
                symbol=symbol,
                requested_extraction_date=forecast_date,
            )
        except FileNotFoundError:
            continue
        if resolved_predict_date == forecast_date:
            aligned_dates.append(forecast_date)
    return tuple(aligned_dates)
