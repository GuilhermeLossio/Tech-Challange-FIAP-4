from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import (
    get_future_prediction_service,
    get_future_predictions_use_case,
    get_predict_closing_price_use_case,
    get_standard_predictor_service,
)
from src.api.main import app


ALIGNED_EXTRACTION_DATE = date(2026, 5, 19)
LATEST_TRAINED_EXTRACTION_DATE = date(2026, 5, 23)
SUPPORTED_SYMBOLS = ["AMD", "ASML", "NVDA", "QCOM", "TSM"]


class _FakePredictorService:
    def get_supported_symbols(self) -> tuple[str, ...]:
        return tuple(SUPPORTED_SYMBOLS)

    def get_latest_extraction_date(self) -> date:
        return LATEST_TRAINED_EXTRACTION_DATE

    def describe_registry(self) -> dict[str, object]:
        return {
            "latest_extraction_date": LATEST_TRAINED_EXTRACTION_DATE.isoformat(),
            "default_serving_extraction_date": ALIGNED_EXTRACTION_DATE.isoformat(),
            "supported_symbols": SUPPORTED_SYMBOLS,
            "default_model_name": "lstm_nvda",
            "default_model_path": "models/training_runs/.../lstm_nvda.keras",
            "online_quantum_inference_enabled": False,
            "promotion_policy_enabled": True,
        }

    def resolve_serving_extraction_date(
        self,
        *,
        symbol: str,
        requested_extraction_date: date | None = None,
    ) -> date:
        normalized_symbol = symbol.strip().upper()
        if normalized_symbol not in SUPPORTED_SYMBOLS:
            raise FileNotFoundError(f"Unsupported symbol {normalized_symbol!r}.")
        if (
            requested_extraction_date is not None
            and requested_extraction_date < ALIGNED_EXTRACTION_DATE
        ):
            raise FileNotFoundError(
                f"No approved serving artifact is available on or before "
                f"{requested_extraction_date.isoformat()}."
            )
        return ALIGNED_EXTRACTION_DATE


class _FakeFuturePredictionService:
    def has_materialized_forecasts(self) -> bool:
        return True

    def list_symbols(self) -> tuple[str, ...]:
        return tuple(SUPPORTED_SYMBOLS)

    def list_available_extraction_dates(
        self,
        *,
        symbol: str,
        lookback: int | None = None,
        horizon_days: int | None = None,
    ) -> tuple[date, ...]:
        if symbol.strip().upper() not in SUPPORTED_SYMBOLS:
            return tuple()
        return (ALIGNED_EXTRACTION_DATE,)

    def resolve_effective_extraction_date(
        self,
        *,
        symbol: str,
        requested_extraction_date: date | None = None,
        lookback: int | None = None,
        horizon_days: int | None = None,
    ) -> date:
        if symbol.strip().upper() not in SUPPORTED_SYMBOLS:
            raise FileNotFoundError(f"Unsupported symbol {symbol!r}.")
        if requested_extraction_date is None:
            return ALIGNED_EXTRACTION_DATE
        if requested_extraction_date < ALIGNED_EXTRACTION_DATE:
            raise FileNotFoundError(
                f"No materialized forecast is available on or before "
                f"{requested_extraction_date.isoformat()}."
            )
        return ALIGNED_EXTRACTION_DATE


class _FakePredictUseCase:
    def __init__(self, predictor_service: _FakePredictorService) -> None:
        self._predictor_service = predictor_service
        self.requests: list[object] = []

    def execute(self, request: object) -> SimpleNamespace:
        self.requests.append(request)
        resolved_extraction_date = self._predictor_service.resolve_serving_extraction_date(
            symbol=request.symbol,
            requested_extraction_date=request.extraction_date,
        )
        prices = request.prices or [100.0] * 60
        return SimpleNamespace(
            symbol=request.symbol.strip().upper(),
            predicted_close=123.45,
            lower_bound=120.0,
            upper_bound=126.9,
            confidence=0.95,
            currency="USD",
            model="lstm_nvda",
            timestamp="2026-05-05T12:00:00+00:00",
            extraction_date=resolved_extraction_date.isoformat(),
            target_column="close",
            predict_type="normal",
            input_mode="client_prices" if request.prices is not None else "historical_auto_window",
            prices_provided_count=len(prices),
            requested_reference_date=(
                request.reference_date.isoformat()
                if request.reference_date is not None
                else None
            ),
            resolved_window_start_date=None,
            resolved_window_end_date=None,
        )


class _FakeForecastUseCase:
    def __init__(self) -> None:
        self.requests: list[object] = []

    def execute(self, request: object) -> SimpleNamespace:
        self.requests.append(request)
        if request.extraction_date != ALIGNED_EXTRACTION_DATE:
            raise FileNotFoundError(
                f"Materialized future predictions were not found for extraction_date="
                f"{request.extraction_date.isoformat()}."
            )
        return SimpleNamespace(
            symbol=request.symbol.strip().upper(),
            source="yfinance",
            extraction_date=request.extraction_date.isoformat(),
            generated_at="20260505T120000Z",
            generated_at_utc="2026-05-05T12:00:00+00:00",
            lookback=request.lookback or 60,
            horizon_days=request.horizon_days or 30,
            available_predict_types=("normal", "quant"),
            available_forecast_start_date="2026-05-20",
            available_forecast_end_date="2026-12-31",
            returned_forecast_start_date="2026-05-20",
            returned_forecast_end_date="2026-05-21",
            last_observed_date="2026-05-19",
            last_observed_close=220.61,
            row_count=2,
            rows=(
                {
                    "forecast_step": 1,
                    "forecast_date": "2026-05-20",
                    "predict_type": "normal",
                    "model_family": "lstm",
                    "model_name": "lstm_nvda",
                    "predicted_close": 210.5,
                    "predicted_direction": 1,
                    "predicted_direction_label": "up",
                    "is_price_proxy": False,
                    "price_proxy_method": None,
                    "step_elapsed_ms": 17.6,
                    "prediction_constraint_applied": False,
                    "dynamic_cumulative_return_cap": 0.35,
                },
                {
                    "forecast_step": 2,
                    "forecast_date": "2026-05-21",
                    "predict_type": "quant",
                    "model_family": "vqc",
                    "model_name": "quantum_vqc_nvda",
                    "predicted_close": 211.0,
                    "predicted_direction": 1,
                    "predicted_direction_label": "up",
                    "is_price_proxy": True,
                    "price_proxy_method": "directional_proxy",
                    "step_elapsed_ms": 343.1,
                    "prediction_constraint_applied": True,
                    "dynamic_cumulative_return_cap": 0.35,
                },
            ),
            local_path="data/processed/future_predict/.../future_predict.parquet",
        )


@pytest.fixture
def client_with_overrides() -> tuple[TestClient, _FakePredictUseCase, _FakeForecastUseCase]:
    predictor_service = _FakePredictorService()
    future_prediction_service = _FakeFuturePredictionService()
    predict_use_case = _FakePredictUseCase(predictor_service)
    forecast_use_case = _FakeForecastUseCase()

    app.dependency_overrides[get_standard_predictor_service] = lambda: predictor_service
    app.dependency_overrides[get_future_prediction_service] = (
        lambda: future_prediction_service
    )
    app.dependency_overrides[get_predict_closing_price_use_case] = (
        lambda: predict_use_case
    )
    app.dependency_overrides[get_future_predictions_use_case] = (
        lambda: forecast_use_case
    )

    try:
        with TestClient(app) as client:
            yield client, predict_use_case, forecast_use_case
    finally:
        app.dependency_overrides.clear()


def test_health_methods_and_data_usage_share_aligned_dates(
    client_with_overrides: tuple[TestClient, _FakePredictUseCase, _FakeForecastUseCase],
) -> None:
    client, _, _ = client_with_overrides

    health_response = client.get("/health")
    methods_response = client.get("/methods")
    data_usage_response = client.get("/data-usage")

    assert health_response.status_code == 200
    assert methods_response.status_code == 200
    assert data_usage_response.status_code == 200

    for payload in (
        health_response.json(),
        methods_response.json(),
        data_usage_response.json(),
    ):
        assert payload["latest_extraction_date"] == ALIGNED_EXTRACTION_DATE.isoformat()
        assert (
            payload["latest_trained_extraction_date"]
            == LATEST_TRAINED_EXTRACTION_DATE.isoformat()
        )


def test_predict_defaults_to_promoted_serving_date(
    client_with_overrides: tuple[TestClient, _FakePredictUseCase, _FakeForecastUseCase],
) -> None:
    client, predict_use_case, _ = client_with_overrides

    response = client.post(
        "/predict",
        json={
            "symbol": "NVDA",
            "prices": [208.27] * 60,
        },
    )

    assert response.status_code == 200
    assert response.json()["extraction_date"] == ALIGNED_EXTRACTION_DATE.isoformat()
    assert predict_use_case.requests[-1].extraction_date is None


def test_predict_requested_newer_partition_falls_back_to_promoted_artifact(
    client_with_overrides: tuple[TestClient, _FakePredictUseCase, _FakeForecastUseCase],
) -> None:
    client, _, _ = client_with_overrides

    response = client.post(
        "/predict",
        json={
            "symbol": "NVDA",
            "prices": [208.27] * 60,
            "extraction_date": LATEST_TRAINED_EXTRACTION_DATE.isoformat(),
        },
    )

    assert response.status_code == 200
    assert response.json()["extraction_date"] == ALIGNED_EXTRACTION_DATE.isoformat()


def test_forecasts_without_extraction_date_use_aligned_default(
    client_with_overrides: tuple[TestClient, _FakePredictUseCase, _FakeForecastUseCase],
) -> None:
    client, _, forecast_use_case = client_with_overrides

    response = client.get("/forecasts/NVDA")

    assert response.status_code == 200
    payload = response.json()
    assert payload["extraction_date"] == ALIGNED_EXTRACTION_DATE.isoformat()
    assert payload["runtime_ratio_vqc_over_lstm"] == pytest.approx(19.494, rel=1e-3)
    assert payload["guardrail_inconsistency_detected"] is False
    assert payload["model_summaries"][1]["uses_price_proxy"] is True
    assert payload["model_summaries"][1]["guardrail_activations_total"] == 1
    assert forecast_use_case.requests[-1].extraction_date == ALIGNED_EXTRACTION_DATE


def test_forecasts_with_explicit_unavailable_date_return_not_found(
    client_with_overrides: tuple[TestClient, _FakePredictUseCase, _FakeForecastUseCase],
) -> None:
    client, _, _ = client_with_overrides

    response = client.get(
        "/forecasts/NVDA",
        params={"extraction_date": LATEST_TRAINED_EXTRACTION_DATE.isoformat()},
    )

    assert response.status_code == 404
    assert "Materialized future predictions were not found" in response.json()["detail"]
