from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import (
    get_future_prediction_service,
    get_future_predictions_use_case,
    get_metrics_service,
    get_standard_predictor_service,
)
from src.api.serving_defaults import (
    DEFAULT_HORIZON_DAYS,
    resolve_symbol_default_forecast_extraction_date,
)
from src.api.schemas.forecast_response import (
    ForecastItemResponse,
    ForecastModelSummaryResponse,
    ForecastResponse,
)
from src.application.services.api_metrics_service import ApiMetricsService
from src.application.services.future_prediction_service import FuturePredictionService
from src.application.services.predictor_service import StandardPredictorService
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
    horizon_days: int = Query(default=DEFAULT_HORIZON_DAYS, ge=1),
    limit: int | None = Query(default=None, ge=1),
    use_case: GetFuturePredictionsUseCase = Depends(get_future_predictions_use_case),
    predictor_service: StandardPredictorService = Depends(get_standard_predictor_service),
    future_prediction_service: FuturePredictionService = Depends(get_future_prediction_service),
    metrics_service: ApiMetricsService = Depends(get_metrics_service),
) -> ForecastResponse:
    route_name = "forecasts"
    metrics_service.record_request(route_name)
    try:
        effective_extraction_date = extraction_date
        if effective_extraction_date is None:
            effective_extraction_date = resolve_symbol_default_forecast_extraction_date(
                symbol=symbol,
                predictor_service=predictor_service,
                future_prediction_service=future_prediction_service,
                lookback=lookback,
                horizon_days=horizon_days,
            )
        result = use_case.execute(
            GetFuturePredictionsRequest(
                symbol=symbol,
                extraction_date=effective_extraction_date,
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

    model_summaries = _build_model_summaries(
        rows=result.rows,
        horizon_days=int(result.horizon_days),
        last_observed_close=float(result.last_observed_close),
    )
    runtime_ratio = _compute_runtime_ratio(model_summaries)
    guardrail_inconsistency_detected = any(
        item.guardrail_activations_total > item.forecast_horizon_steps
        for item in model_summaries
    )

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
        runtime_ratio_vqc_over_lstm=runtime_ratio,
        guardrail_inconsistency_detected=guardrail_inconsistency_detected,
        items=[
            ForecastItemResponse(
                forecast_step=int(row["forecast_step"]),
                forecast_date=str(row["forecast_date"]),
                predict_type=_canonical_predict_type(row),
                stored_predict_type=str(row["predict_type"]),
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
                predicted_step_return=_optional_float(row.get("predicted_step_return")),
                horizon_return_from_last_observed=_optional_float(
                    row.get("horizon_return_from_last_observed")
                ),
                raw_model_predicted_close=_optional_float(
                    row.get("raw_model_predicted_close")
                ),
                prediction_constraint_applied=_optional_bool(
                    row.get("prediction_constraint_applied")
                ),
                prediction_return_cap=_optional_float(row.get("prediction_return_cap")),
                prediction_constraint_method=(
                    str(row.get("prediction_constraint_method"))
                    if row.get("prediction_constraint_method") is not None
                    else None
                ),
                price_proxy_return=_optional_float(row.get("price_proxy_return")),
                step_elapsed_ms=_optional_float(row.get("step_elapsed_ms")),
                hit_lower_band=_optional_bool(row.get("hit_lower_band")),
                hit_upper_band=_optional_bool(row.get("hit_upper_band")),
                dynamic_cumulative_return_cap=_optional_float(
                    row.get("dynamic_cumulative_return_cap")
                ),
                model_prediction_target_mode=(
                    str(row.get("model_prediction_target_mode"))
                    if row.get("model_prediction_target_mode") is not None
                    else None
                ),
            )
            for row in result.rows
        ],
        model_summaries=model_summaries,
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    numeric_value = float(value)
    if numeric_value != numeric_value:
        return None
    return numeric_value


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _canonical_predict_type(row: dict[str, object]) -> str:
    model_family = str(row.get("model_family", "")).lower()
    model_name = str(row.get("model_name", "")).lower()
    is_price_proxy = bool(row.get("is_price_proxy", False))
    if "quantum" in model_family or "vqc" in model_family or "quantum" in model_name:
        return "quant"
    if is_price_proxy:
        return "quant"
    if "keras" in model_family or "lstm" in model_family or "lstm" in model_name:
        return "normal"
    return str(row.get("predict_type", "")).lower()


def _build_model_summaries(
    *,
    rows: tuple[dict[str, object], ...],
    horizon_days: int,
    last_observed_close: float,
) -> list[ForecastModelSummaryResponse]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(_canonical_predict_type(row), []).append(row)

    summaries: list[ForecastModelSummaryResponse] = []
    for predict_type, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda item: int(item["forecast_step"]))
        if not ordered:
            continue

        elapsed_values = [
            numeric_value
            for numeric_value in (
                _optional_float(row.get("step_elapsed_ms")) for row in ordered
            )
            if numeric_value is not None
        ]
        total_runtime = sum(elapsed_values) if elapsed_values else None
        average_runtime = (
            total_runtime / len(elapsed_values)
            if total_runtime is not None and elapsed_values
            else None
        )
        final_close = _optional_float(ordered[-1].get("predicted_close"))
        cumulative_return = (
            (final_close / last_observed_close - 1.0)
            if final_close is not None and abs(last_observed_close) > 1e-8
            else None
        )
        guardrail_steps = sum(
            1 for row in ordered if bool(row.get("prediction_constraint_applied"))
        )
        price_proxy_methods = sorted(
            {
                str(row.get("price_proxy_method"))
                for row in ordered
                if row.get("price_proxy_method") is not None
            }
        )
        cumulative_caps = [
            cap
            for cap in (
                _optional_float(row.get("dynamic_cumulative_return_cap"))
                for row in ordered
            )
            if cap is not None
        ]
        cumulative_cap = max(cumulative_caps) if cumulative_caps else 0.35
        limited_by_cap = (
            cumulative_return is not None
            and abs(abs(cumulative_return) - cumulative_cap) <= 0.001
        )

        summaries.append(
            ForecastModelSummaryResponse(
                predict_type=predict_type,
                model="LSTM - Keras" if predict_type == "normal" else "VQC - Qiskit",
                family=str(ordered[0].get("model_family", "")),
                total_runtime_ms=total_runtime,
                average_runtime_ms=average_runtime,
                minimum_runtime_ms=min(elapsed_values) if elapsed_values else None,
                maximum_runtime_ms=max(elapsed_values) if elapsed_values else None,
                up_rate=(
                    sum(int(row.get("predicted_direction", 0)) for row in ordered)
                    / len(ordered)
                    if ordered
                    else None
                ),
                cumulative_return=cumulative_return,
                guardrail_steps=min(guardrail_steps, horizon_days),
                forecast_horizon_steps=horizon_days,
                guardrail_activations_total=guardrail_steps,
                uses_price_proxy=any(bool(row.get("is_price_proxy")) for row in ordered),
                price_proxy_method=", ".join(price_proxy_methods) or None,
                limited_by_cap=limited_by_cap,
            )
        )
    return summaries


def _compute_runtime_ratio(
    summaries: list[ForecastModelSummaryResponse],
) -> float | None:
    by_type = {item.predict_type: item for item in summaries}
    normal = by_type.get("normal")
    quant = by_type.get("quant")
    if (
        normal is None
        or quant is None
        or normal.average_runtime_ms is None
        or quant.average_runtime_ms is None
        or normal.average_runtime_ms <= 0
    ):
        return None
    return quant.average_runtime_ms / normal.average_runtime_ms
