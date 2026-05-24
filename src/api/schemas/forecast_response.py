from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ForecastItemResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forecast_step: int
    forecast_date: str
    predict_type: str
    stored_predict_type: str | None = None
    model_family: str
    model_name: str
    predicted_close: float
    predicted_direction: int
    predicted_direction_label: str
    is_price_proxy: bool
    price_proxy_method: str | None = None
    predicted_step_return: float | None = None
    horizon_return_from_last_observed: float | None = None
    raw_model_predicted_close: float | None = None
    prediction_constraint_applied: bool | None = None
    prediction_return_cap: float | None = None
    prediction_constraint_method: str | None = None
    price_proxy_return: float | None = None
    step_elapsed_ms: float | None = None
    hit_lower_band: bool | None = None
    hit_upper_band: bool | None = None
    dynamic_cumulative_return_cap: float | None = None
    model_prediction_target_mode: str | None = None


class ForecastModelSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predict_type: str
    model: str
    family: str
    total_runtime_ms: float | None = None
    average_runtime_ms: float | None = None
    minimum_runtime_ms: float | None = None
    maximum_runtime_ms: float | None = None
    up_rate: float | None = None
    cumulative_return: float | None = None
    guardrail_steps: int
    forecast_horizon_steps: int
    guardrail_activations_total: int
    uses_price_proxy: bool
    price_proxy_method: str | None = None
    limited_by_cap: bool


class ForecastResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    source: str
    extraction_date: str
    generated_at: str
    generated_at_utc: str
    lookback: int
    horizon_days: int
    available_predict_types: list[str]
    available_forecast_start_date: str
    available_forecast_end_date: str
    returned_forecast_start_date: str
    returned_forecast_end_date: str
    last_observed_date: str
    last_observed_close: float
    online_quantum_inference_enabled: bool
    row_count: int
    runtime_ratio_vqc_over_lstm: float | None = None
    guardrail_inconsistency_detected: bool = False
    items: list[ForecastItemResponse]
    model_summaries: list[ForecastModelSummaryResponse] = []
