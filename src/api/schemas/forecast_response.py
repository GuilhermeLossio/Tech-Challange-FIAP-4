from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ForecastItemResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forecast_step: int
    forecast_date: str
    predict_type: str
    model_family: str
    model_name: str
    predicted_close: float
    predicted_direction: int
    predicted_direction_label: str
    is_price_proxy: bool
    price_proxy_method: str | None = None


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
    items: list[ForecastItemResponse]
