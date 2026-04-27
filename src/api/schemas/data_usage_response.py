from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DataUsageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_market_source: str
    training_target: str
    target_column: str
    lookback: int
    forecast_horizon_days: int
    supported_symbols: list[str]
    latest_extraction_date: str
    processed_forecast_dataset: str
    online_quantum_inference_enabled: bool
    materialized_forecasts_available: bool
    notes: list[str]
