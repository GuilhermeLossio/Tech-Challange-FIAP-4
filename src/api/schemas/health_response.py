from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    model: str
    version: str
    uptime_seconds: float
    supported_symbols: list[str]
    latest_extraction_date: str
    latest_trained_extraction_date: str
    online_quantum_inference_enabled: bool
    future_predict_ready: bool
    materialized_forecast_symbols: list[str]
