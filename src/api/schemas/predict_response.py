from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    predicted_close: float
    lower_bound: float
    upper_bound: float
    confidence: float
    currency: str
    model: str
    timestamp: str
    extraction_date: str
    predict_type: str
    target_column: str
    input_mode: str
    prices_provided_count: int
    requested_reference_date: str | None = None
    resolved_window_start_date: str | None = None
    resolved_window_end_date: str | None = None
