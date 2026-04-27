from __future__ import annotations

from datetime import date
import math

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "symbol": "NVDA",
                    "prices": [432.10] * 60,
                    "extraction_date": "2026-04-26",
                },
                {
                    "symbol": "NVDA",
                    "extraction_date": "2026-04-27",
                    "reference_date": "2025-12-31",
                },
            ]
        },
    )

    symbol: str = Field(
        min_length=1,
        examples=["NVDA"],
        description="Ticker symbol to predict, for example NVDA or AMD.",
    )
    prices: list[float] | None = Field(
        default=None,
        min_length=60,
        max_length=60,
        examples=[[432.10] * 60],
        description=(
            "Optional list of exactly 60 closing prices. "
            "When omitted, the API builds the 60-day window automatically "
            "from the local historical dataset."
        ),
    )
    extraction_date: date | None = Field(
        default=None,
        description=(
            "Optional pipeline partition date. When the exact date is unavailable, "
            "the API uses the latest trained extraction_date on or before it."
        ),
    )
    reference_date: date | None = Field(
        default=None,
        description=(
            "Optional historical cutoff date used only when `prices` is omitted. "
            "The automatic 60-day window is built using rows up to this date."
        ),
    )

    @model_validator(mode="after")
    def validate_prices(self) -> "PredictRequest":
        if self.prices is not None:
            if len(self.prices) != 60:
                raise ValueError("`prices` must contain exactly 60 values.")
            if any(not math.isfinite(float(value)) for value in self.prices):
                raise ValueError("`prices` must contain only finite numeric values.")
        if self.prices is not None and self.reference_date is not None:
            raise ValueError(
                "`reference_date` can only be used when `prices` is omitted."
            )
        return self
