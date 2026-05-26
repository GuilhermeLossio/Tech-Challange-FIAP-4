from __future__ import annotations

from pydantic import ConfigDict, Field

from src.api.schemas.predict_request import PredictRequest


class EnrichedPredictRequest(PredictRequest):
    model_config = ConfigDict(extra="forbid")

    news_headlines: list[str] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Optional headlines to score with the local keyword fallback sentiment "
            "analyzer. When omitted, enriched prediction falls back to the standard "
            "classical prediction with neutral sentiment."
        ),
    )
