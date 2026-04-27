from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_metrics_service
from src.application.services.api_metrics_service import ApiMetricsService


router = APIRouter(tags=["predict"])


@router.post("/predict/enriched")
def predict_enriched(
    metrics_service: ApiMetricsService = Depends(get_metrics_service),
) -> dict[str, str]:
    route_name = "predict_enriched"
    metrics_service.record_request(route_name)
    metrics_service.record_error(route_name)
    raise HTTPException(
        status_code=501,
        detail=(
            "Sentiment-enriched prediction is not available in this API version yet."
        ),
    )
