from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from src.api.dependencies import get_metrics_service
from src.application.services.api_metrics_service import ApiMetricsService


router = APIRouter(tags=["metrics"])


@router.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
def get_metrics(
    metrics_service: ApiMetricsService = Depends(get_metrics_service),
) -> str:
    return metrics_service.render_prometheus()
