from __future__ import annotations

from fastapi import FastAPI

from src.api.dependencies import API_TITLE, API_VERSION
from src.api.routes.forecasts import router as forecasts_router
from src.api.routes.health import router as health_router
from src.api.routes.metadata import router as metadata_router
from src.api.routes.metrics import router as metrics_router
from src.api.routes.news import router as news_router
from src.api.routes.predict import router as predict_router
from src.api.routes.predict_enriched import router as predict_enriched_router


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "Baseline forecasting API for Tech Challenge Phase 4. "
        "The API serves standard LSTM next-day predictions online and exposes "
        "materialized future forecasts, including precomputed quantum outputs, "
        "without triggering live quantum inference."
    ),
)

app.include_router(health_router)
app.include_router(metrics_router)
app.include_router(predict_router)
app.include_router(predict_enriched_router)
app.include_router(forecasts_router)
app.include_router(metadata_router)
app.include_router(news_router)
