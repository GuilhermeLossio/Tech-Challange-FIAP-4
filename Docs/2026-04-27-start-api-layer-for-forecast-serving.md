# Update Note - 2026-04-27

## Topic
Initial FastAPI layer for forecast serving and frontend-facing method documentation.

## Status
Completed for the baseline API scope.

## Executive Summary
The project now includes a first usable API layer built with FastAPI. This phase focuses on three responsibilities:

- serve standard next-day predictions through the local Keras model
- expose materialized future predictions from `future_predict`
- explain the prediction methods and data usage to a future frontend consumer

## Recorded Decisions

### 1. Keep live inference limited to the classical model

`POST /predict` uses only the standard local Keras LSTM model. The API does not perform online quantum inference.

### 2. Serve quantum outputs only from stored batch artifacts

`GET /forecasts/{symbol}` can expose both `predict_type=normal` and `predict_type=quant`, but only from the materialized `future_predict` dataset generated offline.

### 3. Add explicit explanation endpoints for the frontend

The API now includes:

- `GET /methods`
- `GET /data-usage`

These endpoints are intended to support a future Flask interface with structured descriptions of the prediction logic and data lineage.

### 4. Keep unsupported product surface explicit

The following routes exist but intentionally return `501 Not Implemented` for now:

- `POST /predict/enriched`
- `GET /news/{symbol}`

This avoids suggesting that live sentiment enrichment is already production-ready.

## Added Artifacts

- `src/api/main.py`
- `src/api/dependencies.py`
- `src/api/routes/`
- `src/api/schemas/`
- `src/application/services/predictor_service.py`
- `src/application/services/future_prediction_service.py`
- `src/application/services/api_metrics_service.py`
- `src/application/use_cases/predict_closing_price.py`
- `src/application/use_cases/get_future_predictions.py`

## Endpoints Added

- `GET /health`
- `POST /predict`
- `GET /forecasts/{symbol}`
- `GET /methods`
- `GET /data-usage`
- `GET /metrics`

## Operational Notes

- The API reads future predictions from local materialized parquet data under `data/processed/future_predict/...`.
- Quantum predictions served by the API never call IBM Quantum Runtime or consume API tokens during the request.
- `POST /predict` still depends on trained Keras artifacts and refined scaler metadata already existing locally.

## Suggested Next Steps

1. Add integration tests for `POST /predict` and `GET /forecasts/{symbol}`.
2. Add Athena-backed retrieval as an optional serving mode for environments where local parquet is not mounted.
3. Implement the enriched sentiment path behind `/predict/enriched`.
4. Implement news ingestion and delivery behind `/news/{symbol}`.
