# Software Architecture - Tech Challenge Phase 4

> **Scope:** This document describes the architecture implemented at the end of the delivery cycle. Components excluded from the final delivery include live news adapters, transformer-based sentiment inference, online quantum inference, prediction caches, and observer-based monitoring classes.
>
> `POST /predict/enriched` and `GET /news/{symbol}` are delivered as offline fallback endpoints. They do not fetch live news and do not perform feature-fusion model inference.

---

## Table of Contents

- [Overview](#overview)
- [System Layers](#system-layers)
- [Multi-Asset Coverage](#multi-asset-coverage)
- [Data and Model Flow](#data-and-model-flow)
- [API Surface](#api-surface)
- [Classical-Quantum Boundary](#classical-quantum-boundary)
- [Forecast Quality and Dashboard](#forecast-quality-and-dashboard)
- [Monitoring and Deployment](#monitoring-and-deployment)
- [Testing and CI](#testing-and-ci)
- [Architecture Decisions](#architecture-decisions)

---

## Overview

The system uses a layered architecture centered on FastAPI, application services, domain contracts, and filesystem-backed storage for datasets and model artifacts.

![Clean architecture layers](Docs/graphs/clean_architecture_layers.svg)

```text
┌─────────────────────────────────────────────────────────┐
│  API & UI                                               │
│  FastAPI routes · schemas · Flask dashboard · site/     │
├─────────────────────────────────────────────────────────┤
│  Application                                            │
│  Prediction · forecast · training · raw/refined pipes   │
├─────────────────────────────────────────────────────────┤
│  Domain                                                 │
│  NewsSignal entity · repository interfaces · exceptions │
├─────────────────────────────────────────────────────────┤
│  Infrastructure                                         │
│  yfinance adapter · local/S3 stores · settings          │
├─────────────────────────────────────────────────────────┤
│  Artifacts & Ops                                        │
│  Parquet/JSON/Keras/dill · CI · Docker · Monitoring     │
└─────────────────────────────────────────────────────────┘
```

**Deliberate operational split:**

| Path | Mode | Notes |
|---|---|---|
| FastAPI `/predict` | Online | Promoted Keras artifacts only |
| `/forecasts/{symbol}` | Materialized | Precomputed parquet rows for `normal` and `quant` forecasts |
| Quantum experiments | Offline | IBM Runtime requires `--confirm-ibm-runtime-cost` |
| `/predict/enriched`, `/news/{symbol}` | Fallback | No live news; API contract preserved |

---

## System Layers

### API Layer

Responsible for HTTP routing, request validation, response schemas, and dependency wiring. Business rules stay outside the route handlers.

```text
src/api/
|-- main.py
|-- dependencies.py
|-- serving_defaults.py
|-- routes/
|   |-- health.py
|   |-- metrics.py
|   |-- predict.py
|   |-- predict_enriched.py
|   |-- forecasts.py
|   |-- metadata.py
|   `-- news.py
`-- schemas/
    |-- predict_request.py
    |-- predict_response.py
    |-- enriched_predict_request.py
    |-- enriched_predict_response.py
    |-- forecast_response.py
    |-- health_response.py
    |-- method_response.py
    |-- data_usage_response.py
    `-- news_response.py
```

### Application Layer

Coordinates model serving, training, forecast generation, dataset generation, and auditable metadata.

```text
src/application/use_cases/
|-- predict_closing_price.py
|-- get_future_predictions.py
|-- generate_feature_dataset.py
|-- generate_forecast_batch.py
|-- train_model.py
|-- train_model_quantum.py
|-- provision_athena_catalog.py
`-- _dataset_loading.py

src/application/services/
|-- predictor_service.py
|-- future_prediction_service.py
|-- forecast_guardrails.py
|-- historical_market_service.py
|-- model_promotion_registry.py
|-- news_aggregator_service.py
|-- sentiment_analysis_service.py
|-- training_catalog_service.py
|-- raw_data_pipeline_service.py
|-- refined_data_pipeline_service.py
`-- api_metrics_service.py
```

`PredictorService` resolves promoted model artifacts and supports both direct close-price models and return-target models such as `lstm_return_*`.

### Domain Layer

Contains the minimal set of contracts and entities used by the delivered system. Model and scaler artifact handling lives in application/infrastructure services, not in the domain layer.

```text
src/domain/
|-- entities/
|   `-- news_signal.py
|-- interfaces/
|   |-- i_news_repository.py
|   `-- i_stock_repository.py
`-- exceptions/
    `-- news_fetch_error.py
```

### Infrastructure Layer

Implements external data access and artifact storage. Filesystem storage is the primary delivery path; S3 raw-zone support exists for optional cloud persistence.

```text
src/infrastructure/
|-- config/
|   `-- settings.py
|-- repositories/
|   `-- yfinance_repository.py
`-- storage/
    |-- local_raw_store.py
    |-- local_processed_store.py
    |-- local_model_store.py
    `-- s3_raw_store.py
```

### Frontend Layer

Read-only Flask dashboard plus static export for GitHub Pages.

```text
src/front/
|-- app.py
|-- templates/
|   |-- dashboard.html
|   `-- components/forecast_dashboard.html
`-- static/app.css

site/   <- public static snapshot generated from the Flask view
```

---

## Multi-Asset Coverage

Five semiconductor companies represent different points of the production chain, from chip design to foundry capacity and lithography equipment.

![Semiconductor supply chain](Docs/graphs/semiconductor_supply_chain.svg)

| Symbol | Company | Supply Chain Role | Exchange |
|---|---|---|---|
| `NVDA` | NVIDIA Corporation | GPU design, AI accelerators | NASDAQ |
| `AMD` | Advanced Micro Devices | CPU/GPU design, direct NVDA competitor | NASDAQ |
| `TSM` | Taiwan Semiconductor Mfg. | Contract foundry | NYSE ADR |
| `ASML` | ASML Holding | EUV lithography equipment | NASDAQ |
| `QCOM` | Qualcomm | Mobile SoC design, 5G semiconductors | NASDAQ |

```text
[ASML] ---- supplies EUV machines ----> [TSM]
                                          |
                               fabricates chips for
                                          |
                              +-----------+-----------+
                              v           v           v
                           [NVDA]       [AMD]       [QCOM]
```

The delivered model-serving path is symbol-specific. The multi-asset framing is used for project rationale, data collection, comparison, and dashboard review.

---

## Data and Model Flow

### Dataset Pipeline

```text
yfinance
  `-> data/raw/extraction_date=YYYY-MM-DD
        `-> data/processed/refined
              `-> data/processed/features
                    `-> models/training_runs
                          `-> models/serving_promotions.json
                                `-> FastAPI /predict
```

Raw partitions preserve extraction lineage. Refined and feature partitions are immutable inputs for training and forecast generation. Serving is controlled by `models/serving_promotions.json`.

**Current promoted package:**

| Item | Value |
|---|---|
| Extraction date | `2026-05-19` |
| Training run | `20260525T131034Z` |
| Forecast run | `20260525T143920Z` |
| Forecast horizon | `162` calendar days |
| Symbols | `NVDA`, `AMD`, `TSM`, `ASML`, `QCOM` |

### Forecast Materialization

`scripts/generate_forecast.py` and `src/application/use_cases/generate_forecast_batch.py` generate the serving dataset under `data/processed/future_predict`.

![Hybrid classical-quantum pipeline](Docs/graphs/hybrid_classical_quantum_pipeline.svg)

Each symbol receives:

- `normal` rows: promoted classical LSTM forecast path.
- `quant` rows: offline quantum directional benchmark materialized as a deterministic price proxy.

The API and dashboard read these materialized rows instead of running expensive forecast generation per request.

### Temporal Cross-Validation

Classical training supports optional expanding-window cross-validation through `scripts/train_keras.py --cross-validation-folds N`. Folds are built only from `train + validation` rows; the `test` split remains the final holdout. Each enabled run writes `cross_validation.json` and includes the fold summary in `training_report.md`.

---

## API Surface

![API request flow](Docs/graphs/api_request_flow.svg)

| Endpoint | Status | Responsibility |
|---|---|---|
| `GET /health` | Active | Runtime health metadata |
| `GET /metrics` | Active | Prometheus-compatible metrics |
| `GET /methods` | Active | Method metadata and quantum/normal descriptions |
| `GET /data-usage` | Active | Data lineage metadata |
| `GET /forecasts/{symbol}` | Active | Materialized forecast rows for one symbol |
| `POST /predict` | Active | Online classical prediction with promoted artifacts |
| `POST /predict/enriched` | Fallback | Classical prediction plus local sentiment summary |
| `GET /news/{symbol}` | Fallback | Local keyword-based sentiment response |

> The two fallback endpoints exist to keep the API contract complete. They are not live-data production paths.

---

## Classical-Quantum Boundary

### Classical Path (online)

```text
request payload
  `-> PredictClosingPrice use case
        `-> PredictorService
              `-> LocalModelStore
                    `-> promoted .keras model + scaler metadata
                          `-> response
```

The promotion policy prevents unapproved models from reaching `/predict`.

### Quantum Path (offline)

![Quantum VQC circuit architecture](Docs/graphs/quantum_vqc_circuit_architecture.svg)

```text
window/features
  `-> StandardScaler
        `-> PCA
              `-> angle scaling
                    `-> Qiskit VQC
                          `-> directional class
                                `-> materialized forecast row (proxy)
```

**Quantum serving constraints:**

![Quantum hardware execution loop](Docs/graphs/quantum_hardware_execution_loop.svg)

- No IBM Quantum job is submitted during API requests.
- IBM Runtime submissions require the `--confirm-ibm-runtime-cost` flag.
- API metadata must continue reporting online quantum inference as disabled.
- `quant` rows in `/forecasts/{symbol}` come from precomputed parquet data.

---

## Forecast Quality and Dashboard

Forecast quality is audited by `scripts/forecast_quality_audit.py`.

```text
data/processed/audits/forecast_quality/
|-- forecast_quality_steps.csv
|-- forecast_quality_summary.csv
|-- forecast_quality_scaler_split_audit.csv
|-- nvda_forecast_quality.svg
|-- amd_forecast_quality.svg
|-- tsm_forecast_quality.svg
|-- asml_forecast_quality.svg
|-- qcom_forecast_quality.svg
`-- forecast_quality_report.md
```

**Known quality interpretation:**

- Realized-price comparison is available only where post-cutoff actual data exists.
- The latest audit has a small realized overlap and should be treated as a delivery validation artifact, not a full production backtest.
- Residual monotonic behavior remains documented for some paths and should be monitored in later retraining cycles.

Dashboard validation is documented in `Docs/2026-05-25-dashboard-data-exposure-validation.md` and covered by `tests/test_front_dashboard.py`.

---

## Monitoring and Deployment

```text
Dockerfile
docker-compose.yml
monitoring/
|-- prometheus.yml
`-- grafana/
    |-- dashboards/tech-challenge-api.json
    `-- provisioning/
        |-- dashboards/default.yml
        `-- datasources/prometheus.yml
```

**Runtime services in the Compose stack:**

| Service | Port | Purpose |
|---|---:|---|
| API | `8000` | FastAPI application |
| Prometheus | `9090` | Scrapes `/metrics` |
| Grafana | `3000` | Local observability dashboard |

`/metrics` is implemented directly in the API through `ApiMetricsService`. The project does not implement observer classes; monitoring is handled through endpoint metrics plus Prometheus/Grafana configuration.

---

## Testing and CI

**Local validation commands:**

```bash
python -m compileall src scripts
python -m flake8 src tests scripts --count --select=E9,F63,F7,F82 --show-source --statistics
python -m pytest tests -q
```

**Implemented tests:**

```text
tests/
|-- test_api_endpoints.py
|-- test_front_dashboard.py
|-- test_training_cross_validation.py
`-- test_news_aggregator_service.py
```

**GitHub Actions:**

```text
.github/workflows/
|-- ci.yml      <- install deps - compileall - flake8 - pytest
`-- pages.yml   <- publishes committed static site/
```

`ci.yml` installs runtime and development dependencies, compiles `src` and `scripts`, runs fatal-error flake8 checks, and executes pytest. `pages.yml` publishes the committed static `site/` directory.

---

## Architecture Decisions

### ADR-001 - FastAPI for the Serving API

**Decision:** Use FastAPI for runtime prediction and forecast endpoints.

**Rationale:** FastAPI provides typed request/response contracts, OpenAPI documentation, and low-ceremony dependency wiring.

---

### ADR-002 - Native Keras Artifacts

**Decision:** Store promoted classical models as `.keras` artifacts with companion scaler and metadata files.

**Rationale:** Native Keras artifacts preserve model structure and are straightforward to reload in the serving process.

---

### ADR-003 - Explicit Model Promotion

**Decision:** Control serving through `models/serving_promotions.json`.

**Rationale:** Training output and serving approval are separate lifecycle events. This avoids accidentally serving incomplete, stale, or degraded artifacts.

---

### ADR-004 - Offline Quantum Materialization

**Decision:** Keep quantum execution offline and serve only materialized quantum forecast rows.

**Rationale:** IBM Quantum execution has queue time, noisy measurements, and possible runtime cost. Request-time quantum inference would be slow and hard to control.

---

### ADR-005 - Fallback News Contract

**Decision:** Keep `/news/{symbol}` and `/predict/enriched` active as local fallback endpoints, excluding live news adapters and transformer sentiment inference from the final architecture.

**Rationale:** This preserves the API contract without claiming unsupported live-data behavior.

---

### ADR-006 - Static Dashboard plus Local Monitoring

**Decision:** Use a read-only Flask dashboard, committed static export, `/metrics`, Docker Compose, Prometheus, and Grafana for delivery visibility.

**Rationale:** The stack is simple to review locally and separates presentation from serving-time model execution.
