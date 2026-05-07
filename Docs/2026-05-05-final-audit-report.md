# Final Audit Report

Date: 2026-05-05
Repository: `Tech-Challange-FIAP-4`

## Scope

This report summarizes the technical review performed on the application, the prioritized findings that were reproduced during the audit, the fixes that were applied, and the residual risks that remain after the remediation work.

## Executive Summary

Four priority items were addressed:

1. Online prediction serving was selecting the newest training partition even when the model quality was degraded.
2. API metadata exposed an extraction date that was not actually aligned with the forecast-serving path.
3. The news aggregation deduplication flow could crash when timestamps mixed timezone-aware values and missing values.
4. The repository documentation claimed support for a QA toolchain that was not declared in the local dependency manifest.

All four items were fixed in the codebase and validated locally.

## Findings And Resolutions

### High Priority

#### 1. Serving selected the latest training partition instead of the best eligible artifact

Observed behavior:

- `POST /predict` defaulted to the most recent training partition.
- The default path selected degraded artifacts such as `2026-04-26`.
- In local reproduction, `NVDA` produced a raw predicted close far outside the expected range and only looked plausible after guardrail correction.

Resolution applied:

- Updated the serving model resolver in `src/application/services/predictor_service.py`.
- Serving now selects the best eligible manifest-backed artifact by metric quality instead of simply choosing the newest partition.
- Legacy manifests that only pointed to global alias files such as `models/lstm_*.keras` are excluded from the automatic quality ranking when they cannot guarantee immutable artifact linkage.

Result after fix:

- Default online serving no longer selects `2026-04-26`.
- `NVDA` now resolves to `2026-04-22` for default serving.

#### 2. API metadata advertised an extraction date that was not truly servable

Observed behavior:

- `/methods`, `/data-usage`, and `/health` exposed the latest training extraction date.
- `/forecasts/{symbol}` used a different default resolution path and only had materialized forecast data up to `2026-04-22`.
- Clients could receive metadata pointing to `2026-04-26` and then hit missing forecast data for that same date.

Resolution applied:

- Added a shared resolution layer in `src/api/serving_defaults.py`.
- Updated `src/api/routes/metadata.py`, `src/api/routes/health.py`, and `src/api/routes/forecasts.py`.
- The API now exposes:
  - `latest_extraction_date`: latest partition aligned with effective prediction and forecast serving
  - `latest_trained_extraction_date`: latest training partition detected locally

Result after fix:

- Metadata now reports `latest_extraction_date=2026-04-22`.
- The latest local training partition remains visible as `latest_trained_extraction_date=2026-04-26`.
- `GET /forecasts/{symbol}` without `extraction_date` now resolves consistently with the API defaults.

### Medium Priority

#### 3. News deduplication crashed on mixed timezone-aware and missing timestamps

Observed behavior:

- `NewsAggregatorService._deduplicate()` sorted by `published_at_utc or datetime.min`.
- Mixing timezone-aware datetimes with `None` produced a comparison between aware and naive timestamps.
- This raised `TypeError: can't compare offset-naive and offset-aware datetimes`.

Resolution applied:

- Updated `src/application/services/news_aggregator_service.py`.
- Added timestamp normalization to UTC before sorting and before computing deduplication buckets.
- Added regression coverage in `tests/test_news_aggregator_service.py`.

Result after fix:

- Deduplication now handles mixed aware and missing timestamps safely.
- Naive timestamps are normalized as UTC for bucketing.

#### 4. QA and test dependencies were undocumented in the actual install manifests

Observed behavior:

- The README referenced `pytest`, `pytest-cov`, `black`, `flake8`, `pre-commit`, and API testing flows.
- The local dependency manifest only contained runtime dependencies.
- `FastAPI TestClient` could not run until `httpx` was installed.

Resolution applied:

- Added `requirements-dev.txt`.
- Included:
  - `pytest`
  - `pytest-cov`
  - `httpx`
  - `black`
  - `flake8`
  - `pre-commit`
- Updated the README to distinguish runtime dependencies from development and QA dependencies.

Result after fix:

- The repository now declares the toolchain it documents.
- Local QA commands and `TestClient` usage are supported after installing `requirements-dev.txt`.

## Validation Performed

The following validation steps were executed locally after the changes:

- Compiled modified modules with `compileall`.
- Imported and smoke-tested the core API and front-end modules.
- Called the Flask dashboard route and confirmed successful rendering.
- Exercised the prediction and forecast services directly against local artifacts.
- Verified FastAPI `TestClient` after installing `httpx`.
- Called `GET /health` through `TestClient` and confirmed a `200` response.
- Ran `python -m unittest discover -s tests`.
- Ran `pytest tests -q --maxfail=1`.

Observed validation results:

- `unittest`: `OK`
- `pytest`: `2 passed`
- `GET /health`: `200`

## Files Changed During Remediation

- `src/application/services/predictor_service.py`
- `src/application/services/future_prediction_service.py`
- `src/application/services/news_aggregator_service.py`
- `src/api/routes/metadata.py`
- `src/api/routes/health.py`
- `src/api/routes/forecasts.py`
- `src/api/serving_defaults.py`
- `src/api/schemas/method_response.py`
- `src/api/schemas/data_usage_response.py`
- `src/api/schemas/health_response.py`
- `tests/test_news_aggregator_service.py`
- `requirements-dev.txt`
- `README.md`

## Residual Risks

The following risks remain:

- Some symbols still rely heavily on serving guardrails, which suggests that model-quality issues may still exist in specific training artifacts even though the serving selector is now safer.
- `POST /predict/enriched` and `GET /news/{symbol}` still intentionally return `501` and remain implementation gaps rather than regressions.
- Parts of the README still describe legacy structure and historical plans that do not fully match the live repository state.

## Recommended Next Steps

1. Add broader regression coverage for `predict`, `forecasts`, and metadata alignment through FastAPI endpoint tests.
2. Introduce an explicit model promotion policy so only approved artifacts become serving candidates.
3. Clean up legacy manifests or backfill immutable artifact references where possible.
4. Reconcile remaining README sections with the actual repository structure and current implementation status.
