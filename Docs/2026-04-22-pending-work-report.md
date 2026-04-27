# Pending Work Report

## Scope

This report lists what is still missing to match the implementation promised by the current project documentation in `README.md` and `ARCHITECTURE.md`.

## Already Delivered

- Raw market data ingestion exists through `scripts/generate_raw.py`.
- Refined dataset generation exists through `scripts/generate_refined.py`.
- Feature dataset generation exists through `scripts/generate_features.py`.
- Monthly forecast batch generation exists through `scripts/generate_forecast.py`.
- Athena catalog provisioning exists through `scripts/provision_athena.py`.
- Classical Keras training exists through `scripts/train_keras.py` and `src/application/use_cases/train_model.py`.
- Quantum training exists through `scripts/train_model_quantum.py` and `src/application/use_cases/train_model_quantum.py`.
- Classical-versus-quantum comparison exists through `scripts/train_and_compare_models.py`.
- News aggregation logic exists in `src/application/services/news_aggregator_service.py`.

## Pending Work

### 1. API layer is still missing

The documentation describes a FastAPI application with `POST /predict`, `POST /predict/enriched`, `GET /health`, `GET /metrics`, and news endpoints. Those files and folders do not exist yet.

Missing paths:

- `src/api/main.py`
- `src/api/routes/`
- `src/api/schemas/` or DTO equivalents
- `src/application/services/predictor_service.py`
- `src/application/services/enriched_predictor_service.py`

### 2. Sentiment-enriched prediction is not implemented end-to-end

The documentation presents the enriched workflow as a working prediction path, but the repository currently contains only the aggregation service and the `NewsSignal` entity. The rest of the sentiment stack is missing.

Missing or unfinished items:

- news source adapters such as `news_repository.py`
- sentiment analyzers such as `finbert_analyzer.py`
- fallback analyzer implementation
- enriched prediction use case
- enriched model loading/inference path
- enriched model artifact `models/lstm_nvda_enriched.keras`
- `/predict/enriched` and `/news/{symbol}` endpoints

### 3. Multi-asset context is not implemented in training

The documentation says the model benefits from cross-asset structure and can exploit all five assets together. The current training code does not do that.

Current behavior:

- `src/application/use_cases/train_model.py` iterates `for symbol in request.symbols`
- each dataset is loaded independently per symbol
- sequence columns are built from that symbol only

What is still needed:

- a joined multi-asset training dataset
- cross-asset features or merged sequences
- model input definitions that actually include other symbols
- evaluation showing the benefit of multi-asset training

### 4. Infrastructure adapters promised by the docs are missing

The architecture documentation references several adapters and repositories that are not present in the codebase.

Missing paths:

- `src/infrastructure/repositories/model_repository.py`
- `src/infrastructure/repositories/news_repository.py`
- `src/infrastructure/cache/prediction_cache.py`
- `src/infrastructure/nlp/finbert_analyzer.py`

The same applies to several domain and application files listed in the documented folder structure but absent from the repository.

### 5. Test suite and quality gates are still missing

The documentation promises unit tests, integration tests, model tests, NLP tests, coverage, and CI quality gates. No `tests/` directory exists.

Missing items:

- `tests/unit/`
- `tests/integration/`
- API endpoint tests
- training/use case tests
- sentiment pipeline tests
- coverage setup wired into automation

### 6. CI/CD automation is still missing

The documentation references GitHub Actions and a workflow file, but `.github/workflows/ci.yml` is not present.

Missing items:

- `.github/workflows/ci.yml`
- automated lint/test execution
- PR validation gates

### 7. Docker and deployment assets are still missing

The README documents Docker and Compose usage, but the repository does not currently contain those files.

Missing items:

- `Dockerfile`
- `docker-compose.yml`
- container wiring for API, Prometheus, and Grafana

### 8. Monitoring assets are still missing

The README documents metrics, Prometheus scraping, Grafana dashboards, and alerts. Those assets are not present in the repository.

Missing items:

- `monitoring/prometheus.yml`
- `monitoring/grafana/`
- metrics endpoint implementation
- prediction observers and alerting hooks

### 9. Documentation is ahead of the implementation

Today the documentation describes a larger product than what the codebase actually ships. This creates a planning task of its own:

- either implement the missing API, monitoring, NLP, and deployment layers
- or reduce the documentation so it reflects the actual delivered scope

## Recommended Order

1. Decide whether the documented FastAPI + enriched prediction product remains in scope.
2. Implement the baseline API layer for the already trained Keras model.
3. Add tests and CI before expanding the surface area further.
4. Implement the missing news adapters and sentiment analyzers.
5. Deliver the enriched inference path and enriched model artifact.
6. Add Docker, monitoring, and deployment assets.
7. Rework training if true multi-asset context is a real requirement rather than a documentation placeholder.
