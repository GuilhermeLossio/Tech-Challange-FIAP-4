# Software Architecture — Tech Challenge Phase 4

> Detailed technical documentation of the architecture, design patterns, engineering decisions, and news sentiment integration for the multi-asset semiconductor LSTM forecasting system.

---

## Table of Contents

| Section | Description |
|---|---|
| [Overview](#overview) | Architectural philosophy and layer diagram |
| [System Layers](#system-layers) | Clean Architecture layer breakdown |
| [Multi-Asset Coverage](#multi-asset-coverage) | Covered semiconductor companies |
| [News Intelligence System](#news-intelligence-system) | NLP pipeline and sentiment integration |
| [Design Patterns](#design-patterns) | Strategy, Repository, Factory, Observer |
| [SOLID Principles](#solid-principles) | Applied principles with concrete examples |
| [Data Flow](#detailed-data-flow) | End-to-end request flow with news context |
| [Testing Strategy](#testing-strategy) | Pyramid, unit, integration examples |
| [Dependency Injection](#dependency-injection) | Manual composition at entry point |
| [CI/CD Pipeline](#cicd-pipeline) | Stages, gates, and branch strategy |
| [Architecture Decisions](#architecture-decisions-adr) | ADR-001 through ADR-006 |

---

## Overview

This project is structured into independent, decoupled layers following **Clean Architecture** and **Domain-Driven Design (DDD)** principles. The system extends the classic LSTM price forecasting pipeline with a **News Intelligence Layer** that aggregates real-time semiconductor news from multiple sources, applies NLP sentiment analysis, and injects the resulting signal as an additional feature into the model input.

```
+----------------------------------------------------------+
|                  API Layer  (FastAPI)                    |
|       Input validation · Routing · Serialization        |
+----------------------------------------------------------+
|                 Application Layer                        |
|        Use Cases · Services · Orchestration             |
+----------------------------------------------------------+
|                   Domain Layer                           |
|       Entities · Interfaces · Business Rules            |
+----------------------------------------------------------+
|               Infrastructure Layer                       |
|   yfinance · Keras · NewsAPI · Scrapers · Cache         |
+----------------------------------------------------------+
|              News Intelligence Layer  [NEW]              |
|    Aggregator · NLP Pipeline · Sentiment Store          |
+----------------------------------------------------------+
```

---

## System Layers

### 1. Infrastructure Layer

Responsible for all communication with the outside world: financial data APIs, trained model persistence, prediction caching, raw news ingestion, and raw market data persistence in the local raw zone plus AWS S3.

```
src/infrastructure/
├── repositories/
│   ├── yfinance_repository.py       # Fetches OHLCV data per asset
│   ├── model_repository.py          # Loads/saves .keras models
│   └── news_repository.py           # Raw news fetching (API + scraper)
├── storage/
│   ├── local_raw_store.py           # Persists raw files under data/raw
│   ├── local_processed_store.py     # Persists refined parquet files under data/processed
│   └── s3_raw_store.py              # Uploads raw files and manifests to S3 in parquet
├── cache/
│   └── prediction_cache.py          # In-memory TTL cache (5 min)
└── config/
    └── settings.py                  # Environment configuration for raw/S3 settings
```

### 2. Domain Layer

The system core. Contains no external dependencies. Defines abstract contracts and domain entities for both financial data and news signals.

```
src/domain/
├── entities/
│   ├── stock.py                     # Stock entity (symbol, prices, metadata)
│   ├── prediction.py                # Prediction (value, confidence, timestamp)
│   └── news_signal.py               # NewsSignal (headline, source, sentiment, score)
├── interfaces/
│   ├── i_stock_repository.py        # Financial data contract
│   ├── i_model.py                   # Predictive model contract
│   ├── i_scaler.py                  # Normalization contract
│   ├── i_news_repository.py         # News ingestion contract
│   └── i_sentiment_analyzer.py      # NLP sentiment contract
└── exceptions/
    ├── stock_not_found.py
    ├── insufficient_data.py
    └── news_fetch_error.py
```

### 3. Application Layer

Orchestrates use cases. Coordinates domain and infrastructure without containing business logic. The `EnrichedPredictorService` merges price sequences with live sentiment scores before inference.

```
src/application/
├── use_cases/
│   ├── predict_closing_price.py     # Price-only prediction
│   ├── predict_with_sentiment.py    # Sentiment-enriched prediction  [NEW]
│   ├── train_model.py               # Training use case
│   └── evaluate_model.py            # Evaluation use case
└── services/
    ├── predictor_service.py         # Core LSTM inference
    ├── enriched_predictor_service.py # LSTM + sentiment fusion  [NEW]
    ├── news_aggregator_service.py   # Multi-source news fusion  [NEW]
    ├── data_pipeline_service.py      # Full preprocessing pipeline
    └── refined_data_pipeline_service.py # Raw-to-refined preprocessing pipeline
```

### 4. API Layer

HTTP interface. Validates inputs, formats outputs, no business logic.

```
src/api/
├── main.py                          # FastAPI app factory
├── routes/
│   ├── predict.py                   # POST /predict
│   ├── predict_enriched.py          # POST /predict/enriched  [NEW]
│   ├── news.py                      # GET  /news/{symbol}     [NEW]
│   ├── health.py                    # GET  /health
│   └── metrics.py                   # GET  /metrics
└── schemas/
    ├── predict_request.py
    ├── predict_response.py
    ├── enriched_predict_request.py   # [NEW]
    └── news_response.py              # [NEW]
```

---

## Multi-Asset Coverage

The system supports five semiconductor companies that represent the full production chain — from chip design to fabrication equipment — providing cross-asset correlation signals that improve individual predictions.

| Symbol | Company | Role in Supply Chain | Exchange |
|---|---|---|---|
| `NVDA` | NVIDIA Corporation | GPU design, AI accelerators | NASDAQ |
| `AMD` | Advanced Micro Devices | CPU/GPU design, direct NVDA competitor | NASDAQ |
| `TSM` | Taiwan Semiconductor Mfg. | World's largest contract foundry | NYSE (ADR) |
| `ASML` | ASML Holding | Monopoly on EUV lithography equipment | NASDAQ |
| `QCOM` | Qualcomm | Mobile SoC design, 5G semiconductors | NASDAQ |

### Why this set?

```
[ASML] ──── supplies lithography machines to ────► [TSM]
                                                      │
                                          fabricates chips for
                                                      │
                                          ┌───────────┼───────────┐
                                          ▼           ▼           ▼
                                        [NVDA]      [AMD]       [QCOM]
```

Movements in ASML and TSM often lead NVDA/AMD/QCOM by days or weeks — a causal structure the LSTM can exploit when trained on all five assets simultaneously.

### Multi-Asset Data Collection

```python
SYMBOLS = ["NVDA", "AMD", "TSM", "ASML", "QCOM"]
START_DATE = "2018-01-01"
END_DATE   = "2025-12-31"

def fetch_all(symbols: list[str]) -> dict[str, pd.DataFrame]:
    return {
        symbol: yf.download(symbol, start=START_DATE, end=END_DATE)[["Close"]]
        for symbol in symbols
    }

assets = fetch_all(SYMBOLS)
```

---

## News Intelligence System

The News Intelligence Layer fetches headlines and articles from multiple sources, applies NLP sentiment analysis, and produces a daily sentiment score per asset that is fused with the price sequence before LSTM inference.

### Architecture

```
[NewsAPI]  [Alpha Vantage News]  [Reuters RSS]  [SeekingAlpha Scraper]
     │              │                  │                   │
     └──────────────┴──────────────────┴───────────────────┘
                                │
                    [NewsAggregatorService]
                     Deduplication · Filtering · Normalization
                                │
                    [SentimentPipeline]
                     FinBERT · Scoring · Aggregation
                                │
                    [SentimentStore]
                     Daily scores per symbol (SQLite / Redis)
                                │
                    [EnrichedPredictorService]
                     Fuses price_sequence + sentiment_vector
                                │
                    [LSTM Model]
                     Input: (60 timesteps × 2 features)
```

### News Sources

| Source | Type | Coverage | Rate Limit |
|---|---|---|---|
| NewsAPI | REST API | General financial news | 100 req/day (free) |
| Alpha Vantage News | REST API | Stock-specific news + sentiment | 25 req/day (free) |
| Reuters RSS | RSS Feed | Wire news, no rate limit | Unlimited |
| SeekingAlpha | Web scraper | Analyst articles, earnings commentary | Respectful crawl delay |

### NewsAggregatorService

```python
class NewsAggregatorService:
    def __init__(
        self,
        sources: list[INewsRepository],
        dedup_window_hours: int = 6
    ):
        self._sources = sources
        self._dedup_window = dedup_window_hours

    def fetch(self, symbol: str, date: str) -> list[NewsSignal]:
        raw: list[NewsSignal] = []
        for source in self._sources:
            try:
                raw.extend(source.fetch(symbol=symbol, date=date))
            except NewsFetchError:
                continue                          # graceful degradation

        return self._deduplicate(raw)

    def _deduplicate(self, signals: list[NewsSignal]) -> list[NewsSignal]:
        seen, result = set(), []
        for s in signals:
            key = (s.symbol, s.headline[:60])     # fuzzy key
            if key not in seen:
                seen.add(key)
                result.append(s)
        return result
```

### NLP Sentiment Pipeline

The system uses **FinBERT** — a BERT model fine-tuned on financial text — as the primary sentiment analyzer, with a TextBlob fallback for when the model is unavailable.

```python
from transformers import pipeline

class FinBERTSentimentAnalyzer(ISentimentAnalyzer):
    def __init__(self):
        self._pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert"
        )

    def analyze(self, text: str) -> float:
        """Returns a score in [-1.0, +1.0]: negative → positive."""
        result = self._pipe(text[:512])[0]
        label, score = result["label"], result["score"]
        if label == "positive":
            return +score
        if label == "negative":
            return -score
        return 0.0                                 # neutral


class TextBlobFallbackAnalyzer(ISentimentAnalyzer):
    def analyze(self, text: str) -> float:
        from textblob import TextBlob
        return TextBlob(text).sentiment.polarity   # [-1.0, +1.0]
```

### Daily Sentiment Aggregation

Multiple articles per day are reduced to a single composite score per asset:

```python
def aggregate_daily_sentiment(
    signals: list[NewsSignal],
    method: str = "weighted_mean"
) -> float:
    if not signals:
        return 0.0                                 # neutral on no-news days

    scores  = [s.sentiment_score for s in signals]
    weights = [s.source_weight   for s in signals]

    if method == "weighted_mean":
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    elif method == "mean":
        return sum(scores) / len(scores)
```

Source weights reflect reliability:

| Source | Weight |
|---|---|
| Reuters | 1.0 |
| Alpha Vantage | 0.9 |
| NewsAPI | 0.7 |
| SeekingAlpha | 0.6 |

### Feature Fusion

The daily sentiment score is appended as a second feature channel to the price sequence, giving the LSTM a 2-feature input:

```python
# price_seq:     shape (60, 1) — normalized closing prices
# sentiment_seq: shape (60, 1) — daily sentiment scores [-1, +1]

X_enriched = np.concatenate([price_seq, sentiment_seq], axis=-1)
# X_enriched: shape (60, 2)

# Model input_shape becomes (60, 2) instead of (60, 1)
model = build_model(input_shape=(60, 2))
```

### New API Endpoint

```
POST /predict/enriched

Request:
{
  "symbol": "NVDA",
  "prices": [...60 floats...],
  "include_live_news": true       // fetches today's news automatically
}

Response:
{
  "symbol": "NVDA",
  "predicted_close": 487.32,
  "sentiment_score": 0.43,
  "sentiment_label": "positive",
  "news_count": 12,
  "top_headline": "NVIDIA beats Q2 estimates on AI demand surge",
  "lower_bound": 479.10,
  "upper_bound": 495.54,
  "confidence": 0.95,
  "currency": "USD",
  "model": "lstm_nvda_enriched",
  "timestamp": "2024-07-21T14:32:00Z"
}
```

---

## Design Patterns

### Strategy Pattern — Data Normalization

Allows the scaling algorithm to be swapped without pipeline changes.

```python
from abc import ABC, abstractmethod
import numpy as np

class IScaler(ABC):
    @abstractmethod
    def fit_transform(self, data: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray: ...


class MinMaxStrategy(IScaler):
    def __init__(self, feature_range=(0, 1)):
        from sklearn.preprocessing import MinMaxScaler
        self._scaler = MinMaxScaler(feature_range=feature_range)

    def fit_transform(self, data):
        return self._scaler.fit_transform(data)

    def inverse_transform(self, data):
        return self._scaler.inverse_transform(data)


class StandardStrategy(IScaler):
    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler()

    def fit_transform(self, data):
        return self._scaler.fit_transform(data)

    def inverse_transform(self, data):
        return self._scaler.inverse_transform(data)
```

---

### Repository Pattern — Data Collection

Decouples data sources from business logic. Both financial data and news sources follow the same pattern.

```python
from abc import ABC, abstractmethod
import pandas as pd

class IStockRepository(ABC):
    @abstractmethod
    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame: ...


class YFinanceRepository(IStockRepository):
    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        import yfinance as yf
        df = yf.download(symbol, start=start, end=end)
        if df.empty:
            raise StockNotFoundException(f"No data for {symbol}")
        return df[["Close"]].dropna()


class AlphaVantageRepository(IStockRepository):
    """Alternative implementation — same interface, zero domain impact."""
    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame: ...


# News follows the exact same contract
class INewsRepository(ABC):
    @abstractmethod
    def fetch(self, symbol: str, date: str) -> list[NewsSignal]: ...


class NewsAPIRepository(INewsRepository): ...
class ReutersRSSRepository(INewsRepository): ...
class SeekingAlphaRepository(INewsRepository): ...
```

---

### Factory Pattern — Model Instantiation

Centralized, extensible model registry. Adding GRU or Transformer requires zero changes to existing code.

```python
class ModelFactory:
    _registry: dict = {}

    @classmethod
    def register(cls, name: str):
        def decorator(model_cls):
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, model_type: str, **kwargs) -> IModel:
        if model_type not in cls._registry:
            raise ValueError(f"Unregistered model: '{model_type}'")
        return cls._registry[model_type](**kwargs)


@ModelFactory.register("lstm")
class LSTMModel(IModel): ...

@ModelFactory.register("lstm_enriched")
class LSTMEnrichedModel(IModel): ...       # 2-feature input

@ModelFactory.register("gru")
class GRUModel(IModel): ...
```

---

### Observer Pattern — Monitoring and Logging

Adds cross-cutting behaviors (metrics, logging, alerts) without touching core services.

```python
class PredictionObserver(ABC):
    @abstractmethod
    def on_prediction(self, symbol: str, value: float, latency_ms: float): ...


class PrometheusObserver(PredictionObserver):
    def on_prediction(self, symbol, value, latency_ms):
        prediction_counter.labels(symbol=symbol).inc()
        latency_histogram.observe(latency_ms / 1000)


class LogObserver(PredictionObserver):
    def on_prediction(self, symbol, value, latency_ms):
        logger.info(f"[{symbol}] predicted={value:.4f} latency={latency_ms:.1f}ms")


class SentimentDriftObserver(PredictionObserver):
    """Alerts when sentiment diverges significantly from recent baseline."""
    def on_prediction(self, symbol, value, latency_ms):
        ...


class PredictorService:
    def __init__(self, observers: list[PredictionObserver]):
        self._observers = observers

    def predict(self, symbol: str, prices: list[float]) -> float:
        import time
        start  = time.perf_counter()
        result = self._model.predict(prices)
        latency = (time.perf_counter() - start) * 1000
        for obs in self._observers:
            obs.on_prediction(symbol, result, latency)
        return result
```

---

## SOLID Principles

| Principle | Application in the Project |
|---|---|
| **S** ingle Responsibility | `Preprocessor` only transforms; `LSTMModel` only trains/predicts; `NewsAggregatorService` only aggregates |
| **O** pen/Closed | New models via `@ModelFactory.register`; new news sources via `INewsRepository` — zero changes to existing code |
| **L** iskov Substitution | `YFinanceRepository`, `AlphaVantageRepository`; `FinBERTAnalyzer`, `TextBlobFallbackAnalyzer` — fully interchangeable |
| **I** nterface Segregation | `IModel`, `IScaler`, `IStockRepository`, `INewsRepository`, `ISentimentAnalyzer` — each minimal and focused |
| **D** ependency Inversion | All services receive abstract interfaces through constructors; no `new` calls inside business logic |

---

## Detailed Data Flow

### Standard Prediction

```
POST /predict
     │
     ▼
[PredictRequest]                  ← Pydantic validation
     │
     ▼
[PredictorService]                ← Application Layer
     │
     ├──► [PredictionCache]       ← Return cached result if TTL valid
     │
     ├──► [IScaler.transform()]   ← Normalize price input (0–1)
     │
     ├──► [IModel.predict()]      ← LSTM inference  (60 × 1)
     │
     ├──► [IScaler.inverse()]     ← Denormalize to USD
     │
     ├──► [Observers.notify()]    ← Logs + Prometheus
     │
     └──► [PredictResponse]       ← JSON output
```

### Sentiment-Enriched Prediction

```
POST /predict/enriched
     │
     ▼
[EnrichedPredictRequest]          ← Pydantic validation
     │
     ├──► [NewsAggregatorService] ← Fetch & deduplicate today's news
     │         │
     │         ├── NewsAPIRepository
     │         ├── AlphaVantageNewsRepository
     │         ├── ReutersRSSRepository
     │         └── SeekingAlphaRepository
     │
     ├──► [FinBERTSentimentAnalyzer]  ← Score each headline [-1, +1]
     │
     ├──► [aggregate_daily_sentiment] ← Weighted mean → single score
     │
     ├──► [FeatureFusion]         ← Concat price_seq + sentiment_seq → (60, 2)
     │
     ├──► [IScaler.transform()]   ← Normalize both channels
     │
     ├──► [LSTMEnrichedModel]     ← Inference on (60, 2) input
     │
     ├──► [IScaler.inverse()]     ← Denormalize
     │
     ├──► [Observers.notify()]    ← Logs + Prometheus + SentimentDrift
     │
     └──► [EnrichedPredictResponse] ← JSON with price + sentiment context
```

---

## Testing Strategy

### Testing Pyramid

```
            /\
           /  \        E2E (1–2 tests — full Docker stack)
          /----\
         /      \      Integration (API endpoints + DB)
        /--------\
       /          \    Unit (domain, services, NLP pipeline)
      /------------\
```

### Unit — Predictor Service

```python
# tests/unit/test_predictor_service.py
from unittest.mock import MagicMock
from src.application.services.predictor_service import PredictorService

def test_predict_returns_denormalized_value():
    mock_model  = MagicMock()
    mock_scaler = MagicMock()
    mock_model.predict.return_value           = [[0.75]]
    mock_scaler.inverse_transform.return_value = [[487.32]]

    service = PredictorService(model=mock_model, scaler=mock_scaler, observers=[])
    result  = service.predict("NVDA", prices=[1.0] * 60)

    assert result == pytest.approx(487.32, rel=1e-3)
```

### Unit — Sentiment Pipeline

```python
# tests/unit/test_sentiment.py
from src.infrastructure.nlp.finbert_analyzer import FinBERTSentimentAnalyzer

def test_positive_headline_returns_positive_score():
    analyzer = FinBERTSentimentAnalyzer()
    score = analyzer.analyze("NVIDIA beats earnings expectations on AI demand surge")
    assert score > 0.0

def test_negative_headline_returns_negative_score():
    score = analyzer.analyze("ASML misses revenue forecast amid export restrictions")
    assert score < 0.0

def test_neutral_headline_returns_near_zero():
    score = analyzer.analyze("TSMC to hold quarterly earnings call next Tuesday")
    assert abs(score) < 0.3
```

### Integration — API

```python
# tests/integration/test_predict_endpoint.py
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_standard_predict_returns_200():
    payload = {"symbol": "NVDA", "prices": [float(400 + i * 0.5) for i in range(60)]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_close" in response.json()

def test_enriched_predict_includes_sentiment():
    payload = {
        "symbol": "NVDA",
        "prices": [float(400 + i * 0.5) for i in range(60)],
        "include_live_news": False
    }
    response = client.post("/predict/enriched", json=payload)
    data = response.json()
    assert "sentiment_score" in data
    assert "sentiment_label" in data
```

### Execution

```bash
# Full suite with HTML coverage report
pytest tests/ -v --cov=src --cov-report=html
open htmlcov/index.html

# Unit only
pytest tests/unit/ -v

# Integration only
pytest tests/integration/ -v
```

| Layer | Location | Coverage Target |
|---|---|---|
| Unit | `tests/unit/` | >= 80% |
| Integration | `tests/integration/` | All endpoints |
| Model | `tests/test_model.py` | Output shapes, value ranges |
| NLP | `tests/unit/test_sentiment.py` | Score direction, edge cases |

---

## Dependency Injection

System composition happens at the entry point (`main.py`) — no DI framework magic, full transparency.

```python
# src/api/main.py

def build_predictor_service() -> PredictorService:
    scaler     = MinMaxStrategy()
    model      = ModelRepository.load("models/lstm_nvda.keras")
    observers  = [PrometheusObserver(), LogObserver()]
    return PredictorService(model=model, scaler=scaler, observers=observers)


def build_enriched_predictor_service() -> EnrichedPredictorService:
    price_scaler     = MinMaxStrategy()
    sentiment_scaler = MinMaxStrategy(feature_range=(-1, 1))
    model            = ModelRepository.load("models/lstm_nvda_enriched.keras")

    news_sources = [
        NewsAPIRepository(api_key=settings.NEWSAPI_KEY),
        AlphaVantageNewsRepository(api_key=settings.ALPHAVANTAGE_KEY),
        ReutersRSSRepository(),
        SeekingAlphaRepository(crawl_delay=2.0),
    ]
    aggregator = NewsAggregatorService(sources=news_sources)
    analyzer   = FinBERTSentimentAnalyzer()
    observers  = [PrometheusObserver(), LogObserver(), SentimentDriftObserver()]

    return EnrichedPredictorService(
        model=model,
        price_scaler=price_scaler,
        sentiment_scaler=sentiment_scaler,
        aggregator=aggregator,
        analyzer=analyzer,
        observers=observers
    )


predictor_service          = build_predictor_service()
enriched_predictor_service = build_enriched_predictor_service()
```

---

## CI/CD Pipeline

```
push / pull_request
        │
        ├── [lint]      black --check · flake8
        │
        ├── [test]      pytest --cov=src
        │               fails if line coverage < 80%
        │
        ├── [build]     docker build -t lstm-api:$SHA .
        │
        ├── [scan]      trivy image (CVE vulnerability scan)
        │
        └── [deploy]    docker push → production environment
```

Branch strategy: `main` (production) ← `develop` ← `feature/*`  
Commit convention: [Conventional Commits](https://www.conventionalcommits.org) — `feat:`, `fix:`, `docs:`, `test:`, `chore:`

---

## Architecture Decisions (ADR)

### ADR-001 — FastAPI over Flask

**Decision:** Use FastAPI.  
**Reason:** Automatic Pydantic validation, native OpenAPI 3.0 docs, async support, and measurably better I/O-bound performance. Schema-first design enforces contract discipline across all endpoints.

---

### ADR-002 — Model serialized as `.keras` (not `.h5`)

**Decision:** Native Keras format.  
**Reason:** More robust than HDF5 for custom layer subclasses, natively supports arbitrary Python objects, and is the recommended default from TensorFlow 2.12 onward.

---

### ADR-003 — In-memory cache (not Redis) for MVP

**Decision:** TTL cache in process memory.  
**Reason:** Removes an external service dependency in the initial phase, reducing operational complexity. The `ICache` interface means Redis can be injected at the infrastructure layer with zero domain impact when needed.

---

### ADR-004 — FinBERT over general-purpose BERT for sentiment

**Decision:** Use `ProsusAI/finbert` as the primary sentiment model.  
**Reason:** FinBERT was trained on 10,000+ financial news articles and analyst reports. General-purpose sentiment models misclassify financial language — for example, "strong resistance" (bearish in trading) would score as positive with standard BERT. A TextBlob fallback is retained for availability when GPU inference is not accessible.

---

### ADR-005 — Multi-source news aggregation with deduplication

**Decision:** Combine NewsAPI + Alpha Vantage News + Reuters RSS + SeekingAlpha scraper with an in-process deduplication step.  
**Reason:** No single source provides complete coverage of semiconductor sector news. Reuters RSS is free and reliable for wire-level events; Alpha Vantage provides stock-specific signals; SeekingAlpha covers analyst commentary not found on wires. Deduplication by (symbol, headline prefix) prevents sentiment score inflation from identical stories republished across outlets.

---

### ADR-006 — Feature fusion (price + sentiment) as 2-channel input

**Decision:** Concatenate normalized sentiment scores as a second feature channel rather than training a separate sentiment model.  
**Reason:** Late fusion (separate models combined at output) requires calibrated probability estimates that are difficult to obtain from regression. Early fusion (concatenated input) allows the LSTM to learn interaction effects between price momentum and sentiment directly — for example, detecting when a positive price trend is contradicted by deteriorating news sentiment, which often precedes reversals.

---

## Author

**Guilherme Lossio**  
Postgraduate Program in Machine Learning Engineering — Tech Challenge Phase 4
