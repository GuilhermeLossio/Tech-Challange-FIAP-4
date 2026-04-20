<div align="center">

# 📈 Tech Challenge Phase 4
## Stock Price Forecasting with LSTM Neural Networks

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white)](https://github.com/features/actions)

> End-to-end deep learning solution for semiconductor stock forecasting with LSTM models, multi-asset context, and optional news sentiment enrichment.

**Primary asset:** `NVDA` · **Multi-asset coverage:** `NVDA`, `AMD`, `TSM`, `ASML`, `QCOM` · **Period:** 2018-2025

</div>

---

## 📋 Table of Contents

| Section | Description |
|---|---|
| [About](#-about-the-project) | Academic context, scope, and forecasting modes |
| [Architecture](#-system-architecture) | System layers, news intelligence, and data flow |
| [Technologies](#-technologies) | Core stack, NLP, and external integrations |
| [Folder Structure](#-folder-structure) | Project layout and main modules |
| [Quickstart](#-quickstart) | Installation, environment, and execution |
| [Data Pipeline](#-data-pipeline) | Multi-asset collection, preprocessing, sentiment fusion |
| [LSTM Model](#-lstm-model) | Baseline and enriched model variants |
| [Evaluation](#-evaluation-metrics) | MAE, RMSE, MAPE, and comparison criteria |
| [API Reference](#-api-reference) | Standard and enriched endpoints |
| [Docker](#-docker--deployment) | Build, compose, and environment variables |
| [Monitoring](#-monitoring) | Metrics, alerting, and enriched-path observability |
| [Software Engineering](#-software-engineering) | Patterns, SOLID, tests, CI/CD |
| [Results](#-results) | Baseline vs enriched benchmarks |
| [Author](#-author) | Contact information |

---

## 🎯 About the Project

This project was developed as the capstone deliverable for **Tech Challenge Phase 4** of the Machine Learning Engineering postgraduate program. It covers the full machine learning lifecycle for financial time-series forecasting, from raw data ingestion to a production-ready API.

### Problem Statement

Predicting short-term stock closing prices is a sequence learning problem with non-linear temporal dependencies. Classical statistical methods such as ARIMA or ETS struggle to capture abrupt regime changes, delayed reactions, and cross-asset interactions. **LSTM networks** are well suited to this setting because they learn long-range dependencies through gated memory.

This project documents two prediction modes:

- **Standard forecasting:** price-only inference from historical closing prices
- **Sentiment-enriched forecasting:** price inference augmented with live semiconductor news sentiment

### Why Semiconductors?

The semiconductor sector was selected based on three criteria:

- **Signal richness:** high volatility creates expressive temporal patterns for sequence models
- **Economic sensitivity:** the sector reacts quickly to macro cycles, AI demand, supply chain disruptions, and export restrictions
- **Cross-company structure:** foundries, equipment vendors, and chip designers influence each other in observable ways

### Scope

| Item | Detail |
|---|---|
| Primary asset | NVDA - NVIDIA Corporation (NASDAQ) |
| Multi-asset universe | `NVDA`, `AMD`, `TSM`, `ASML`, `QCOM` |
| Historical window | 2018-01-01 -> 2025-12-31 |
| Prediction target | Next-day closing price (D+1) |
| Data sources | Yahoo Finance via `yfinance` and live news feeds for enrichment |
| Prediction modes | `standard` and `sentiment-enriched` |
| Model variants | `lstm_nvda.keras` and `lstm_nvda_enriched.keras` |

---

## 🏗️ System Architecture

The system follows a **Clean Architecture** core with a dedicated **News Intelligence Layer** that enriches the forecasting pipeline with live market context. The technical architecture, ADRs, and implementation rationale are documented in [ARCHITECTURE.md](ARCHITECTURE.md).

```text
+----------------------------------------------------------+
|                  API Layer  (FastAPI)                    |
|       Input validation · Routing · Serialization         |
+----------------------------------------------------------+
|                 Application Layer                        |
|        Use Cases · Services · Orchestration              |
+----------------------------------------------------------+
|                   Domain Layer                           |
|       Entities · Interfaces · Business Rules             |
+----------------------------------------------------------+
|               Infrastructure Layer                       |
|   yfinance · Keras · News APIs · Scrapers · Cache        |
+----------------------------------------------------------+
|              News Intelligence Layer                     |
|   Aggregation · NLP Sentiment · Feature Enrichment       |
+----------------------------------------------------------+
```

### End-to-End Data Flow

```text
[Yahoo Finance] --------------------> [Financial Repositories]
                                            |
                                            v
                                     [Preprocessor]
                                            |
                                            +----> [Standard LSTM Model] ----> [POST /predict]
                                            |
[NewsAPI / Alpha Vantage / RSS / Scrapers] -> [NewsAggregatorService]
                                                     |
                                                     v
                                           [FinBERT / Fallback Analyzer]
                                                     |
                                                     v
                                           [Feature Fusion (price + sentiment)]
                                                     |
                                                     v
                                         [Enriched LSTM Model] ----> [POST /predict/enriched]
```

---

## 🛠️ Technologies

### Core Stack

| Category | Library / Service | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.11 | Runtime |
| Data collection | yfinance | 0.2.x | Historical OHLCV retrieval |
| Data processing | pandas | 2.x | DataFrame manipulation |
| Numerical computing | NumPy | 1.26 | Array operations |
| Deep learning | TensorFlow / Keras | 2.15 | Baseline and enriched LSTM models |
| Scaling | scikit-learn | 1.4 | Feature normalization |
| API framework | FastAPI | 0.111 | REST API |
| ASGI server | Uvicorn | 0.29 | Local and production serving |
| Data validation | Pydantic | 2.x | Request and response schemas |
| NLP sentiment | transformers / FinBERT | latest | Financial text sentiment scoring |
| Fallback sentiment | TextBlob | latest | Lightweight fallback polarity scoring |
| Containerization | Docker + Compose | 26.x | Deployment |
| Monitoring | Prometheus + Grafana | latest | Observability |
| Visualization | Matplotlib / Seaborn | 3.8 | Charts and diagnostics |

### External Integrations

| Integration | Type | Purpose |
|---|---|---|
| Yahoo Finance | Market data | Historical closing prices |
| NewsAPI | REST API | Broad financial news coverage |
| Alpha Vantage News | REST API | Stock-specific news signals |
| Reuters RSS | Feed | Low-latency wire news |
| Scraper sources | HTML/RSS | Additional sector commentary |

### Development & Quality

| Tool | Purpose |
|---|---|
| pytest + pytest-cov | Test suite and coverage |
| black + flake8 | Code formatting and linting |
| GitHub Actions | CI/CD pipeline |
| pre-commit | Local quality gates |

---

## 📁 Folder Structure

```text
tech-challenge-phase4/
|
+-- .github/
|   \-- workflows/
|       \-- ci.yml
|
+-- data/
|   +-- raw/
|   \-- processed/
|
+-- notebooks/
|   +-- 01_data_collection.ipynb
|   +-- 02_preprocessing.ipynb
|   +-- 03_lstm_training.ipynb
|   \-- 04_model_evaluation.ipynb
|
+-- src/
|   +-- domain/
|   |   +-- entities/
|   |   |   +-- stock.py
|   |   |   +-- prediction.py
|   |   |   \-- news_signal.py
|   |   \-- interfaces/
|   |       +-- i_stock_repository.py
|   |       +-- i_model.py
|   |       +-- i_scaler.py
|   |       +-- i_news_repository.py
|   |       \-- i_sentiment_analyzer.py
|   |
|   +-- infrastructure/
|   |   +-- repositories/
|   |   |   +-- yfinance_repository.py
|   |   |   +-- model_repository.py
|   |   |   \-- news_repository.py
|   |   +-- nlp/
|   |   |   \-- finbert_analyzer.py
|   |   \-- config/
|   |       \-- settings.py
|   |
|   +-- application/
|   |   +-- use_cases/
|   |   |   +-- predict_closing_price.py
|   |   |   +-- predict_with_sentiment.py
|   |   |   +-- train_model.py
|   |   |   \-- evaluate_model.py
|   |   \-- services/
|   |       +-- predictor_service.py
|   |       +-- enriched_predictor_service.py
|   |       \-- news_aggregator_service.py
|   |
|   \-- api/
|       +-- main.py
|       +-- routes/
|       |   +-- predict.py
|       |   +-- predict_enriched.py
|       |   +-- news.py
|       |   +-- health.py
|       |   \-- metrics.py
|       \-- schemas/
|           +-- predict_request.py
|           +-- predict_response.py
|           +-- enriched_predict_request.py
|           \-- news_response.py
|
+-- models/
|   +-- lstm_nvda.keras
|   \-- lstm_nvda_enriched.keras
|
+-- monitoring/
|   +-- prometheus.yml
|   \-- grafana/
|       \-- dashboard.json
|
+-- tests/
|   +-- unit/
|   |   +-- test_preprocessor.py
|   |   +-- test_predictor_service.py
|   |   \-- test_sentiment.py
|   +-- integration/
|   |   \-- test_predict_endpoint.py
|   \-- test_model.py
|
+-- docker/
|   +-- Dockerfile
|   \-- docker-compose.yml
|
+-- requirements.txt
+-- requirements-dev.txt
+-- .env.example
+-- ARCHITECTURE.md
\-- README.md
```

---

## 🚀 Quickstart

### Prerequisites

- Python 3.11+
- Docker 26+ and Docker Compose v2
- Git

### Option A - Local (virtualenv)

```bash
# 1. Clone
git clone https://github.com/guilherme-lossio/tech-challenge-phase4.git
cd tech-challenge-phase4

# 2. Virtual environment
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows

# 3. Dependencies
pip install -r requirements.txt

# 4. Environment variables
cp .env.example .env              # Linux / macOS
copy .env.example .env            # Windows

# 5. Start API
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Environment Variables

Configure at least the model paths. News credentials are required only for live enrichment features.

```env
MODEL_PATH=models/lstm_nvda.keras
ENRICHED_MODEL_PATH=models/lstm_nvda_enriched.keras
RAW_LOCAL_DIR=data/raw
AWS_REGION=us-east-1
S3_BUCKET_RAW=your-raw-bucket-name
S3_RAW_PREFIX=raw
NEWSAPI_KEY=your_newsapi_key
ALPHAVANTAGE_KEY=your_alpha_vantage_key
```

Notes:

- `POST /predict` works with the standard model only
- `POST /predict/enriched` is the endpoint that uses live or recent news context
- news ingestion should degrade gracefully when a configured source is unavailable, but coverage will be reduced

### Generate Raw Market Data

Use the raw generator to materialize the historical OHLCV zone locally and optionally mirror it to S3. Local files stay in `csv/json`; S3 objects are written in `.parquet`.

```bash
# Local raw zone only
python scripts/generate_raw.py --skip-s3

# Local + S3 upload (requires S3_BUCKET_RAW and AWS credentials)
python scripts/generate_raw.py
```

Raw files are partitioned by source, symbol, and extraction date:

```text
data/raw/
├── market_data/source=yfinance/symbol=NVDA/extraction_date=2026-04-19/ohlcv.csv
└── manifests/extraction_date=2026-04-19/raw_manifest.json
```

```text
s3://your-raw-bucket-name/raw/
├── market_data/source=yfinance/symbol=NVDA/extraction_date=2026-04-19/ohlcv.parquet
└── manifests/extraction_date=2026-04-19/raw_manifest.parquet
```

### Option B - Docker (recommended)

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d

# Check logs
docker-compose logs -f api
```

### Access Points

| Service | URL |
|---|---|
| API (Swagger UI) | http://localhost:8000/docs |
| API (ReDoc) | http://localhost:8000/redoc |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

---

## 🔄 Data Pipeline

### 1. Multi-Asset Collection

```python
import yfinance as yf

SYMBOLS = ["NVDA", "AMD", "TSM", "ASML", "QCOM"]
START_DATE = "2018-01-01"
END_DATE = "2025-12-31"

assets = {
    symbol: yf.download(symbol, start=START_DATE, end=END_DATE)[["Close"]].dropna()
    for symbol in SYMBOLS
}
```

### 2. Normalization

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_nvda = scaler.fit_transform(assets["NVDA"])
```

> Persist the scaler alongside the model so inverse transformation remains consistent in production.

### 3. Sliding Window

```python
import numpy as np

LOOKBACK = 60  # trading days

def create_sequences(data: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_nvda, LOOKBACK)
X = X.reshape(X.shape[0], X.shape[1], 1)
```

### 4. News Sentiment Enrichment

Live or recent news can be aggregated into a daily sentiment score per asset and fused with the price sequence:

```python
# price_seq: shape (60, 1)
# sentiment_seq: shape (60, 1)

X_enriched = np.concatenate([price_seq, sentiment_seq], axis=-1)
# shape: (60, 2)
```

### 5. Train / Validation / Test Split

```python
n = len(X)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]
```

| Split | Proportion | Purpose |
|---|---|---|
| Train | 70% | Weight optimization |
| Validation | 15% | Hyperparameter tuning and early stopping |
| Test | 15% | Final unbiased evaluation |

---

## 🧠 LSTM Model

### Architecture

The project maintains two closely related LSTM variants:

| Variant | Input Shape | Features | Output |
|---|---|---|---|
| Baseline | `(60, 1)` | normalized closing prices | next-day close |
| Enriched | `(60, 2)` | normalized prices + sentiment score | next-day close |

Both variants use the same stacked architecture:

```text
Input Layer       -> 60 timesteps
LSTM Layer 1      -> 128 units, return_sequences=True
Dropout           -> 0.20
LSTM Layer 2      -> 64 units, return_sequences=False
Dropout           -> 0.20
Dense Layer       -> 32 units, activation=relu
Output Layer      -> 1 unit
```

### Keras Implementation

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_model(input_shape: tuple) -> Sequential:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
```

### Training

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint("models/lstm_nvda.keras", save_best_only=True)
]

model = build_model(input_shape=(60, 1))
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)
```

Use the same training flow for the enriched model, changing the input shape to `(60, 2)` and the output checkpoint to `models/lstm_nvda_enriched.keras`.

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Lookback window | 60 days | enough context without excessive drift |
| LSTM units L1 | 128 | captures multi-scale temporal patterns |
| LSTM units L2 | 64 | compresses learned representation |
| Dropout rate | 0.20 | regularization without strong underfitting |
| Optimizer | Adam | robust default for non-stationary series |
| Learning rate | 0.001 | standard starting point |
| Loss function | MSE | penalizes large deviations |
| Batch size | 32 | stable training for sequence batches |
| Max epochs | 100 | upper bound with early stopping |

---

## 📊 Evaluation Metrics

```python
import numpy as np

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
```

| Metric | Interpretation |
|---|---|
| MAE | average absolute error in USD |
| RMSE | stronger penalty for large errors |
| MAPE | scale-independent percentage error |

For the current project scope, the relevant comparison is:

- **Baseline vs enriched model**
- **Validation vs test split**
- **Price-only vs price-plus-sentiment input**

---

## 🌐 API Reference

The API is built with **FastAPI** and documented through OpenAPI at `/docs`.

### `GET /health`

Returns service status and loaded model metadata.

**Response `200 OK`:**
```json
{
  "status": "ok",
  "model": "lstm_nvda",
  "version": "1.0.0",
  "uptime_seconds": 3821
}
```

### `POST /predict`

Accepts 60 historical closing prices and returns the next-day forecast using the baseline model.

**Request:**
```json
{
  "symbol": "NVDA",
  "prices": [432.10, 435.30, 440.00, "... 60 float values total ..."]
}
```

**Response `200 OK`:**
```json
{
  "symbol": "NVDA",
  "predicted_close": 447.82,
  "lower_bound": 439.60,
  "upper_bound": 456.04,
  "confidence": 0.95,
  "currency": "USD",
  "model": "lstm_nvda",
  "timestamp": "2024-07-21T14:32:00Z"
}
```

### `POST /predict/enriched`

Accepts 60 historical prices and enriches the forecast with news sentiment.

**Request:**
```json
{
  "symbol": "NVDA",
  "prices": [432.10, 435.30, 440.00, "... 60 float values total ..."],
  "include_live_news": true
}
```

**Response `200 OK`:**
```json
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

### `GET /news/{symbol}`

Returns recent news items and aggregated sentiment context for a supported symbol.

### `GET /metrics`

Exposes Prometheus-compatible metrics for scraping.

### Error Codes

| Status | Condition |
|---|---|
| `422` | Invalid payload, wrong input length, or non-numeric values |
| `503` | Model unavailable or enrichment dependency unavailable |

---

## 🐳 Docker & Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: "3.9"

services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - MODEL_PATH=models/lstm_nvda.keras
      - ENRICHED_MODEL_PATH=models/lstm_nvda_enriched.keras
      - NEWSAPI_KEY=${NEWSAPI_KEY}
      - ALPHAVANTAGE_KEY=${ALPHAVANTAGE_KEY}
    depends_on: [prometheus]
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    volumes:
      - ./monitoring/grafana:/etc/grafana/provisioning
```

| Service | Port | Description |
|---|---|---|
| `api` | 8000 | FastAPI inference server |
| `prometheus` | 9090 | Metrics collection and storage |
| `grafana` | 3000 | Monitoring dashboards |

---

## 📡 Monitoring

### Tracked Metrics

| Metric | Type | Description |
|---|---|---|
| `predictions_total` | Counter | Total prediction requests by symbol |
| `prediction_latency_seconds` | Histogram | Inference latency distribution |
| `prediction_errors_total` | Counter | Prediction failures by type |
| `news_fetch_errors_total` | Counter | Failed external news fetch attempts |
| `sentiment_score_live` | Gauge | Current aggregated sentiment score |
| `model_mae_live` | Gauge | Rolling MAE on live predictions |
| `container_cpu_usage` | Gauge | CPU utilization |
| `container_memory_usage_bytes` | Gauge | Memory consumption |

### Alerting Rules

```yaml
groups:
  - name: lstm_api
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, prediction_latency_seconds_bucket) > 0.5
        for: 2m
        labels: { severity: warning }
        annotations:
          summary: "P95 latency above 500ms"
      - alert: HighErrorRate
        expr: rate(prediction_errors_total[5m]) > 0.05
        for: 1m
        labels: { severity: critical }
      - alert: NewsSourceFailure
        expr: rate(news_fetch_errors_total[10m]) > 0
        for: 5m
        labels: { severity: warning }
```

---

## ⚙️ Software Engineering

### Design Patterns

**Strategy** - swappable normalization and sentiment analysis components:

```python
class IScaler(ABC): ...
class MinMaxStrategy(IScaler): ...

class ISentimentAnalyzer(ABC): ...
class FinBERTSentimentAnalyzer(ISentimentAnalyzer): ...
```

**Repository** - decouples both market and news data sources from business logic:

```python
class IStockRepository(ABC): ...
class YFinanceRepository(IStockRepository): ...

class INewsRepository(ABC): ...
class NewsAPIRepository(INewsRepository): ...
```

**Factory** - controlled model instantiation for baseline and enriched variants:

```python
@ModelFactory.register("lstm")
class LSTMModel(IModel): ...

@ModelFactory.register("lstm_enriched")
class LSTMEnrichedModel(IModel): ...
```

### SOLID Principles

| Principle | Implementation |
|---|---|
| **S** ingle Responsibility | `Preprocessor`, `PredictorService`, and `NewsAggregatorService` each own one concern |
| **O** pen/Closed | New models and new news sources are added through registries and interfaces |
| **L** iskov Substitution | Repository and analyzer implementations are interchangeable |
| **I** nterface Segregation | `IModel`, `IScaler`, `IStockRepository`, `INewsRepository`, `ISentimentAnalyzer` remain focused |
| **D** ependency Inversion | Services depend on interfaces injected at composition time |

### Testing

```bash
pytest tests/ -v --cov=src --cov-report=html
```

| Layer | Location | Coverage Target |
|---|---|---|
| Unit | `tests/unit/` | >= 80% |
| Integration | `tests/integration/` | Standard and enriched endpoints |
| Model | `tests/test_model.py` | Output shapes and value ranges |
| NLP | `tests/unit/test_sentiment.py` | Sentiment direction and edge cases |

### CI/CD Pipeline

```text
push / pull_request
        |
        +-- [lint]    black --check · flake8
        +-- [test]    pytest --cov=src
        +-- [build]   docker build -t lstm-api:$SHA
        +-- [scan]    trivy image
        \-- [deploy]  docker push -> production
```

Branch strategy: `main` <- `develop` <- `feature/*`  
Commit convention: [Conventional Commits](https://www.conventionalcommits.org) (`feat:`, `fix:`, `docs:`, `test:`, `chore:`)

---

## 📈 Results

> To be completed after final training and benchmark comparison.

| Metric | Baseline Validation | Baseline Test | Enriched Validation | Enriched Test |
|---|---|---|---|---|
| MAE (USD) | - | - | - | - |
| RMSE (USD) | - | - | - | - |
| MAPE (%) | - | - | - | - |

Use this section to report whether news sentiment enrichment improves directional consistency, interval quality, or raw regression error relative to the baseline.

Forecast vs. actual charts and residual analysis are available in `notebooks/04_model_evaluation.ipynb`.

---

## 🎥 Demo Video

> Link to be added after recording.

[![Watch the demo](https://img.shields.io/badge/Watch_Demo-YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtube.com)

---

## 👨‍💻 Author

<div align="center">

**Guilherme Lossio**  
Postgraduate Program in Machine Learning Engineering  
Tech Challenge - Phase 4

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/guilherme-lossio)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/guilherme-lossio)

</div>

---

<div align="center">
<sub>Developed for academic purposes as part of a postgraduate program in Machine Learning Engineering.</sub>
</div>
