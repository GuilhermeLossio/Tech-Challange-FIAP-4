# Update Note - 2026-05-25

## Topic

Align API serving, promoted model artifacts, and the latest materialized forecast package.

## Status

Implemented.

## Decision

Promote the `2026-05-19` classical Keras artifacts instead of regenerating the final delivery from the older `2026-04-22` partition.

The selected promoted training run is:

| Item | Value |
|---|---|
| Extraction date | `2026-05-19` |
| Trained at | `20260525T030113Z` |
| Model prefix | `lstm_return` |
| Symbols | `AMD`, `ASML`, `NVDA`, `QCOM`, `TSM` |
| Manifest | `models/manifests/extraction_date=2026-05-19/trained_at=20260525T030113Z/keras_training_manifest.json` |

The aligned forecast package is:

| Item | Value |
|---|---|
| Dataset root | `data/processed/future_predict` |
| Generated at | `20260525T143920Z` |
| Forecast window | `2026-05-20` to `2026-12-31` |
| Horizon | `162` business days |
| Predict types | `normal`, `quant` |
| Unified report | `data/processed/future_predict/unified_forecast_report.md` |

## Implementation Summary

- Updated `models/serving_promotions.json` to policy version `2`.
- Replaced the previous `2026-04-22` approvals with immutable `2026-05-19` `lstm_return_*` artifacts.
- Updated `StandardPredictorService` so promoted return-target models are valid serving candidates.
- Updated the default forecast horizon used by API serving metadata and `/forecasts/{symbol}` to `162`, matching the latest materialized package.
- Updated `README.md` with the current promoted extraction date, training run, forecast package, and GitHub workflow status.

## API Outcome

With the local artifacts present, the API default serving date is now aligned across:

- `GET /health`
- `GET /methods`
- `GET /data-usage`
- `POST /predict`
- `GET /forecasts/{symbol}`

The shared default is `latest_extraction_date=2026-05-19`.
