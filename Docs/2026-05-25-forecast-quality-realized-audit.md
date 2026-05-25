# Update Note - 2026-05-25

## Topic

Forecast quality audit with realized prices and baseline comparison.

## Status

Implemented.

## Input Data

The audit now targets the active delivery package:

| Item | Value |
|---|---|
| Forecast extraction date | `2026-05-19` |
| Forecast generation token | `20260525T143920Z` |
| Forecast horizon | `162` business days |
| Forecast window | `2026-05-20` to `2026-12-31` |
| Actual-price raw partition | `2026-05-25` |
| Realized rows compared | `3` per symbol/model |

The realized rows cover the first three business days after the forecast seed date. The rest of the horizon remains in the future and is intentionally excluded from MAE/RMSE/MAPE.

## Results

| Symbol | Normal MAPE | Quant MAPE | Best baseline / model | Best MAPE | Guardrail rate |
|---|---:|---:|---|---:|---:|
| AMD | 9.46% | 10.41% | `baseline_recent_return_20` | 5.13% | 0.00% |
| ASML | 8.02% | 9.47% | `baseline_sma_5` | 4.46% | 0.00% |
| NVDA | 1.74% | 1.06% | `quant` | 1.06% | 0.00% |
| QCOM | 9.75% | 7.63% | `baseline_recent_return_20` | 6.24% | 0.00% |
| TSM | 2.80% | 2.55% | `baseline_sma_5` | 0.65% | 0.00% |

## Findings

- The prior `compared_rows=0` issue is resolved. Every active `normal` and `quant` series now has `3` realized comparison rows.
- The prior guardrail concern is not present in the current package. The maximum observed constraint rate is `0.00%`.
- The remaining quality limitation is recursive monotonicity in the `normal` LSTM path for all five symbols.
- The quantum series remains a direction-classification output converted to a price proxy, not a direct price regressor.
- Short realized-window metrics show that simple baselines still outperform the model path for AMD, ASML, QCOM, and TSM. NVDA is the exception in this three-row window, where the quantum proxy has the lowest MAPE.

## Generated Local Artifacts

The audit generated local files under `data/processed/audits/forecast_quality/`:

- `forecast_quality_report.md`
- `forecast_quality_summary.csv`
- `forecast_quality_steps.csv`
- `forecast_quality_scaler_split_audit.csv`
- per-symbol SVG charts

These files are intentionally local data artifacts. This note records the delivery-relevant metrics in versioned documentation.

## Validation Command

```bash
python scripts/forecast_quality_audit.py --actual-extraction-date 2026-05-25
```
