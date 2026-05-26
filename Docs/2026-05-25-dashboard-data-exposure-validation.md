# Update Note - 2026-05-25

## Topic

Dashboard data exposure and forecast table/chart consistency validation.

## Status

Validated.

## Findings Addressed

The non-versioned dashboard exposure note identified three presentation risks:

| Risk | Status | Validation |
|---|---|---|
| Optional forecast fields render as blank or `None` | Covered | `_with_canonical_predict_type` now normalizes missing return, latency, direction, proxy, and guardrail fields into display labels. |
| `normal` / `quant` classification can be ambiguous when `is_price_proxy=true` | Covered | `_canonical_predict_type` prioritizes stored `predict_type` before model metadata and proxy flags. A `normal` row is not reclassified as `quant` solely because `is_price_proxy=true`. |
| Chart and table can read different forecast sources | Covered | The dashboard builds `comparison_forecast_rows` once, canonicalizes those rows, and uses that same tuple for the six-month chart and forecast table filtering. |

## Regression Coverage

Added `tests/test_front_dashboard.py` with checks for:

- stored `predict_type` precedence over `is_price_proxy`
- fallback display labels for optional fields
- table filtering through canonical predict-type classification

## Residual Note

The dashboard still presents a long-horizon recursive forecast, so monotonic model paths can appear when the model output is monotonic. That is a model-quality characteristic already documented in the forecast quality audit, not a dashboard data exposure bug.
