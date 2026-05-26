from __future__ import annotations

from src.front.app import (
    _canonical_predict_type,
    _filter_display_forecast_rows,
    _with_canonical_predict_type,
)


def test_canonical_predict_type_prefers_stored_type_over_price_proxy() -> None:
    row = {
        "predict_type": "normal",
        "model_family": "keras_lstm_regression",
        "model_name": "lstm_return_nvda",
        "is_price_proxy": True,
    }

    assert _canonical_predict_type(row) == "normal"


def test_forecast_row_fallback_labels_optional_fields() -> None:
    row = _with_canonical_predict_type(
        {
            "predict_type": "quant",
            "model_family": "vqc",
            "model_name": "quantum_vqc_nvda",
            "predicted_direction": 1,
            "is_price_proxy": True,
        }
    )

    assert row["predicted_step_return_text"] == "N/A"
    assert row["step_elapsed_ms_text"] == "N/A"
    assert row["predicted_direction_label"] == "up"
    assert row["guardrail_label"] == "N/A"
    assert row["price_proxy_label"] == "yes"


def test_forecast_table_filter_uses_canonical_predict_type() -> None:
    rows = (
        _with_canonical_predict_type(
            {
                "forecast_step": 1,
                "predict_type": "normal",
                "model_family": "keras_lstm_regression",
                "model_name": "lstm_return_nvda",
            }
        ),
        _with_canonical_predict_type(
            {
                "forecast_step": 1,
                "predict_type": "quant",
                "model_family": "quantum_vqc_classifier",
                "model_name": "quantum_vqc_nvda",
            }
        ),
    )

    filtered = _filter_display_forecast_rows(
        rows=rows,
        selected_predict_type="quant",
        row_limit=None,
    )

    assert len(filtered) == 1
    assert filtered[0]["predict_type"] == "quant"
