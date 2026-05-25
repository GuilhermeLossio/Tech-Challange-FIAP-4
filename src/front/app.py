from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from flask import Flask, render_template, request

from src.application.services.future_prediction_service import FuturePredictionService
from src.application.services.historical_market_service import HistoricalMarketService
from src.application.services.predictor_service import StandardPredictorService
from src.application.services.training_catalog_service import TrainingCatalogService
from src.infrastructure.config.settings import ForecastPipelineSettings


FRONT_TITLE = "Signal Deck"
DEFAULT_SYMBOL = "NVDA"
DEFAULT_HORIZON_DAYS = 30
SIX_MONTH_TRADING_DAYS = 126
DEFAULT_CUMULATIVE_RETURN_CAP = 0.35


def create_app() -> Flask:
    app_root = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(app_root / "templates"),
        static_folder=str(app_root / "static"),
    )

    settings = ForecastPipelineSettings.from_env()
    predictor_service = StandardPredictorService(
        raw_root_dir=settings.local_raw_dir,
        processed_root_dir=settings.local_processed_dir,
        models_root_dir=settings.local_models_dir,
    )
    future_prediction_service = FuturePredictionService(
        processed_root_dir=settings.local_processed_dir,
    )
    historical_market_service = HistoricalMarketService(
        raw_root_dir=settings.local_raw_dir,
    )
    training_catalog_service = TrainingCatalogService(
        models_root_dir=settings.local_models_dir,
    )

    @app.route("/", methods=["GET"])
    def dashboard() -> str:
        warnings: list[str] = []
        supported_symbols = tuple(
            sorted(
                set(predictor_service.get_supported_symbols())
                | set(future_prediction_service.list_symbols())
            )
        )
        symbol = (
            request.args.get("symbol", DEFAULT_SYMBOL).strip().upper() or DEFAULT_SYMBOL
        )
        if symbol not in supported_symbols and supported_symbols:
            symbol = supported_symbols[0]

        selected_predict_type = request.args.get("predict_type", "all").strip().lower() or "all"
        if selected_predict_type not in {"all", "normal", "quant"}:
            selected_predict_type = "all"

        selected_extraction_date = request.args.get("extraction_date") or None
        selected_reference_date = request.args.get("reference_date") or None
        selected_forecast_date_from = request.args.get("forecast_date_from") or None
        selected_forecast_date_to = request.args.get("forecast_date_to") or None

        lookback = _parse_positive_int(request.args.get("lookback"), default=60)
        horizon_days = _resolve_front_horizon_days(
            raw_value=request.args.get("horizon_days"),
            future_prediction_service=future_prediction_service,
            symbol=symbol,
            lookback=lookback,
            requested_extraction_date=_parse_iso_date(selected_extraction_date),
        )
        row_limit = _parse_optional_positive_int(request.args.get("limit"))

        future_result = None
        comparison_future_result = None
        historical_window = None
        training_summary = None
        next_day_prediction = None

        resolved_extraction_date = selected_extraction_date
        if resolved_extraction_date is None:
            try:
                future_result = future_prediction_service.load_forecasts(
                    symbol=symbol,
                    predict_type=selected_predict_type,
                    lookback=lookback,
                    horizon_days=horizon_days,
                    forecast_date_from=_parse_iso_date(selected_forecast_date_from),
                    forecast_date_to=_parse_iso_date(selected_forecast_date_to),
                    limit=row_limit,
                )
                resolved_extraction_date = future_result.extraction_date
            except (FileNotFoundError, ValueError) as exc:
                warnings.append(str(exc))

        try:
            training_summary = training_catalog_service.get_dashboard_summary(
                symbol=symbol,
                extraction_date=resolved_extraction_date,
            )
            resolved_extraction_date = training_summary.extraction_date
        except FileNotFoundError as exc:
            warnings.append(str(exc))

        if future_result is None:
            try:
                future_result = future_prediction_service.load_forecasts(
                    symbol=symbol,
                    extraction_date=_parse_iso_date(resolved_extraction_date),
                    predict_type=selected_predict_type,
                    lookback=lookback,
                    horizon_days=horizon_days,
                    forecast_date_from=_parse_iso_date(selected_forecast_date_from),
                    forecast_date_to=_parse_iso_date(selected_forecast_date_to),
                    limit=row_limit,
                )
            except (FileNotFoundError, ValueError) as exc:
                warnings.append(str(exc))

        try:
            comparison_future_result = future_prediction_service.load_forecasts(
                symbol=symbol,
                extraction_date=_parse_iso_date(resolved_extraction_date),
                predict_type="all",
                lookback=lookback,
                horizon_days=horizon_days,
                forecast_date_from=_parse_iso_date(selected_forecast_date_from),
                forecast_date_to=_parse_iso_date(selected_forecast_date_to),
            )
        except (FileNotFoundError, ValueError) as exc:
            warnings.append(str(exc))

        effective_reference_date = selected_reference_date
        if effective_reference_date is None:
            if comparison_future_result is not None:
                effective_reference_date = comparison_future_result.last_observed_date
            elif future_result is not None:
                effective_reference_date = future_result.last_observed_date

        try:
            next_day_prediction = predictor_service.predict(
                symbol=symbol,
                prices=None,
                extraction_date=_parse_iso_date(resolved_extraction_date),
                reference_date=_parse_iso_date(effective_reference_date),
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            warnings.append(str(exc))

        history_anchor_date = None
        if comparison_future_result is not None:
            history_anchor_date = comparison_future_result.last_observed_date
        elif next_day_prediction and next_day_prediction.resolved_window_end_date:
            history_anchor_date = next_day_prediction.resolved_window_end_date
        elif effective_reference_date:
            history_anchor_date = effective_reference_date

        try:
            historical_window = historical_market_service.load_close_history(
                symbol=symbol,
                extraction_date=_parse_iso_date(resolved_extraction_date),
                as_of_date=_parse_iso_date(history_anchor_date),
                trading_days=SIX_MONTH_TRADING_DAYS,
            )
        except (FileNotFoundError, ValueError) as exc:
            warnings.append(str(exc))

        market_context_chart = _build_market_context_chart(
            historical_window.rows if historical_window else tuple(),
            comparison_future_result.rows if comparison_future_result else tuple(),
        )
        training_chart = _build_training_chart(
            training_summary.keras.history if training_summary and training_summary.keras else {}
        )
        forecast_rows = (
            [_with_canonical_predict_type(row) for row in future_result.rows]
            if future_result
            else []
        )
        forecast_window_summary = _build_forecast_window_summary(
            future_result=future_result,
            comparison_future_result=comparison_future_result,
            selected_predict_type=selected_predict_type,
            row_limit=row_limit,
        )
        outlook_cards = _build_outlook_cards(
            comparison_future_result.rows if comparison_future_result else tuple(),
            base_close=(
                comparison_future_result.last_observed_close
                if comparison_future_result is not None
                else (
                    float(historical_window.rows[-1]["close"])
                    if historical_window and historical_window.rows
                    else None
                )
            ),
            next_day_prediction=next_day_prediction,
        )
        overview_metrics = _build_overview_metrics(
            historical_window=historical_window,
            comparison_future_result=comparison_future_result,
            next_day_prediction=next_day_prediction,
        )
        forecast_report = _build_forecast_report(
            comparison_future_result=comparison_future_result,
            training_summary=training_summary,
        )
        company_snapshot_rows = _build_company_snapshot_rows(
            supported_symbols=supported_symbols,
            selected_symbol=symbol,
            comparison_future_result=comparison_future_result,
            future_prediction_service=future_prediction_service,
            extraction_date=resolved_extraction_date,
            lookback=lookback,
            horizon_days=horizon_days,
        )
        method_summary = _build_method_summary()
        data_usage_summary = _build_data_usage_summary(
            supported_symbols=supported_symbols,
            lookback=lookback,
            horizon_days=horizon_days,
            materialized_forecasts_available=future_prediction_service.has_materialized_forecasts(),
        )

        return render_template(
            "dashboard.html",
            front_title=FRONT_TITLE,
            supported_symbols=supported_symbols,
            symbol=symbol,
            selected_predict_type=selected_predict_type,
            selected_extraction_date=resolved_extraction_date or "",
            selected_reference_date=effective_reference_date or "",
            selected_forecast_date_from=selected_forecast_date_from or "",
            selected_forecast_date_to=selected_forecast_date_to or "",
            lookback=lookback,
            horizon_days=horizon_days,
            row_limit=row_limit,
            row_limit_value=str(row_limit) if row_limit is not None else "",
            next_day_prediction=next_day_prediction,
            future_result=future_result,
            comparison_future_result=comparison_future_result,
            forecast_window_summary=forecast_window_summary,
            historical_window=historical_window,
            training_summary=training_summary,
            market_context_chart=market_context_chart,
            training_chart=training_chart,
            forecast_rows=forecast_rows,
            outlook_cards=outlook_cards,
            overview_metrics=overview_metrics,
            forecast_report=forecast_report,
            company_snapshot_rows=company_snapshot_rows,
            method_summary=method_summary,
            data_usage_summary=data_usage_summary,
            warnings=warnings,
            today_utc=datetime.utcnow().strftime("%Y-%m-%d"),
            static_export=bool(app.config.get("STATIC_EXPORT", False)),
        )

    return app


def _parse_iso_date(raw_value: str | None) -> date | None:
    if raw_value is None or not raw_value.strip():
        return None
    return datetime.strptime(raw_value.strip(), "%Y-%m-%d").date()


def _parse_positive_int(raw_value: str | None, *, default: int) -> int:
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return value if value > 0 else default


def _parse_optional_positive_int(raw_value: str | None) -> int | None:
    if raw_value is None or not raw_value.strip():
        return None
    try:
        value = int(raw_value)
    except ValueError:
        return None
    return value if value > 0 else None


def _resolve_front_horizon_days(
    *,
    raw_value: str | None,
    future_prediction_service: FuturePredictionService,
    symbol: str,
    lookback: int,
    requested_extraction_date: date | None = None,
) -> int:
    if raw_value is not None and raw_value.strip():
        return _parse_positive_int(raw_value, default=DEFAULT_HORIZON_DAYS)

    latest_horizon = future_prediction_service.resolve_latest_horizon_days(
        symbol=symbol,
        lookback=lookback,
        requested_extraction_date=requested_extraction_date,
    )
    if latest_horizon is not None:
        return latest_horizon

    available_horizons = future_prediction_service.list_available_horizon_days(
        symbol=symbol,
        lookback=lookback,
    )
    if available_horizons:
        if DEFAULT_HORIZON_DAYS in available_horizons:
            return DEFAULT_HORIZON_DAYS
        return available_horizons[0]
    return DEFAULT_HORIZON_DAYS


def _build_forecast_window_summary(
    *,
    future_result: Any | None,
    comparison_future_result: Any | None,
    selected_predict_type: str,
    row_limit: int | None,
) -> dict[str, Any] | None:
    result = future_result or comparison_future_result
    full_result = comparison_future_result or future_result
    if result is None:
        return None

    available_rows = tuple(full_result.rows) if full_result is not None else tuple(result.rows)
    returned_rows = tuple(result.rows)
    predict_types = (
        ", ".join(result.available_predict_types)
        if selected_predict_type == "all"
        else selected_predict_type
    )
    model_cards = _build_forecast_model_cards(available_rows)
    return {
        "symbol": result.symbol,
        "generated_at": result.generated_at,
        "generated_at_utc": result.generated_at_utc,
        "extraction_date": result.extraction_date,
        "last_observed_date": result.last_observed_date,
        "last_observed_close": float(result.last_observed_close),
        "available_forecast_start_date": result.available_forecast_start_date,
        "available_forecast_end_date": result.available_forecast_end_date,
        "returned_forecast_start_date": result.returned_forecast_start_date,
        "returned_forecast_end_date": result.returned_forecast_end_date,
        "available_row_count": len(available_rows),
        "returned_row_count": len(returned_rows),
        "predict_types": predict_types,
        "model_cards": model_cards,
        "row_limit": row_limit,
        "is_limited": row_limit is not None and len(returned_rows) <= row_limit,
    }


def _build_forecast_model_cards(
    forecast_rows: tuple[dict[str, Any], ...],
) -> tuple[dict[str, str], ...]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in forecast_rows:
        grouped[_canonical_predict_type(row)].append(dict(row))

    cards: list[dict[str, str]] = []
    for predict_type, rows in sorted(grouped.items()):
        if not rows:
            continue
        rows.sort(key=lambda item: int(item.get("forecast_step", 0)))
        first_row = rows[0]
        model_name = str(first_row.get("model_name") or "unknown")
        feature_input_mode = str(first_row.get("feature_input_mode") or "n/a")
        sequence_input_kind = str(first_row.get("sequence_input_kind") or "n/a")
        target_mode = str(first_row.get("model_prediction_target_mode") or "n/a")
        model_family = str(first_row.get("model_family") or "n/a")
        cards.append(
            {
                "predict_type": predict_type,
                "title": "Standard forecast" if predict_type == "normal" else "Quantum forecast",
                "model_name": model_name,
                "feature_input_mode": feature_input_mode,
                "sequence_input_kind": sequence_input_kind,
                "target_mode": target_mode,
                "model_family": model_family,
            }
        )
    return tuple(cards)


def _build_market_context_chart(
    historical_rows: tuple[dict[str, Any], ...],
    forecast_rows: tuple[dict[str, Any], ...],
) -> dict[str, Any] | None:
    if not historical_rows:
        return None

    history = [
        (str(row["date"]), float(row["close"]))
        for row in historical_rows
    ]
    grouped_forecasts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in forecast_rows:
        grouped_forecasts[_canonical_predict_type(row)].append(dict(row))

    for rows in grouped_forecasts.values():
        rows.sort(key=lambda item: int(item["forecast_step"]))

    history_count = len(history)
    max_forecast_step = max(
        (int(row["forecast_step"]) for row in forecast_rows),
        default=0,
    )
    span_count = max(history_count + max_forecast_step, 2)

    width = 920.0
    height = 320.0
    padding = 30.0
    all_values = [value for _, value in history] + [
        float(row["predicted_close"]) for row in forecast_rows
    ]
    min_value = min(all_values)
    max_value = max(all_values)
    if abs(max_value - min_value) < 1e-9:
        max_value = min_value + 1.0
    y_padding = max((max_value - min_value) * 0.08, abs(history[-1][1]) * 0.01, 1.0)
    min_value = max(min_value - y_padding, 0.0)
    max_value = max_value + y_padding

    def x_for(index: int) -> float:
        return padding + index * ((width - 2 * padding) / (span_count - 1))

    def y_for(value: float) -> float:
        return height - padding - ((value - min_value) / (max_value - min_value)) * (
            height - 2 * padding
        )

    history_polyline = " ".join(
        f"{x_for(index):.2f},{y_for(value):.2f}"
        for index, (_, value) in enumerate(history)
    )

    forecast_series = []
    colors = {
        "historical": "#174b83",
        "normal": "#0b6e69",
        "quant": "#c46d1a",
    }
    dasharrays = {
        "normal": "",
        "quant": "9 6",
    }
    last_history_index = history_count - 1
    last_history_date, last_history_value = history[-1]
    last_history_x = x_for(last_history_index)
    last_history_y = y_for(last_history_value)
    for predict_type, rows in sorted(grouped_forecasts.items()):
        points = [f"{last_history_x:.2f},{last_history_y:.2f}"]
        for row in rows:
            index = last_history_index + int(row["forecast_step"])
            points.append(
                f"{x_for(index):.2f},{y_for(float(row['predicted_close'])):.2f}"
            )
        first_row = rows[0]
        last_row = rows[-1]
        last_value = float(last_row["predicted_close"])
        day_one_value = float(first_row["predicted_close"])
        final_delta_abs = last_value - float(last_history_value)
        final_delta_pct = (
            (last_value / float(last_history_value) - 1.0) * 100.0
            if abs(float(last_history_value)) > 1e-8
            else 0.0
        )
        endpoint_x = x_for(last_history_index + int(last_row["forecast_step"]))
        endpoint_y = y_for(last_value)
        forecast_series.append(
            {
                "name": predict_type,
                "label": "Standard forecast" if predict_type == "normal" else "Quantum forecast",
                "color": colors.get(predict_type, "#334155"),
                "dasharray": dasharrays.get(predict_type, ""),
                "polyline": " ".join(points),
                "day_one_value": day_one_value,
                "day_one_date": str(first_row["forecast_date"]),
                "last_value": last_value,
                "last_date": str(last_row["forecast_date"]),
                "final_delta_abs": final_delta_abs,
                "final_delta_pct": final_delta_pct,
                "final_delta_label": (
                    f"{final_delta_pct:+.2f}% (${final_delta_abs:+.2f})"
                ),
                "endpoint_x": endpoint_x,
                "endpoint_y": endpoint_y,
                "endpoint_tone": (
                    "positive" if final_delta_abs > 0 else "negative" if final_delta_abs < 0 else "neutral"
                ),
            }
        )

    y_guides = []
    for value in (
        max_value,
        max_value - (max_value - min_value) * 0.25,
        (max_value + min_value) / 2.0,
        min_value + (max_value - min_value) * 0.25,
        min_value,
    ):
        y_guides.append(
            {
                "value": value,
                "y": y_for(value),
            }
        )

    forecast_end_date = (
        max((str(row["forecast_date"]) for row in forecast_rows), default=last_history_date)
    )
    return {
        "width": width,
        "height": height,
        "history_polyline": history_polyline,
        "history_color": colors["historical"],
        "forecast_series": forecast_series,
        "boundary_x": last_history_x,
        "boundary_label": last_history_date,
        "base_value": float(last_history_value),
        "base_y": last_history_y,
        "base_label": f"Last close ${float(last_history_value):.2f}",
        "x_labels": (
            {"label": str(history[0][0]), "x": x_for(0)},
            {"label": last_history_date, "x": last_history_x},
            {"label": forecast_end_date, "x": x_for(span_count - 1)},
        ),
        "y_guides": tuple(y_guides),
    }


def _build_outlook_cards(
    forecast_rows: tuple[dict[str, Any], ...],
    *,
    base_close: float | None,
    next_day_prediction: Any | None,
) -> tuple[dict[str, Any], ...]:
    if not forecast_rows or base_close is None:
        return tuple()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in forecast_rows:
        grouped[_canonical_predict_type(row)].append(dict(row))

    cards: list[dict[str, Any]] = []
    for predict_type, rows in sorted(grouped.items()):
        rows.sort(key=lambda item: int(item["forecast_step"]))
        first_row = rows[0]
        last_row = rows[-1]
        best_row = max(rows, key=lambda item: float(item["predicted_close"]))
        worst_row = min(rows, key=lambda item: float(item["predicted_close"]))
        proxy_methods = sorted(
            {
                str(row["price_proxy_method"])
                for row in rows
                if row.get("price_proxy_method")
            }
        )

        def delta_payload(value: float) -> tuple[float, float]:
            delta_abs = value - base_close
            delta_pct = (delta_abs / base_close * 100.0) if abs(base_close) > 1e-9 else 0.0
            return delta_abs, delta_pct

        day_one_delta_abs, day_one_delta_pct = delta_payload(float(first_row["predicted_close"]))
        horizon_delta_abs, horizon_delta_pct = delta_payload(float(last_row["predicted_close"]))
        best_delta_abs, best_delta_pct = delta_payload(float(best_row["predicted_close"]))
        worst_delta_abs, worst_delta_pct = delta_payload(float(worst_row["predicted_close"]))

        if predict_type == "normal":
            acquisition_summary = (
                "Generated from the offline Keras recursive forecast stored in "
                "`future_predict`, seeded by a 60-day close window."
            )
            acquisition_points = [
                f"The batch path rolls predicted closes forward for {len(rows)} business days.",
                "The line shown here is directly materialized from the stored forecast dataset.",
            ]
            if next_day_prediction is not None:
                acquisition_points.append(
                    "The live D+1 Keras request also exposes a 95% interval for the first step."
                )
            if bool(first_row.get("prediction_constraint_applied")):
                raw_model_close = first_row.get("raw_model_predicted_close")
                return_cap = first_row.get("prediction_return_cap")
                if raw_model_close is not None and return_cap is not None:
                    acquisition_points.append(
                        "The first offline standard step was constrained from "
                        f"${float(raw_model_close):.2f} to "
                        f"${float(first_row['predicted_close']):.2f} using a "
                        f"+/-{float(return_cap) * 100.0:.2f}% daily return cap."
                    )
            if next_day_prediction is not None and next_day_prediction.prediction_constraint_applied:
                acquisition_points.append(
                    "The live D+1 standard estimate was also constrained by the "
                    "recent realized volatility guardrail."
                )
        else:
            acquisition_summary = (
                "Generated from the offline VQC direction classifier stored in "
                "`future_predict`; no quantum runtime token is consumed in the UI."
            )
            acquisition_points = [
                "The batch path is reconstructed from direction outputs over the forecast horizon.",
                (
                    f"Price proxy method: {', '.join(proxy_methods)}."
                    if proxy_methods
                    else "The stored rows do not declare a specific proxy method."
                ),
            ]

        cards.append(
            {
                "predict_type": predict_type,
                "title": "Standard Forecast" if predict_type == "normal" else "Quantum Forecast",
                "base_close": base_close,
                "base_date": str(first_row["last_observed_date"]),
                "day_one_date": str(first_row["forecast_date"]),
                "day_one_close": float(first_row["predicted_close"]),
                "day_one_delta_abs": day_one_delta_abs,
                "day_one_delta_pct": day_one_delta_pct,
                "horizon_end_date": str(last_row["forecast_date"]),
                "horizon_end_close": float(last_row["predicted_close"]),
                "horizon_delta_abs": horizon_delta_abs,
                "horizon_delta_pct": horizon_delta_pct,
                "best_date": str(best_row["forecast_date"]),
                "best_close": float(best_row["predicted_close"]),
                "best_delta_abs": best_delta_abs,
                "best_delta_pct": best_delta_pct,
                "worst_date": str(worst_row["forecast_date"]),
                "worst_close": float(worst_row["predicted_close"]),
                "worst_delta_abs": worst_delta_abs,
                "worst_delta_pct": worst_delta_pct,
                "status_tone": "positive" if horizon_delta_abs >= 0 else "negative",
                "acquisition_summary": acquisition_summary,
                "acquisition_points": tuple(acquisition_points),
                "live_interval": (
                    {
                        "lower": float(next_day_prediction.lower_bound),
                        "upper": float(next_day_prediction.upper_bound),
                    }
                    if predict_type == "normal" and next_day_prediction is not None
                    else None
                ),
            }
        )

    return tuple(cards)


def _build_forecast_report(
    *,
    comparison_future_result: Any | None,
    training_summary: Any | None,
) -> dict[str, Any] | None:
    if comparison_future_result is None or not comparison_future_result.rows:
        return None

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comparison_future_result.rows:
        grouped[_canonical_predict_type(row)].append(dict(row))
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["forecast_step"]))

    last_close = float(comparison_future_result.last_observed_close)
    horizon_days = int(comparison_future_result.horizon_days)
    model_summaries = []
    for predict_type, rows in sorted(grouped.items()):
        if not rows:
            continue
        first_row = rows[0]
        last_row = rows[-1]
        elapsed_values = [
            float(row["step_elapsed_ms"])
            for row in rows
            if row.get("step_elapsed_ms") is not None
            and float(row["step_elapsed_ms"]) == float(row["step_elapsed_ms"])
        ]
        total_ms = float(sum(elapsed_values)) if elapsed_values else None
        avg_ms = float(total_ms / len(elapsed_values)) if elapsed_values else None
        min_ms = min(elapsed_values) if elapsed_values else None
        max_ms = max(elapsed_values) if elapsed_values else None
        guardrail_count = sum(
            1 for row in rows if bool(row.get("prediction_constraint_applied"))
        )
        final_price = float(last_row["predicted_close"])
        cumulative_return_pct = _compute_delta_pct(
            value=final_price,
            base_close=last_close,
        ) or 0.0
        up_count = sum(1 for row in rows if int(row.get("predicted_direction", 0)) == 1)
        cumulative_cap = _resolve_cumulative_return_cap(rows)
        limited_by_cap = _is_limited_by_cap(
            cumulative_return_pct=cumulative_return_pct,
            cumulative_cap=cumulative_cap,
        )
        proxy_methods = sorted(
            {
                str(row["price_proxy_method"])
                for row in rows
                if row.get("price_proxy_method")
            }
        )
        uses_price_proxy = any(bool(row.get("is_price_proxy")) for row in rows)
        formatted_cumulative_return = formatPercent(cumulative_return_pct / 100.0)
        model_summaries.append(
            {
                "predict_type": predict_type,
                "short_name": "LSTM" if predict_type == "normal" else "VQC",
                "display_name": "LSTM - Keras" if predict_type == "normal" else "VQC - Qiskit",
                "final_label": (
                    "LSTM final price" if predict_type == "normal" else "VQC estimated price"
                ),
                "family": str(first_row.get("model_family", "")),
                "final_price": final_price,
                "cumulative_return_pct": cumulative_return_pct,
                "cumulative_return_text": formatted_cumulative_return,
                "cumulative_return_tone": (
                    "positive" if cumulative_return_pct > 0 else "negative" if cumulative_return_pct < 0 else "neutral"
                ),
                "guardrail_count": guardrail_count,
                "constraint_rate": guardrail_count / len(rows) if rows else 0.0,
                "up_rate": up_count / len(rows) if rows else 0.0,
                "total_ms": total_ms,
                "avg_ms": avg_ms,
                "min_ms": min_ms,
                "max_ms": max_ms,
                "row_count": len(rows),
                "horizon_steps": horizon_days,
                "uses_price_proxy": uses_price_proxy,
                "price_proxy_label": (
                    f"Yes, {', '.join(proxy_methods)}"
                    if proxy_methods
                    else "Yes, volatility proxy"
                    if uses_price_proxy
                    else "No"
                ),
                "cumulative_cap": cumulative_cap,
                "limited_by_cap": limited_by_cap,
                "total_runtime_text": formatMs(total_ms),
                "avg_runtime_text": formatMs(avg_ms),
                "min_runtime_text": formatMs(min_ms),
                "max_runtime_text": formatMs(max_ms),
                "up_rate_text": _format_unsigned_percent(
                    up_count / len(rows) if rows else None
                ),
                "guardrail_text": f"{guardrail_count}/{horizon_days}",
                "price_proxy_badge": "Price proxy" if uses_price_proxy else "",
                "limit_badge": "Limited by cap" if limited_by_cap else "",
            }
        )

    summary_by_type = {item["predict_type"]: item for item in model_summaries}
    normal = summary_by_type.get("normal")
    quant = summary_by_type.get("quant")
    timing_ratio = None
    if normal and quant and normal.get("avg_ms") and quant.get("avg_ms"):
        timing_ratio = float(quant["avg_ms"] / normal["avg_ms"])
    delta_time_ms = None
    if normal and quant and normal.get("total_ms") is not None and quant.get("total_ms") is not None:
        delta_time_ms = float(quant["total_ms"] or 0.0) - float(normal["total_ms"] or 0.0)

    seed = 42
    if training_summary is not None and training_summary.keras is not None:
        seed = 42

    metric_cards: list[dict[str, str]] = []
    for item in model_summaries:
        metric_cards.append(
            {
                "label": item["final_label"],
                "value": f"${float(item['final_price']):.2f}",
                "detail": f"{item['cumulative_return_text']} vs last close",
            }
        )
    metric_cards.append(
        {
            "label": "Razao tempo VQC/LSTM",
            "value": formatRatio(timing_ratio),
            "detail": "per step",
        }
    )
    for item in model_summaries:
        metric_cards.append(
            {
                "label": f"{item['short_name']} - guardrails",
                "value": item["guardrail_text"],
                "detail": f"{float(item['constraint_rate']):.0%} of rows",
            }
        )
    metric_cards.append(
        {
            "label": "Lookback",
            "value": f"{int(comparison_future_result.lookback)}",
            "detail": "session window",
        }
    )

    executive_cards = _build_executive_summary_cards(
        normal=normal,
        quant=quant,
        horizon_days=horizon_days,
        timing_ratio=timing_ratio,
    )
    alerts = _build_forecast_alerts(
        model_summaries=model_summaries,
        horizon_days=horizon_days,
        timing_ratio=timing_ratio,
        normal=normal,
        quant=quant,
    )
    timing_copy = _build_timing_copy(normal=normal, quant=quant, timing_ratio=timing_ratio)
    detailed_rows = _build_model_comparison_rows(model_summaries)
    return {
        "symbol": comparison_future_result.symbol,
        "last_observed_close": last_close,
        "last_observed_date": comparison_future_result.last_observed_date,
        "available_forecast_start_date": comparison_future_result.available_forecast_start_date,
        "available_forecast_end_date": comparison_future_result.available_forecast_end_date,
        "returned_forecast_start_date": comparison_future_result.returned_forecast_start_date,
        "returned_forecast_end_date": comparison_future_result.returned_forecast_end_date,
        "generated_at": comparison_future_result.generated_at,
        "horizon_days": horizon_days,
        "seed": seed,
        "metric_cards": tuple(metric_cards),
        "executive_cards": tuple(executive_cards),
        "model_summaries": tuple(model_summaries),
        "detailed_rows": tuple(detailed_rows),
        "alerts": tuple(alerts),
        "timing_ratio": timing_ratio,
        "timing_ratio_text": formatRatio(timing_ratio),
        "delta_time_text": formatSeconds(delta_time_ms),
        "timing_copy": timing_copy,
        "uses_price_proxy": bool(quant and quant.get("uses_price_proxy")),
        "same_capped_return": bool(
            normal
            and quant
            and normal.get("limited_by_cap")
            and quant.get("limited_by_cap")
            and abs(
                float(normal.get("cumulative_return_pct") or 0.0)
                - float(quant.get("cumulative_return_pct") or 0.0)
            )
            < 0.01
        ),
    }


def _build_executive_summary_cards(
    *,
    normal: dict[str, Any] | None,
    quant: dict[str, Any] | None,
    horizon_days: int,
    timing_ratio: float | None,
) -> list[dict[str, str]]:
    cards = [
        {
            "label": "Total LSTM runtime",
            "value": normal["total_runtime_text"] if normal else "N/A",
            "detail": f"{normal['avg_runtime_text']} per step" if normal else "Keras regression",
            "tone": "neutral",
        },
        {
            "label": "Total VQC runtime",
            "value": quant["total_runtime_text"] if quant else "N/A",
            "detail": f"{quant['avg_runtime_text']} per step" if quant else "Qiskit direction model",
            "tone": "cost" if timing_ratio is not None and timing_ratio > 10.0 else "neutral",
        },
        {
            "label": "Runtime ratio",
            "value": formatRatio(timing_ratio),
            "detail": "VQC / LSTM per step",
            "tone": "cost" if timing_ratio is not None and timing_ratio > 10.0 else "neutral",
        },
        {
            "label": "Forecast horizon",
            "value": f"{horizon_days}",
            "detail": "business days",
            "tone": "neutral",
        },
    ]
    for model in (normal, quant):
        if not model:
            continue
        cards.append(
            {
                "label": f"{model['short_name']} cumulative return",
                "value": model["cumulative_return_text"],
                "detail": "Limited by cap" if model["limited_by_cap"] else "vs last observed close",
                "tone": model["cumulative_return_tone"],
            }
        )
    return cards


def _build_forecast_alerts(
    *,
    model_summaries: list[dict[str, Any]],
    horizon_days: int,
    timing_ratio: float | None,
    normal: dict[str, Any] | None,
    quant: dict[str, Any] | None,
) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []
    high_guardrail_models = [
        item["short_name"]
        for item in model_summaries
        if item["guardrail_count"] >= max(3, int(horizon_days * 0.3))
    ]
    if high_guardrail_models:
        alerts.append(
            {
                "tone": "warning",
                "title": "Guardrail activity is elevated",
                "message": (
                    "Warning: many steps activated volatility guardrails. Final prices "
                    "reflect the cumulative cap and may not represent the models' "
                    "unconstrained predictions."
                ),
            }
        )

    inconsistent_models = [
        item["short_name"]
        for item in model_summaries
        if item["guardrail_count"] > horizon_days
    ]
    if inconsistent_models:
        alerts.append(
            {
                "tone": "note",
                "title": "Guardrail count exceeds horizon",
                "message": (
                    "Note: guardrail activations exceed the displayed forecast horizon. "
                    "Verify whether this metric represents internal activations, "
                    "accumulated windows, or multiple simulations."
                ),
            }
        )

    if timing_ratio is not None and timing_ratio > 10.0:
        alerts.append(
            {
                "tone": "cost",
                "title": "Computational cost",
                "message": (
                    f"The VQC is {formatRatio(timing_ratio)} slower than the LSTM "
                    "per step, which is expected for circuit execution and bitstring "
                    "interpretation."
                ),
            }
        )

    if normal and quant and normal.get("limited_by_cap") and quant.get("limited_by_cap"):
        if abs(
            float(normal.get("cumulative_return_pct") or 0.0)
            - float(quant.get("cumulative_return_pct") or 0.0)
        ) < 0.01:
            alerts.append(
                {
                    "tone": "note",
                    "title": "Equal returns are cap-driven",
                    "message": (
                        "Both models show the same cumulative return because the cap "
                        "was reached. This does not mean both models produced the same "
                        "unconstrained forecast."
                    ),
                }
            )

    if quant and quant.get("uses_price_proxy"):
        alerts.append(
            {
                "tone": "proxy",
                "title": "VQC uses a price proxy",
                "message": (
                    "The VQC does not directly predict price in this view. It predicts "
                    "movement direction or volatility, which is converted into an "
                    "approximate price estimate."
                ),
            }
        )
    return alerts


def _build_model_comparison_rows(
    model_summaries: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "model": item["display_name"],
            "family": item["family"] or "N/A",
            "total_runtime": item["total_runtime_text"],
            "avg_runtime": item["avg_runtime_text"],
            "min_runtime": item["min_runtime_text"],
            "max_runtime": item["max_runtime_text"],
            "up_rate": item["up_rate_text"],
            "cumulative_return": item["cumulative_return_text"],
            "guardrails": item["guardrail_text"],
            "price_proxy": item["price_proxy_label"],
            "tone": item["cumulative_return_tone"],
            "limited_by_cap": item["limited_by_cap"],
            "uses_price_proxy": item["uses_price_proxy"],
        }
        for item in model_summaries
    ]


def _resolve_cumulative_return_cap(rows: list[dict[str, Any]]) -> float:
    caps = [
        float(row["dynamic_cumulative_return_cap"])
        for row in rows
        if row.get("dynamic_cumulative_return_cap") is not None
        and float(row["dynamic_cumulative_return_cap"]) == float(row["dynamic_cumulative_return_cap"])
    ]
    return max(caps) if caps else DEFAULT_CUMULATIVE_RETURN_CAP


def _is_limited_by_cap(
    *,
    cumulative_return_pct: float,
    cumulative_cap: float,
) -> bool:
    return abs(abs(cumulative_return_pct) / 100.0 - cumulative_cap) <= 0.001


def _build_timing_copy(
    *,
    normal: dict[str, Any] | None,
    quant: dict[str, Any] | None,
    timing_ratio: float | None,
) -> str:
    if not normal or not quant:
        return "Timing per step is unavailable for this comparison window."
    if normal.get("avg_ms") is None or quant.get("avg_ms") is None:
        return "Timing per step is unavailable for this comparison window."
    delta_time_ms = float(quant["total_ms"] or 0.0) - float(normal["total_ms"] or 0.0)
    return (
        f"The VQC is approximately {formatRatio(timing_ratio)} slower than the "
        f"LSTM per step. Over {int(normal['horizon_steps'])} steps, the total "
        f"runtime difference is approximately {formatSeconds(delta_time_ms)}. "
        "This is expected because each VQC step may involve running a variational "
        "quantum circuit, sampler execution, transpilation, and bitstring "
        "interpretation, while the LSTM mainly performs optimized matrix operations."
    )


def _build_company_snapshot_rows(
    *,
    supported_symbols: tuple[str, ...],
    selected_symbol: str,
    comparison_future_result: Any | None,
    future_prediction_service: FuturePredictionService,
    extraction_date: str | None,
    lookback: int,
    horizon_days: int,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    parsed_extraction_date = _parse_iso_date(extraction_date)
    ordered_symbols = sorted(
        supported_symbols,
        key=lambda item: (item != selected_symbol, item),
    )

    for item in ordered_symbols:
        try:
            result = (
                comparison_future_result
                if item == selected_symbol and comparison_future_result is not None
                else _load_company_snapshot_forecasts(
                    future_prediction_service=future_prediction_service,
                    symbol=item,
                    extraction_date=parsed_extraction_date,
                    lookback=lookback,
                    horizon_days=horizon_days,
                )
            )
        except (FileNotFoundError, ValueError):
            rows.append(
                {
                    "symbol": item,
                    "is_selected": item == selected_symbol,
                    "availability": "unavailable",
                    "base_close": None,
                    "base_date": None,
                    "standard_day_one": None,
                    "standard_day_one_delta_pct": None,
                    "standard_horizon_end": None,
                    "quantum_day_one": None,
                    "quantum_day_one_delta_pct": None,
                    "standard_constraint_applied": False,
                    "standard_return_cap_pct": None,
                }
            )
            continue

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in result.rows:
            grouped[_canonical_predict_type(row)].append(dict(row))

        for predict_type_rows in grouped.values():
            predict_type_rows.sort(key=lambda value: int(value["forecast_step"]))

        base_close = float(result.last_observed_close)
        normal_rows = grouped.get("normal", [])
        quant_rows = grouped.get("quant", [])
        standard_day_one = (
            float(normal_rows[0]["predicted_close"]) if normal_rows else None
        )
        standard_horizon_end = (
            float(normal_rows[-1]["predicted_close"]) if normal_rows else None
        )
        quantum_day_one = float(quant_rows[0]["predicted_close"]) if quant_rows else None

        rows.append(
            {
                "symbol": item,
                "is_selected": item == selected_symbol,
                "availability": _describe_predict_type_availability(
                    normal_rows=normal_rows,
                    quant_rows=quant_rows,
                ),
                "base_close": base_close,
                "base_date": result.last_observed_date,
                "standard_day_one": standard_day_one,
                "standard_day_one_delta_pct": _compute_delta_pct(
                    value=standard_day_one,
                    base_close=base_close,
                ),
                "standard_horizon_end": standard_horizon_end,
                "quantum_day_one": quantum_day_one,
                "quantum_day_one_delta_pct": _compute_delta_pct(
                    value=quantum_day_one,
                    base_close=base_close,
                ),
                "standard_constraint_applied": bool(
                    normal_rows and normal_rows[0].get("prediction_constraint_applied")
                ),
                "standard_return_cap_pct": (
                    float(normal_rows[0]["prediction_return_cap"]) * 100.0
                    if normal_rows and normal_rows[0].get("prediction_return_cap") is not None
                    else None
                ),
                "extraction_date": result.extraction_date,
                "horizon_days": int(result.horizon_days),
                "generated_at": result.generated_at,
            }
        )

    return tuple(rows)


def _load_company_snapshot_forecasts(
    *,
    future_prediction_service: FuturePredictionService,
    symbol: str,
    extraction_date: date | None,
    lookback: int,
    horizon_days: int,
) -> Any:
    try:
        return future_prediction_service.load_forecasts(
            symbol=symbol,
            extraction_date=extraction_date,
            predict_type="all",
            lookback=lookback,
            horizon_days=horizon_days,
        )
    except (FileNotFoundError, ValueError):
        latest_horizon_days = future_prediction_service.resolve_latest_horizon_days(
            symbol=symbol,
            lookback=lookback,
        )
        return future_prediction_service.load_forecasts(
            symbol=symbol,
            predict_type="all",
            lookback=lookback,
            horizon_days=latest_horizon_days or horizon_days,
        )


def _build_overview_metrics(
    *,
    historical_window: Any | None,
    comparison_future_result: Any | None,
    next_day_prediction: Any | None,
) -> tuple[dict[str, str], ...]:
    base_close = None
    base_date = None
    if comparison_future_result is not None:
        base_close = float(comparison_future_result.last_observed_close)
        base_date = comparison_future_result.last_observed_date
    elif historical_window and historical_window.rows:
        base_close = float(historical_window.rows[-1]["close"])
        base_date = str(historical_window.rows[-1]["date"])

    standard_day_one = None
    quantum_day_one = None
    if comparison_future_result is not None:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in comparison_future_result.rows:
            grouped[_canonical_predict_type(row)].append(dict(row))
        if grouped.get("normal"):
            grouped["normal"].sort(key=lambda item: int(item["forecast_step"]))
            standard_day_one = float(grouped["normal"][0]["predicted_close"])
        if grouped.get("quant"):
            grouped["quant"].sort(key=lambda item: int(item["forecast_step"]))
            quantum_day_one = float(grouped["quant"][0]["predicted_close"])

    return (
        {
            "label": "Last Observed Close",
            "value": f"${base_close:.2f}" if base_close is not None else "--",
            "detail": base_date or "Base price unavailable",
        },
        {
            "label": "Six-Month History",
            "value": (
                f"{historical_window.row_count} sessions"
                if historical_window is not None
                else "--"
            ),
            "detail": (
                f"{historical_window.start_date} to {historical_window.end_date}"
                if historical_window is not None
                else "Historical window unavailable"
            ),
        },
        {
            "label": "Standard D+1",
            "value": (
                f"${standard_day_one:.2f}"
                if standard_day_one is not None
                else (
                    f"${next_day_prediction.predicted_close:.2f}"
                    if next_day_prediction is not None
                    else "--"
                )
            ),
            "detail": "Keras batch/live baseline",
        },
        {
            "label": "Quantum D+1",
            "value": f"${quantum_day_one:.2f}" if quantum_day_one is not None else "--",
            "detail": "Offline VQC batch path",
        },
    )


def _build_training_chart(history: dict[str, list[float]]) -> dict[str, Any] | None:
    loss_series = history.get("loss") or []
    validation_series = history.get("val_loss") or []
    if not loss_series and not validation_series:
        return None

    width = 760.0
    height = 220.0
    padding = 22.0
    all_values = [*loss_series, *validation_series]
    min_value = min(all_values)
    max_value = max(all_values)
    if abs(max_value - min_value) < 1e-9:
        max_value = min_value + 1.0

    def build_polyline(series: list[float]) -> str:
        if not series:
            return ""
        points: list[str] = []
        max_len = max(len(loss_series), len(validation_series), 1)
        for index, value in enumerate(series):
            x = padding if max_len <= 1 else padding + index * ((width - 2 * padding) / (max_len - 1))
            y = height - padding - ((value - min_value) / (max_value - min_value)) * (height - 2 * padding)
            points.append(f"{x:.2f},{y:.2f}")
        return " ".join(points)

    return {
        "width": width,
        "height": height,
        "padding": padding,
        "min_value": min_value,
        "max_value": max_value,
        "loss_polyline": build_polyline(loss_series),
        "validation_polyline": build_polyline(validation_series),
        "epochs": max(len(loss_series), len(validation_series)),
    }


def _build_method_summary() -> tuple[dict[str, Any], ...]:
    return (
        {
            "title": "Classical next-day prediction",
            "availability": "online",
            "predict_type": "normal",
            "summary": (
                "Uses the trained Keras LSTM model and a 60-day closing-price window "
                "to estimate the next trading session close."
            ),
            "limitations": (
                "Served live.",
                "Depends on locally available Keras artifacts and scaler metadata.",
            ),
        },
        {
            "title": "Quantum future forecast",
            "availability": "batch only",
            "predict_type": "quant",
            "summary": (
                "Serves precomputed quantum rows from the materialized future_predict "
                "dataset without triggering IBM Quantum token usage."
            ),
            "limitations": (
                "Never executed online by the client view.",
                "Some rows may expose a price proxy instead of a direct regressed close.",
            ),
        },
    )


def _build_data_usage_summary(
    *,
    supported_symbols: tuple[str, ...],
    lookback: int,
    horizon_days: int,
    materialized_forecasts_available: bool,
) -> dict[str, Any]:
    return {
        "raw_market_source": "Yahoo Finance via yfinance",
        "training_target": "Next-day closing price (D+1)",
        "target_column": "close",
        "lookback": lookback,
        "forecast_horizon_days": horizon_days,
        "supported_symbols": supported_symbols,
        "processed_forecast_dataset": "data/processed/future_predict",
        "materialized_forecasts_available": materialized_forecasts_available,
        "notes": (
            "The dashboard is read-only and exposes no write path to raw, processed, or model data.",
            "Online prediction uses only the classical model.",
            "Quantum rows are served only when they were generated offline and stored locally or in S3/Athena.",
        ),
    }


def _describe_predict_type_availability(
    *,
    normal_rows: list[dict[str, Any]],
    quant_rows: list[dict[str, Any]],
) -> str:
    if normal_rows and quant_rows:
        return "standard + quantum"
    if normal_rows:
        return "standard only"
    if quant_rows:
        return "quantum only"
    return "unavailable"


def _canonical_predict_type(row: dict[str, Any]) -> str:
    model_family = str(row.get("model_family", "")).lower()
    model_name = str(row.get("model_name", "")).lower()
    is_price_proxy = bool(row.get("is_price_proxy", False))
    if "quantum" in model_family or "vqc" in model_family or "quantum" in model_name:
        return "quant"
    if is_price_proxy:
        return "quant"
    if "keras" in model_family or "lstm" in model_family or "lstm" in model_name:
        return "normal"
    return str(row.get("predict_type", "")).lower()


def _with_canonical_predict_type(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload.setdefault("stored_predict_type", payload.get("predict_type"))
    payload["predict_type"] = _canonical_predict_type(payload)
    return payload


def formatMs(value: float | None) -> str:
    if value is None:
        return "N/A"
    if abs(value) >= 1000.0:
        return formatSeconds(value)
    return f"{value:.1f}ms"


def formatSeconds(value_ms: float | None) -> str:
    if value_ms is None:
        return "N/A"
    return f"{value_ms / 1000.0:.2f}s"


def formatPercent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100.0:+.1f}%"


def _format_unsigned_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100.0:.1f}%"


def formatRatio(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}x"


def _compute_delta_pct(*, value: float | None, base_close: float | None) -> float | None:
    if value is None or base_close is None or abs(base_close) <= 1e-9:
        return None
    return (value - base_close) / base_close * 100.0


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
