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
SIX_MONTH_TRADING_DAYS = 126


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
        )
        row_limit = _parse_positive_int(request.args.get("limit"), default=12)

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
        forecast_rows = list(future_result.rows) if future_result else []
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
            next_day_prediction=next_day_prediction,
            future_result=future_result,
            comparison_future_result=comparison_future_result,
            historical_window=historical_window,
            training_summary=training_summary,
            market_context_chart=market_context_chart,
            training_chart=training_chart,
            forecast_rows=forecast_rows,
            outlook_cards=outlook_cards,
            overview_metrics=overview_metrics,
            company_snapshot_rows=company_snapshot_rows,
            method_summary=method_summary,
            data_usage_summary=data_usage_summary,
            warnings=warnings,
            today_utc=datetime.utcnow().strftime("%Y-%m-%d"),
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


def _resolve_front_horizon_days(
    *,
    raw_value: str | None,
    future_prediction_service: FuturePredictionService,
    symbol: str,
    lookback: int,
) -> int:
    if raw_value is not None and raw_value.strip():
        return _parse_positive_int(raw_value, default=30)

    available_horizons = future_prediction_service.list_available_horizon_days(
        symbol=symbol,
        lookback=lookback,
    )
    if available_horizons:
        return available_horizons[-1]
    return 30


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
        grouped_forecasts[str(row["predict_type"]).lower()].append(dict(row))

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
        last_row = rows[-1]
        forecast_series.append(
            {
                "name": predict_type,
                "color": colors.get(predict_type, "#334155"),
                "dasharray": dasharrays.get(predict_type, ""),
                "polyline": " ".join(points),
                "last_value": float(last_row["predicted_close"]),
                "last_date": str(last_row["forecast_date"]),
            }
        )

    y_guides = []
    for value in (max_value, (max_value + min_value) / 2.0, min_value):
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
        grouped[str(row["predict_type"]).lower()].append(dict(row))

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
                else future_prediction_service.load_forecasts(
                    symbol=item,
                    extraction_date=parsed_extraction_date,
                    predict_type="all",
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
            grouped[str(row["predict_type"]).lower()].append(dict(row))

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
            }
        )

    return tuple(rows)


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
            grouped[str(row["predict_type"]).lower()].append(dict(row))
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


def _compute_delta_pct(*, value: float | None, base_close: float | None) -> float | None:
    if value is None or base_close is None or abs(base_close) <= 1e-9:
        return None
    return (value - base_close) / base_close * 100.0


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
