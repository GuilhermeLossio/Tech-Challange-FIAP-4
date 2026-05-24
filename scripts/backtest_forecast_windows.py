from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "TensorFlow is required for historical forecast backtests. "
        "Install the project dependencies before running this script."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.services.forecast_guardrails import apply_standard_forecast_guardrail  # noqa: E402
from src.application.use_cases.generate_feature_dataset import GenerateFeatureDatasetUseCase  # noqa: E402
from src.infrastructure.config.settings import ForecastPipelineSettings  # noqa: E402


DEFAULT_SYMBOLS = ("NVDA", "AMD", "TSM", "ASML", "QCOM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest the normal forecast path on historical windows. Example: "
            "hide 2025, forecast it from the last 2024 window, then compare with actual 2025 closes."
        )
    )
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--extraction-date", default=None)
    parser.add_argument("--source", default="yfinance")
    parser.add_argument("--target-column", default="close")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--train-through", default="2024-12-31")
    parser.add_argument("--forecast-start", default="2025-01-01")
    parser.add_argument("--forecast-end", default="2025-12-31")
    parser.add_argument("--model-name-prefix", default="lstm")
    parser.add_argument(
        "--prediction-target-mode",
        choices=("price", "return"),
        default="price",
        help="Use `return` when the selected model was trained with --prediction-target-mode return.",
    )
    parser.add_argument("--output-dir", default="data/processed/backtests/forecast_windows")
    return parser.parse_args()


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def detect_latest_extraction_date(processed_root: Path) -> date:
    manifests_root = processed_root / "manifests"
    candidates: list[date] = []
    for partition in manifests_root.glob("extraction_date=*"):
        if (partition / "refined_manifest.json").exists():
            try:
                candidates.append(parse_iso_date(partition.name.split("=", 1)[-1]))
            except ValueError:
                continue
    if not candidates:
        raise SystemExit(f"No refined manifests found under {manifests_root}.")
    return max(candidates)


def load_scaler_metadata(
    *,
    processed_root: Path,
    extraction_date: date,
    source: str,
    symbol: str,
    target_column: str,
    lookback: int,
) -> dict[str, float]:
    manifest_path = (
        processed_root
        / "manifests"
        / f"extraction_date={extraction_date.isoformat()}"
        / "refined_manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    request = payload.get("request", {})
    if request.get("source") != source or request.get("target_column") != target_column:
        raise ValueError(f"Refined manifest request does not match for {symbol}: {manifest_path}")
    asset = next(
        (
            item
            for item in payload.get("assets", [])
            if str(item.get("symbol", "")).upper() == symbol.upper()
            and int(item.get("feature_count", lookback)) == lookback
        ),
        None,
    )
    if asset is None:
        raise ValueError(f"No refined scaler asset for {symbol} lookback={lookback}.")
    return {
        "min_offset": float(asset["scaler_min_offset"]),
        "scale": float(asset["scaler_scale"]),
    }


def scale_array(values: np.ndarray, *, min_offset: float, scale: float) -> np.ndarray:
    return values * scale + min_offset


def inverse_scale_array(values: np.ndarray, *, min_offset: float, scale: float) -> np.ndarray:
    if scale == 0:
        raise ValueError("Cannot inverse scale values because scale is zero.")
    return (values - min_offset) / scale


def compute_dynamic_cumulative_return_cap(
    *,
    recent_window: np.ndarray,
    forecast_step: int,
    horizon_days: int,
) -> float:
    daily_returns = GenerateFeatureDatasetUseCase._compute_daily_returns(recent_window)  # type: ignore[attr-defined]
    if len(daily_returns) == 0:
        realized_volatility = 0.012
        realized_move = 0.01
    else:
        effective_returns = daily_returns[-min(60, len(daily_returns)) :]
        realized_volatility = float(np.std(effective_returns, ddof=0))
        realized_move = float(np.mean(np.abs(effective_returns)))
    time_scale = np.sqrt(max(float(forecast_step), 1.0))
    horizon_scale = np.sqrt(max(float(horizon_days), 1.0) / 30.0)
    volatility_cap = (2.6 * realized_volatility + 1.2 * realized_move) * time_scale
    min_cap = min(0.22, 0.025 * time_scale * horizon_scale + 0.035)
    max_cap = min(0.65, 0.18 + 0.018 * max(float(horizon_days), 1.0))
    return float(min(max(volatility_cap, min_cap), max_cap))


def run_backtest_for_symbol(
    *,
    raw_root: Path,
    processed_root: Path,
    models_root: Path,
    extraction_date: date,
    source: str,
    symbol: str,
    target_column: str,
    lookback: int,
    train_through: date,
    forecast_start: date,
    forecast_end: date,
    model_name_prefix: str,
    prediction_target_mode: str,
) -> pd.DataFrame:
    raw_path = (
        raw_root
        / "market_data"
        / f"source={source}"
        / f"symbol={symbol.upper()}"
        / f"extraction_date={extraction_date.isoformat()}"
        / "ohlcv.csv"
    )
    raw = pd.read_csv(raw_path, parse_dates=["date"])
    raw = raw.loc[:, ["date", target_column]].dropna().sort_values("date").reset_index(drop=True)
    hidden = raw.loc[
        (raw["date"] >= pd.Timestamp(forecast_start))
        & (raw["date"] <= pd.Timestamp(forecast_end))
    ].copy()
    history = raw.loc[raw["date"] <= pd.Timestamp(train_through)].copy()
    if len(history.index) < lookback:
        raise ValueError(f"Not enough pre-cutoff rows for {symbol}.")
    if hidden.empty:
        raise ValueError(f"No actual rows found in hidden window for {symbol}.")

    model_path = models_root / f"{model_name_prefix}_{symbol.lower()}.keras"
    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = load_scaler_metadata(
        processed_root=processed_root,
        extraction_date=extraction_date,
        source=source,
        symbol=symbol,
        target_column=target_column,
        lookback=lookback,
    )

    forecast_dates = pd.bdate_range(pd.Timestamp(forecast_start), pd.Timestamp(forecast_end))
    raw_window = history[target_column].tail(lookback).to_numpy(dtype=np.float32)
    scaled_window = scale_array(
        raw_window,
        min_offset=scaler["min_offset"],
        scale=scaler["scale"],
    ).astype(np.float32)
    last_observed_close = float(raw_window[-1])

    rows: list[dict[str, Any]] = []
    actual_by_date = {
        pd.Timestamp(row.date).strftime("%Y-%m-%d"): float(getattr(row, target_column))
        for row in hidden.itertuples(index=False)
    }
    for step, forecast_date in enumerate(forecast_dates, start=1):
        input_close = float(raw_window[-1])
        model_output = float(model.predict(scaled_window.reshape(1, lookback, 1), verbose=0).reshape(-1)[0])
        if prediction_target_mode == "return":
            raw_predicted_close = input_close * (1.0 + model_output)
        else:
            raw_predicted_close = float(
                inverse_scale_array(
                    np.asarray([model_output], dtype=np.float32),
                    min_offset=scaler["min_offset"],
                    scale=scaler["scale"],
                )[0]
            )
        guardrail = apply_standard_forecast_guardrail(
            raw_model_close=raw_predicted_close,
            current_close=input_close,
            recent_window=raw_window.astype(np.float64),
        )
        predicted_close = guardrail.constrained_close
        cap = compute_dynamic_cumulative_return_cap(
            recent_window=raw_window.astype(np.float64),
            forecast_step=step,
            horizon_days=len(forecast_dates),
        )
        lower = max(last_observed_close * (1.0 - cap), 0.0)
        upper = last_observed_close * (1.0 + cap)
        predicted_close = float(np.clip(predicted_close, lower, upper))
        actual_close = actual_by_date.get(forecast_date.strftime("%Y-%m-%d"))
        error = None if actual_close is None else predicted_close - actual_close
        rows.append(
            {
                "symbol": symbol.upper(),
                "forecast_step": step,
                "forecast_date": forecast_date.strftime("%Y-%m-%d"),
                "predicted_close": predicted_close,
                "actual_close": actual_close,
                "error": error,
                "abs_error": abs(error) if error is not None else None,
                "ape": abs(error / actual_close) if actual_close and error is not None else None,
                "raw_model_predicted_close": raw_predicted_close,
                "prediction_constraint_applied": bool(
                    guardrail.applied or predicted_close != guardrail.constrained_close
                ),
                "dynamic_cumulative_return_cap": cap,
                "prediction_target_mode": prediction_target_mode,
            }
        )
        raw_window = np.concatenate([raw_window[1:], np.asarray([predicted_close], dtype=np.float32)])
        scaled_next = scale_array(
            np.asarray([predicted_close], dtype=np.float32),
            min_offset=scaler["min_offset"],
            scale=scaler["scale"],
        )
        scaled_window = np.concatenate([scaled_window[1:], scaled_next.astype(np.float32)])
    return pd.DataFrame(rows)


def write_report(*, destination: Path, frame: pd.DataFrame, generated_at_utc: str) -> None:
    lines = [
        "# Historical Forecast Backtest",
        "",
        f"Generated at UTC: `{generated_at_utc}`",
        "",
        "| Symbol | Compared rows | MAE | RMSE | MAPE | Constraint rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    comparable = frame.dropna(subset=["actual_close"]).copy()
    for symbol, group in comparable.groupby("symbol"):
        errors = group["error"].astype(float)
        abs_errors = group["abs_error"].astype(float)
        ape = group["ape"].astype(float)
        lines.append(
            f"| {symbol} | {len(group.index)} | "
            f"{abs_errors.mean():.4f} | "
            f"{np.sqrt(np.mean(np.square(errors))):.4f} | "
            f"{ape.mean() * 100.0:.2f}% | "
            f"{group['prediction_constraint_applied'].astype(bool).mean():.1%} |"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    settings = ForecastPipelineSettings.from_env()
    extraction_date = (
        parse_iso_date(args.extraction_date)
        if args.extraction_date
        else detect_latest_extraction_date(settings.local_processed_dir)
    )
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    generated_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    frames = [
        run_backtest_for_symbol(
            raw_root=settings.local_raw_dir,
            processed_root=settings.local_processed_dir,
            models_root=settings.local_models_dir,
            extraction_date=extraction_date,
            source=args.source,
            symbol=symbol.upper(),
            target_column=args.target_column,
            lookback=args.lookback,
            train_through=parse_iso_date(args.train_through),
            forecast_start=parse_iso_date(args.forecast_start),
            forecast_end=parse_iso_date(args.forecast_end),
            model_name_prefix=args.model_name_prefix,
            prediction_target_mode=args.prediction_target_mode,
        )
        for symbol in args.symbols
    ]
    result = pd.concat(frames, ignore_index=True)
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output_dir / f"forecast_backtest_{token}.csv"
    report_path = output_dir / f"forecast_backtest_{token}.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(csv_path, index=False)
    write_report(destination=report_path, frame=result, generated_at_utc=generated_at_utc)
    print(f"Backtest CSV: {csv_path}")
    print(f"Backtest report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
