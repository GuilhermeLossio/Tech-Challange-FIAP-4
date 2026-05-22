from __future__ import annotations

import argparse
from datetime import date, datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.application.services.predictor_service import StandardPredictorService  # noqa: E402


DEFAULT_SYMBOLS = ("NVDA", "TSM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit scaler fitting, temporal splits, and raw LSTM forecast output."
    )
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--extraction-date", default="2026-04-26")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--source", default="yfinance")
    parser.add_argument("--target-column", default="close")
    parser.add_argument("--model-name-prefix", default="lstm")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--models-root", default="models")
    parser.add_argument(
        "--output-dir",
        default="data/processed/audits/forecast_pipeline",
        help="Directory for markdown, CSV, and SVG audit outputs.",
    )
    return parser.parse_args()


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_refined_manifest(processed_root: Path, extraction_date: date) -> dict[str, Any]:
    manifest_path = (
        processed_root
        / "manifests"
        / f"extraction_date={extraction_date.isoformat()}"
        / "refined_manifest.json"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Refined manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def audit_scaler_and_splits(
    *,
    manifest: dict[str, Any],
    symbols: list[str],
    lookback: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    assets = manifest.get("assets", [])
    for symbol in symbols:
        asset = next(
            (
                item
                for item in assets
                if str(item.get("symbol", "")).upper() == symbol.upper()
                and int(item.get("feature_count", lookback)) == lookback
            ),
            None,
        )
        if asset is None:
            rows.append({"symbol": symbol.upper(), "status": "missing_refined_asset"})
            continue

        refined_path = Path(str(asset["local_path"]))
        frame = pd.read_parquet(refined_path)
        frame["target_date"] = pd.to_datetime(frame["target_date"])
        frame["window_start_date"] = pd.to_datetime(frame["window_start_date"])
        frame["window_end_date"] = pd.to_datetime(frame["window_end_date"])
        split = frame["split"].astype(str).str.lower()
        train = frame.loc[split == "train"].copy()
        validation = frame.loc[split == "validation"].copy()
        test = frame.loc[split == "test"].copy()

        scaler_fit_end = pd.Timestamp(asset["scaler_fit_end_date"])
        train_target_max = train["target_date"].max()
        validation_target_min = validation["target_date"].min() if len(validation) else pd.NaT
        test_target_min = test["target_date"].min() if len(test) else pd.NaT
        train_has_post_2025 = bool((train["target_date"] > pd.Timestamp("2025-12-31")).any())

        rows.append(
            {
                "symbol": symbol.upper(),
                "status": "ok",
                "lookback": lookback,
                "refined_path": str(refined_path),
                "history_start": asset.get("history_start_date"),
                "history_end": asset.get("history_end_date"),
                "scaler_fit_start": asset.get("scaler_fit_start_date"),
                "scaler_fit_end": asset.get("scaler_fit_end_date"),
                "train_target_start": train["target_date"].min().strftime("%Y-%m-%d"),
                "train_target_end": train_target_max.strftime("%Y-%m-%d"),
                "validation_target_start": (
                    validation_target_min.strftime("%Y-%m-%d")
                    if not pd.isna(validation_target_min)
                    else ""
                ),
                "test_target_start": (
                    test_target_min.strftime("%Y-%m-%d") if not pd.isna(test_target_min) else ""
                ),
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "test_rows": int(len(test)),
                "scaler_fit_matches_train_end": bool(scaler_fit_end == train_target_max),
                "scaler_before_validation": bool(
                    pd.isna(validation_target_min) or scaler_fit_end < validation_target_min
                ),
                "scaler_before_test": bool(pd.isna(test_target_min) or scaler_fit_end < test_target_min),
                "train_has_post_2025_target": train_has_post_2025,
            }
        )
    return rows


def run_unconstrained_lstm_forecast(
    *,
    raw_root: Path,
    processed_root: Path,
    models_root: Path,
    symbol: str,
    extraction_date: date,
    source: str,
    target_column: str,
    lookback: int,
    horizon_days: int,
    model_name_prefix: str,
) -> pd.DataFrame:
    import tensorflow as tf

    service = StandardPredictorService(
        raw_root_dir=raw_root,
        processed_root_dir=processed_root,
        models_root_dir=models_root,
        source=source,
        target_column=target_column,
        lookback=lookback,
        model_name_prefix=model_name_prefix,
    )
    resolution = service._resolve_serving_model_resolution(  # noqa: SLF001
        symbol=symbol.upper(),
        requested_extraction_date=extraction_date,
    )
    scaler = service._load_scaler_metadata(  # noqa: SLF001
        symbol=symbol.upper(),
        extraction_date=resolution.extraction_date,
    )
    model = tf.keras.models.load_model(str(resolution.model_local_path), compile=False)

    raw_path = (
        raw_root
        / "market_data"
        / f"source={source}"
        / f"symbol={symbol.upper()}"
        / f"extraction_date={resolution.extraction_date.isoformat()}"
        / "ohlcv.csv"
    )
    raw = pd.read_csv(raw_path, parse_dates=["date"])
    raw = raw.loc[:, ["date", target_column]].dropna().sort_values("date").reset_index(drop=True)
    raw_window = raw[target_column].tail(lookback).to_numpy(dtype=np.float32)
    scaled_window = service._scale_array(  # noqa: SLF001
        raw_window,
        min_offset=scaler["min_offset"],
        scale=scaler["scale"],
    ).astype(np.float32)

    last_observed_date = pd.Timestamp(raw["date"].iloc[-1])
    last_observed_close = float(raw_window[-1])
    forecast_dates = pd.bdate_range(last_observed_date + pd.offsets.BDay(1), periods=horizon_days)

    rows = []
    for step, forecast_date in enumerate(forecast_dates, start=1):
        input_end_close = float(raw_window[-1])
        predicted_scaled = float(model.predict(scaled_window.reshape(1, lookback, 1), verbose=0).reshape(-1)[0])
        predicted_close = float(
            service._inverse_scale_array(  # noqa: SLF001
                np.asarray([predicted_scaled], dtype=np.float32),
                min_offset=scaler["min_offset"],
                scale=scaler["scale"],
            )[0]
        )
        rows.append(
            {
                "symbol": symbol.upper(),
                "forecast_step": step,
                "forecast_date": forecast_date.strftime("%Y-%m-%d"),
                "input_window_end_close": input_end_close,
                "raw_lstm_predicted_close": predicted_close,
                "raw_lstm_predicted_scaled": predicted_scaled,
                "raw_step_return": (
                    predicted_close / input_end_close - 1.0 if abs(input_end_close) > 1e-8 else 0.0
                ),
                "raw_horizon_return_from_last_observed": (
                    predicted_close / last_observed_close - 1.0
                    if abs(last_observed_close) > 1e-8
                    else 0.0
                ),
                "model_local_path": str(resolution.model_local_path),
            }
        )
        raw_window = np.concatenate([raw_window[1:], np.asarray([predicted_close], dtype=np.float32)])
        scaled_window = np.concatenate([scaled_window[1:], np.asarray([predicted_scaled], dtype=np.float32)])

    return pd.DataFrame(rows)


def load_latest_materialized_forecast(
    *,
    processed_root: Path,
    symbol: str,
    source: str,
    lookback: int,
    extraction_date: date,
) -> pd.DataFrame | None:
    root = (
        processed_root
        / "future_predict"
        / f"source={source}"
        / f"symbol={symbol.upper()}"
        / f"lookback={lookback}"
        / f"extraction_date={extraction_date.isoformat()}"
    )
    candidates = sorted(root.glob("horizon_days=*/generated_at=*/future_predict.parquet"))
    if not candidates:
        return None
    return pd.read_parquet(candidates[-1])


def write_raw_plot(
    *,
    output_path: Path,
    symbol: str,
    raw_frame: pd.DataFrame,
    materialized_frame: pd.DataFrame | None,
) -> None:
    plt.figure(figsize=(11, 5.8))
    x = pd.to_datetime(raw_frame["forecast_date"])
    plt.plot(x, raw_frame["raw_lstm_predicted_close"], label="LSTM raw recursive", linewidth=2.0)
    if materialized_frame is not None and not materialized_frame.empty:
        normal = materialized_frame.loc[
            materialized_frame["predict_type"].astype(str).str.lower() == "normal"
        ].sort_values("forecast_step")
        if not normal.empty:
            plt.plot(
                pd.to_datetime(normal["forecast_date"]),
                normal["predicted_close"].astype(float),
                label="Materialized constrained forecast",
                linewidth=1.7,
                linestyle="--",
            )
            if "raw_model_predicted_close" in normal.columns:
                plt.plot(
                    pd.to_datetime(normal["forecast_date"]),
                    normal["raw_model_predicted_close"].astype(float),
                    label="Stored raw before step constraint",
                    linewidth=1.2,
                    linestyle=":",
                )
    plt.title(f"{symbol.upper()} LSTM Raw Output vs Constrained Forecast")
    plt.xlabel("Forecast date")
    plt.ylabel("Close")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="svg")
    plt.close()


def write_markdown_report(
    *,
    output_path: Path,
    scaler_rows: list[dict[str, Any]],
    raw_rows: pd.DataFrame,
    generated_files: list[Path],
) -> None:
    lines = [
        "# Forecast Pipeline Audit",
        "",
        f"Generated at UTC: `{datetime.utcnow().replace(microsecond=0).isoformat()}Z`",
        "",
        "## Scaler and temporal split checks",
        "",
        "| Symbol | Lookback | Scaler fit | Train targets | Validation starts | Test starts | Fit=train end | Before val/test | Train > 2025 |",
        "|---|---:|---|---|---|---|---:|---:|---:|",
    ]
    for row in scaler_rows:
        if row.get("status") != "ok":
            lines.append(f"| {row.get('symbol')} | | {row.get('status')} | | | | | | |")
            continue
        lines.append(
            "| {symbol} | {lookback} | {scaler_fit_start} -> {scaler_fit_end} | "
            "{train_target_start} -> {train_target_end} | {validation_target_start} | "
            "{test_target_start} | {scaler_fit_matches_train_end} | {before} | "
            "{train_has_post_2025_target} |".format(
                before=bool(row["scaler_before_validation"] and row["scaler_before_test"]),
                **row,
            )
        )

    lines.extend(
        [
            "",
            "## Raw LSTM forecast summary",
            "",
            "| Symbol | Steps | First raw close | Last raw close | Min raw close | Max raw close | Last raw horizon return |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for symbol, group in raw_rows.groupby("symbol"):
        ordered = group.sort_values("forecast_step")
        lines.append(
            f"| {symbol} | {len(ordered)} | "
            f"{ordered['raw_lstm_predicted_close'].iloc[0]:.4f} | "
            f"{ordered['raw_lstm_predicted_close'].iloc[-1]:.4f} | "
            f"{ordered['raw_lstm_predicted_close'].min():.4f} | "
            f"{ordered['raw_lstm_predicted_close'].max():.4f} | "
            f"{ordered['raw_horizon_return_from_last_observed'].iloc[-1] * 100:.2f}% |"
        )

    lines.extend(["", "## Generated files", ""])
    for path in generated_files:
        lines.append(f"- `{path}`")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    extraction_date = parse_iso_date(args.extraction_date)
    symbols = [symbol.upper() for symbol in args.symbols]
    processed_root = resolve_project_path(args.processed_root)
    raw_root = resolve_project_path(args.raw_root)
    models_root = resolve_project_path(args.models_root)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_refined_manifest(processed_root, extraction_date)
    scaler_rows = audit_scaler_and_splits(
        manifest=manifest,
        symbols=symbols,
        lookback=args.lookback,
    )
    scaler_csv = output_dir / f"scaler_split_audit_lookback={args.lookback}.csv"
    pd.DataFrame(scaler_rows).to_csv(scaler_csv, index=False)

    raw_frames = []
    generated_files: list[Path] = [scaler_csv]
    for symbol in symbols:
        raw_frame = run_unconstrained_lstm_forecast(
            raw_root=raw_root,
            processed_root=processed_root,
            models_root=models_root,
            symbol=symbol,
            extraction_date=extraction_date,
            source=args.source,
            target_column=args.target_column,
            lookback=args.lookback,
            horizon_days=args.horizon_days,
            model_name_prefix=args.model_name_prefix,
        )
        raw_frames.append(raw_frame)
        raw_csv = output_dir / f"{symbol.lower()}_raw_lstm_unconstrained_lookback={args.lookback}.csv"
        raw_frame.to_csv(raw_csv, index=False)
        generated_files.append(raw_csv)
        materialized = load_latest_materialized_forecast(
            processed_root=processed_root,
            symbol=symbol,
            source=args.source,
            lookback=args.lookback,
            extraction_date=extraction_date,
        )
        plot_path = output_dir / f"{symbol.lower()}_raw_vs_constrained_lookback={args.lookback}.svg"
        write_raw_plot(
            output_path=plot_path,
            symbol=symbol,
            raw_frame=raw_frame,
            materialized_frame=materialized,
        )
        generated_files.append(plot_path)

    raw_rows = pd.concat(raw_frames, ignore_index=True)
    report_path = output_dir / f"forecast_pipeline_audit_lookback={args.lookback}.md"
    write_markdown_report(
        output_path=report_path,
        scaler_rows=scaler_rows,
        raw_rows=raw_rows,
        generated_files=generated_files,
    )
    print(f"Audit report: {report_path}")
    for path in generated_files:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
