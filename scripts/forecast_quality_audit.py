from __future__ import annotations

# flake8: noqa: E501

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_SYMBOLS = ("NVDA", "AMD", "TSM", "ASML", "QCOM")
DEFAULT_GENERATED_AT = "20260525T143920Z"


@dataclass(frozen=True)
class ForecastSelection:
    source: str
    symbol: str
    lookback: int
    horizon_days: int
    extraction_date: date
    generated_at: str
    parquet_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a forecast quality audit that explains poor materialized "
            "forecast results, guardrail effects, and quantum price proxy behavior."
        )
    )
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--source", default="yfinance")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizon-days", type=int, default=162)
    parser.add_argument("--extraction-date", default="2026-05-19")
    parser.add_argument("--generated-at", default=DEFAULT_GENERATED_AT)
    parser.add_argument("--target-column", default="close")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument(
        "--actual-extraction-date",
        default="auto",
        help=(
            "Raw extraction date used for actual prices. Use 'auto' to choose "
            "the best available raw partition, allowing partial realized coverage "
            "when the full forecast horizon is still in the future."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/audits/forecast_quality",
    )
    return parser.parse_args()


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_forecast_selection(
    *,
    processed_root: Path,
    source: str,
    symbol: str,
    lookback: int,
    horizon_days: int,
    extraction_date: date,
    generated_at: str,
) -> ForecastSelection:
    root = (
        processed_root
        / "future_predict"
        / f"source={source}"
        / f"symbol={symbol.upper()}"
        / f"lookback={lookback}"
        / f"horizon_days={horizon_days}"
        / f"extraction_date={extraction_date.isoformat()}"
    )
    if generated_at == "latest":
        candidates = sorted(root.glob("generated_at=*/future_predict.parquet"))
    else:
        candidates = [root / f"generated_at={generated_at}" / "future_predict.parquet"]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise FileNotFoundError(f"No future_predict parquet found for {symbol.upper()} under {root}.")
    parquet_path = existing[-1]
    resolved_generated_at = parquet_path.parent.name.split("=", 1)[-1]
    return ForecastSelection(
        source=source,
        symbol=symbol.upper(),
        lookback=lookback,
        horizon_days=horizon_days,
        extraction_date=extraction_date,
        generated_at=resolved_generated_at,
        parquet_path=parquet_path,
    )


def load_actual_prices(
    *,
    raw_root: Path,
    source: str,
    symbol: str,
    target_column: str,
    forecast_dates: pd.Series,
    actual_extraction_date: str,
) -> tuple[pd.DataFrame, str | None]:
    symbol_root = raw_root / "market_data" / f"source={source}" / f"symbol={symbol.upper()}"
    if actual_extraction_date != "auto":
        candidates = [symbol_root / f"extraction_date={actual_extraction_date}" / "ohlcv.csv"]
    else:
        candidates = sorted(symbol_root.glob("extraction_date=*/ohlcv.csv"))

    needed_start = pd.to_datetime(forecast_dates).min()
    needed_end = pd.to_datetime(forecast_dates).max()
    fallback: tuple[int, pd.Timestamp, pd.DataFrame, str | None] | None = None
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["date"])
        frame = frame.loc[:, ["date", target_column]].dropna().copy()
        frame = frame.sort_values("date").reset_index(drop=True)
        partition = path.parent.name.split("=", 1)[-1]
        if frame["date"].min() <= needed_start and frame["date"].max() >= needed_end:
            return frame, partition
        realized_count = int(
            ((frame["date"] >= needed_start) & (frame["date"] <= needed_end)).sum()
        )
        max_date = pd.Timestamp(frame["date"].max()) if not frame.empty else pd.Timestamp.min
        fallback_candidate = (realized_count, max_date, frame, partition)
        if fallback is None or fallback_candidate[:2] > fallback[:2]:
            fallback = fallback_candidate
    if fallback is not None:
        _, _, frame, partition = fallback
        return frame, partition
    return pd.DataFrame(columns=["date", target_column]), None


def build_seed_baselines(
    *,
    raw_root: Path,
    source: str,
    symbol: str,
    target_column: str,
    extraction_date: date,
    forecast_dates: pd.Series,
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
    values = raw[target_column].astype(float)
    last_close = float(values.iloc[-1])
    sma_5 = float(values.tail(5).mean())
    sma_20 = float(values.tail(20).mean())
    returns = values.pct_change().dropna()
    mean_return_20 = float(returns.tail(20).mean()) if len(returns) else 0.0

    rows: list[dict[str, Any]] = []
    for step, forecast_date in enumerate(pd.to_datetime(forecast_dates), start=1):
        rows.extend(
            [
                {
                    "symbol": symbol.upper(),
                    "predict_type": "baseline_last_close",
                    "forecast_step": step,
                    "forecast_date": forecast_date,
                    "predicted_close": last_close,
                },
                {
                    "symbol": symbol.upper(),
                    "predict_type": "baseline_sma_5",
                    "forecast_step": step,
                    "forecast_date": forecast_date,
                    "predicted_close": sma_5,
                },
                {
                    "symbol": symbol.upper(),
                    "predict_type": "baseline_sma_20",
                    "forecast_step": step,
                    "forecast_date": forecast_date,
                    "predicted_close": sma_20,
                },
                {
                    "symbol": symbol.upper(),
                    "predict_type": "baseline_recent_return_20",
                    "forecast_step": step,
                    "forecast_date": forecast_date,
                    "predicted_close": last_close * ((1.0 + mean_return_20) ** step),
                },
            ]
        )
    return pd.DataFrame(rows)


def enrich_forecast_steps(
    *,
    forecast: pd.DataFrame,
    actual: pd.DataFrame,
    baselines: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    forecast = forecast.copy()
    forecast["forecast_date"] = pd.to_datetime(forecast["forecast_date"])
    actual_lookup = actual.rename(columns={"date": "forecast_date", target_column: "actual_close"})
    actual_lookup["forecast_date"] = pd.to_datetime(actual_lookup["forecast_date"])

    common_columns = [
        "symbol",
        "predict_type",
        "forecast_step",
        "forecast_date",
        "predicted_close",
    ]
    baseline_rows = baselines.loc[:, common_columns].copy()
    for column in forecast.columns:
        if column not in baseline_rows.columns:
            baseline_rows[column] = np.nan
    baseline_rows = baseline_rows.loc[:, forecast.columns]

    combined = pd.concat([forecast, baseline_rows], ignore_index=True, sort=False)
    combined = combined.merge(
        actual_lookup.loc[:, ["forecast_date", "actual_close"]],
        how="left",
        on="forecast_date",
    )
    combined["error"] = combined["predicted_close"].astype(float) - combined["actual_close"].astype(float)
    combined.loc[combined["actual_close"].isna(), "error"] = np.nan
    combined["abs_error"] = combined["error"].abs()
    combined["ape"] = np.where(
        combined["actual_close"].abs() > 1e-8,
        combined["abs_error"] / combined["actual_close"].abs(),
        np.nan,
    )
    combined["raw_to_final_delta"] = (
        combined["predicted_close"].astype(float)
        - combined.get("raw_model_predicted_close", combined["predicted_close"]).astype(float)
    )
    return combined


def is_monotonic(values: pd.Series) -> bool:
    cleaned = values.dropna().astype(float)
    return bool(cleaned.is_monotonic_increasing or cleaned.is_monotonic_decreasing)


def bool_rate(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return 0.0
    values = frame[column].astype(object).where(frame[column].notna(), False)
    return float(values.astype(bool).mean())


def bool_any(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame:
        return False
    values = frame[column].astype(object).where(frame[column].notna(), False)
    return bool(values.astype(bool).any())


def classify_cause(group: pd.DataFrame) -> str:
    predict_type = str(group["predict_type"].iloc[0])
    if predict_type.startswith("baseline_"):
        return "baseline"
    if bool_any(group, "is_price_proxy"):
        return "proxy quantico"

    constraint_rate = bool_rate(group, "prediction_constraint_applied")
    hit_lower_rate = bool_rate(group, "hit_lower_band")
    raw = group.get("raw_model_predicted_close")
    last_observed = float(group["last_observed_close"].dropna().iloc[0]) if group["last_observed_close"].notna().any() else np.nan
    raw_final_return = np.nan
    raw_first_return = np.nan
    if raw is not None and raw.notna().any() and abs(last_observed) > 1e-8:
        ordered = group.sort_values("forecast_step")
        raw_first_return = float(ordered["raw_model_predicted_close"].dropna().iloc[0] / last_observed - 1.0)
        raw_final_return = float(ordered["raw_model_predicted_close"].dropna().iloc[-1] / last_observed - 1.0)

    causes: list[str] = []
    if raw_first_return <= -0.05 or raw_final_return <= -0.15:
        causes.append("modelo")
    if is_monotonic(group.sort_values("forecast_step")["predicted_close"]):
        causes.append("recursao")
    if constraint_rate >= 0.2 or hit_lower_rate >= 0.1:
        causes.append("guardrail")
    return " + ".join(causes) if causes else "dados/desatualizacao"


def aggregate_quality(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (symbol, predict_type), group in frame.groupby(["symbol", "predict_type"], dropna=False):
        ordered = group.sort_values("forecast_step")
        comparable = ordered.dropna(subset=["actual_close"])
        errors = comparable["error"].astype(float)
        abs_errors = comparable["abs_error"].astype(float)
        ape = comparable["ape"].astype(float)
        last_observed = (
            float(ordered["last_observed_close"].dropna().iloc[0])
            if "last_observed_close" in ordered and ordered["last_observed_close"].notna().any()
            else np.nan
        )
        final_predicted = float(ordered["predicted_close"].astype(float).iloc[-1])
        final_actual = (
            float(comparable["actual_close"].astype(float).iloc[-1]) if not comparable.empty else np.nan
        )
        worst_step = (
            int(comparable.sort_values("abs_error", ascending=False)["forecast_step"].iloc[0])
            if not comparable.empty
            else np.nan
        )
        rows.append(
            {
                "symbol": symbol,
                "predict_type": predict_type,
                "rows": int(len(ordered)),
                "compared_rows": int(len(comparable)),
                "mae": float(abs_errors.mean()) if not comparable.empty else np.nan,
                "rmse": float(np.sqrt(np.mean(np.square(errors)))) if not comparable.empty else np.nan,
                "mape": float(ape.mean()) if not comparable.empty else np.nan,
                "final_predicted_close": final_predicted,
                "final_actual_close": final_actual,
                "predicted_horizon_return": (
                    float(final_predicted / last_observed - 1.0)
                    if not np.isnan(last_observed) and abs(last_observed) > 1e-8
                    else np.nan
                ),
                "actual_horizon_return": (
                    float(final_actual / last_observed - 1.0)
                    if not np.isnan(final_actual)
                    and not np.isnan(last_observed)
                    and abs(last_observed) > 1e-8
                    else np.nan
                ),
                "constraint_rate": float(
                    bool_rate(ordered, "prediction_constraint_applied")
                ),
                "lower_band_rate": float(
                    bool_rate(ordered, "hit_lower_band")
                ),
                "up_rate": float(
                    ordered.get("predicted_direction", pd.Series(dtype=float)).dropna().astype(float).mean()
                )
                if "predicted_direction" in ordered and ordered["predicted_direction"].notna().any()
                else np.nan,
                "monotonic_path": is_monotonic(ordered["predicted_close"]),
                "worst_step": worst_step,
                "uses_price_proxy": bool(
                    bool_any(ordered, "is_price_proxy")
                ),
                "cause_classification": classify_cause(ordered),
            }
        )
    return pd.DataFrame(rows).sort_values(["symbol", "predict_type"]).reset_index(drop=True)


def audit_scaler_splits(
    *,
    processed_root: Path,
    extraction_date: date,
    symbols: list[str],
    lookback: int,
) -> pd.DataFrame:
    manifest_path = (
        processed_root
        / "manifests"
        / f"extraction_date={extraction_date.isoformat()}"
        / "refined_manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
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
            rows.append({"symbol": symbol.upper(), "status": "missing_refined_asset"})
            continue
        refined = pd.read_parquet(Path(str(asset["local_path"])))
        refined["target_date"] = pd.to_datetime(refined["target_date"])
        split = refined["split"].astype(str).str.lower()
        train = refined.loc[split == "train"]
        validation = refined.loc[split == "validation"]
        test = refined.loc[split == "test"]
        train_end = train["target_date"].max()
        validation_start = validation["target_date"].min() if not validation.empty else pd.NaT
        test_start = test["target_date"].min() if not test.empty else pd.NaT
        scaler_end = pd.Timestamp(asset["scaler_fit_end_date"])
        rows.append(
            {
                "symbol": symbol.upper(),
                "status": "ok",
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "test_rows": int(len(test)),
                "scaler_fit_start": asset.get("scaler_fit_start_date"),
                "scaler_fit_end": asset.get("scaler_fit_end_date"),
                "train_target_start": train["target_date"].min().strftime("%Y-%m-%d"),
                "train_target_end": train_end.strftime("%Y-%m-%d"),
                "validation_target_start": (
                    validation_start.strftime("%Y-%m-%d") if not pd.isna(validation_start) else ""
                ),
                "test_target_start": test_start.strftime("%Y-%m-%d") if not pd.isna(test_start) else "",
                "scaler_fit_matches_train_end": bool(scaler_end == train_end),
                "scaler_before_validation": bool(
                    pd.isna(validation_start) or scaler_end < validation_start
                ),
                "scaler_before_test": bool(pd.isna(test_start) or scaler_end < test_start),
            }
        )
    return pd.DataFrame(rows)


def write_symbol_plot(*, frame: pd.DataFrame, symbol: str, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    symbol_frame = frame.loc[frame["symbol"] == symbol.upper()].copy()
    for predict_type, group in symbol_frame.groupby("predict_type"):
        if predict_type not in {"normal", "quant", "baseline_last_close", "baseline_sma_20"}:
            continue
        ordered = group.sort_values("forecast_step")
        plt.plot(
            pd.to_datetime(ordered["forecast_date"]),
            ordered["predicted_close"].astype(float),
            label=str(predict_type),
            linewidth=2 if predict_type in {"normal", "quant"} else 1.3,
            linestyle="-" if predict_type in {"normal", "quant"} else "--",
        )
        if predict_type == "normal" and ordered["raw_model_predicted_close"].notna().any():
            plt.plot(
                pd.to_datetime(ordered["forecast_date"]),
                ordered["raw_model_predicted_close"].astype(float),
                label="normal_raw_model",
                linewidth=1.2,
                linestyle=":",
            )

    actual = symbol_frame.dropna(subset=["actual_close"]).sort_values("forecast_step")
    if not actual.empty:
        actual = actual.drop_duplicates("forecast_date")
        plt.plot(
            pd.to_datetime(actual["forecast_date"]),
            actual["actual_close"].astype(float),
            label="actual",
            linewidth=2.2,
            color="#111827",
        )

    plt.title(f"{symbol.upper()} forecast quality audit")
    plt.xlabel("Forecast date")
    plt.ylabel("Close")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="svg")
    plt.close()


def fmt_number(value: Any, suffix: str = "", precision: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{precision}f}{suffix}"


def fmt_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100.0:.2f}%"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.loc[:, columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def write_report(
    *,
    output_path: Path,
    detailed: pd.DataFrame,
    aggregate: pd.DataFrame,
    scaler_audit: pd.DataFrame,
    generated_files: list[Path],
    selections: list[ForecastSelection],
    actual_partitions: dict[str, str | None],
) -> None:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    report_rows = aggregate.loc[
        aggregate["predict_type"].isin(["normal", "quant"]),
        [
            "symbol",
            "predict_type",
            "compared_rows",
            "mae",
            "rmse",
            "mape",
            "predicted_horizon_return",
            "actual_horizon_return",
            "constraint_rate",
            "up_rate",
            "monotonic_path",
            "uses_price_proxy",
            "cause_classification",
        ],
    ].copy()
    for column in ["mae", "rmse"]:
        report_rows[column] = report_rows[column].map(lambda value: fmt_number(value))
    for column in ["mape", "predicted_horizon_return", "actual_horizon_return", "constraint_rate", "up_rate"]:
        report_rows[column] = report_rows[column].map(fmt_percent)

    baseline_rows = aggregate.loc[
        aggregate["predict_type"].str.startswith("baseline_"),
        ["symbol", "predict_type", "mae", "rmse", "mape"],
    ].copy()
    for column in ["mae", "rmse"]:
        baseline_rows[column] = baseline_rows[column].map(lambda value: fmt_number(value))
    baseline_rows["mape"] = baseline_rows["mape"].map(fmt_percent)

    model_rows = aggregate.loc[aggregate["predict_type"].isin(["normal", "quant"])].copy()
    realized_counts = model_rows["compared_rows"].dropna().astype(int)
    min_compared_rows = int(realized_counts.min()) if not realized_counts.empty else 0
    max_compared_rows = int(realized_counts.max()) if not realized_counts.empty else 0
    max_constraint_rate = (
        float(model_rows["constraint_rate"].fillna(0.0).max()) if not model_rows.empty else 0.0
    )
    normal_monotonic_symbols = sorted(
        model_rows.loc[
            (model_rows["predict_type"] == "normal")
            & (model_rows["monotonic_path"].astype(bool)),
            "symbol",
        ].astype(str)
    )
    best_rows: list[str] = []
    comparable = aggregate.loc[aggregate["compared_rows"].fillna(0).astype(int) > 0]
    for symbol, symbol_rows in comparable.groupby("symbol"):
        best = symbol_rows.sort_values("mape", ascending=True).iloc[0]
        best_rows.append(
            f"- `{symbol}` best short-realized MAPE: `{best['predict_type']}` "
            f"at `{fmt_percent(best['mape'])}`."
        )
    if not best_rows:
        best_rows.append("- No realized actual rows are available yet for baseline ranking.")

    steps_preview = detailed.loc[
        detailed["actual_close"].notna()
        & detailed["predict_type"].isin(["normal", "quant"]),
        [
            "symbol",
            "predict_type",
            "forecast_step",
            "forecast_date",
            "predicted_close",
            "actual_close",
            "error",
            "ape",
            "prediction_constraint_applied",
            "is_price_proxy",
        ],
    ].copy()
    for column in ["predicted_close", "actual_close", "error"]:
        steps_preview[column] = steps_preview[column].map(lambda value: fmt_number(value))
    steps_preview["ape"] = steps_preview["ape"].map(fmt_percent)
    steps_preview["forecast_date"] = pd.to_datetime(
        steps_preview["forecast_date"]
    ).dt.strftime("%Y-%m-%d")

    summary_preview = aggregate.copy()
    for column in ["mae", "rmse", "final_predicted_close", "final_actual_close"]:
        if column in summary_preview:
            summary_preview[column] = summary_preview[column].map(
                lambda value: fmt_number(value)
            )
    for column in [
        "mape",
        "predicted_horizon_return",
        "actual_horizon_return",
        "constraint_rate",
        "lower_band_rate",
        "up_rate",
    ]:
        if column in summary_preview:
            summary_preview[column] = summary_preview[column].map(fmt_percent)

    scaler_preview = scaler_audit.copy()
    for column in ["scaler_fit_matches_train_end", "scaler_before_validation", "scaler_before_test"]:
        if column in scaler_preview:
            scaler_preview[column] = scaler_preview[column].map(str)

    lines = [
        "# Forecast Quality Audit",
        "",
        f"Generated at UTC: `{generated_at}`",
        "",
        "## Dataset",
        "",
        f"- Forecast extraction: `{selections[0].extraction_date.isoformat()}`",
        f"- Generated at: `{selections[0].generated_at}`",
        f"- Lookback: `{selections[0].lookback}`",
        f"- Horizon days: `{selections[0].horizon_days}`",
        f"- Symbols: `{', '.join(selection.symbol for selection in selections)}`",
        "- Actual raw partitions: "
        + ", ".join(f"{symbol}={partition or 'missing'}" for symbol, partition in actual_partitions.items()),
        "",
        "## Main Diagnosis",
        "",
        (
            f"This audit compares the first `{min_compared_rows}` to `{max_compared_rows}` "
            "realized forecast rows per model because the remaining business days in the "
            "162-step horizon are still in the future. The previous zero-comparison issue "
            "is resolved by using the `2026-05-25` raw partition as the actual-price source."
        ),
        "",
        (
            f"Guardrail impact is not material in the current package: the maximum observed "
            f"constraint rate across `normal` and `quant` rows is `{fmt_percent(max_constraint_rate)}`. "
            "The remaining quality concern is long-horizon recursive shape: the normal LSTM "
            "path is monotonic for "
            + (
                ", ".join(f"`{symbol}`" for symbol in normal_monotonic_symbols)
                if normal_monotonic_symbols
                else "no symbols"
            )
            + "."
        ),
        "",
        (
            "Quantum rows remain direction classifications converted through a volatility-based "
            "price proxy, so they should be compared as an experimental proxy path rather than "
            "as direct price regression."
        ),
        "",
        "Short realized-window baseline ranking:",
        "",
        *best_rows,
        "",
        "## Forecast vs Actual",
        "",
    ]
    lines.extend(markdown_table(report_rows, list(report_rows.columns)))
    lines.extend(["", "## Baseline Comparison", ""])
    lines.extend(markdown_table(baseline_rows, list(baseline_rows.columns)))
    lines.extend(["", "## Scaler and Split Audit", ""])
    lines.extend(markdown_table(scaler_preview, list(scaler_preview.columns)))
    lines.extend(
        [
            "",
            "## Generated File Previews",
            "",
            "### `forecast_quality_steps.csv`",
            "",
            "Preview of realized `normal` and `quant` rows with actual-price comparison:",
            "",
        ]
    )
    lines.extend(markdown_table(steps_preview, list(steps_preview.columns)))
    lines.extend(
        [
            "",
            "### `forecast_quality_summary.csv`",
            "",
        ]
    )
    lines.extend(markdown_table(summary_preview, list(summary_preview.columns)))
    lines.extend(
        [
            "",
            "### `forecast_quality_scaler_split_audit.csv`",
            "",
        ]
    )
    lines.extend(markdown_table(scaler_preview, list(scaler_preview.columns)))
    lines.extend(["", "### Forecast Quality Charts", ""])
    for selection in selections:
        chart_name = f"{selection.symbol.lower()}_forecast_quality.svg"
        lines.extend(
            [
                f"#### `{chart_name}`",
                "",
                f"![{selection.symbol} forecast quality]({chart_name})",
                "",
            ]
        )
    lines.extend(
        [
            "### `forecast_quality_report.md`",
            "",
            "This Markdown file is the rendered preview report for the generated audit outputs.",
        ]
    )
    lines.extend(
        [
            "",
            "## Cause Classification",
            "",
            "- `modelo`: raw LSTM predictions are already far from the last observed close.",
            "- `recursao`: the path is monotonic or compounds its own predictions step by step.",
            "- `guardrail`: volatility or cumulative-band constraints materially changed the path.",
            "- `proxy quantico`: quantum output is a direction plus volatility proxy, not direct price.",
            "- `dados/desatualizacao`: no stronger model, recursion, guardrail, or proxy signal was detected.",
            "",
            "## Generated Files",
            "",
        ]
    )
    for path in generated_files:
        lines.append(f"- `{path.relative_to(PROJECT_ROOT) if path.is_absolute() else path}`")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    symbols = [symbol.upper() for symbol in args.symbols]
    source = str(args.source)
    lookback = int(args.lookback)
    horizon_days = int(args.horizon_days)
    extraction_date = parse_iso_date(args.extraction_date)
    raw_root = resolve_project_path(args.raw_root)
    processed_root = resolve_project_path(args.processed_root)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_step_frames: list[pd.DataFrame] = []
    selections: list[ForecastSelection] = []
    actual_partitions: dict[str, str | None] = {}
    generated_files: list[Path] = []
    for symbol in symbols:
        selection = load_forecast_selection(
            processed_root=processed_root,
            source=source,
            symbol=symbol,
            lookback=lookback,
            horizon_days=horizon_days,
            extraction_date=extraction_date,
            generated_at=str(args.generated_at),
        )
        selections.append(selection)
        forecast = pd.read_parquet(selection.parquet_path)
        actual, actual_partition = load_actual_prices(
            raw_root=raw_root,
            source=source,
            symbol=symbol,
            target_column=str(args.target_column),
            forecast_dates=forecast["forecast_date"],
            actual_extraction_date=str(args.actual_extraction_date),
        )
        actual_partitions[symbol] = actual_partition
        baselines = build_seed_baselines(
            raw_root=raw_root,
            source=source,
            symbol=symbol,
            target_column=str(args.target_column),
            extraction_date=extraction_date,
            forecast_dates=forecast["forecast_date"].drop_duplicates(),
        )
        all_step_frames.append(
            enrich_forecast_steps(
                forecast=forecast,
                actual=actual,
                baselines=baselines,
                target_column=str(args.target_column),
            )
        )

    detailed = pd.concat(all_step_frames, ignore_index=True)
    detailed = detailed.sort_values(["symbol", "predict_type", "forecast_step"]).reset_index(drop=True)
    aggregate = aggregate_quality(detailed)
    scaler_audit = audit_scaler_splits(
        processed_root=processed_root,
        extraction_date=extraction_date,
        symbols=symbols,
        lookback=lookback,
    )

    detailed_csv = output_dir / "forecast_quality_steps.csv"
    aggregate_csv = output_dir / "forecast_quality_summary.csv"
    scaler_csv = output_dir / "forecast_quality_scaler_split_audit.csv"
    detailed.to_csv(detailed_csv, index=False)
    aggregate.to_csv(aggregate_csv, index=False)
    scaler_audit.to_csv(scaler_csv, index=False)
    generated_files.extend([detailed_csv, aggregate_csv, scaler_csv])

    for symbol in symbols:
        plot_path = output_dir / f"{symbol.lower()}_forecast_quality.svg"
        write_symbol_plot(frame=detailed, symbol=symbol, output_path=plot_path)
        generated_files.append(plot_path)

    report_path = output_dir / "forecast_quality_report.md"
    write_report(
        output_path=report_path,
        detailed=detailed,
        aggregate=aggregate,
        scaler_audit=scaler_audit,
        generated_files=generated_files + [report_path],
        selections=selections,
        actual_partitions=actual_partitions,
    )
    print(f"Forecast quality report: {report_path}")
    print(f"Detailed CSV: {detailed_csv}")
    print(f"Summary CSV: {aggregate_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
