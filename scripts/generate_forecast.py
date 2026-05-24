from __future__ import annotations

import argparse
import json
from datetime import date, datetime
import os
from pathlib import Path
import sys

import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.use_cases.generate_forecast_batch import (  # noqa: E402
    ForecastBatchRequest,
    GenerateForecastBatchUseCase,
)
from src.application.use_cases.provision_athena_catalog import (  # noqa: E402
    AthenaProvisionRequest,
    ProvisionAthenaCatalogUseCase,
)
from src.infrastructure.config.settings import (  # noqa: E402
    AthenaCatalogSettings,
    ForecastPipelineSettings,
)
from src.infrastructure.storage.local_processed_store import LocalProcessedStore  # noqa: E402
from src.infrastructure.storage.s3_raw_store import S3RawStore  # noqa: E402


DEFAULT_SYMBOLS = ("NVDA", "AMD", "TSM", "ASML", "QCOM")
DEFAULT_ATHENA_OUTPUT_S3_URI = "s3://quantumprojects3/athena-results/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate future predictions for the selected symbols using both the "
            "normal LSTM model and the quantum classifier, then persist them to "
            "Parquet, S3, and Athena."
        )
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_SYMBOLS),
        help="Ticker symbols to forecast from the refined/raw zones.",
    )
    parser.add_argument(
        "--extraction-date",
        default=None,
        help="Pipeline extraction date in YYYY-MM-DD format. Defaults to the latest available refined partition.",
    )
    parser.add_argument(
        "--source",
        default="yfinance",
        help="Dataset source partition to read from.",
    )
    parser.add_argument(
        "--target-column",
        default="close",
        help="Target price column used by the trained model.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Sliding window size in trading days.",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=30,
        help="Forecast horizon in future business days.",
    )
    parser.add_argument(
        "--forecast-end-date",
        default=None,
        help=(
            "Forecast through this business date in YYYY-MM-DD format. "
            "When provided, it overrides --horizon-days by deriving the number "
            "of future business days from the last observed raw date."
        ),
    )
    parser.add_argument(
        "--model-name-prefix",
        default="lstm",
        help="Prefix used when resolving Keras model files, for example `lstm` -> `models/lstm_nvda.keras`.",
    )
    parser.add_argument(
        "--quantum-model-name-prefix",
        default="quantum_vqc",
        help="Prefix used when resolving quantum model files, for example `quantum_vqc` -> `models/quantum_vqc_nvda.json`.",
    )
    parser.add_argument(
        "--skip-normal",
        action="store_true",
        help="Skip normal LSTM predictions.",
    )
    parser.add_argument(
        "--skip-quant",
        action="store_true",
        help="Skip quantum predictions.",
    )
    parser.add_argument(
        "--quantum-runtime-mode",
        choices=("local", "cloud"),
        default="local",
        help=(
            "Runtime used only for quantum forecast inference. Training never uses "
            "IBM Quantum in this workflow. Use cloud only for small, confirmed runs."
        ),
    )
    parser.add_argument(
        "--quantum-backend",
        default=None,
        help="Real backend name for cloud quantum prediction. Omit to use least_busy.",
    )
    parser.add_argument(
        "--quantum-shots",
        type=int,
        default=256,
        help="Shots for quantum forecast inference. Default: 256.",
    )
    parser.add_argument(
        "--quantum-optimization-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=0,
        help="Transpiler optimization level for quantum forecast inference.",
    )
    parser.add_argument(
        "--max-cloud-quantum-predictions",
        type=int,
        default=5,
        help=(
            "Safety limit for cloud quantum predictions. Effective requested jobs "
            "are roughly symbols * horizon_days."
        ),
    )
    parser.add_argument(
        "--confirm-ibm-runtime-cost",
        action="store_true",
        help=(
            "Required with --quantum-runtime-mode cloud. Confirms that forecast "
            "inference may submit IBM Quantum Runtime jobs."
        ),
    )
    parser.add_argument(
        "--skip-s3",
        action="store_true",
        help="Only persist forecast files locally and skip S3 upload.",
    )
    parser.add_argument(
        "--skip-athena",
        action="store_true",
        help="Skip Athena catalog provisioning after the S3 upload completes.",
    )
    parser.add_argument(
        "--athena-database",
        default=None,
        help="Athena database name. Defaults to ATHENA_DATABASE or tech_challenge_phase4.",
    )
    parser.add_argument(
        "--athena-workgroup",
        default=None,
        help="Athena workgroup. Defaults to ATHENA_WORKGROUP or primary.",
    )
    parser.add_argument(
        "--athena-output-s3-uri",
        default=None,
        help=(
            "S3 URI for Athena query results. Defaults to ATHENA_OUTPUT_S3_URI "
            f"or {DEFAULT_ATHENA_OUTPUT_S3_URI}."
        ),
    )
    parser.add_argument(
        "--replace-athena-tables",
        action="store_true",
        help="Drop and recreate Athena external tables before running the repair step.",
    )
    parser.add_argument(
        "--skip-athena-repair",
        action="store_true",
        help="Skip `MSCK REPAIR TABLE` after creating the Athena tables.",
    )
    return parser.parse_args()


def parse_iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"Invalid date {value!r}. Use YYYY-MM-DD.") from exc


def detect_latest_extraction_date(
    processed_root: Path,
    *,
    source: str,
    target_column: str,
    lookback: int,
) -> date:
    manifests_root = processed_root / "manifests"
    if not manifests_root.exists():
        raise SystemExit(
            f"Could not locate refined manifests under {manifests_root}. "
            "Run `python scripts/generate_refined.py --skip-s3` first."
        )

    candidates: list[date] = []
    for partition in manifests_root.glob("extraction_date=*"):
        manifest_path = partition / "refined_manifest.json"
        if not manifest_path.exists():
            continue

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        request_payload = payload.get("request", {})
        if (
            request_payload.get("source") != source
            or request_payload.get("target_column") != target_column
            or int(request_payload.get("lookback", -1)) != lookback
        ):
            continue

        token = partition.name.split("=", 1)[-1]
        try:
            candidates.append(parse_iso_date(token))
        except SystemExit:
            continue

    if not candidates:
        raise SystemExit(
            "No compatible refined extraction_date partitions were found under "
            f"{manifests_root} for source={source!r}, target_column={target_column!r}, "
            f"lookback={lookback}."
        )

    return max(candidates)


def build_processed_s3_store(
    settings: ForecastPipelineSettings,
    upload_to_s3: bool,
) -> S3RawStore | None:
    if not upload_to_s3:
        return None

    if not settings.processed_s3_enabled:
        raise SystemExit(
            "S3 upload was requested, but no processed bucket was configured. "
            "Set S3_BUCKET_PROCESSED or reuse S3_BUCKET_REFINED/S3_BUCKET_RAW."
        )

    return S3RawStore(
        bucket=settings.s3_bucket_processed or "",
        prefix=settings.s3_processed_prefix,
        region_name=settings.aws_region,
        endpoint_url=settings.aws_endpoint_url,
    )


def detect_last_observed_date(
    *,
    raw_root_dir: Path,
    source: str,
    symbol: str,
    extraction_date: date,
) -> date:
    raw_path = (
        raw_root_dir
        / "market_data"
        / f"source={source}"
        / f"symbol={symbol.upper()}"
        / f"extraction_date={extraction_date.isoformat()}"
        / "ohlcv.csv"
    )
    if not raw_path.exists():
        raise SystemExit(f"Raw input not found for symbol {symbol!r}: {raw_path}")

    frame = pd.read_csv(raw_path, usecols=["date"], parse_dates=["date"])
    if frame.empty:
        raise SystemExit(f"Raw input for symbol {symbol!r} is empty: {raw_path}")
    return pd.Timestamp(frame["date"].max()).date()


def resolve_horizon_days(
    *,
    requested_horizon_days: int,
    forecast_end_date: date | None,
    raw_root_dir: Path,
    source: str,
    symbols: tuple[str, ...],
    extraction_date: date,
) -> int:
    if forecast_end_date is None:
        return requested_horizon_days

    last_observed_by_symbol = {
        symbol: detect_last_observed_date(
            raw_root_dir=raw_root_dir,
            source=source,
            symbol=symbol,
            extraction_date=extraction_date,
        )
        for symbol in symbols
    }
    unique_last_observed_dates = set(last_observed_by_symbol.values())
    if len(unique_last_observed_dates) != 1:
        details = ", ".join(
            f"{symbol}={last_date.isoformat()}"
            for symbol, last_date in sorted(last_observed_by_symbol.items())
        )
        raise SystemExit(
            "Cannot derive one shared --horizon-days because symbols have different "
            f"last observed dates: {details}."
        )

    last_observed_date = next(iter(unique_last_observed_dates))
    first_forecast_date = pd.Timestamp(last_observed_date) + pd.offsets.BDay(1)
    forecast_dates = pd.bdate_range(
        first_forecast_date,
        pd.Timestamp(forecast_end_date),
    )
    if len(forecast_dates) == 0:
        raise SystemExit(
            "--forecast-end-date must be after the last observed business date. "
            f"last_observed_date={last_observed_date.isoformat()} "
            f"forecast_end_date={forecast_end_date.isoformat()}"
        )

    return int(len(forecast_dates))


def main() -> int:
    args = parse_args()
    settings = ForecastPipelineSettings.from_env()
    athena_settings = AthenaCatalogSettings.from_env()
    upload_to_s3 = not args.skip_s3
    extraction_date = (
        parse_iso_date(args.extraction_date)
        if args.extraction_date
        else detect_latest_extraction_date(
            settings.local_processed_dir,
            source=args.source,
            target_column=args.target_column,
            lookback=args.lookback,
        )
    )
    symbols = tuple(symbol.upper() for symbol in args.symbols)
    horizon_days = resolve_horizon_days(
        requested_horizon_days=args.horizon_days,
        forecast_end_date=(
            parse_iso_date(args.forecast_end_date)
            if args.forecast_end_date
            else None
        ),
        raw_root_dir=settings.local_raw_dir,
        source=args.source,
        symbols=symbols,
        extraction_date=extraction_date,
    )

    use_case = GenerateForecastBatchUseCase(
        raw_root_dir=settings.local_raw_dir,
        processed_root_dir=settings.local_processed_dir,
        models_root_dir=settings.local_models_dir,
        local_store=LocalProcessedStore(settings.local_processed_dir),
        s3_store=build_processed_s3_store(settings, upload_to_s3),
    )

    request = ForecastBatchRequest(
        symbols=symbols,
        extraction_date=extraction_date,
        source=args.source,
        target_column=args.target_column,
        lookback=args.lookback,
        horizon_days=horizon_days,
        model_name_prefix=args.model_name_prefix,
        quantum_model_name_prefix=args.quantum_model_name_prefix,
        include_normal=not args.skip_normal,
        include_quantum=not args.skip_quant,
        upload_to_s3=upload_to_s3,
        quantum_runtime_mode=args.quantum_runtime_mode,
        quantum_backend_name=args.quantum_backend,
        quantum_shots=args.quantum_shots,
        quantum_optimization_level=args.quantum_optimization_level,
        confirm_ibm_runtime_cost=args.confirm_ibm_runtime_cost,
        max_cloud_quantum_predictions=args.max_cloud_quantum_predictions,
    )

    if args.forecast_end_date:
        print(
            "Forecast end date requested: "
            f"{args.forecast_end_date}; derived horizon_days={horizon_days}."
        )

    try:
        result = use_case.generate(request)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc

    print("Forecast generation completed.")
    print(f"Manifest local: {result.manifest_local_path}")
    print(f"Unified report local: {result.unified_report_local_path}")
    print(f"Unified latest report local: {result.unified_latest_report_local_path}")
    if result.manifest_s3_uri:
        print(f"Manifest S3: {result.manifest_s3_uri}")
    if result.unified_report_s3_uri:
        print(f"Unified report S3: {result.unified_report_s3_uri}")

    for asset in result.assets:
        print(
            f"- {asset.symbol}: "
            f"{asset.forecast_start_date} -> {asset.forecast_end_date} "
            f"({asset.row_count} rows; {', '.join(asset.predict_types)})"
        )
        print(f"  last observed: {asset.last_observed_date} close={asset.last_observed_close:.4f}")
        print(f"  local: {asset.local_path}")
        print(f"  report: {asset.report_local_path}")
        print(f"  chart: {asset.chart_local_path}")
        if asset.normal_model_local_path:
            print(f"  normal model: {asset.normal_model_local_path}")
        if asset.quantum_model_local_path:
            print(f"  quantum model: {asset.quantum_model_local_path}")
        if asset.s3_uri:
            print(f"  s3: {asset.s3_uri}")
        if asset.report_s3_uri:
            print(f"  report s3: {asset.report_s3_uri}")
        if asset.chart_s3_uri:
            print(f"  chart s3: {asset.chart_s3_uri}")

    if upload_to_s3 and not args.skip_athena:
        provision_request = AthenaProvisionRequest(
            database_name=(args.athena_database or athena_settings.athena_database).lower(),
            raw_bucket=athena_settings.s3_bucket_raw or "",
            raw_prefix=athena_settings.s3_raw_prefix,
            refined_bucket=athena_settings.s3_bucket_refined or "",
            refined_prefix=athena_settings.s3_refined_prefix,
            processed_bucket=(
                athena_settings.s3_bucket_processed or settings.s3_bucket_processed or ""
            ),
            processed_prefix=athena_settings.s3_processed_prefix,
            target_column=args.target_column,
            lookback=args.lookback,
            repair_tables=not args.skip_athena_repair,
            replace_tables=args.replace_athena_tables,
            workgroup=args.athena_workgroup or athena_settings.athena_workgroup,
            output_s3_uri=(
                args.athena_output_s3_uri
                or athena_settings.athena_output_s3_uri
                or DEFAULT_ATHENA_OUTPUT_S3_URI
            ),
            execute=True,
        )
        try:
            athena_result = ProvisionAthenaCatalogUseCase(
                region_name=athena_settings.aws_region,
                endpoint_url=athena_settings.aws_endpoint_url,
            ).execute(provision_request)
        except (RuntimeError, ValueError, TimeoutError) as exc:
            raise SystemExit(f"Athena provisioning failed: {exc}") from exc

        print("Athena catalog provision completed.")
        print(
            f"Table ready: {athena_result.database_name}."
            f"{athena_result.future_predict_table_name}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
