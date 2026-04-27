from __future__ import annotations

import argparse
from datetime import date, datetime
import os
from pathlib import Path
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.use_cases.generate_forecast_batch import (  # noqa: E402
    ForecastBatchRequest,
    GenerateForecastBatchUseCase,
)
from src.infrastructure.config.settings import ForecastPipelineSettings  # noqa: E402
from src.infrastructure.storage.local_processed_store import LocalProcessedStore  # noqa: E402
from src.infrastructure.storage.s3_raw_store import S3RawStore  # noqa: E402


DEFAULT_SYMBOLS = ("NVDA", "AMD", "TSM", "ASML", "QCOM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate recursive monthly forecasts from the latest market window and "
            "persist them as flat parquet files that Athena can query directly."
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
        "--model-name-prefix",
        default="lstm",
        help="Prefix used when resolving model files, for example `lstm` -> `models/lstm_nvda.keras`.",
    )
    parser.add_argument(
        "--skip-s3",
        action="store_true",
        help="Only persist forecast files locally and skip S3 upload.",
    )
    return parser.parse_args()


def parse_iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"Invalid date {value!r}. Use YYYY-MM-DD.") from exc


def detect_latest_extraction_date(processed_root: Path) -> date:
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

        token = partition.name.split("=", 1)[-1]
        try:
            candidates.append(parse_iso_date(token))
        except SystemExit:
            continue

    if not candidates:
        raise SystemExit(
            f"No refined extraction_date partitions were found under {manifests_root}."
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


def main() -> int:
    args = parse_args()
    settings = ForecastPipelineSettings.from_env()
    upload_to_s3 = not args.skip_s3
    extraction_date = (
        parse_iso_date(args.extraction_date)
        if args.extraction_date
        else detect_latest_extraction_date(settings.local_processed_dir)
    )

    use_case = GenerateForecastBatchUseCase(
        raw_root_dir=settings.local_raw_dir,
        processed_root_dir=settings.local_processed_dir,
        models_root_dir=settings.local_models_dir,
        local_store=LocalProcessedStore(settings.local_processed_dir),
        s3_store=build_processed_s3_store(settings, upload_to_s3),
    )

    request = ForecastBatchRequest(
        symbols=tuple(symbol.upper() for symbol in args.symbols),
        extraction_date=extraction_date,
        source=args.source,
        target_column=args.target_column,
        lookback=args.lookback,
        horizon_days=args.horizon_days,
        model_name_prefix=args.model_name_prefix,
        upload_to_s3=upload_to_s3,
    )

    try:
        result = use_case.generate(request)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    print("Forecast generation completed.")
    print(f"Manifest local: {result.manifest_local_path}")
    if result.manifest_s3_uri:
        print(f"Manifest S3: {result.manifest_s3_uri}")

    for asset in result.assets:
        print(
            f"- {asset.symbol}: "
            f"{asset.forecast_start_date} -> {asset.forecast_end_date} "
            f"({asset.row_count} rows)"
        )
        print(f"  last observed: {asset.last_observed_date} close={asset.last_observed_close:.4f}")
        print(f"  model: {asset.model_local_path}")
        print(f"  local: {asset.local_path}")
        if asset.training_manifest_local_path:
            print(f"  training manifest: {asset.training_manifest_local_path}")
        if asset.s3_uri:
            print(f"  s3: {asset.s3_uri}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
