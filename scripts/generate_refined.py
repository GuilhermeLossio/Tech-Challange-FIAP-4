from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.services.refined_data_pipeline_service import (  # noqa: E402
    RefinedDataPipelineService,
    RefinedDatasetRequest,
)
from src.infrastructure.config.settings import RefinedPipelineSettings  # noqa: E402
from src.infrastructure.storage.local_processed_store import LocalProcessedStore  # noqa: E402
from src.infrastructure.storage.s3_raw_store import S3RawStore  # noqa: E402


DEFAULT_SYMBOLS = ("NVDA", "AMD", "TSM", "ASML", "QCOM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate refined datasets from the local raw zone."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_SYMBOLS),
        help="Ticker symbols to refine from the raw zone.",
    )
    parser.add_argument(
        "--extraction-date",
        default=None,
        help="Raw extraction date in YYYY-MM-DD format. Defaults to the latest available raw partition.",
    )
    parser.add_argument(
        "--source",
        default="yfinance",
        help="Raw data source partition to read from.",
    )
    parser.add_argument(
        "--target-column",
        default="close",
        help="Target column to scale and convert into time windows.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Sliding window size in trading days.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--skip-s3",
        action="store_true",
        help="Only persist refined files locally and skip S3 upload.",
    )
    return parser.parse_args()


def parse_iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"Invalid date {value!r}. Use YYYY-MM-DD.") from exc


def detect_latest_extraction_date(raw_root: Path) -> date:
    manifests_root = raw_root / "manifests"
    if not manifests_root.exists():
        raise SystemExit(
            f"Could not locate raw manifests under {manifests_root}. "
            "Run `python scripts/generate_raw.py --skip-s3` first."
        )

    candidates: list[date] = []
    for partition in manifests_root.glob("extraction_date=*"):
        token = partition.name.split("=", 1)[-1]
        try:
            candidates.append(parse_iso_date(token))
        except SystemExit:
            continue

    if not candidates:
        raise SystemExit(
            f"No raw extraction_date partitions were found under {manifests_root}."
        )

    return max(candidates)


def build_s3_store(settings: RefinedPipelineSettings, upload_to_s3: bool) -> S3RawStore | None:
    if not upload_to_s3:
        return None

    if not settings.s3_enabled:
        raise SystemExit(
            "S3 upload was requested, but no refined bucket was configured. "
            "Set S3_BUCKET_REFINED or reuse S3_BUCKET_RAW."
        )

    return S3RawStore(
        bucket=settings.s3_bucket_refined or "",
        prefix=settings.s3_refined_prefix,
        region_name=settings.aws_region,
        endpoint_url=settings.aws_endpoint_url,
    )


def main() -> int:
    args = parse_args()
    settings = RefinedPipelineSettings.from_env()
    upload_to_s3 = not args.skip_s3
    extraction_date = (
        parse_iso_date(args.extraction_date)
        if args.extraction_date
        else detect_latest_extraction_date(settings.local_raw_dir)
    )

    service = RefinedDataPipelineService(
        raw_root_dir=settings.local_raw_dir,
        local_store=LocalProcessedStore(settings.local_processed_dir),
        s3_store=build_s3_store(settings, upload_to_s3),
    )

    request = RefinedDatasetRequest(
        symbols=tuple(symbol.upper() for symbol in args.symbols),
        extraction_date=extraction_date,
        source=args.source,
        target_column=args.target_column,
        lookback=args.lookback,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        upload_to_s3=upload_to_s3,
    )
    result = service.generate(request)

    print("Refined generation completed.")
    print(f"Manifest local: {result.manifest_local_path}")
    if result.manifest_s3_uri:
        print(f"Manifest S3: {result.manifest_s3_uri}")

    for asset in result.assets:
        print(f"- {asset.symbol}: {asset.row_count} rows")
        print(f"  local: {asset.local_path}")
        print(f"  splits: {asset.split_counts}")
        if asset.s3_uri:
            print(f"  s3: {asset.s3_uri}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
