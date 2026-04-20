from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.services.raw_data_pipeline_service import (  # noqa: E402
    RawDataPipelineService,
    RawIngestionRequest,
)
from src.infrastructure.config.settings import RawPipelineSettings  # noqa: E402
from src.infrastructure.repositories.yfinance_repository import YFinanceRepository  # noqa: E402
from src.infrastructure.storage.local_raw_store import LocalRawStore  # noqa: E402
from src.infrastructure.storage.s3_raw_store import S3RawStore  # noqa: E402


DEFAULT_SYMBOLS = ("NVDA", "AMD", "TSM", "ASML", "QCOM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate raw OHLCV market data and optionally upload it to S3."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_SYMBOLS),
        help="Ticker symbols to ingest into the raw zone.",
    )
    parser.add_argument(
        "--start-date",
        default="2018-01-01",
        help="Inclusive start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        default="2025-12-31",
        help="Inclusive end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--extraction-date",
        default=date.today().isoformat(),
        help="Partition date used in the raw zone layout.",
    )
    parser.add_argument(
        "--skip-s3",
        action="store_true",
        help="Only persist raw files locally and skip S3 upload.",
    )
    return parser.parse_args()


def parse_iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"Invalid date {value!r}. Use YYYY-MM-DD.") from exc


def build_s3_store(settings: RawPipelineSettings, upload_to_s3: bool) -> S3RawStore | None:
    if not upload_to_s3:
        return None

    if not settings.s3_enabled:
        raise SystemExit(
            "S3 upload was requested, but S3_BUCKET_RAW is not configured in the environment."
        )

    return S3RawStore(
        bucket=settings.s3_bucket_raw or "",
        prefix=settings.s3_raw_prefix,
        region_name=settings.aws_region,
        endpoint_url=settings.aws_endpoint_url,
    )


def main() -> int:
    args = parse_args()
    settings = RawPipelineSettings.from_env()
    upload_to_s3 = not args.skip_s3

    service = RawDataPipelineService(
        stock_repository=YFinanceRepository(),
        local_store=LocalRawStore(settings.local_raw_dir),
        s3_store=build_s3_store(settings, upload_to_s3),
    )

    request = RawIngestionRequest(
        symbols=tuple(symbol.upper() for symbol in args.symbols),
        start_date=parse_iso_date(args.start_date),
        end_date=parse_iso_date(args.end_date),
        extraction_date=parse_iso_date(args.extraction_date),
        upload_to_s3=upload_to_s3,
    )

    result = service.generate(request)

    print("Raw generation completed.")
    print(f"Manifest local: {result.manifest_local_path}")
    if result.manifest_s3_uri:
        print(f"Manifest S3: {result.manifest_s3_uri}")

    for asset in result.assets:
        print(f"- {asset.symbol}: {asset.row_count} rows")
        print(f"  local: {asset.local_path}")
        if asset.s3_uri:
            print(f"  s3: {asset.s3_uri}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
