from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.use_cases.provision_athena_catalog import (  # noqa: E402
    AthenaProvisionRequest,
    ProvisionAthenaCatalogUseCase,
)
from src.infrastructure.config.settings import AthenaCatalogSettings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an Athena database and external tables that point to the "
            "raw, refined, feature, and forecast datasets stored in S3."
        )
    )
    parser.add_argument(
        "--database-name",
        default=None,
        help="Athena database name. Defaults to ATHENA_DATABASE or tech_challenge_phase4.",
    )
    parser.add_argument(
        "--target-column",
        default="close",
        help="Target column name used in refined/feature/forecast datasets.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Lookback used when generating the refined/feature schema.",
    )
    parser.add_argument(
        "--workgroup",
        default=None,
        help="Athena workgroup. Defaults to ATHENA_WORKGROUP or primary.",
    )
    parser.add_argument(
        "--output-s3-uri",
        default=None,
        help=(
            "S3 URI for Athena query results, for example "
            "s3://my-bucket/athena-results/. "
            "Optional when the selected workgroup already defines an output location."
        ),
    )
    parser.add_argument(
        "--replace-tables",
        action="store_true",
        help="Drop existing tables before recreating them.",
    )
    parser.add_argument(
        "--skip-repair",
        action="store_true",
        help="Skip `MSCK REPAIR TABLE` after the tables are created.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the SQL statements without executing them in Athena.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=2.0,
        help="Polling interval while waiting for Athena query completion.",
    )
    parser.add_argument(
        "--wait-timeout-seconds",
        type=float,
        default=300.0,
        help="Maximum wait time for each Athena statement.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = AthenaCatalogSettings.from_env()

    request = AthenaProvisionRequest(
        database_name=(args.database_name or settings.athena_database).lower(),
        raw_bucket=settings.s3_bucket_raw or "",
        raw_prefix=settings.s3_raw_prefix,
        refined_bucket=settings.s3_bucket_refined or "",
        refined_prefix=settings.s3_refined_prefix,
        processed_bucket=settings.s3_bucket_processed or "",
        processed_prefix=settings.s3_processed_prefix,
        target_column=args.target_column,
        lookback=args.lookback,
        repair_tables=not args.skip_repair,
        replace_tables=args.replace_tables,
        workgroup=args.workgroup or settings.athena_workgroup,
        output_s3_uri=args.output_s3_uri or settings.athena_output_s3_uri,
        execute=not args.print_only,
        poll_interval_seconds=args.poll_interval_seconds,
        wait_timeout_seconds=args.wait_timeout_seconds,
    )

    use_case = ProvisionAthenaCatalogUseCase(
        region_name=settings.aws_region,
        endpoint_url=settings.aws_endpoint_url,
    )
    result = use_case.execute(request)

    if args.print_only:
        print(f"Athena database: {result.database_name}")
        print(f"Raw table:      {result.raw_table_name}")
        print(f"Refined table:  {result.refined_table_name}")
        print(f"Feature table:  {result.feature_table_name}")
        print(f"Forecast table: {result.forecast_table_name}")
        print("")
        for execution in result.executions:
            print(f"-- {execution.name}")
            print(execution.sql)
            print("")
        return 0

    print("Athena catalog provision completed.")
    print(f"Database: {result.database_name}")
    print(f"Raw table: {result.raw_table_name}")
    print(f"Refined table: {result.refined_table_name}")
    print(f"Feature table: {result.feature_table_name}")
    print(f"Forecast table: {result.forecast_table_name}")
    for execution in result.executions:
        print(f"- {execution.name}: {execution.state} ({execution.query_execution_id})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
