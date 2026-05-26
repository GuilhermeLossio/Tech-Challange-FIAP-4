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

from src.application.use_cases.train_model import (  # noqa: E402
    KerasTrainingRequest,
    KerasTrainingService,
    TrainingInterruptedError,
)
from src.infrastructure.config.settings import TrainingPipelineSettings  # noqa: E402
from src.infrastructure.storage.local_model_store import LocalModelStore  # noqa: E402
from src.infrastructure.storage.s3_raw_store import S3RawStore  # noqa: E402


DEFAULT_SYMBOLS = ("NVDA",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline LSTM models from the refined parquet datasets using Keras."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_SYMBOLS),
        help="Ticker symbols to train from the refined zone.",
    )
    parser.add_argument(
        "--extraction-date",
        default=None,
        help="Refined extraction date in YYYY-MM-DD format. Defaults to the latest available refined partition.",
    )
    parser.add_argument(
        "--source",
        default="yfinance",
        help="Refined dataset source partition to read from.",
    )
    parser.add_argument(
        "--target-column",
        default="close",
        help="Target column used when the refined dataset was generated.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Sliding window size in trading days.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size used during training.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for NumPy and TensorFlow.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="Keras fit verbosity.",
    )
    parser.add_argument(
        "--model-name-prefix",
        default="lstm",
        help="Prefix used when publishing models, for example `lstm` -> `models/lstm_nvda.keras`.",
    )
    parser.add_argument(
        "--prediction-target-mode",
        choices=("price", "return"),
        default="price",
        help=(
            "Train the LSTM to predict the next price (`price`) or the next "
            "percent return (`return`). Use a distinct --model-name-prefix, "
            "for example lstm_return, when training return models."
        ),
    )
    parser.add_argument(
        "--feature-input-mode",
        choices=("sequence_price", "technical_returns"),
        default="sequence_price",
        help=(
            "Model input shape. Use technical_returns to train a two-input model "
            "from return sequences plus engineered feature_* columns."
        ),
    )
    parser.add_argument(
        "--cross-validation-folds",
        type=int,
        default=0,
        help=(
            "Run expanding-window temporal cross-validation over train+validation "
            "before writing the manifest. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--cross-validation-min-train-size",
        type=int,
        default=0,
        help=(
            "Minimum number of train+validation rows used by the first CV fold. "
            "Defaults to an automatic expanding-window size."
        ),
    )
    parser.add_argument(
        "--skip-s3",
        action="store_true",
        help="Only persist training artifacts locally and skip S3 upload.",
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


def build_model_s3_store(
    settings: TrainingPipelineSettings,
    upload_to_s3: bool,
) -> S3RawStore | None:
    if not upload_to_s3:
        return None

    if not settings.model_s3_enabled:
        raise SystemExit(
            "S3 upload was requested, but no model bucket was configured. "
            "Set S3_BUCKET_MODEL or reuse S3_BUCKET_PROCESSED/S3_BUCKET_REFINED/S3_BUCKET_RAW."
        )

    return S3RawStore(
        bucket=settings.s3_bucket_model or "",
        prefix=settings.s3_model_prefix,
        region_name=settings.aws_region,
        endpoint_url=settings.aws_endpoint_url,
    )


def main() -> int:
    args = parse_args()
    settings = TrainingPipelineSettings.from_env()
    upload_to_s3 = not args.skip_s3
    extraction_date = (
        parse_iso_date(args.extraction_date)
        if args.extraction_date
        else detect_latest_extraction_date(settings.local_processed_dir)
    )

    service = KerasTrainingService(
        processed_root_dir=settings.local_processed_dir,
        local_store=LocalModelStore(settings.local_models_dir),
        s3_store=build_model_s3_store(settings, upload_to_s3),
    )

    request = KerasTrainingRequest(
        symbols=tuple(symbol.upper() for symbol in args.symbols),
        extraction_date=extraction_date,
        source=args.source,
        target_column=args.target_column,
        lookback=args.lookback,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        learning_rate=args.learning_rate,
        seed=args.seed,
        verbose=args.verbose,
        model_name_prefix=args.model_name_prefix,
        prediction_target_mode=args.prediction_target_mode,
        feature_input_mode=args.feature_input_mode,
        cross_validation_folds=args.cross_validation_folds,
        cross_validation_min_train_size=args.cross_validation_min_train_size,
    )
    try:
        result = service.train(request)
    except TrainingInterruptedError as exc:
        raise SystemExit(str(exc)) from exc
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc

    print("Keras training completed.")
    print(f"Manifest local: {result.manifest_local_path}")
    if result.manifest_s3_uri:
        print(f"Manifest S3: {result.manifest_s3_uri}")

    for asset in result.assets:
        print(f"- {asset.symbol}: model saved to {asset.model_local_path}")
        print(
            "  metrics: "
            f"val_mae={asset.validation_metrics.mae!r}, "
            f"test_mae={asset.test_metrics.mae!r}, "
            f"test_rmse={asset.test_metrics.rmse!r}"
        )
        print(f"  history: {asset.history_local_path}")
        print(f"  report: {asset.report_local_path}")
        if asset.cross_validation_local_path:
            print(f"  cross validation: {asset.cross_validation_local_path}")
        print(f"  loss chart: {asset.loss_chart_local_path}")
        print(f"  metrics chart: {asset.metrics_chart_local_path}")
        if asset.model_s3_uri:
            print(f"  model S3: {asset.model_s3_uri}")
        if asset.history_s3_uri:
            print(f"  history S3: {asset.history_s3_uri}")
        if asset.report_s3_uri:
            print(f"  report S3: {asset.report_s3_uri}")
        if asset.loss_chart_s3_uri:
            print(f"  loss chart S3: {asset.loss_chart_s3_uri}")
        if asset.metrics_chart_s3_uri:
            print(f"  metrics chart S3: {asset.metrics_chart_s3_uri}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
