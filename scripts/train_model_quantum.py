from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
import sys

# Simulador local — usa SIMULATOR_DEFAULTS automaticamente
#python train_quantum.py --mode local

# Hardware real — usa QUANTUM_HARDWARE_DEFAULTS automaticamente
#python train_quantum.py --mode cloud --backend ibm_sherbrooke

# Override pontual ainda funciona normalmente
#python train_quantum.py --mode cloud --shots 512 --optimizer-maxiter 20

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.use_cases.train_model_quantum import (  # noqa: E402
    QuantumTrainingInterruptedError,
    QuantumTrainingRequest,
    TrainQuantumModelUseCase,
)
from src.infrastructure.config.settings import TrainingPipelineSettings  # noqa: E402
from src.infrastructure.storage.local_model_store import LocalModelStore  # noqa: E402
from src.infrastructure.storage.s3_raw_store import S3RawStore  # noqa: E402


DEFAULT_SYMBOLS = ("NVDA",)

# ---------------------------------------------------------------------------
# Mode-specific defaults
# ---------------------------------------------------------------------------
# These are applied automatically based on --mode.
# Any value explicitly passed via CLI overrides the mode default.

SIMULATOR_DEFAULTS: dict[str, int | str] = {
    "shots": 1024,
    "optimizer": "cobyla",
    "optimizer_maxiter": 50,
    "max_train_samples": 128,
    "max_validation_samples": 64,
    "max_test_samples": 64,
    "optimization_level": 1,
}

# On real hardware every shot costs real queue + execution time.
# SPSA estimates gradients with only 2 circuit executions per iteration
# regardless of parameter count — it was designed for noisy quantum hardware.
QUANTUM_HARDWARE_DEFAULTS: dict[str, int | str] = {
    "shots": 256,
    "optimizer": "spsa",
    "optimizer_maxiter": 15,
    "max_train_samples": 24,
    "max_validation_samples": 16,
    "max_test_samples": 16,
    "optimization_level": 0,
}

MODE_DEFAULTS: dict[str, dict[str, int | str]] = {
    "local": SIMULATOR_DEFAULTS,
    "cloud": QUANTUM_HARDWARE_DEFAULTS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a hybrid quantum classifier on refined market windows using "
            "IBM Quantum Runtime locally (simulator) or on a real backend.\n\n"
            "Sensible defaults are applied automatically per --mode so that "
            "cloud runs do not waste quantum time. Any flag can still be "
            "overridden explicitly."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- data / model identity ---
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--extraction-date", default=None)
    parser.add_argument("--source", default="yfinance")
    parser.add_argument("--target-column", default="close")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--model-name-prefix", default="quantum_vqc")
    parser.add_argument("--seed", type=int, default=42)

    # --- execution mode ---
    parser.add_argument(
        "--mode",
        choices=("local", "cloud"),
        default="local",
        help=(
            "local  → simulator (high shots, many samples, COBYLA). "
            "cloud  → real backend (low shots, few samples, SPSA). "
            "Defaults are set automatically; override with the flags below."
        ),
    )
    parser.add_argument("--backend", dest="backend_name", default=None)
    parser.add_argument("--list-backends", action="store_true")

    # --- circuit architecture ---
    parser.add_argument("--num-qubits", type=int, default=2)
    parser.add_argument("--feature-map-reps", type=int, default=1)
    parser.add_argument("--ansatz-reps", type=int, default=1)

    # --- runtime knobs (mode defaults applied in main) ---
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="Shots per circuit execution. Default: 1024 (local) / 256 (cloud).",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=None,
        help="Transpiler optimisation level. Default: 1 (local) / 0 (cloud).",
    )
    parser.add_argument(
        "--optimizer",
        choices=("cobyla", "spsa"),
        default=None,
        help="Classical optimiser. Default: cobyla (local) / spsa (cloud).",
    )
    parser.add_argument(
        "--optimizer-maxiter",
        type=int,
        default=None,
        help="Max optimiser iterations. Default: 50 (local) / 15 (cloud).",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Training samples. Default: 128 (local) / 24 (cloud).",
    )
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=None,
        help="Validation samples. Default: 64 (local) / 16 (cloud).",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Test samples. Default: 64 (local) / 16 (cloud).",
    )

    # --- storage ---
    parser.add_argument(
        "--skip-s3",
        action="store_true",
        help="Only persist artifacts locally; skip S3 upload.",
    )

    return parser.parse_args()


def apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in None values with the appropriate mode defaults."""
    defaults = MODE_DEFAULTS[args.mode]
    for key, value in defaults.items():
        attr = key.replace("-", "_")
        if getattr(args, attr, None) is None:
            setattr(args, attr, value)
    return args


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


def print_mode_summary(args: argparse.Namespace) -> None:
    """Print a summary of the effective configuration so the user can confirm."""
    print(f"Mode          : {args.mode.upper()}")
    print(f"Symbols       : {args.symbols}")
    print(f"Optimizer     : {args.optimizer}  (maxiter={args.optimizer_maxiter})")
    print(f"Shots         : {args.shots}")
    print(
        f"Samples       : train={args.max_train_samples}  "
        f"val={args.max_validation_samples}  "
        f"test={args.max_test_samples}"
    )
    print(f"Opt. level    : {args.optimization_level}")
    print(f"Qubits        : {args.num_qubits}  "
          f"(feature_map_reps={args.feature_map_reps}, ansatz_reps={args.ansatz_reps})")
    print("-" * 48)


def main() -> int:
    args = parse_args()
    args = apply_mode_defaults(args)

    settings = TrainingPipelineSettings.from_env()
    upload_to_s3 = not args.skip_s3

    extraction_date = (
        parse_iso_date(args.extraction_date)
        if args.extraction_date
        else detect_latest_extraction_date(settings.local_processed_dir)
    )

    if args.list_backends:
        if args.mode != "cloud":
            raise SystemExit("--list-backends is only meaningful with --mode cloud.")
        use_case = TrainQuantumModelUseCase(
            processed_root_dir=settings.local_processed_dir,
            local_store=LocalModelStore(settings.local_models_dir),
            s3_store=None,
        )
        rows = use_case.list_cloud_backends()
        if not rows:
            print("No real backend was found for the current account.")
            return 0

        print("Accessible real backends:")
        for row in rows:
            print(
                f"- {row['name']}: qubits={row['num_qubits']}, "
                f"operational={row['operational']}"
            )
        return 0

    print_mode_summary(args)

    use_case = TrainQuantumModelUseCase(
        processed_root_dir=settings.local_processed_dir,
        local_store=LocalModelStore(settings.local_models_dir),
        s3_store=build_model_s3_store(settings, upload_to_s3),
    )

    request = QuantumTrainingRequest(
        symbols=tuple(symbol.upper() for symbol in args.symbols),
        extraction_date=extraction_date,
        source=args.source,
        target_column=args.target_column,
        lookback=args.lookback,
        execution_mode=args.mode,
        backend_name=args.backend_name,
        num_qubits=args.num_qubits,
        feature_map_reps=args.feature_map_reps,
        ansatz_reps=args.ansatz_reps,
        shots=args.shots,
        optimization_level=args.optimization_level,
        optimizer_name=args.optimizer,
        optimizer_maxiter=args.optimizer_maxiter,
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
        model_name_prefix=args.model_name_prefix,
    )

    try:
        result = use_case.train(request)
    except QuantumTrainingInterruptedError as exc:
        raise SystemExit(str(exc)) from exc
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    print("Quantum training completed.")
    print(f"Manifest local: {result.manifest_local_path}")
    if result.manifest_s3_uri:
        print(f"Manifest S3:    {result.manifest_s3_uri}")

    for asset in result.assets:
        print(f"\n[{asset.symbol}]")
        print(f"  backend        : {asset.backend_name}  mode={asset.execution_mode}")
        print(
            f"  val_acc        : {asset.validation_metrics.accuracy!r}\n"
            f"  test_acc       : {asset.test_metrics.accuracy!r}\n"
            f"  test_f1        : {asset.test_metrics.f1!r}"
        )
        print(f"  model bundle   : {asset.model_local_path}")
        print(f"  preprocessor   : {asset.preprocessor_local_path}")
        print(f"  training detail: {asset.training_details_local_path}")
        if asset.model_s3_uri:
            print(f"  model S3       : {asset.model_s3_uri}")
        if asset.preprocessor_s3_uri:
            print(f"  preprocessor S3: {asset.preprocessor_s3_uri}")
        if asset.training_details_s3_uri:
            print(f"  training det S3: {asset.training_details_s3_uri}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())