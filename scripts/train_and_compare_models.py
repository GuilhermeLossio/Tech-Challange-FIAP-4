from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import os
import sys
import time
from typing import Any


def _argv_requests_unconfirmed_cloud_run(argv: list[str]) -> bool:
    """Fast preflight before heavy imports to avoid accidental cloud runs."""
    mode_is_cloud = "--quantum-mode=cloud" in argv
    for index, token in enumerate(argv[:-1]):
        if token == "--quantum-mode" and argv[index + 1] == "cloud":
            mode_is_cloud = True
            break

    return mode_is_cloud and "--confirm-ibm-runtime-cost" not in argv


if _argv_requests_unconfirmed_cloud_run(sys.argv[1:]):
    raise SystemExit(
        "Refusing to run with --quantum-mode cloud without explicit confirmation.\n"
        "Cloud mode may submit jobs to IBM Quantum and consume runtime minutes.\n"
        "Re-run with --confirm-ibm-runtime-cost after reviewing the run budget."
    )

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "TensorFlow is required for the comparison script because the Keras model "
        "must be reloaded for directional evaluation.\n"
        f"Technical details: {exc}"
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.use_cases._dataset_loading import load_preferred_training_frame  # noqa: E402
from src.application.use_cases.train_model import (  # noqa: E402
    KerasTrainingRequest,
    KerasTrainingService,
    TrainingInterruptedError,
)
from src.application.use_cases.train_model_quantum import (  # noqa: E402
    QuantumTrainingInterruptedError,
    QuantumTrainingRequest,
    TrainQuantumModelUseCase,
)
from src.infrastructure.config.settings import TrainingPipelineSettings  # noqa: E402
from src.infrastructure.storage.local_model_store import LocalModelStore  # noqa: E402
from src.infrastructure.storage.s3_raw_store import S3RawStore  # noqa: E402


DEFAULT_SYMBOLS = ("NVDA", "AMD", "TSM", "ASML", "QCOM")

# ---------------------------------------------------------------------------
# Mode-specific defaults for the quantum portion of the comparison
# ---------------------------------------------------------------------------
# Applied automatically based on --quantum-mode.
# Any value explicitly passed via CLI overrides the mode default.

SIMULATOR_DEFAULTS: dict[str, int | str] = {
    "quantum_shots": 1024,
    "quantum_optimizer": "cobyla",
    "quantum_optimizer_maxiter": 50,
    "quantum_max_train_samples": 128,
    "quantum_max_validation_samples": 64,
    "quantum_max_test_samples": 64,
    "quantum_optimization_level": 1,
}

# On real hardware every shot costs real queue + execution time.
# SPSA estimates gradients with only 2 circuit executions per iteration
# regardless of parameter count — it was designed for noisy quantum hardware.
QUANTUM_HARDWARE_DEFAULTS: dict[str, int | str] = {
    "quantum_shots": 256,
    "quantum_optimizer": "spsa",
    "quantum_optimizer_maxiter": 15,
    "quantum_max_train_samples": 24,
    "quantum_max_validation_samples": 16,
    "quantum_max_test_samples": 16,
    "quantum_optimization_level": 0,
}

MODE_DEFAULTS: dict[str, dict[str, int | str]] = {
    "local": SIMULATOR_DEFAULTS,
    "cloud": QUANTUM_HARDWARE_DEFAULTS,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DirectionComparisonMetrics:
    sample_count: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: dict[str, int]


@dataclass(frozen=True)
class ComparisonAssetArtifact:
    symbol: str
    keras_training_seconds: float
    quantum_training_seconds: float
    keras_model_local_path: str
    quantum_model_local_path: str
    dashboard_local_path: str
    confusion_matrix_local_path: str
    report_local_path: str
    keras_price_metrics: dict[str, float | None]
    keras_direction_metrics: DirectionComparisonMetrics
    quantum_direction_metrics: DirectionComparisonMetrics
    quantum_execution_mode: str
    quantum_backend_name: str
    quantum_shots: int
    quantum_optimizer_name: str
    quantum_optimizer_maxiter: int
    quantum_function_evaluations: int | None
    quantum_objective_value: float | None
    quantum_num_qubits: int
    quantum_train_samples: int
    quantum_validation_samples: int
    quantum_test_samples: int


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the classical Keras LSTM and the hybrid quantum model, then "
            "compare execution time, prediction behavior, and test metrics.\n\n"
            "Quantum defaults are applied automatically per --quantum-mode so that "
            "cloud runs do not waste quantum time. Any flag can still be overridden."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- data / model identity ---
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_SYMBOLS),
        help="Ticker symbols to train and compare.",
    )
    parser.add_argument(
        "--extraction-date",
        default=None,
        help="Refined extraction date in YYYY-MM-DD format. Defaults to the latest available.",
    )
    parser.add_argument("--source", default="yfinance")
    parser.add_argument("--target-column", default="close")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)

    # --- Keras (classical) knobs ---
    parser.add_argument("--keras-epochs", type=int, default=30)
    parser.add_argument("--keras-batch-size", type=int, default=32)
    parser.add_argument("--keras-patience", type=int, default=5)
    parser.add_argument("--keras-learning-rate", type=float, default=0.001)
    parser.add_argument("--keras-verbose", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--keras-model-name-prefix", default="lstm")

    # --- Quantum execution mode ---
    parser.add_argument(
        "--quantum-mode",
        choices=("local", "cloud"),
        default="local",
        help=(
            "local  → simulator (high shots, many samples, COBYLA). "
            "cloud  → real backend (low shots, few samples, SPSA). "
            "Defaults are set automatically; override with the flags below."
        ),
    )
    parser.add_argument(
        "--quantum-backend",
        default=None,
        help="Backend name for cloud mode. Omit to use least_busy.",
    )
    parser.add_argument(
        "--confirm-ibm-runtime-cost",
        action="store_true",
        help=(
            "Required with --quantum-mode cloud. Confirms that this run may "
            "submit jobs to IBM Quantum and consume runtime minutes."
        ),
    )
    parser.add_argument("--quantum-model-name-prefix", default="quantum_vqc")

    # --- Quantum circuit architecture ---
    parser.add_argument("--quantum-num-qubits", type=int, default=2)
    parser.add_argument("--quantum-feature-map-reps", type=int, default=1)
    parser.add_argument("--quantum-ansatz-reps", type=int, default=1)

    # --- Quantum runtime knobs (mode defaults applied in main) ---
    parser.add_argument(
        "--quantum-shots",
        type=int,
        default=None,
        help="Shots per circuit execution. Default: 1024 (local) / 256 (cloud).",
    )
    parser.add_argument(
        "--quantum-optimization-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=None,
        help="Transpiler optimisation level. Default: 1 (local) / 0 (cloud).",
    )
    parser.add_argument(
        "--quantum-optimizer",
        choices=("cobyla", "spsa"),
        default=None,
        help="Classical optimiser. Default: cobyla (local) / spsa (cloud).",
    )
    parser.add_argument(
        "--quantum-optimizer-maxiter",
        type=int,
        default=None,
        help="Max optimiser iterations. Default: 50 (local) / 15 (cloud).",
    )
    parser.add_argument(
        "--quantum-max-train-samples",
        type=int,
        default=None,
        help="Training samples. Default: 128 (local) / 24 (cloud).",
    )
    parser.add_argument(
        "--quantum-max-validation-samples",
        type=int,
        default=None,
        help="Validation samples. Default: 64 (local) / 16 (cloud).",
    )
    parser.add_argument(
        "--quantum-max-test-samples",
        type=int,
        default=None,
        help="Test samples. Default: 64 (local) / 16 (cloud).",
    )

    # --- output ---
    parser.add_argument("--sample-plot-points", type=int, default=30)
    parser.add_argument(
        "--skip-s3",
        action="store_true",
        help="Only persist trained model artifacts locally; skip S3 upload.",
    )

    return parser.parse_args()


def apply_quantum_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in None quantum values with the appropriate mode defaults."""
    defaults = MODE_DEFAULTS[args.quantum_mode]
    for key, value in defaults.items():
        attr = key.replace("-", "_")
        if getattr(args, attr, None) is None:
            setattr(args, attr, value)
    return args


def require_ibm_runtime_confirmation(args: argparse.Namespace) -> None:
    """Prevent accidental IBM Quantum hardware submissions."""
    if args.quantum_mode != "cloud" or args.confirm_ibm_runtime_cost:
        return

    raise SystemExit(
        "Refusing to run with --quantum-mode cloud without explicit confirmation.\n"
        "Cloud mode may submit jobs to IBM Quantum and consume runtime minutes.\n"
        "Re-run with --confirm-ibm-runtime-cost after reviewing the run budget."
    )


def print_run_summary(args: argparse.Namespace) -> None:
    """Print effective configuration before any training starts."""
    print("=" * 56)
    print("COMPARISON RUN CONFIGURATION")
    print("=" * 56)
    print(f"Symbols          : {args.symbols}")
    print(f"Lookback         : {args.lookback}")
    print()
    print("[Keras LSTM]")
    print(f"  epochs         : {args.keras_epochs}")
    print(f"  batch size     : {args.keras_batch_size}")
    print(f"  patience       : {args.keras_patience}")
    print(f"  learning rate  : {args.keras_learning_rate}")
    print()
    print(f"[Quantum VQC — mode={args.quantum_mode.upper()}]")
    print(f"  optimizer      : {args.quantum_optimizer}  (maxiter={args.quantum_optimizer_maxiter})")
    print(f"  shots          : {args.quantum_shots}")
    print(
        f"  samples        : train={args.quantum_max_train_samples}"
        f"  val={args.quantum_max_validation_samples}"
        f"  test={args.quantum_max_test_samples}"
    )
    print(f"  opt. level     : {args.quantum_optimization_level}")
    print(f"  qubits         : {args.quantum_num_qubits}"
          f"  (fm_reps={args.quantum_feature_map_reps}"
          f", ansatz_reps={args.quantum_ansatz_reps})")
    print("=" * 56)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def load_training_frame_with_scaler(
    *,
    processed_root_dir: Path,
    source: str,
    symbol: str,
    extraction_date: date,
    lookback: int,
    target_column: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    frame, scaler_metadata, _ = load_preferred_training_frame(
        processed_root_dir=processed_root_dir,
        source=source,
        symbol=symbol,
        extraction_date=extraction_date,
        lookback=lookback,
        target_column=target_column,
    )
    return frame, scaler_metadata


def inverse_scale(values: np.ndarray, *, min_offset: float, scale: float) -> np.ndarray:
    if scale == 0:
        raise ValueError("Cannot inverse scale values because scale is zero.")
    return (values - min_offset) / scale


def compute_direction_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> DirectionComparisonMetrics:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return DirectionComparisonMetrics(
        sample_count=int(len(y_true)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        confusion_matrix={
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    )


def build_keras_direction_evaluation(
    *,
    model_path: Path,
    frame: pd.DataFrame,
    scaler_metadata: dict[str, float],
    target_column: str,
    lookback: int,
) -> tuple[DirectionComparisonMetrics, pd.DataFrame]:
    feature_columns = [
        f"{target_column}_t_minus_{lag}"
        for lag in range(lookback, 0, -1)
    ]
    target_raw_column = f"y_{target_column}"
    current_scaled_column = f"{target_column}_t_minus_1"

    ordered = frame.copy()
    ordered["target_date"] = pd.to_datetime(ordered["target_date"])
    ordered = ordered.sort_values("target_date").reset_index(drop=True)

    test_frame = ordered.loc[ordered["split"].astype(str).str.lower() == "test"].copy()
    if test_frame.empty:
        raise ValueError("Refined dataset does not contain a test split.")

    X_test = test_frame.loc[:, feature_columns].to_numpy(dtype=np.float32).reshape(
        -1, lookback, 1
    )
    actual_next_close = test_frame[target_raw_column].to_numpy(dtype=np.float32)
    current_scaled = test_frame[current_scaled_column].to_numpy(dtype=np.float32)
    current_close = inverse_scale(
        current_scaled,
        min_offset=scaler_metadata["min_offset"],
        scale=scaler_metadata["scale"],
    )

    model = tf.keras.models.load_model(model_path)
    predicted_scaled = model.predict(X_test, verbose=0).reshape(-1)
    predicted_next_close = inverse_scale(
        predicted_scaled,
        min_offset=scaler_metadata["min_offset"],
        scale=scaler_metadata["scale"],
    )

    actual_direction = (actual_next_close > current_close).astype(int)
    predicted_direction = (predicted_next_close > current_close).astype(int)
    metrics = compute_direction_metrics(y_true=actual_direction, y_pred=predicted_direction)

    comparison_frame = pd.DataFrame(
        {
            "target_date": test_frame["target_date"].dt.strftime("%Y-%m-%d"),
            "current_close": current_close,
            "actual_next_close": actual_next_close,
            "predicted_next_close": predicted_next_close,
            "actual_direction": actual_direction,
            "predicted_direction": predicted_direction,
        }
    )
    return metrics, comparison_frame


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_dashboard(
    *,
    destination: Path,
    symbol: str,
    keras_seconds: float,
    quantum_seconds: float,
    keras_price_metrics: dict[str, float | None],
    keras_direction_metrics: DirectionComparisonMetrics,
    quantum_direction_metrics: DirectionComparisonMetrics,
    price_comparison_frame: pd.DataFrame,
    sample_plot_points: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Model Comparison Dashboard - {symbol}", fontsize=16, fontweight="bold")

    axes[0, 0].bar(
        ["Keras LSTM", "Quantum VQC"],
        [keras_seconds, quantum_seconds],
        color=["#1f77b4", "#ff7f0e"],
    )
    axes[0, 0].set_title("Training Time")
    axes[0, 0].set_ylabel("Seconds")
    for index, value in enumerate([keras_seconds, quantum_seconds]):
        axes[0, 0].text(index, value, f"{value:.2f}s", ha="center", va="bottom")

    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    keras_values = [
        keras_direction_metrics.accuracy,
        keras_direction_metrics.precision,
        keras_direction_metrics.recall,
        keras_direction_metrics.f1,
    ]
    quantum_values = [
        quantum_direction_metrics.accuracy,
        quantum_direction_metrics.precision,
        quantum_direction_metrics.recall,
        quantum_direction_metrics.f1,
    ]
    positions = np.arange(len(metric_names))
    width = 0.35
    axes[0, 1].bar(positions - width / 2, keras_values, width=width, label="Keras LSTM")
    axes[0, 1].bar(positions + width / 2, quantum_values, width=width, label="Quantum VQC")
    axes[0, 1].set_title("Directional Test Metrics")
    axes[0, 1].set_xticks(positions)
    axes[0, 1].set_xticklabels(metric_names)
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].legend()

    plot_frame = price_comparison_frame.head(sample_plot_points).copy()
    x_values = np.arange(len(plot_frame.index))
    axes[1, 0].plot(
        x_values,
        plot_frame["actual_next_close"],
        label="Actual next close",
        linewidth=2,
    )
    axes[1, 0].plot(
        x_values,
        plot_frame["predicted_next_close"],
        label="Predicted next close",
        linewidth=2,
        linestyle="--",
    )
    axes[1, 0].set_title("Keras Price Forecast vs Actual")
    axes[1, 0].set_xlabel("Test sample index")
    axes[1, 0].set_ylabel("Close price")
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    explanation_text = "\n".join(
        [
            "How to explain the models:",
            "",
            "Keras LSTM:",
            "1. Reads the last 60 closing prices.",
            "2. Learns a time pattern in those prices.",
            "3. Predicts the next closing price.",
            "",
            "Quantum VQC:",
            "1. Compresses the 60-price window into a few latent factors.",
            "2. Encodes those factors as qubit rotation angles.",
            "3. Uses circuit measurements to classify the next move as up or down.",
            "",
            f"Keras price MAE: {keras_price_metrics['mae']:.2f}",
            f"Keras price RMSE: {keras_price_metrics['rmse']:.2f}",
            f"Keras price MAPE: {keras_price_metrics['mape']:.2f}%",
        ]
    )
    axes[1, 1].text(
        0.0,
        1.0,
        explanation_text,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_chart(
    *,
    destination: Path,
    symbol: str,
    keras_metrics: DirectionComparisonMetrics,
    quantum_metrics: DirectionComparisonMetrics,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Directional Confusion Matrices - {symbol}", fontsize=15, fontweight="bold")

    models = [
        ("Keras LSTM", keras_metrics.confusion_matrix),
        ("Quantum VQC", quantum_metrics.confusion_matrix),
    ]
    for axis, (title, matrix_payload) in zip(axes, models):
        matrix = np.array(
            [
                [matrix_payload["tn"], matrix_payload["fp"]],
                [matrix_payload["fn"], matrix_payload["tp"]],
            ]
        )
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(title)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class")
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["Down", "Up"])
        axis.set_yticks([0, 1])
        axis.set_yticklabels(["Down", "Up"])
        for row in range(2):
            for col in range(2):
                axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


"""
Improved replacement for write_markdown_report in compare_models.py

Drop this function in place of the original in scripts/compare_models.py.
The signature is identical — no other part of the script needs to change.
"""

def _bar(value: float, width: int = 20) -> str:
    """ASCII progress bar proportional to a value between 0 and 1."""
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def _confusion_block(matrix: dict[str, int]) -> str:
    """Render a confusion matrix as formatted plain text."""
    tn = matrix["tn"]
    fp = matrix["fp"]
    fn = matrix["fn"]
    tp = matrix["tp"]
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total if total else 0.0

    lines = [
        "```",
        "                  Predicted",
        "                Down      Up",
        f"  Actual  Down  [{tn:>6}]  [{fp:>6}]",
        f"          Up    [{fn:>6}]  [{tp:>6}]",
        "```",
        "",
        f"- **True negatives (TN):**  {tn}  — correctly predicted a downward move",
        f"- **False positives (FP):** {fp}  — predicted up, actual was down",
        f"- **False negatives (FN):** {fn}  — predicted down, actual was up",
        f"- **True positives (TP):**  {tp}  — correctly predicted an upward move",
        f"- **Implied accuracy:**     {accuracy:.1%}",
    ]
    return "\n".join(lines)


def _interpret_direction(metrics: DirectionComparisonMetrics, model_name: str) -> str:
    """Generate an automatic plain-English interpretation of directional metrics."""
    notes: list[str] = []

    if metrics.accuracy >= 0.60:
        notes.append(
            f"**{model_name}** predicted direction correctly in **{metrics.accuracy:.1%}** "
            f"of cases — above the 60% reference threshold."
        )
    elif metrics.accuracy >= 0.50:
        notes.append(
            f"**{model_name}** predicted direction correctly in **{metrics.accuracy:.1%}** "
            f"of cases — marginally above chance."
        )
    else:
        notes.append(
            f"**{model_name}** predicted direction correctly in only **{metrics.accuracy:.1%}** "
            f"of cases — below chance; consider revisiting features or hyperparameters."
        )

    if metrics.precision < 0.50:
        notes.append(
            "Low precision: many false positives — the model tends to predict upward moves that do not materialise."
        )
    if metrics.recall < 0.50:
        notes.append(
            "Low recall: many false negatives — the model misses actual upward moves."
        )
    if metrics.f1 >= 0.60:
        notes.append(
            f"F1 of **{metrics.f1:.2f}** indicates a reasonable balance between precision and recall."
        )

    return "  \n".join(notes) if notes else "Metrics are within expected range for the test split."


def _format_optional_float(value: float | None, precision: int = 4) -> str:
    if value is None:
        return "not captured"
    return f"{value:.{precision}f}"


def _format_optional_int(value: int | None) -> str:
    if value is None:
        return "not captured"
    return str(value)


def _describe_quantum_environment(execution_mode: str, backend_name: str) -> tuple[str, str, str]:
    normalized_mode = execution_mode.lower()
    if normalized_mode == "cloud":
        return (
            "IBM Quantum real hardware",
            "yes",
            (
                f"`{backend_name}` is treated as a physical IBM Quantum backend. "
                "Results are affected by real NISQ noise: decoherence, gate errors, "
                "readout errors, queue latency, and shot noise."
            ),
        )

    return (
        "Qiskit simulator / fake backend",
        "no",
        (
            f"`{backend_name}` is a local simulated backend path. It is useful for "
            "repeatable development and report generation, but it is not evidence "
            "of physical QPU execution."
        ),
    )


def _build_quantum_methodology_section(
    *,
    execution_mode: str,
    backend_name: str,
    shots: int,
    optimizer_name: str,
    optimizer_maxiter: int,
    function_evaluations: int | None,
    objective_value: float | None,
    num_qubits: int,
    train_samples: int,
    validation_samples: int,
    test_samples: int,
    quantum_seconds: float,
) -> str:
    environment_label, consumes_ibm_minutes, environment_note = _describe_quantum_environment(
        execution_mode=execution_mode,
        backend_name=backend_name,
    )
    qpu_seconds = "not captured by the current VQC abstraction"
    payg_estimate = "not estimated until QPU seconds are captured"
    if execution_mode.lower() != "cloud":
        qpu_seconds = "0 for IBM hardware; local simulator time only"
        payg_estimate = "$0.00 IBM QPU cost"

    return f"""## 5. Execution environment and cost profile

### 5.1 Current quantum run

| Field | Value |
|---|---:|
| Execution environment | {environment_label} |
| Execution mode | `{execution_mode}` |
| Backend | `{backend_name}` |
| IBM runtime minutes consumed | {consumes_ibm_minutes} |
| Wall-clock quantum training time | `{quantum_seconds:.2f} s` |
| QPU seconds consumed | {qpu_seconds} |
| Pay-as-you-go equivalent estimate | {payg_estimate} |
| Shots | `{shots}` |
| Optimizer | `{optimizer_name}` |
| Optimizer max iterations | `{optimizer_maxiter}` |
| Function evaluations | `{_format_optional_int(function_evaluations)}` |
| Objective value | `{_format_optional_float(objective_value)}` |
| Qubits | `{num_qubits}` |
| Samples | train=`{train_samples}`, validation=`{validation_samples}`, test=`{test_samples}` |

{environment_note}

### 5.2 What changes on IBM Quantum real hardware

When `execution_mode=cloud`, the experiment should be described as **IBM Quantum via API key on real superconducting NISQ hardware**, not as local simulation. The backend executes circuits on physical qubits, so the methodology must report hardware noise and operational latency.

| Category | IBM real-hardware variable | Article note |
|---|---|---|
| Latency | Queue time | Can range from minutes to hours depending on backend load |
| Latency | Real job execution time | Usually seconds for small circuits, but not under API request control |
| Latency | Total wall-clock time | Queue + transpilation + QPU execution + result retrieval |
| Serving | D+1 online inference latency | Not viable for real-time API serving; use offline materialization |
| Noise | Gate error rate | Backend-reported calibration value; often order-of-magnitude 0.1%-1% per gate |
| Noise | Readout error | Backend-reported calibration value; often order-of-magnitude 1%-5% |
| Noise | T1 / T2 coherence | Backend calibration should be captured with the run |
| Mitigation | Error mitigation | Compare unmitigated vs Qiskit Runtime mitigation when enabled |

### 5.3 IBM job observability checklist

The current VQC training artifact captures backend, shots, optimizer, objective value, and function evaluations. For a publication-grade IBM hardware experiment, extend the runner to persist:

| Item to log | Why it matters |
|---|---|
| `job.result().metadata` | Execution metadata returned by Runtime |
| `job.metrics()` | Queue time, execution time, and backend system timing when available |
| Backend name | Example: `ibm_brisbane`, `ibm_sherbrooke` |
| QPU seconds | Required for free-plan budget and paid-plan cost simulation |
| Shots | Controls variance and runtime cost |
| Circuit depth after transpilation | Directly affects noise exposure |
| Two-qubit gate count after transpilation | Usually the dominant hardware error source |
| Bitstring histogram | Shows measured quantum output distribution |
| Expectation value | Provides observable-level summary |
| Variance across runs | Repeat 3-5 times and report mean + standard deviation |
| Mitigation mode | Compare no mitigation vs available Runtime mitigation options |

### 5.4 Cost model note

For the article, record the Open Plan budget as consumed QPU seconds and keep the paid cost as a transparent simulation. If using the 2025 article assumption of **US$1.60 per QPU second**, compute:

```text
estimated_paid_cost_usd = qpu_seconds * 1.60
```

Do not treat that number as a current IBM price unless it is verified at writing time; keep it labelled as the article's cost-simulation assumption.

---
"""


def write_markdown_report(
    *,
    destination: Path,
    symbol: str,
    extraction_date: date,
    generated_at_utc: str,
    keras_seconds: float,
    quantum_seconds: float,
    keras_price_metrics: dict[str, float | None],
    keras_direction_metrics: DirectionComparisonMetrics,
    quantum_direction_metrics: DirectionComparisonMetrics,
    quantum_execution_mode: str,
    quantum_backend_name: str,
    quantum_shots: int,
    quantum_optimizer_name: str,
    quantum_optimizer_maxiter: int,
    quantum_function_evaluations: int | None,
    quantum_objective_value: float | None,
    quantum_num_qubits: int,
    quantum_train_samples: int,
    quantum_validation_samples: int,
    quantum_test_samples: int,
    dashboard_path: Path,
    confusion_chart_path: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    # --- timing ---
    faster_model = "Keras LSTM" if keras_seconds < quantum_seconds else "Quantum VQC"
    speedup = max(keras_seconds, quantum_seconds) / min(keras_seconds, quantum_seconds)

    # --- direction winner ---
    keras_f1 = keras_direction_metrics.f1
    quantum_f1 = quantum_direction_metrics.f1
    direction_winner = "Keras LSTM" if keras_f1 >= quantum_f1 else "Quantum VQC"
    direction_delta = abs(keras_f1 - quantum_f1)

    # --- confusion blocks ---
    keras_confusion = _confusion_block(keras_direction_metrics.confusion_matrix)
    quantum_confusion = _confusion_block(quantum_direction_metrics.confusion_matrix)

    # --- shorthand aliases ---
    k = keras_direction_metrics
    q = quantum_direction_metrics
    quantum_methodology_section = _build_quantum_methodology_section(
        execution_mode=quantum_execution_mode,
        backend_name=quantum_backend_name,
        shots=quantum_shots,
        optimizer_name=quantum_optimizer_name,
        optimizer_maxiter=quantum_optimizer_maxiter,
        function_evaluations=quantum_function_evaluations,
        objective_value=quantum_objective_value,
        num_qubits=quantum_num_qubits,
        train_samples=quantum_train_samples,
        validation_samples=quantum_validation_samples,
        test_samples=quantum_test_samples,
        quantum_seconds=quantum_seconds,
    )

    report = f"""# Model Comparison Report — `{symbol}`

> **Extraction date:** `{extraction_date.isoformat()}`  
> **Generated at (UTC):** `{generated_at_utc}`  
> **Evaluated samples — Keras:** {k.sample_count} | **Quantum:** {q.sample_count}

---

## Executive summary

| | Keras LSTM | Quantum VQC |
|---|:---:|:---:|
| Primary task | Price regression | Direction classification |
| Directional F1 | `{k.f1:.4f}` | `{q.f1:.4f}` |
| Directional accuracy | `{k.accuracy:.1%}` | `{q.accuracy:.1%}` |
| Training time | `{keras_seconds:.1f} s` | `{quantum_seconds:.1f} s` |
| Price MAE | `{keras_price_metrics["mae"]:.4f}` | — |

**Fastest model:** {faster_model} ({speedup:.1f}× faster)  
**Best directional F1:** {direction_winner} (Δ F1 = {direction_delta:.4f})

---

## 1. Training time

| Model | Time (s) | Share |
|---|---:|---:|
| Keras LSTM | {keras_seconds:.2f} | {keras_seconds / (keras_seconds + quantum_seconds):.1%} |
| Quantum VQC | {quantum_seconds:.2f} | {quantum_seconds / (keras_seconds + quantum_seconds):.1%} |
| **Total** | **{keras_seconds + quantum_seconds:.2f}** | 100% |

> {faster_model} trained **{speedup:.1f}×** faster.  
> Quantum models are typically slower because they rely on circuit simulation or real quantum hardware access.

---

## 2. Price metrics — Keras LSTM

The Keras model performs **direct price regression**; the Quantum VQC does not produce a price output.

| Metric | Value | Interpretation |
|---|---:|---|
| MAE | `{keras_price_metrics["mae"]:.4f}` | Mean absolute error in price units (MinMax scale) |
| RMSE | `{keras_price_metrics["rmse"]:.4f}` | Penalises large errors; compare against MAE to detect outliers |
| MAPE | `{keras_price_metrics["mape"]:.4f}%` | Mean absolute percentage error |

> **Reading tip:** RMSE > MAE means a few days with large errors are pulling the average up.  
> MAPE below 5% is generally considered acceptable for financial time series.

---

## 3. Directional metrics — side by side

Both models are evaluated on their ability to predict whether the next close is higher or lower than the current one.

| Metric | Keras LSTM | Quantum VQC | Δ (Keras − Quantum) |
|---|:---:|:---:|:---:|
| Accuracy | `{k.accuracy:.4f}` | `{q.accuracy:.4f}` | `{k.accuracy - q.accuracy:+.4f}` |
| Precision | `{k.precision:.4f}` | `{q.precision:.4f}` | `{k.precision - q.precision:+.4f}` |
| Recall | `{k.recall:.4f}` | `{q.recall:.4f}` | `{k.recall - q.recall:+.4f}` |
| F1 | `{k.f1:.4f}` | `{q.f1:.4f}` | `{k.f1 - q.f1:+.4f}` |
| Samples | {k.sample_count} | {q.sample_count} | — |

### Quick F1 visualisation

```
Keras LSTM   {_bar(k.f1)}  {k.f1:.2f}
Quantum VQC  {_bar(q.f1)}  {q.f1:.2f}
             0%                    100%
```

### Interpretation — Keras LSTM

{_interpret_direction(keras_direction_metrics, "Keras LSTM")}

### Interpretation — Quantum VQC

{_interpret_direction(quantum_direction_metrics, "Quantum VQC")}

---

## 4. Confusion matrices

### 4.1 Keras LSTM

{keras_confusion}

### 4.2 Quantum VQC

{quantum_confusion}

---

{quantum_methodology_section}

## 6. Article comparison table

| Dimension | Keras LSTM | Qiskit / IBM real hardware |
|---|---|---|
| Model role | Production baseline | Experimental NISQ benchmark |
| Primary output | Next-day price regression | Next-day direction classification |
| Trainable parameters | Thousands, depending on LSTM shape | Dozens of circuit angles, depending on ansatz |
| Training time | `{keras_seconds:.2f} s` in this run | `{quantum_seconds:.2f} s` wall-clock in this run |
| Inference latency | Millisecond-scale locally; article shorthand can use ~5 ms | Minutes or more on hardware because of queue time |
| Estimated experiment cost | Near-zero locally; article may model ~$0.01 EC2-equivalent | Measure via QPU seconds / `job.metrics()` for paid-cost estimate |
| Intrinsic noise | Deterministic with fixed seed and hardware | Stochastic: shots, readout noise, gate noise, decoherence |
| Reproducibility | High with fixed seeds and artifacts | Partial; repeat runs and report mean + standard deviation |
| API serving fit | Suitable for `POST /predict` | Serve only from offline materialized parquet |
| Scientific contribution | Strong baseline for comparison | Real-hardware behavior is more publishable than simulator-only QML |

**Methodology requirement:** for real IBM hardware, repeat each quantum experiment at least **3-5 times** with the same configuration and report mean plus standard deviation. This is necessary because shot noise and hardware calibration drift can change results between runs.

---

## 7. How to explain the models to investors

### Keras LSTM

1. Receives the **last 60 closing prices** as a time-ordered sequence.
2. Learns short-term temporal patterns using LSTM (Long Short-Term Memory) layers.
3. Outputs an **estimated next closing price**.
4. Direction (up/down) is derived by comparing the prediction against the current close.

### Quantum VQC

1. The 60-price window is compressed by PCA into a small number of latent factors.
2. Those factors are encoded as **qubit rotation angles** inside a quantum circuit.
3. A classical optimiser tunes the circuit to classify the next move as **up or down**.
4. Output is a binary label directly — no absolute price estimate is produced.

---

## 8. Recommendations

| # | Recommendation |
|---|---|
| 1 | Use **Keras LSTM** for absolute price forecasting and level-based entry/exit signal generation. |
| 2 | Treat **Quantum VQC** as an experimental directional model — not a production replacement for Keras yet. |
| 3 | The fairest apples-to-apples comparison between both approaches is the **directional metrics table** (Section 3). |
| 4 | For IBM hardware experiments, report backend, queue/runtime metrics, shots, circuit depth, and repeated-run variance. |
| 5 | If Keras MAPE exceeds 5%, consider adjusting the lookback window or adding volume-based features. |
| 6 | For the Quantum model, increasing `quantum_max_train_samples` improves generalisation at the cost of training time. |

---

## 9. Visual assets

| Artifact | Local path |
|---|---|
| Main dashboard | `{dashboard_path}` |
| Confusion matrices | `{confusion_chart_path}` |

---

*Report generated automatically by the model comparison pipeline.*
"""
    destination.write_text(report, encoding="utf-8")


def build_run_token() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    args = apply_quantum_mode_defaults(args)
    require_ibm_runtime_confirmation(args)

    settings = TrainingPipelineSettings.from_env()
    upload_to_s3 = not args.skip_s3
    extraction_date = (
        parse_iso_date(args.extraction_date)
        if args.extraction_date
        else detect_latest_extraction_date(settings.local_processed_dir)
    )

    print_run_summary(args)

    local_store = LocalModelStore(settings.local_models_dir)
    model_s3_store = build_model_s3_store(settings, upload_to_s3)
    keras_service = KerasTrainingService(
        processed_root_dir=settings.local_processed_dir,
        local_store=local_store,
        s3_store=model_s3_store,
    )
    quantum_use_case = TrainQuantumModelUseCase(
        processed_root_dir=settings.local_processed_dir,
        local_store=local_store,
        s3_store=model_s3_store,
    )

    generated_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    run_token = build_run_token()
    assets: list[ComparisonAssetArtifact] = []

    for raw_symbol in args.symbols:
        symbol = raw_symbol.upper()
        print(f"\nRunning comparison for {symbol}...")

        # --- Keras ---
        keras_request = KerasTrainingRequest(
            symbols=(symbol,),
            extraction_date=extraction_date,
            source=args.source,
            target_column=args.target_column,
            lookback=args.lookback,
            epochs=args.keras_epochs,
            batch_size=args.keras_batch_size,
            patience=args.keras_patience,
            learning_rate=args.keras_learning_rate,
            seed=args.seed,
            verbose=args.keras_verbose,
            model_name_prefix=args.keras_model_name_prefix,
        )

        keras_start = time.perf_counter()
        try:
            keras_result = keras_service.train(keras_request)
        except TrainingInterruptedError as exc:
            raise SystemExit(str(exc)) from exc
        keras_seconds = time.perf_counter() - keras_start
        keras_asset = keras_result.assets[0]
        print(f"  Keras finished in {keras_seconds:.1f}s")

        # --- Quantum ---
        quantum_request = QuantumTrainingRequest(
            symbols=(symbol,),
            extraction_date=extraction_date,
            source=args.source,
            target_column=args.target_column,
            lookback=args.lookback,
            execution_mode=args.quantum_mode,
            backend_name=args.quantum_backend,
            num_qubits=args.quantum_num_qubits,
            feature_map_reps=args.quantum_feature_map_reps,
            ansatz_reps=args.quantum_ansatz_reps,
            shots=args.quantum_shots,
            optimization_level=args.quantum_optimization_level,
            optimizer_name=args.quantum_optimizer,
            optimizer_maxiter=args.quantum_optimizer_maxiter,
            max_train_samples=args.quantum_max_train_samples,
            max_validation_samples=args.quantum_max_validation_samples,
            max_test_samples=args.quantum_max_test_samples,
            seed=args.seed,
            model_name_prefix=args.quantum_model_name_prefix,
        )

        quantum_start = time.perf_counter()
        try:
            quantum_result = quantum_use_case.train(quantum_request)
        except QuantumTrainingInterruptedError as exc:
            raise SystemExit(str(exc)) from exc
        quantum_seconds = time.perf_counter() - quantum_start
        quantum_asset = quantum_result.assets[0]
        print(f"  Quantum finished in {quantum_seconds:.1f}s")

        # --- Evaluation ---
        refined_frame, scaler_metadata = load_training_frame_with_scaler(
            processed_root_dir=settings.local_processed_dir,
            source=args.source,
            symbol=symbol,
            extraction_date=extraction_date,
            lookback=args.lookback,
            target_column=args.target_column,
        )
        keras_direction_metrics, keras_price_frame = build_keras_direction_evaluation(
            model_path=Path(keras_asset.model_local_path),
            frame=refined_frame,
            scaler_metadata=scaler_metadata,
            target_column=args.target_column,
            lookback=args.lookback,
        )

        quantum_direction_metrics = DirectionComparisonMetrics(
            sample_count=quantum_asset.test_metrics.sample_count,
            accuracy=float(quantum_asset.test_metrics.accuracy or 0.0),
            precision=float(quantum_asset.test_metrics.precision or 0.0),
            recall=float(quantum_asset.test_metrics.recall or 0.0),
            f1=float(quantum_asset.test_metrics.f1 or 0.0),
            confusion_matrix=dict(quantum_asset.test_metrics.confusion_matrix),
        )

        # --- Artifacts ---
        chart_root = (
            Path("comparison_runs")
            / f"extraction_date={extraction_date.isoformat()}"
            / f"generated_at={run_token}"
            / f"symbol={symbol}"
        )
        dashboard_path = local_store.prepare_path(chart_root / "comparison_dashboard.png")
        confusion_chart_path = local_store.prepare_path(
            chart_root / "comparison_confusion_matrices.png"
        )
        report_path = local_store.prepare_path(chart_root / "comparison_report.md")

        keras_price_metrics = {
            "mae": float(keras_asset.test_metrics.mae or 0.0),
            "rmse": float(keras_asset.test_metrics.rmse or 0.0),
            "mape": float(keras_asset.test_metrics.mape or 0.0),
        }

        save_dashboard(
            destination=dashboard_path,
            symbol=symbol,
            keras_seconds=keras_seconds,
            quantum_seconds=quantum_seconds,
            keras_price_metrics=keras_price_metrics,
            keras_direction_metrics=keras_direction_metrics,
            quantum_direction_metrics=quantum_direction_metrics,
            price_comparison_frame=keras_price_frame,
            sample_plot_points=args.sample_plot_points,
        )
        save_confusion_matrix_chart(
            destination=confusion_chart_path,
            symbol=symbol,
            keras_metrics=keras_direction_metrics,
            quantum_metrics=quantum_direction_metrics,
        )
        write_markdown_report(
            destination=report_path,
            symbol=symbol,
            extraction_date=extraction_date,
            generated_at_utc=generated_at_utc,
            keras_seconds=keras_seconds,
            quantum_seconds=quantum_seconds,
            keras_price_metrics=keras_price_metrics,
            keras_direction_metrics=keras_direction_metrics,
            quantum_direction_metrics=quantum_direction_metrics,
            quantum_execution_mode=quantum_asset.execution_mode,
            quantum_backend_name=quantum_asset.backend_name,
            quantum_shots=args.quantum_shots,
            quantum_optimizer_name=quantum_asset.optimizer_name,
            quantum_optimizer_maxiter=quantum_asset.optimizer_maxiter,
            quantum_function_evaluations=quantum_asset.function_evaluations,
            quantum_objective_value=quantum_asset.objective_value,
            quantum_num_qubits=quantum_asset.num_qubits,
            quantum_train_samples=quantum_asset.sampled_counts.get("train", 0),
            quantum_validation_samples=quantum_asset.sampled_counts.get("validation", 0),
            quantum_test_samples=quantum_asset.sampled_counts.get("test", 0),
            dashboard_path=dashboard_path,
            confusion_chart_path=confusion_chart_path,
        )

        assets.append(
            ComparisonAssetArtifact(
                symbol=symbol,
                keras_training_seconds=keras_seconds,
                quantum_training_seconds=quantum_seconds,
                keras_model_local_path=keras_asset.model_local_path,
                quantum_model_local_path=quantum_asset.model_local_path,
                dashboard_local_path=str(dashboard_path),
                confusion_matrix_local_path=str(confusion_chart_path),
                report_local_path=str(report_path),
                keras_price_metrics=keras_price_metrics,
                keras_direction_metrics=keras_direction_metrics,
                quantum_direction_metrics=quantum_direction_metrics,
                quantum_execution_mode=quantum_asset.execution_mode,
                quantum_backend_name=quantum_asset.backend_name,
                quantum_shots=args.quantum_shots,
                quantum_optimizer_name=quantum_asset.optimizer_name,
                quantum_optimizer_maxiter=quantum_asset.optimizer_maxiter,
                quantum_function_evaluations=quantum_asset.function_evaluations,
                quantum_objective_value=quantum_asset.objective_value,
                quantum_num_qubits=quantum_asset.num_qubits,
                quantum_train_samples=quantum_asset.sampled_counts.get("train", 0),
                quantum_validation_samples=quantum_asset.sampled_counts.get("validation", 0),
                quantum_test_samples=quantum_asset.sampled_counts.get("test", 0),
            )
        )

        print(f"  report    : {report_path}")
        print(f"  dashboard : {dashboard_path}")

    manifest_payload = {
        "generated_at_utc": generated_at_utc,
        "request": {
            "symbols": [s.upper() for s in args.symbols],
            "extraction_date": extraction_date.isoformat(),
            "source": args.source,
            "target_column": args.target_column,
            "lookback": args.lookback,
            "keras_epochs": args.keras_epochs,
            "keras_batch_size": args.keras_batch_size,
            "quantum_mode": args.quantum_mode,
            "quantum_backend": args.quantum_backend,
            "quantum_num_qubits": args.quantum_num_qubits,
            "quantum_optimizer": args.quantum_optimizer,
            "quantum_optimizer_maxiter": args.quantum_optimizer_maxiter,
            "quantum_shots": args.quantum_shots,
            "quantum_max_train_samples": args.quantum_max_train_samples,
            "quantum_max_validation_samples": args.quantum_max_validation_samples,
            "quantum_max_test_samples": args.quantum_max_test_samples,
            "quantum_optimization_level": args.quantum_optimization_level,
        },
        "assets": [asdict(asset) for asset in assets],
    }
    manifest_path = local_store.write_json(
        manifest_payload,
        Path("comparison_runs")
        / f"extraction_date={extraction_date.isoformat()}"
        / f"generated_at={run_token}"
        / "comparison_manifest.json",
    )

    print(f"\nComparison run completed.")
    print(f"Manifest local: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
