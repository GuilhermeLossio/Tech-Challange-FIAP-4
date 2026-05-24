from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from html import escape
from pathlib import Path
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src.application.services.forecast_guardrails import apply_standard_forecast_guardrail
from src.application.services.model_promotion_registry import ModelPromotionRegistry

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    tf = None
    _TENSORFLOW_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _TENSORFLOW_IMPORT_ERROR = None

try:
    from qiskit.circuit.library import real_amplitudes, zz_feature_map
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    from qiskit_machine_learning.algorithms import VQC
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    COBYLA = FakeManilaV2 = QiskitRuntimeService = Sampler = VQC = None
    generate_preset_pass_manager = real_amplitudes = zz_feature_map = None
    _QISKIT_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _QISKIT_IMPORT_ERROR = None

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.application.use_cases.generate_feature_dataset import GenerateFeatureDatasetUseCase
from src.infrastructure.config.settings import load_env_file
from src.infrastructure.storage.local_processed_store import LocalProcessedStore
from src.infrastructure.storage.s3_raw_store import S3RawStore


DEFAULT_SYMBOLS = ("NVDA", "AMD", "TSM", "ASML", "QCOM")


def _print_timing_summary(
    *,
    symbol: str,
    normal_elapsed_ms: float | None,
    normal_per_step_ms: list[float] | None,
    quantum_elapsed_ms: float | None,
    quantum_per_step_ms: list[float] | None,
    timing_ratio: float | None,
) -> None:
    """Print per-symbol model timing to stdout during batch generation."""
    separator = "-" * 56
    print(f"\n{separator}")
    print(f"  Timing report - {symbol.upper()}")
    print(separator)

    if normal_per_step_ms:
        normal_avg = sum(normal_per_step_ms) / len(normal_per_step_ms)
        print(f"  LSTM  total : {normal_elapsed_ms:>10.1f} ms")
        print(
            "        avg/step: "
            f"{normal_avg:>8.2f} ms  |  "
            f"min: {min(normal_per_step_ms):.2f}  "
            f"max: {max(normal_per_step_ms):.2f}"
        )

    if quantum_per_step_ms:
        quantum_avg = sum(quantum_per_step_ms) / len(quantum_per_step_ms)
        print(f"  VQC   total : {quantum_elapsed_ms:>10.1f} ms")
        print(
            "        avg/step: "
            f"{quantum_avg:>8.2f} ms  |  "
            f"min: {min(quantum_per_step_ms):.2f}  "
            f"max: {max(quantum_per_step_ms):.2f}"
        )

    if timing_ratio is not None:
        print(f"  Ratio VQC/LSTM: {timing_ratio:.1f}x")

    print(separator)


@dataclass(frozen=True)
class ForecastBatchRequest:
    symbols: tuple[str, ...]
    extraction_date: date
    source: str = "yfinance"
    target_column: str = "close"
    lookback: int = 60
    horizon_days: int = 30
    model_name_prefix: str = "lstm"
    quantum_model_name_prefix: str = "quantum_vqc"
    include_normal: bool = True
    include_quantum: bool = True
    upload_to_s3: bool = True
    quantum_runtime_mode: str = "local"
    quantum_backend_name: str | None = None
    quantum_shots: int = 1024
    quantum_optimization_level: int = 1
    confirm_ibm_runtime_cost: bool = False
    max_cloud_quantum_predictions: int = 5


@dataclass(frozen=True)
class ForecastAssetArtifact:
    symbol: str
    row_count: int
    predict_types: tuple[str, ...]
    forecast_start_date: str
    forecast_end_date: str
    last_observed_date: str
    last_observed_close: float
    normal_model_local_path: str | None
    quantum_model_local_path: str | None
    normal_elapsed_ms: float | None
    normal_per_step_ms: list[float] | None
    quantum_elapsed_ms: float | None
    quantum_per_step_ms: list[float] | None
    timing_ratio: float | None
    forecast_summary: dict[str, dict[str, Any]]
    local_path: str
    s3_uri: str | None
    report_local_path: str
    report_s3_uri: str | None
    chart_local_path: str
    chart_s3_uri: str | None


@dataclass(frozen=True)
class ForecastBatchResult:
    source: str
    generated_at_utc: str
    manifest_local_path: str
    manifest_s3_uri: str | None
    unified_report_local_path: str
    unified_latest_report_local_path: str
    unified_report_s3_uri: str | None
    assets: tuple[ForecastAssetArtifact, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "generated_at_utc": self.generated_at_utc,
            "manifest_local_path": self.manifest_local_path,
            "manifest_s3_uri": self.manifest_s3_uri,
            "unified_report_local_path": self.unified_report_local_path,
            "unified_latest_report_local_path": self.unified_latest_report_local_path,
            "unified_report_s3_uri": self.unified_report_s3_uri,
            "asset_count": len(self.assets),
            "assets": [asdict(asset) for asset in self.assets],
        }


@dataclass(frozen=True)
class ForecastStepStabilizationResult:
    predicted_close: float
    raw_model_close: float
    predicted_scaled: float | None
    prediction_return_cap: float
    prediction_constraint_applied: bool
    prediction_constraint_method: str | None
    hit_lower_band: bool
    hit_upper_band: bool
    dynamic_cumulative_return_cap: float


class GenerateForecastBatchUseCase:
    """
    Generate a combined future prediction dataset for both model families.
    """

    def __init__(
        self,
        raw_root_dir: Path,
        processed_root_dir: Path,
        models_root_dir: Path,
        local_store: LocalProcessedStore,
        s3_store: S3RawStore | None = None,
    ) -> None:
        self._raw_root_dir = raw_root_dir
        self._processed_root_dir = processed_root_dir
        self._models_root_dir = models_root_dir
        self._local_store = local_store
        self._s3_store = s3_store
        self._promotion_registry = ModelPromotionRegistry(
            models_root_dir=self._models_root_dir,
        )

    def generate(self, request: ForecastBatchRequest) -> ForecastBatchResult:
        if request.include_normal:
            self._ensure_tensorflow_available()
        if request.include_quantum:
            self._ensure_quantum_dependencies_available()
        self._validate_request(request)

        generated_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        generated_at_token = self._to_path_safe_timestamp(generated_at_utc)
        artifacts: list[ForecastAssetArtifact] = []

        for symbol in request.symbols:
            raw_frame = self._load_raw_frame(
                source=request.source,
                symbol=symbol,
                extraction_date=request.extraction_date,
                target_column=request.target_column,
            )
            scaler_metadata = self._load_scaler_metadata(
                source=request.source,
                symbol=symbol,
                extraction_date=request.extraction_date,
                lookback=request.lookback,
                target_column=request.target_column,
            )

            combined_rows: list[dict[str, Any]] = []
            normal_model_local_path: str | None = None
            quantum_model_local_path: str | None = None
            predict_types: list[str] = []
            normal_elapsed_ms: float | None = None
            normal_per_step_ms: list[float] | None = None
            quantum_elapsed_ms: float | None = None
            quantum_per_step_ms: list[float] | None = None
            timing_ratio: float | None = None

            if request.include_normal:
                normal_metadata = self._resolve_keras_model_metadata(
                    symbol=symbol,
                    extraction_date=request.extraction_date,
                    source=request.source,
                    target_column=request.target_column,
                    lookback=request.lookback,
                    model_name_prefix=request.model_name_prefix,
                )
                normal_model_local_path = str(normal_metadata["model_local_path"])
                normal_model = tf.keras.models.load_model(  # type: ignore[union-attr]
                    normal_model_local_path,
                    compile=False,
                )
                normal_start = time.perf_counter()
                normal_rows, normal_per_step_ms = self._build_normal_rows_timed(
                    raw_frame=raw_frame,
                    request=request,
                    symbol=symbol,
                    generated_at_utc=generated_at_utc,
                    generated_at_token=generated_at_token,
                    model=normal_model,
                    scaler_metadata=scaler_metadata,
                    model_metadata=normal_metadata,
                )
                normal_elapsed_ms = (time.perf_counter() - normal_start) * 1_000
                combined_rows.extend(normal_rows)
                predict_types.append("normal")

            if request.include_quantum:
                quantum_metadata = self._resolve_quantum_model_metadata(
                    symbol=symbol,
                    extraction_date=request.extraction_date,
                    source=request.source,
                    target_column=request.target_column,
                    lookback=request.lookback,
                    model_name_prefix=request.quantum_model_name_prefix,
                )
                quantum_model_local_path = str(quantum_metadata["model_local_path"])
                quantum_bundle = joblib.load(str(quantum_metadata["preprocessor_local_path"]))
                quantum_payload = json.loads(
                    Path(str(quantum_metadata["model_local_path"])).read_text(encoding="utf-8")
                )
                quantum_model = self._build_quantum_predictor(
                    model_payload=quantum_payload,
                    request=request,
                )
                quantum_start = time.perf_counter()
                quantum_rows, quantum_per_step_ms = self._build_quantum_rows_timed(
                    raw_frame=raw_frame,
                    request=request,
                    symbol=symbol,
                    generated_at_utc=generated_at_utc,
                    generated_at_token=generated_at_token,
                    model=quantum_model,
                    scaler_metadata=scaler_metadata,
                    quantum_bundle=quantum_bundle,
                    model_metadata=quantum_metadata,
                )
                quantum_elapsed_ms = (time.perf_counter() - quantum_start) * 1_000
                combined_rows.extend(quantum_rows)
                predict_types.append("quant")

            if normal_per_step_ms and quantum_per_step_ms:
                normal_average = sum(normal_per_step_ms) / len(normal_per_step_ms)
                quantum_average = sum(quantum_per_step_ms) / len(quantum_per_step_ms)
                timing_ratio = quantum_average / normal_average if normal_average > 0 else None

            _print_timing_summary(
                symbol=symbol,
                normal_elapsed_ms=normal_elapsed_ms,
                normal_per_step_ms=normal_per_step_ms,
                quantum_elapsed_ms=quantum_elapsed_ms,
                quantum_per_step_ms=quantum_per_step_ms,
                timing_ratio=timing_ratio,
            )

            future_predict_frame = pd.DataFrame(combined_rows)
            future_predict_frame = future_predict_frame.sort_values(
                by=["forecast_step", "predict_type"],
            ).reset_index(drop=True)
            forecast_summary = self._summarize_future_predict_frame(future_predict_frame)
            forecast_summary["_timing"] = {
                "normal_elapsed_ms": normal_elapsed_ms,
                "normal_per_step_ms_mean": (
                    sum(normal_per_step_ms) / len(normal_per_step_ms)
                    if normal_per_step_ms
                    else None
                ),
                "quantum_elapsed_ms": quantum_elapsed_ms,
                "quantum_per_step_ms_mean": (
                    sum(quantum_per_step_ms) / len(quantum_per_step_ms)
                    if quantum_per_step_ms
                    else None
                ),
                "timing_ratio_quant_over_normal": timing_ratio,
            }

            relative_path = self._build_future_predict_relative_path(
                source=request.source,
                symbol=symbol,
                lookback=request.lookback,
                horizon_days=request.horizon_days,
                extraction_date=request.extraction_date,
                generated_at_token=generated_at_token,
            )
            local_path = self._local_store.write_frame(future_predict_frame, relative_path)
            s3_uri = None
            if request.upload_to_s3 and self._s3_store is not None:
                s3_uri = self._s3_store.upload_dataframe(
                    frame=future_predict_frame,
                    relative_path=relative_path,
                )

            chart_relative_path = relative_path.with_name("forecast_chart.svg")
            chart_local_path = self._write_forecast_chart(
                frame=future_predict_frame,
                relative_path=chart_relative_path,
                symbol=symbol,
            )
            chart_s3_uri = None
            if request.upload_to_s3 and self._s3_store is not None:
                chart_s3_uri = self._s3_store.upload_file(
                    local_path=chart_local_path,
                    relative_path=chart_relative_path,
                )

            forecast_dates = pd.to_datetime(future_predict_frame["forecast_date"])
            report_relative_path = relative_path.with_name("forecast_report.md")
            report_local_path = self._write_forecast_report(
                frame=future_predict_frame,
                relative_path=report_relative_path,
                symbol=symbol,
                request=request,
                generated_at_utc=generated_at_utc,
                forecast_summary=forecast_summary,
                normal_model_local_path=normal_model_local_path,
                quantum_model_local_path=quantum_model_local_path,
                chart_local_path=chart_local_path,
            )
            report_s3_uri = None
            if request.upload_to_s3 and self._s3_store is not None:
                report_s3_uri = self._s3_store.upload_file(
                    local_path=report_local_path,
                    relative_path=report_relative_path,
                )
            artifacts.append(
                ForecastAssetArtifact(
                    symbol=symbol.upper(),
                    row_count=int(len(future_predict_frame.index)),
                    predict_types=tuple(predict_types),
                    forecast_start_date=forecast_dates.iloc[0].strftime("%Y-%m-%d"),
                    forecast_end_date=forecast_dates.iloc[-1].strftime("%Y-%m-%d"),
                    last_observed_date=str(future_predict_frame["last_observed_date"].iloc[0]),
                    last_observed_close=float(
                        future_predict_frame["last_observed_close"].iloc[0]
                    ),
                    normal_model_local_path=normal_model_local_path,
                    quantum_model_local_path=quantum_model_local_path,
                    normal_elapsed_ms=normal_elapsed_ms,
                    normal_per_step_ms=normal_per_step_ms,
                    quantum_elapsed_ms=quantum_elapsed_ms,
                    quantum_per_step_ms=quantum_per_step_ms,
                    timing_ratio=timing_ratio,
                    forecast_summary=forecast_summary,
                    local_path=str(local_path),
                    s3_uri=s3_uri,
                    report_local_path=str(report_local_path),
                    report_s3_uri=report_s3_uri,
                    chart_local_path=str(chart_local_path),
                    chart_s3_uri=chart_s3_uri,
                )
            )

        unified_report_relative_path = self._build_unified_report_relative_path(
            extraction_date=request.extraction_date,
            generated_at_token=generated_at_token,
        )
        unified_report_local_path = self._write_unified_forecast_report(
            relative_path=unified_report_relative_path,
            request=request,
            generated_at_utc=generated_at_utc,
            assets=artifacts,
        )
        unified_latest_report_relative_path = Path("future_predict") / "unified_forecast_report.md"
        unified_latest_report_local_path = self._write_unified_forecast_report(
            relative_path=unified_latest_report_relative_path,
            request=request,
            generated_at_utc=generated_at_utc,
            assets=artifacts,
        )
        unified_report_s3_uri = None
        if request.upload_to_s3 and self._s3_store is not None:
            unified_report_s3_uri = self._s3_store.upload_file(
                local_path=unified_report_local_path,
                relative_path=unified_report_relative_path,
            )

        manifest_payload = {
            "source": request.source,
            "generated_at_utc": generated_at_utc,
            "unified_report_local_path": str(unified_report_local_path),
            "unified_latest_report_local_path": str(unified_latest_report_local_path),
            "unified_report_s3_uri": unified_report_s3_uri,
            "request": {
                "symbols": list(request.symbols),
                "extraction_date": request.extraction_date.isoformat(),
                "source": request.source,
                "target_column": request.target_column,
                "lookback": request.lookback,
                "horizon_days": request.horizon_days,
                "model_name_prefix": request.model_name_prefix,
                "quantum_model_name_prefix": request.quantum_model_name_prefix,
                "include_normal": request.include_normal,
                "include_quantum": request.include_quantum,
                "upload_to_s3": request.upload_to_s3,
            },
            "asset_count": len(artifacts),
            "assets": [asdict(artifact) for artifact in artifacts],
        }
        manifest_relative_path = self._build_manifest_relative_path(
            extraction_date=request.extraction_date,
            generated_at_token=generated_at_token,
        )
        manifest_local_path = self._local_store.write_json(
            manifest_payload,
            manifest_relative_path,
        )
        manifest_s3_uri = None
        if request.upload_to_s3 and self._s3_store is not None:
            manifest_s3_uri = self._s3_store.upload_file(
                local_path=manifest_local_path,
                relative_path=manifest_relative_path,
            )

        return ForecastBatchResult(
            source=request.source,
            generated_at_utc=generated_at_utc,
            manifest_local_path=str(manifest_local_path),
            manifest_s3_uri=manifest_s3_uri,
            unified_report_local_path=str(unified_report_local_path),
            unified_latest_report_local_path=str(unified_latest_report_local_path),
            unified_report_s3_uri=unified_report_s3_uri,
            assets=tuple(artifacts),
        )

    def _ensure_tensorflow_available(self) -> None:
        if _TENSORFLOW_IMPORT_ERROR is not None:
            raise RuntimeError(
                "TensorFlow is required for normal forecast generation. "
                "Install the project dependencies again after adding TensorFlow."
            ) from _TENSORFLOW_IMPORT_ERROR

    def _ensure_quantum_dependencies_available(self) -> None:
        if _QISKIT_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Qiskit Runtime and Qiskit Machine Learning are required for "
                "quantum forecast generation."
            ) from _QISKIT_IMPORT_ERROR

    def _validate_request(self, request: ForecastBatchRequest) -> None:
        if not request.symbols:
            raise ValueError("At least one symbol must be provided.")
        if request.lookback <= 0:
            raise ValueError("lookback must be greater than zero.")
        if request.horizon_days <= 0:
            raise ValueError("horizon_days must be greater than zero.")
        if not request.include_normal and not request.include_quantum:
            raise ValueError("At least one predict type must be enabled.")
        if request.include_normal and not request.model_name_prefix.strip():
            raise ValueError("model_name_prefix must not be blank.")
        if request.include_quantum and not request.quantum_model_name_prefix.strip():
            raise ValueError("quantum_model_name_prefix must not be blank.")
        if request.quantum_runtime_mode not in {"local", "cloud"}:
            raise ValueError("quantum_runtime_mode must be either 'local' or 'cloud'.")
        if request.quantum_shots <= 0:
            raise ValueError("quantum_shots must be greater than zero.")
        if request.quantum_optimization_level not in {0, 1, 2, 3}:
            raise ValueError("quantum_optimization_level must be 0, 1, 2, or 3.")
        if request.max_cloud_quantum_predictions <= 0:
            raise ValueError("max_cloud_quantum_predictions must be greater than zero.")
        if request.include_quantum and request.quantum_runtime_mode == "cloud":
            requested_predictions = len(request.symbols) * request.horizon_days
            if not request.confirm_ibm_runtime_cost:
                raise ValueError(
                    "Cloud quantum prediction requires --confirm-ibm-runtime-cost. "
                    "Training must stay off IBM Quantum; only forecast inference may "
                    "submit Runtime jobs."
                )
            if requested_predictions > request.max_cloud_quantum_predictions:
                raise ValueError(
                    "Refusing to submit "
                    f"{requested_predictions} cloud quantum predictions. "
                    "Lower --horizon-days/--symbols or raise "
                    "--max-cloud-quantum-predictions explicitly."
                )

    def _load_raw_frame(
        self,
        *,
        source: str,
        symbol: str,
        extraction_date: date,
        target_column: str,
    ) -> pd.DataFrame:
        relative_path = (
            Path("market_data")
            / f"source={source}"
            / f"symbol={symbol.upper()}"
            / f"extraction_date={extraction_date.isoformat()}"
            / "ohlcv.csv"
        )
        raw_path = self._raw_root_dir / relative_path
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw input not found for symbol {symbol!r}: {raw_path}"
            )

        frame = pd.read_csv(raw_path, parse_dates=["date"])
        if "date" not in frame.columns:
            raise ValueError(f"Raw input for symbol {symbol!r} does not contain `date`.")
        if target_column not in frame.columns:
            raise ValueError(
                f"Raw input for symbol {symbol!r} does not contain "
                f"target column {target_column!r}."
            )
        return frame

    def _load_scaler_metadata(
        self,
        *,
        source: str,
        symbol: str,
        extraction_date: date,
        lookback: int,
        target_column: str,
    ) -> dict[str, float]:
        manifest_path = (
            self._processed_root_dir
            / "manifests"
            / f"extraction_date={extraction_date.isoformat()}"
            / "refined_manifest.json"
        )
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Refined manifest not found: {manifest_path}. "
                "Run `python scripts/generate_refined.py --skip-s3` first."
            )

        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_request = manifest_payload.get("request", {})
        if manifest_request.get("source") != source:
            raise ValueError(
                f"Refined manifest source {manifest_request.get('source')!r} "
                f"does not match requested source {source!r}."
            )
        if manifest_request.get("target_column") != target_column:
            raise ValueError(
                f"Refined manifest target_column {manifest_request.get('target_column')!r} "
                f"does not match requested target_column {target_column!r}."
            )

        asset_payload = next(
            (
                asset
                for asset in manifest_payload.get("assets", [])
                if asset.get("symbol", "").upper() == symbol.upper()
                and int(asset.get("feature_count", lookback)) == lookback
            ),
            None,
        )
        if asset_payload is None:
            raise ValueError(
                f"Could not find refined scaler metadata for symbol {symbol!r} "
                f"with lookback={lookback} in {manifest_path}."
            )

        return {
            "min_offset": float(asset_payload["scaler_min_offset"]),
            "scale": float(asset_payload["scaler_scale"]),
            "data_min": float(asset_payload.get("data_min", 0.0)),
            "data_max": float(asset_payload.get("data_max", 0.0)),
        }

    def _resolve_keras_model_metadata(
        self,
        *,
        symbol: str,
        extraction_date: date,
        source: str,
        target_column: str,
        lookback: int,
        model_name_prefix: str,
    ) -> dict[str, str | None]:
        expected_model_name = f"{model_name_prefix}_{symbol.lower()}.keras"
        promoted_metadata = self._resolve_promoted_keras_model_metadata(
            symbol=symbol,
            requested_extraction_date=extraction_date,
            source=source,
            target_column=target_column,
            lookback=lookback,
            expected_model_name=expected_model_name,
        )
        if promoted_metadata is not None:
            return promoted_metadata

        manifests_root = (
            self._models_root_dir
            / "manifests"
            / f"extraction_date={extraction_date.isoformat()}"
        )
        if manifests_root.exists():
            best_candidate: tuple[
                tuple[float, float, float, float, int],
                Path,
                dict[str, Any],
                dict[str, Any],
            ] | None = None
            candidate_paths = manifests_root.glob("trained_at=*/keras_training_manifest.json")
            for manifest_path in candidate_paths:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                request_payload = payload.get("request", {})
                if (
                    request_payload.get("source") != source
                    or request_payload.get("target_column") != target_column
                    or int(request_payload.get("lookback", lookback)) != lookback
                ):
                    continue

                asset_payload = next(
                    (
                        asset
                        for asset in payload.get("assets", [])
                        if asset.get("symbol", "").upper() == symbol.upper()
                        and Path(str(asset.get("model_local_path", ""))).name.lower()
                        == expected_model_name.lower()
                    ),
                    None,
                )
                if asset_payload is None:
                    continue

                candidate_score = self._score_regression_asset(
                    asset_payload=asset_payload,
                    manifest_path=manifest_path,
                )
                if best_candidate is None or candidate_score < best_candidate[0]:
                    best_candidate = (
                        candidate_score,
                        manifest_path,
                        payload,
                        asset_payload,
                    )

            if best_candidate is not None:
                _, manifest_path, payload, asset_payload = best_candidate
                return {
                    "model_local_path": str(asset_payload["model_local_path"]),
                    "training_manifest_local_path": str(manifest_path),
                    "training_generated_at_utc": str(payload.get("generated_at_utc")),
                    "prediction_target_mode": str(
                        payload.get("request", {}).get("prediction_target_mode", "price")
                    ),
                }

        fallback_path = self._models_root_dir / expected_model_name
        if fallback_path.exists():
            return {
                "model_local_path": str(fallback_path),
                "training_manifest_local_path": None,
                "training_generated_at_utc": None,
                "prediction_target_mode": "price",
            }

        raise FileNotFoundError(
            f"Could not find a trained Keras model for symbol {symbol!r}. "
            f"Expected manifest under {manifests_root} or fallback model {fallback_path}."
        )

    def _resolve_promoted_keras_model_metadata(
        self,
        *,
        symbol: str,
        requested_extraction_date: date,
        source: str,
        target_column: str,
        lookback: int,
        expected_model_name: str,
    ) -> dict[str, str | None] | None:
        if not self._promotion_registry.exists():
            return None

        for promotion in self._promotion_registry.list_candidates(
            symbol=symbol,
            requested_extraction_date=requested_extraction_date,
        ):
            manifest_path = promotion.manifest_local_path
            model_path = promotion.model_local_path
            if not manifest_path.exists() or not model_path.exists():
                continue
            if model_path.name.lower() != expected_model_name.lower():
                continue

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            request_payload = payload.get("request", {})
            if (
                request_payload.get("source") != source
                or request_payload.get("target_column") != target_column
                or int(request_payload.get("lookback", lookback)) != lookback
            ):
                continue

            asset_payload = next(
                (
                    asset
                    for asset in payload.get("assets", [])
                    if str(asset.get("symbol", "")).upper() == symbol.upper()
                    and Path(str(asset.get("model_local_path", ""))).name.lower()
                    == expected_model_name.lower()
                ),
                None,
            )
            if asset_payload is None:
                continue

            return {
                "model_local_path": str(model_path),
                "training_manifest_local_path": str(manifest_path),
                "training_generated_at_utc": str(payload.get("generated_at_utc")),
                "prediction_target_mode": str(
                    payload.get("request", {}).get("prediction_target_mode", "price")
                ),
            }

        return None

    def _resolve_quantum_model_metadata(
        self,
        *,
        symbol: str,
        extraction_date: date,
        source: str,
        target_column: str,
        lookback: int,
        model_name_prefix: str,
    ) -> dict[str, str | None]:
        expected_model_name = f"{model_name_prefix}_{symbol.lower()}.json"
        manifests_root = (
            self._models_root_dir
            / "manifests"
            / f"extraction_date={extraction_date.isoformat()}"
        )
        if manifests_root.exists():
            candidate_paths = sorted(
                manifests_root.glob("trained_at=*/quantum_training_manifest.json"),
                reverse=True,
            )
            for manifest_path in candidate_paths:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                request_payload = payload.get("request", {})
                if (
                    request_payload.get("source") != source
                    or request_payload.get("target_column") != target_column
                    or int(request_payload.get("lookback", lookback)) != lookback
                ):
                    continue

                asset_payload = next(
                    (
                        asset
                        for asset in payload.get("assets", [])
                        if asset.get("symbol", "").upper() == symbol.upper()
                        and Path(str(asset.get("model_local_path", ""))).name.lower()
                        == expected_model_name.lower()
                    ),
                    None,
                )
                if asset_payload is None:
                    continue

                return {
                    "model_local_path": str(asset_payload["model_local_path"]),
                    "preprocessor_local_path": str(asset_payload["preprocessor_local_path"]),
                    "training_manifest_local_path": str(manifest_path),
                    "training_generated_at_utc": str(payload.get("generated_at_utc")),
                }

        fallback_model_path = self._models_root_dir / expected_model_name
        fallback_preprocessor_root = (
            self._models_root_dir
            / "quantum_training_runs"
            / f"source={source}"
            / f"symbol={symbol.upper()}"
            / f"lookback={lookback}"
            / f"extraction_date={extraction_date.isoformat()}"
        )
        if fallback_model_path.exists() and fallback_preprocessor_root.exists():
            trained_dirs = sorted(fallback_preprocessor_root.glob("trained_at=*"))
            if trained_dirs:
                latest_trained_dir = trained_dirs[-1]
                preprocessor_path = latest_trained_dir / "preprocessor.joblib"
                if preprocessor_path.exists():
                    return {
                        "model_local_path": str(fallback_model_path),
                        "preprocessor_local_path": str(preprocessor_path),
                        "training_manifest_local_path": None,
                        "training_generated_at_utc": None,
                    }

        raise FileNotFoundError(
            f"Could not find a trained quantum model for symbol {symbol!r}. "
            f"Expected manifest under {manifests_root} or fallback model {fallback_model_path}."
        )

    def _build_normal_rows(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        rows, _ = self._build_normal_rows_timed(**kwargs)
        return rows

    def _build_normal_rows_timed(
        self,
        *,
        raw_frame: pd.DataFrame,
        request: ForecastBatchRequest,
        symbol: str,
        generated_at_utc: str,
        generated_at_token: str,
        model: Any,
        scaler_metadata: dict[str, float],
        model_metadata: dict[str, str | None],
    ) -> tuple[list[dict[str, Any]], list[float]]:
        working = raw_frame.loc[:, ["date", request.target_column]].dropna().copy()
        working["date"] = pd.to_datetime(working["date"])
        working = working.sort_values("date").reset_index(drop=True)

        if len(working.index) < request.lookback:
            raise ValueError(
                f"Not enough rows to build a forecast window for symbol {symbol!r}. "
                f"Need at least {request.lookback} rows."
            )

        raw_window = working[request.target_column].tail(request.lookback).to_numpy(dtype=np.float32)
        scaled_window = self._scale_array(
            raw_window,
            min_offset=scaler_metadata["min_offset"],
            scale=scaler_metadata["scale"],
        ).astype(np.float32)
        window_dates = list(working["date"].tail(request.lookback))

        last_observed_date = pd.Timestamp(window_dates[-1])
        last_observed_close = float(raw_window[-1])
        forecast_dates = pd.bdate_range(
            last_observed_date + pd.offsets.BDay(1),
            periods=request.horizon_days,
        )

        rows: list[dict[str, Any]] = []
        per_step_ms: list[float] = []
        for step, forecast_date in enumerate(forecast_dates, start=1):
            step_start = time.perf_counter()
            input_window_start_date = pd.Timestamp(window_dates[0])
            input_window_end_date = pd.Timestamp(window_dates[-1])
            input_window_end_close = float(raw_window[-1])

            prediction_input = scaled_window.reshape(1, request.lookback, 1)
            predicted_scaled = float(model.predict(prediction_input, verbose=0).reshape(-1)[0])
            model_prediction_target_mode = str(
                model_metadata.get("prediction_target_mode") or "price"
            )
            if model_prediction_target_mode == "return":
                predicted_close = float(input_window_end_close * (1.0 + predicted_scaled))
            else:
                predicted_close = float(
                    self._inverse_scale_array(
                        np.asarray([predicted_scaled], dtype=np.float32),
                        min_offset=scaler_metadata["min_offset"],
                        scale=scaler_metadata["scale"],
                    )[0]
                )
            guardrail = apply_standard_forecast_guardrail(
                raw_model_close=predicted_close,
                current_close=input_window_end_close,
                recent_window=raw_window.astype(np.float64),
            )
            stabilized = self._stabilize_forecast_step(
                guardrail=guardrail,
                recent_window=raw_window.astype(np.float64),
                last_observed_close=last_observed_close,
                forecast_step=step,
                horizon_days=request.horizon_days,
                scaler_metadata=scaler_metadata,
            )
            predicted_close = stabilized.predicted_close
            predicted_scaled = float(stabilized.predicted_scaled or 0.0)
            predicted_direction = int(predicted_close > input_window_end_close)
            predicted_step_return = self._safe_return(
                current_value=predicted_close,
                reference_value=input_window_end_close,
            )
            horizon_return = self._safe_return(
                current_value=predicted_close,
                reference_value=last_observed_close,
            )

            observed_points_in_window = max(request.lookback - (step - 1), 0)
            predicted_points_in_window = min(step - 1, request.lookback)
            step_elapsed_ms = (time.perf_counter() - step_start) * 1_000
            per_step_ms.append(step_elapsed_ms)
            row = self._build_future_predict_row(
                source=request.source,
                symbol=symbol,
                target_column=request.target_column,
                extraction_date=request.extraction_date,
                generated_at_utc=generated_at_utc,
                generated_at_token=generated_at_token,
                predict_type="normal",
                model_family="keras_lstm_regression",
                model_name=Path(str(model_metadata["model_local_path"])).stem,
                model_local_path=str(model_metadata["model_local_path"]),
                training_manifest_local_path=model_metadata["training_manifest_local_path"],
                training_generated_at_utc=model_metadata["training_generated_at_utc"],
                lookback=request.lookback,
                horizon_days=request.horizon_days,
                forecast_step=step,
                forecast_date=forecast_date,
                predicted_close=predicted_close,
                predicted_scaled=predicted_scaled,
                predicted_direction=predicted_direction,
                raw_model_predicted_close=stabilized.raw_model_close,
                prediction_constraint_applied=stabilized.prediction_constraint_applied,
                prediction_constraint_method=stabilized.prediction_constraint_method,
                prediction_return_cap=stabilized.prediction_return_cap,
                hit_lower_band=stabilized.hit_lower_band,
                hit_upper_band=stabilized.hit_upper_band,
                dynamic_cumulative_return_cap=stabilized.dynamic_cumulative_return_cap,
                predicted_step_return=predicted_step_return,
                horizon_return_from_last_observed=horizon_return,
                price_proxy_return=None,
                last_observed_date=last_observed_date,
                last_observed_close=last_observed_close,
                input_window_start_date=input_window_start_date,
                input_window_end_date=input_window_end_date,
                input_window_end_close=input_window_end_close,
                input_window_end_origin="observed" if step == 1 else "predicted",
                observed_points_in_window=observed_points_in_window,
                predicted_points_in_window=predicted_points_in_window,
                is_price_proxy=False,
                price_proxy_method=None,
            )
            row["model_prediction_target_mode"] = model_prediction_target_mode
            row["step_elapsed_ms"] = round(step_elapsed_ms, 3)
            rows.append(row)

            raw_window = np.concatenate(
                [raw_window[1:], np.asarray([predicted_close], dtype=np.float32)]
            )
            scaled_window = np.concatenate(
                [scaled_window[1:], np.asarray([predicted_scaled], dtype=np.float32)]
            )
            window_dates = window_dates[1:] + [forecast_date]

        return rows, per_step_ms

    def _build_quantum_rows(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        rows, _ = self._build_quantum_rows_timed(**kwargs)
        return rows

    def _build_quantum_rows_timed(
        self,
        *,
        raw_frame: pd.DataFrame,
        request: ForecastBatchRequest,
        symbol: str,
        generated_at_utc: str,
        generated_at_token: str,
        model: VQC,
        scaler_metadata: dict[str, float],
        quantum_bundle: dict[str, Any],
        model_metadata: dict[str, str | None],
    ) -> tuple[list[dict[str, Any]], list[float]]:
        working = raw_frame.loc[:, ["date", request.target_column]].dropna().copy()
        working["date"] = pd.to_datetime(working["date"])
        working = working.sort_values("date").reset_index(drop=True)

        if len(working.index) < request.lookback:
            raise ValueError(
                f"Not enough rows to build a forecast window for symbol {symbol!r}. "
                f"Need at least {request.lookback} rows."
            )

        raw_window = working[request.target_column].tail(request.lookback).to_numpy(dtype=np.float64)
        window_dates = list(working["date"].tail(request.lookback))

        last_observed_date = pd.Timestamp(window_dates[-1])
        last_observed_close = float(raw_window[-1])
        forecast_dates = pd.bdate_range(
            last_observed_date + pd.offsets.BDay(1),
            periods=request.horizon_days,
        )

        rows: list[dict[str, Any]] = []
        per_step_ms: list[float] = []
        for step, forecast_date in enumerate(forecast_dates, start=1):
            step_start = time.perf_counter()
            input_window_start_date = pd.Timestamp(window_dates[0])
            input_window_end_date = pd.Timestamp(window_dates[-1])
            input_window_end_close = float(raw_window[-1])

            feature_vector = self._build_quantum_feature_vector(
                raw_window=raw_window,
                feature_columns=list(quantum_bundle["feature_columns"]),
                scaler_metadata=scaler_metadata,
                target_column=request.target_column,
                lookback=request.lookback,
            )
            transformed_vector = self._transform_quantum_feature_vector(
                values=feature_vector,
                bundle=quantum_bundle,
            )
            predicted_direction = self._predict_quantum_direction(
                model=model,
                transformed_values=transformed_vector,
            )
            proxy_return = self._compute_quantum_price_proxy_return(
                raw_window,
                forecast_step=step,
                horizon_days=request.horizon_days,
            )
            raw_proxy_close = float(
                input_window_end_close
                * (1.0 + proxy_return if predicted_direction == 1 else 1.0 - proxy_return)
            )
            guardrail = apply_standard_forecast_guardrail(
                raw_model_close=raw_proxy_close,
                current_close=input_window_end_close,
                recent_window=raw_window.astype(np.float64),
                volatility_multiplier=2.0,
                max_return_cap=0.035,
            )
            stabilized = self._stabilize_forecast_step(
                guardrail=guardrail,
                recent_window=raw_window.astype(np.float64),
                last_observed_close=last_observed_close,
                forecast_step=step,
                horizon_days=request.horizon_days,
                scaler_metadata=None,
            )
            predicted_close = stabilized.predicted_close
            predicted_direction = int(predicted_close > input_window_end_close)
            predicted_step_return = self._safe_return(
                current_value=predicted_close,
                reference_value=input_window_end_close,
            )
            horizon_return = self._safe_return(
                current_value=predicted_close,
                reference_value=last_observed_close,
            )

            observed_points_in_window = max(request.lookback - (step - 1), 0)
            predicted_points_in_window = min(step - 1, request.lookback)
            step_elapsed_ms = (time.perf_counter() - step_start) * 1_000
            per_step_ms.append(step_elapsed_ms)
            row = self._build_future_predict_row(
                source=request.source,
                symbol=symbol,
                target_column=request.target_column,
                extraction_date=request.extraction_date,
                generated_at_utc=generated_at_utc,
                generated_at_token=generated_at_token,
                predict_type="quant",
                model_family="quantum_vqc_direction",
                model_name=Path(str(model_metadata["model_local_path"])).stem,
                model_local_path=str(model_metadata["model_local_path"]),
                training_manifest_local_path=model_metadata["training_manifest_local_path"],
                training_generated_at_utc=model_metadata["training_generated_at_utc"],
                lookback=request.lookback,
                horizon_days=request.horizon_days,
                forecast_step=step,
                forecast_date=forecast_date,
                predicted_close=predicted_close,
                predicted_scaled=None,
                predicted_direction=predicted_direction,
                raw_model_predicted_close=stabilized.raw_model_close,
                prediction_constraint_applied=stabilized.prediction_constraint_applied,
                prediction_constraint_method=stabilized.prediction_constraint_method,
                prediction_return_cap=stabilized.prediction_return_cap,
                hit_lower_band=stabilized.hit_lower_band,
                hit_upper_band=stabilized.hit_upper_band,
                dynamic_cumulative_return_cap=stabilized.dynamic_cumulative_return_cap,
                predicted_step_return=predicted_step_return,
                horizon_return_from_last_observed=horizon_return,
                price_proxy_return=proxy_return,
                last_observed_date=last_observed_date,
                last_observed_close=last_observed_close,
                input_window_start_date=input_window_start_date,
                input_window_end_date=input_window_end_date,
                input_window_end_close=input_window_end_close,
                input_window_end_origin="observed" if step == 1 else "predicted",
                observed_points_in_window=observed_points_in_window,
                predicted_points_in_window=predicted_points_in_window,
                is_price_proxy=True,
                price_proxy_method=(
                    "directional_return_proxy_shrunk_mean_abs_20d_from_quant_signal"
                ),
            )
            row["step_elapsed_ms"] = round(step_elapsed_ms, 3)
            rows.append(row)

            raw_window = np.concatenate(
                [raw_window[1:], np.asarray([predicted_close], dtype=np.float64)]
            )
            window_dates = window_dates[1:] + [forecast_date]

        return rows, per_step_ms

    def _build_quantum_predictor(
        self,
        *,
        model_payload: dict[str, Any],
        request: ForecastBatchRequest,
    ) -> VQC:
        runtime_context = self._build_quantum_runtime_context(
            model_payload=model_payload,
            request=request,
        )
        feature_map = zz_feature_map(
            feature_dimension=int(model_payload["num_qubits"]),
            reps=int(model_payload["feature_map"]["reps"]),
            entanglement=str(model_payload["feature_map"]["entanglement"]),
        )
        ansatz = real_amplitudes(
            num_qubits=int(model_payload["num_qubits"]),
            reps=int(model_payload["ansatz"]["reps"]),
        )
        model = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            sampler=runtime_context["sampler"],
            pass_manager=runtime_context["pass_manager"],
            optimizer=COBYLA(maxiter=1),
        )
        model._fit_result = SimpleNamespace(
            x=np.asarray(model_payload["weights"], dtype=np.float64)
        )
        return model

    def _build_quantum_runtime_context(
        self,
        *,
        model_payload: dict[str, Any],
        request: ForecastBatchRequest,
    ) -> dict[str, Any]:
        execution_mode = request.quantum_runtime_mode
        backend_name = request.quantum_backend_name or str(
            model_payload.get("backend_name", "fake_manila")
        )
        if execution_mode == "cloud" and backend_name.startswith("fake_"):
            backend_name = ""

        if execution_mode == "local":
            backend = FakeManilaV2()
            sampler = Sampler(mode=backend, options={"simulator": {"seed_simulator": 42}})
            sampler.options.default_shots = request.quantum_shots
            pass_manager = generate_preset_pass_manager(
                backend=backend,
                optimization_level=request.quantum_optimization_level,
            )
            return {
                "backend": backend,
                "sampler": sampler,
                "pass_manager": pass_manager,
            }

        service = self._build_cloud_service()
        backend = service.backend(backend_name) if backend_name else service.least_busy(
            operational=True,
            simulator=False,
        )
        sampler = Sampler(mode=backend)
        sampler.options.default_shots = request.quantum_shots
        pass_manager = generate_preset_pass_manager(
            backend=backend,
            optimization_level=request.quantum_optimization_level,
        )
        return {
            "backend": backend,
            "sampler": sampler,
            "pass_manager": pass_manager,
        }

    def _build_cloud_service(self) -> QiskitRuntimeService:
        load_env_file()
        token = os.getenv("IBM_QUANTUM_API_TOKEN")
        instance = os.getenv("IBM_QUANTUM_INSTANCE")

        if token:
            kwargs: dict[str, Any] = {
                "channel": "ibm_quantum_platform",
                "token": token,
            }
            if instance:
                kwargs["instance"] = instance
            return QiskitRuntimeService(**kwargs)

        return QiskitRuntimeService()

    def _build_quantum_feature_vector(
        self,
        *,
        raw_window: np.ndarray,
        feature_columns: list[str],
        scaler_metadata: dict[str, float],
        target_column: str,
        lookback: int,
    ) -> np.ndarray:
        feature_values: dict[str, float] = {}
        feature_values.update(self._build_engineered_feature_map(raw_window))

        scaled_window = self._scale_array(
            raw_window.astype(np.float64),
            min_offset=scaler_metadata["min_offset"],
            scale=scaler_metadata["scale"],
        )
        for position, value in enumerate(scaled_window):
            lag = lookback - position
            feature_values[f"{target_column}_t_minus_{lag}"] = float(value)

        missing_columns = [column for column in feature_columns if column not in feature_values]
        if missing_columns:
            raise ValueError(
                "Could not build all required quantum feature columns for inference: "
                f"{missing_columns}"
            )

        return np.asarray(
            [[feature_values[column] for column in feature_columns]],
            dtype=np.float64,
        )

    def _build_engineered_feature_map(self, raw_window: np.ndarray) -> dict[str, float]:
        feature_engineer = GenerateFeatureDatasetUseCase
        current_price = float(raw_window[-1])
        daily_returns = feature_engineer._compute_daily_returns(raw_window)  # type: ignore[attr-defined]
        sma_5 = feature_engineer._compute_sma(raw_window, 5)  # type: ignore[attr-defined]
        sma_10 = feature_engineer._compute_sma(raw_window, 10)  # type: ignore[attr-defined]
        sma_20 = feature_engineer._compute_sma(raw_window, 20)  # type: ignore[attr-defined]

        return {
            "feature_current_price": current_price,
            "feature_window_mean": float(np.mean(raw_window)),
            "feature_window_std": float(np.std(raw_window)),
            "feature_window_min": float(np.min(raw_window)),
            "feature_window_max": float(np.max(raw_window)),
            "feature_window_range": float(np.max(raw_window) - np.min(raw_window)),
            "feature_return_1d": feature_engineer._compute_window_return(raw_window, 1),  # type: ignore[attr-defined]
            "feature_return_5d": feature_engineer._compute_window_return(raw_window, 5),  # type: ignore[attr-defined]
            "feature_return_10d": feature_engineer._compute_window_return(raw_window, 10),  # type: ignore[attr-defined]
            "feature_return_20d": feature_engineer._compute_window_return(raw_window, 20),  # type: ignore[attr-defined]
            "feature_sma_gap_5d": feature_engineer._compute_gap_ratio(current_price, sma_5),  # type: ignore[attr-defined]
            "feature_sma_gap_10d": feature_engineer._compute_gap_ratio(current_price, sma_10),  # type: ignore[attr-defined]
            "feature_sma_gap_20d": feature_engineer._compute_gap_ratio(current_price, sma_20),  # type: ignore[attr-defined]
            "feature_ema_gap_5d": feature_engineer._compute_gap_ratio(  # type: ignore[attr-defined]
                current_price,
                feature_engineer._compute_ema(raw_window, 5),  # type: ignore[attr-defined]
            ),
            "feature_ema_gap_10d": feature_engineer._compute_gap_ratio(  # type: ignore[attr-defined]
                current_price,
                feature_engineer._compute_ema(raw_window, 10),  # type: ignore[attr-defined]
            ),
            "feature_volatility_5d": feature_engineer._compute_volatility(daily_returns, 5),  # type: ignore[attr-defined]
            "feature_volatility_10d": feature_engineer._compute_volatility(daily_returns, 10),  # type: ignore[attr-defined]
            "feature_trend_slope_10d": feature_engineer._compute_trend_slope(raw_window, 10),  # type: ignore[attr-defined]
            "feature_trend_slope_20d": feature_engineer._compute_trend_slope(raw_window, 20),  # type: ignore[attr-defined]
            "feature_up_day_ratio_5d": feature_engineer._compute_up_day_ratio(daily_returns, 5),  # type: ignore[attr-defined]
            "feature_up_day_ratio_10d": feature_engineer._compute_up_day_ratio(daily_returns, 10),  # type: ignore[attr-defined]
            "feature_position_in_window": feature_engineer._compute_position_in_window(raw_window),  # type: ignore[attr-defined]
            "feature_window_max_drawdown": feature_engineer._compute_max_drawdown(raw_window),  # type: ignore[attr-defined]
        }

    @staticmethod
    def _transform_quantum_feature_vector(
        *,
        values: np.ndarray,
        bundle: dict[str, Any],
    ) -> np.ndarray:
        values_standard = bundle["standard_scaler"].transform(values)
        values_reduced = bundle["pca"].transform(values_standard)
        return bundle["angle_scaler"].transform(values_reduced)

    @staticmethod
    def _predict_quantum_direction(
        *,
        model: VQC,
        transformed_values: np.ndarray,
    ) -> int:
        prediction = np.asarray(model.predict(transformed_values))
        if prediction.ndim == 2 and prediction.shape[1] > 1:
            return int(np.argmax(prediction, axis=1)[0])

        flattened = prediction.reshape(-1)
        value = float(flattened[0])
        if value in {0.0, 1.0}:
            return int(value)
        return int(value > 0.0)

    def _compute_quantum_price_proxy_return(
        self,
        raw_window: np.ndarray,
        *,
        forecast_step: int = 1,
        horizon_days: int = 30,
    ) -> float:
        feature_engineer = GenerateFeatureDatasetUseCase
        daily_returns = feature_engineer._compute_daily_returns(raw_window)  # type: ignore[attr-defined]
        if len(daily_returns) == 0:
            return 0.003

        effective_periods = min(20, len(daily_returns))
        proxy_return = float(np.mean(np.abs(daily_returns[-effective_periods:])))
        long_horizon_factor = 1.0 / (1.0 + 0.04 * max(forecast_step - 1, 0))
        horizon_factor = min(1.0, np.sqrt(30.0 / max(float(horizon_days), 1.0)))
        adjusted_return = proxy_return * 0.60 * long_horizon_factor * horizon_factor
        dynamic_max = max(0.008, 0.025 * long_horizon_factor)
        return float(min(max(adjusted_return, 0.0005), dynamic_max))

    def _apply_cumulative_horizon_band(
        self,
        *,
        predicted_close: float,
        last_observed_close: float,
        recent_window: np.ndarray,
        forecast_step: int,
        horizon_days: int,
    ) -> tuple[float, bool, bool, bool, float]:
        if abs(last_observed_close) <= 1e-8:
            return float(predicted_close), False, False, False, 0.0
        max_cumulative_return = self._compute_dynamic_cumulative_return_cap(
            recent_window=recent_window,
            forecast_step=forecast_step,
            horizon_days=horizon_days,
        )
        lower_bound = max(last_observed_close * (1.0 - max_cumulative_return), 0.0)
        upper_bound = last_observed_close * (1.0 + max_cumulative_return)
        constrained = float(np.clip(predicted_close, lower_bound, upper_bound))
        applied = abs(constrained - predicted_close) > 1e-9
        return (
            constrained,
            applied,
            bool(applied and constrained <= lower_bound + 1e-9),
            bool(applied and constrained >= upper_bound - 1e-9),
            float(max_cumulative_return),
        )

    def _stabilize_forecast_step(
        self,
        *,
        guardrail: Any,
        recent_window: np.ndarray,
        last_observed_close: float,
        forecast_step: int,
        horizon_days: int,
        scaler_metadata: dict[str, float] | None,
    ) -> ForecastStepStabilizationResult:
        predicted_close = float(guardrail.constrained_close)
        (
            predicted_close,
            horizon_clamp_applied,
            hit_lower_band,
            hit_upper_band,
            dynamic_cumulative_return_cap,
        ) = self._apply_cumulative_horizon_band(
            predicted_close=predicted_close,
            last_observed_close=last_observed_close,
            recent_window=recent_window,
            forecast_step=forecast_step,
            horizon_days=horizon_days,
        )
        predicted_scaled = None
        if scaler_metadata is not None:
            predicted_scaled = float(
                self._scale_array(
                    np.asarray([predicted_close], dtype=np.float32),
                    min_offset=scaler_metadata["min_offset"],
                    scale=scaler_metadata["scale"],
                )[0]
            )
        return ForecastStepStabilizationResult(
            predicted_close=predicted_close,
            raw_model_close=float(guardrail.raw_model_close),
            predicted_scaled=predicted_scaled,
            prediction_return_cap=float(guardrail.return_cap),
            prediction_constraint_applied=bool(guardrail.applied or horizon_clamp_applied),
            prediction_constraint_method=self._join_constraint_methods(
                guardrail.method,
                "dynamic_cumulative_horizon_return_band" if horizon_clamp_applied else None,
            ),
            hit_lower_band=hit_lower_band,
            hit_upper_band=hit_upper_band,
            dynamic_cumulative_return_cap=dynamic_cumulative_return_cap,
        )

    @staticmethod
    def _compute_dynamic_cumulative_return_cap(
        *,
        recent_window: np.ndarray,
        forecast_step: int,
        horizon_days: int,
    ) -> float:
        daily_returns = GenerateFeatureDatasetUseCase._compute_daily_returns(recent_window)  # type: ignore[attr-defined]
        if len(daily_returns) == 0:
            realized_volatility = 0.012
            realized_move = 0.01
        else:
            effective_periods = min(60, len(daily_returns))
            effective_returns = daily_returns[-effective_periods:]
            realized_volatility = float(np.std(effective_returns, ddof=0))
            realized_move = float(np.mean(np.abs(effective_returns)))
        time_scale = np.sqrt(max(float(forecast_step), 1.0))
        horizon_scale = np.sqrt(max(float(horizon_days), 1.0) / 30.0)
        volatility_cap = (2.6 * realized_volatility + 1.2 * realized_move) * time_scale
        min_cap = min(0.22, 0.025 * time_scale * horizon_scale + 0.035)
        max_cap = min(0.65, 0.18 + 0.018 * max(float(horizon_days), 1.0))
        return float(min(max(volatility_cap, min_cap), max_cap))

    @staticmethod
    def _safe_return(*, current_value: float, reference_value: float) -> float:
        if abs(reference_value) <= 1e-8:
            return 0.0
        return float(current_value / reference_value - 1.0)

    @staticmethod
    def _join_constraint_methods(*methods: str | None) -> str | None:
        active_methods = [method for method in methods if method]
        return "+".join(active_methods) if active_methods else None

    @staticmethod
    def _summarize_future_predict_frame(
        frame: pd.DataFrame,
    ) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        if frame.empty:
            return summary
        for predict_type, group in frame.groupby("predict_type"):
            ordered = group.sort_values("forecast_step")
            directions = ordered["predicted_direction"].astype(int)
            closes = ordered["predicted_close"].astype(float)
            last_observed_close = float(ordered["last_observed_close"].iloc[0])
            horizon_delta = (
                float(closes.iloc[-1] / last_observed_close - 1.0)
                if abs(last_observed_close) > 1e-8
                else 0.0
            )
            step_returns = ordered["predicted_step_return"].astype(float)
            up_rate = float(directions.mean()) if len(directions) else 0.0
            close_diffs = closes.diff().dropna()
            close_range_pct = (
                float((closes.max() - closes.min()) / last_observed_close)
                if abs(last_observed_close) > 1e-8
                else 0.0
            )
            constraint_count = int(
                ordered["prediction_constraint_applied"].astype(bool).sum()
            )
            row_count = int(len(ordered.index))
            hit_lower_count = (
                int(ordered["hit_lower_band"].astype(bool).sum())
                if "hit_lower_band" in ordered.columns
                else 0
            )
            hit_upper_count = (
                int(ordered["hit_upper_band"].astype(bool).sum())
                if "hit_upper_band" in ordered.columns
                else 0
            )
            summary[str(predict_type)] = {
                "row_count": row_count,
                "up_rate": up_rate,
                "up_count": int(directions.sum()),
                "down_count": int((1 - directions).sum()),
                "horizon_delta_pct": float(horizon_delta * 100.0),
                "mean_abs_step_return_pct": float(step_returns.abs().mean() * 100.0),
                "std_step_return_pct": float(step_returns.std(ddof=0) * 100.0),
                "constraint_applied_count": constraint_count,
                "constraint_rate": float(constraint_count / row_count) if row_count else 0.0,
                "hit_lower_band": bool(hit_lower_count > 0),
                "hit_lower_band_count": hit_lower_count,
                "hit_upper_band": bool(hit_upper_count > 0),
                "hit_upper_band_count": hit_upper_count,
                "flat_path": bool(close_range_pct <= 0.005),
                "monotonic_path": bool(
                    len(close_diffs) > 0
                    and (
                        bool((close_diffs >= -1e-9).all())
                        or bool((close_diffs <= 1e-9).all())
                    )
                ),
                "degenerate_direction_path": bool(up_rate <= 0.05 or up_rate >= 0.95),
                "uses_price_proxy": bool(ordered["is_price_proxy"].astype(bool).any()),
            }
        return summary

    def _write_forecast_report(
        self,
        *,
        frame: pd.DataFrame,
        relative_path: Path,
        symbol: str,
        request: ForecastBatchRequest,
        generated_at_utc: str,
        forecast_summary: dict[str, dict[str, Any]],
        normal_model_local_path: str | None,
        quantum_model_local_path: str | None,
        chart_local_path: Path,
    ) -> Path:
        destination = self._local_store.root_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            self._build_forecast_report_markdown(
                frame=frame,
                symbol=symbol,
                request=request,
                generated_at_utc=generated_at_utc,
                forecast_summary=forecast_summary,
                normal_model_local_path=normal_model_local_path,
                quantum_model_local_path=quantum_model_local_path,
                chart_local_path=chart_local_path,
            ),
            encoding="utf-8",
        )
        return destination

    def _write_forecast_chart(
        self,
        *,
        frame: pd.DataFrame,
        relative_path: Path,
        symbol: str,
    ) -> Path:
        destination = self._local_store.root_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            self._build_forecast_chart_svg(frame=frame, symbol=symbol),
            encoding="utf-8",
        )
        return destination

    def _write_unified_forecast_report(
        self,
        *,
        relative_path: Path,
        request: ForecastBatchRequest,
        generated_at_utc: str,
        assets: list[ForecastAssetArtifact],
    ) -> Path:
        destination = self._local_store.root_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            self._build_unified_forecast_report_markdown(
                destination=destination,
                request=request,
                generated_at_utc=generated_at_utc,
                assets=assets,
            ),
            encoding="utf-8",
        )
        return destination

    def _build_forecast_report_markdown(
        self,
        *,
        frame: pd.DataFrame,
        symbol: str,
        request: ForecastBatchRequest,
        generated_at_utc: str,
        forecast_summary: dict[str, dict[str, Any]],
        normal_model_local_path: str | None,
        quantum_model_local_path: str | None,
        chart_local_path: Path,
    ) -> str:
        ordered = frame.sort_values(["predict_type", "forecast_step"]).reset_index(drop=True)
        last_observed_date = str(ordered["last_observed_date"].iloc[0])
        last_observed_close = float(ordered["last_observed_close"].iloc[0])
        forecast_start = str(ordered["forecast_date"].min())
        forecast_end = str(ordered["forecast_date"].max())
        summary_table = self._build_forecast_summary_table(forecast_summary)
        warning_lines = self._build_forecast_warning_lines(forecast_summary)
        endpoint_table = self._build_forecast_endpoint_table(ordered)
        visual_report = self._build_forecast_visual_report_markdown(
            frame=ordered,
            request=request,
        )
        timing_section = self._build_timing_report_section(
            forecast_summary=forecast_summary,
            steps=request.horizon_days,
        )

        return f"""# Forecast Report - `{symbol.upper()}`

> **Extraction date:** `{request.extraction_date.isoformat()}`
> **Generated at (UTC):** `{generated_at_utc}`
> **Forecast window:** `{forecast_start}` -> `{forecast_end}`
> **Last observed:** `{last_observed_date}` close=`{last_observed_close:.4f}`
> **Horizon days:** `{request.horizon_days}`

---

## Executive Summary

{visual_report}

{summary_table}

{warning_lines}

---

## Endpoint Check

{endpoint_table}

---

## Forecast Chart

![Forecast chart]({chart_local_path.name})

---

## Methodology Notes

- `normal` rows are recursive Keras LSTM price-regression forecasts with per-step volatility guardrails.
- `quant` rows are VQC direction-classifier outputs converted into prices using a recent-volatility proxy; they are not direct price-regression outputs.
- `price_proxy_return` records the magnitude used by the quantum proxy before guardrail and cumulative-band effects.
- `predicted_step_return` measures the model-implied move versus the previous forecast window endpoint.
- `horizon_return_from_last_observed` measures cumulative drift versus the last observed close.
- `prediction_constraint_applied` indicates per-step or cumulative guardrails changed the raw output.

---

## Model Inputs

| Field | Value |
|---|---|
| Source | `{request.source}` |
| Target column | `{request.target_column}` |
| Lookback | `{request.lookback}` |
| Normal model | `{normal_model_local_path or "not generated"}` |
| Quantum model | `{quantum_model_local_path or "not generated"}` |
| Quantum runtime mode | `{request.quantum_runtime_mode}` |
| Quantum shots | `{request.quantum_shots}` |

{timing_section}

---

*Forecast report generated automatically by the forecast pipeline.*
"""

    @staticmethod
    def _build_forecast_visual_report_markdown(
        *,
        frame: pd.DataFrame,
        request: ForecastBatchRequest,
    ) -> str:
        if frame.empty:
            return ""
        last_observed_close = float(frame["last_observed_close"].iloc[0])
        symbol = str(frame["symbol"].iloc[0]).upper()
        sections = [
            "### Forecast Control Report",
            "",
            f"**Ativo:** `{symbol}` - `${last_observed_close:.2f}`",
            f"**Horizonte:** `{request.horizon_days}` dias",
            "**Semente aleatoria:** `42`",
            "",
            "| Metric | Value | Detail |",
            "|---|---:|---|",
        ]
        model_rows = []
        for predict_type, group in frame.groupby("predict_type"):
            ordered = group.sort_values("forecast_step")
            final_close = float(ordered["predicted_close"].iloc[-1])
            final_return_pct = (
                (final_close / last_observed_close - 1.0) * 100.0
                if abs(last_observed_close) > 1e-8
                else 0.0
            )
            elapsed = (
                ordered["step_elapsed_ms"].dropna().astype(float)
                if "step_elapsed_ms" in ordered.columns
                else pd.Series(dtype=float)
            )
            total_ms = float(elapsed.sum()) if len(elapsed) else None
            avg_ms = float(elapsed.mean()) if len(elapsed) else None
            guardrails = int(ordered["prediction_constraint_applied"].astype(bool).sum())
            label = "LSTM" if str(predict_type).lower() == "normal" else "VQC"
            sections.append(
                f"| {label} - preco final | `${final_close:.2f}` | "
                f"{final_return_pct:+.2f}% vs last |"
            )
            model_rows.append(
                {
                    "label": label,
                    "family": str(ordered["model_family"].iloc[0]),
                    "total_ms": total_ms,
                    "avg_ms": avg_ms,
                    "up_rate": float(ordered["predicted_direction"].astype(int).mean()),
                    "return_pct": final_return_pct,
                    "guardrails": guardrails,
                    "row_count": int(len(ordered.index)),
                    "price_proxy": bool(ordered["is_price_proxy"].astype(bool).any()),
                }
            )

        normal = next((row for row in model_rows if row["label"] == "LSTM"), None)
        quant = next((row for row in model_rows if row["label"] == "VQC"), None)
        if normal and quant and normal["avg_ms"] and quant["avg_ms"]:
            ratio = float(quant["avg_ms"] / normal["avg_ms"])
            sections.append(f"| Razao tempo VQC/LSTM | `{ratio:.1f}x` | por step |")
        for row in model_rows:
            sections.append(
                f"| {row['label']} - guardrails | "
                f"`{row['guardrails']}/{row['row_count']}` | "
                f"{(row['guardrails'] / row['row_count']):.0%} dos steps |"
            )
        sections.append(f"| Lookback | `{request.lookback}` | dias de janela |")
        sections.extend(
            [
                "",
                "| Model | Familia | Tempo total | Media/step | Up rate | Retorno acumulado | Guardrails ativos | Price proxy |",
                "|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in model_rows:
            total_value = f"{float(row['total_ms']):.0f} ms" if row["total_ms"] is not None else "--"
            avg_value = f"{float(row['avg_ms']):.1f} ms" if row["avg_ms"] is not None else "--"
            sections.append(
                f"| {row['label']} | `{row['family']}` | {total_value} | "
                f"{avg_value} | {float(row['up_rate']) * 100.0:.1f}% | "
                f"{float(row['return_pct']):+.2f}% | "
                f"{row['guardrails']}/{row['row_count']} | "
                f"{'Sim (volatility proxy)' if row['price_proxy'] else 'Nao (regressao direta)'} |"
            )
        sections.append("")
        return "\n".join(sections)

    @staticmethod
    def _build_timing_report_section(
        *,
        forecast_summary: dict[str, dict[str, Any]],
        steps: int,
    ) -> str:
        timing = forecast_summary.get("_timing", {})
        normal_total = timing.get("normal_elapsed_ms")
        normal_avg = timing.get("normal_per_step_ms_mean")
        quantum_total = timing.get("quantum_elapsed_ms")
        quantum_avg = timing.get("quantum_per_step_ms_mean")
        ratio = timing.get("timing_ratio_quant_over_normal")
        if (
            normal_total is None
            or normal_avg is None
            or quantum_total is None
            or quantum_avg is None
            or ratio is None
        ):
            return ""

        return f"""
---

## Timing Report

| Model | Total (ms) | Average per step (ms) | Steps |
|---|---:|---:|---:|
| LSTM (normal) | {float(normal_total):.1f} | {float(normal_avg):.2f} | {steps} |
| VQC (quant) | {float(quantum_total):.1f} | {float(quantum_avg):.2f} | {steps} |

> **VQC / LSTM ratio:** {float(ratio):.1f}x
> The VQC path runs variational quantum circuit inference per step, while the
> LSTM path runs one matrix-forward pass per step.
"""

    def _build_unified_forecast_report_markdown(
        self,
        *,
        destination: Path,
        request: ForecastBatchRequest,
        generated_at_utc: str,
        assets: list[ForecastAssetArtifact],
    ) -> str:
        summary_rows = [
            "| Symbol | Rows | Types | Window | Last observed | Report | Dataset | Chart |",
            "|---|---:|---|---|---:|---|---|---|",
        ]
        sections: list[str] = []
        for asset in sorted(assets, key=lambda item: item.symbol):
            report_path = Path(asset.report_local_path)
            dataset_path = Path(asset.local_path)
            chart_path = Path(asset.chart_local_path)
            summary_rows.append(
                "| "
                f"{asset.symbol} | "
                f"{asset.row_count} | "
                f"{', '.join(asset.predict_types)} | "
                f"{asset.forecast_start_date} -> {asset.forecast_end_date} | "
                f"{asset.last_observed_close:.4f} | "
                f"{self._markdown_path_link(destination=destination, target=report_path, label='report')} | "
                f"{self._markdown_path_link(destination=destination, target=dataset_path, label='parquet')} | "
                f"{self._markdown_path_link(destination=destination, target=chart_path, label='svg')} |"
            )
            sections.append(
                self._build_unified_asset_section(
                    destination=destination,
                    asset=asset,
                )
            )

        return f"""# Unified Forecast Report

> **Extraction date:** `{request.extraction_date.isoformat()}`
> **Generated at (UTC):** `{generated_at_utc}`
> **Source:** `{request.source}`
> **Lookback:** `{request.lookback}`
> **Horizon days:** `{request.horizon_days}`
> **Quantum runtime mode:** `{request.quantum_runtime_mode}`

---

## Run Summary

{chr(10).join(summary_rows)}

---

{chr(10).join(sections)}

*Unified forecast report generated automatically under `data/processed/future_predict`.*
"""

    def _build_unified_asset_section(
        self,
        *,
        destination: Path,
        asset: ForecastAssetArtifact,
    ) -> str:
        summary_table = self._build_forecast_summary_table(asset.forecast_summary)
        chart_markdown = self._markdown_image(
            destination=destination,
            target=Path(asset.chart_local_path),
            alt=f"{asset.symbol} forecast chart",
        )
        report_link = self._markdown_path_link(
            destination=destination,
            target=Path(asset.report_local_path),
            label="detailed report",
        )
        dataset_link = self._markdown_path_link(
            destination=destination,
            target=Path(asset.local_path),
            label="future_predict parquet",
        )
        normal_model = self._format_markdown_path_or_empty(
            destination=destination,
            target=asset.normal_model_local_path,
        )
        quantum_model = self._format_markdown_path_or_empty(
            destination=destination,
            target=asset.quantum_model_local_path,
        )

        return f"""## {asset.symbol}

{summary_table}

{chart_markdown}

| Item | Value |
|---|---|
| Detailed report | {report_link} |
| Dataset | {dataset_link} |
| Normal model | {normal_model} |
| Quantum model | {quantum_model} |
| Last observed date | `{asset.last_observed_date}` |
| Last observed close | `{asset.last_observed_close:.4f}` |

---
"""

    @staticmethod
    def _build_forecast_chart_svg(*, frame: pd.DataFrame, symbol: str) -> str:
        width, height = 920, 420
        left, right, top, bottom = 70, 32, 48, 62
        plot_width = width - left - right
        plot_height = height - top - bottom
        ordered = frame.sort_values(["predict_type", "forecast_step"])
        values = ordered["predicted_close"].astype(float).to_list()
        if not values:
            return GenerateForecastBatchUseCase._empty_svg(
                f"{symbol.upper()} Forecast",
                "No forecast values available.",
            )
        y_min = min(values)
        y_max = max(values)
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0
        padding = (y_max - y_min) * 0.08
        y_min -= padding
        y_max += padding
        max_step = max(int(value) for value in ordered["forecast_step"].to_list())
        colors = {"normal": "#1f77b4", "quant": "#d62728"}

        def x_for(step: int) -> float:
            denominator = max(max_step - 1, 1)
            return left + ((step - 1) / denominator) * plot_width

        def y_for(value: float) -> float:
            return top + ((y_max - value) / (y_max - y_min)) * plot_height

        polylines: list[str] = []
        legend: list[str] = []
        for index, (predict_type, group) in enumerate(ordered.groupby("predict_type")):
            group = group.sort_values("forecast_step")
            points = " ".join(
                f"{x_for(int(row.forecast_step)):.1f},{y_for(float(row.predicted_close)):.1f}"
                for row in group.itertuples(index=False)
            )
            color = colors.get(str(predict_type), "#2ca02c")
            polylines.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" '
                'stroke-width="2.8" stroke-linejoin="round" stroke-linecap="round" />'
            )
            legend_y = 76 + index * 22
            legend.append(
                f'<rect x="758" y="{legend_y - 10}" width="12" height="12" fill="{color}" />'
                f'<text x="776" y="{legend_y}" font-size="13">{escape(str(predict_type))}</text>'
            )

        y_ticks = []
        for tick in range(5):
            value = y_min + ((y_max - y_min) * tick / 4)
            y = y_for(value)
            y_ticks.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb" />'
                f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{value:.4g}</text>'
            )

        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff" />
<text x="{left}" y="28" font-size="20" font-weight="700">{escape(symbol.upper())} Forecast Path</text>
<text x="{left}" y="{height - 18}" font-size="12">Forecast step</text>
<text x="18" y="{top + plot_height / 2:.1f}" font-size="12" transform="rotate(-90 18 {top + plot_height / 2:.1f})">Predicted close</text>
{''.join(y_ticks)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" />
<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" stroke="#111827" />
{''.join(polylines)}
{''.join(legend)}
</svg>
"""

    @staticmethod
    def _empty_svg(title: str, message: str) -> str:
        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="840" height="260" viewBox="0 0 840 260">
<rect width="100%" height="100%" fill="#ffffff" />
<text x="42" y="42" font-size="20" font-weight="700">{escape(title)}</text>
<text x="42" y="92" font-size="14">{escape(message)}</text>
</svg>
"""

    @staticmethod
    def _markdown_path_link(*, destination: Path, target: Path, label: str) -> str:
        relative = os.path.relpath(target, start=destination.parent).replace("\\", "/")
        return f"[{label}]({relative})"

    @staticmethod
    def _markdown_image(*, destination: Path, target: Path, alt: str) -> str:
        relative = os.path.relpath(target, start=destination.parent).replace("\\", "/")
        return f"![{alt}]({relative})"

    def _format_markdown_path_or_empty(
        self,
        *,
        destination: Path,
        target: str | None,
    ) -> str:
        if not target:
            return "`not generated`"
        path = Path(target)
        if path.suffix.lower() == ".svg":
            return self._markdown_image(destination=destination, target=path, alt=path.stem)
        return self._markdown_path_link(destination=destination, target=path, label=path.name)

    @staticmethod
    def _build_forecast_summary_table(
        forecast_summary: dict[str, dict[str, Any]],
    ) -> str:
        lines = [
            "| Type | Rows | Up rate | Up/Down | Horizon delta | Mean abs step return | Step return std | constraint_rate | hit_lower_band | hit_upper_band | flat_path | monotonic_path | Price proxy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|",
        ]
        for predict_type, payload in sorted(forecast_summary.items()):
            if predict_type.startswith("_"):
                continue
            lines.append(
                "| "
                f"{predict_type} | "
                f"{int(payload['row_count'])} | "
                f"{float(payload['up_rate']):.1%} | "
                f"{int(payload['up_count'])}/{int(payload['down_count'])} | "
                f"{float(payload['horizon_delta_pct']):+.2f}% | "
                f"{float(payload['mean_abs_step_return_pct']):.3f}% | "
                f"{float(payload['std_step_return_pct']):.3f}% | "
                f"{float(payload.get('constraint_rate', 0.0)):.1%} | "
                f"{'yes' if bool(payload.get('hit_lower_band')) else 'no'} | "
                f"{'yes' if bool(payload.get('hit_upper_band')) else 'no'} | "
                f"{'yes' if bool(payload.get('flat_path')) else 'no'} | "
                f"{'yes' if bool(payload.get('monotonic_path')) else 'no'} | "
                f"{'yes' if bool(payload['uses_price_proxy']) else 'no'} |"
            )
        return "\n".join(lines)

    @staticmethod
    def _build_forecast_warning_lines(
        forecast_summary: dict[str, dict[str, Any]],
    ) -> str:
        warnings: list[str] = []
        for predict_type, payload in sorted(forecast_summary.items()):
            if predict_type.startswith("_"):
                continue
            if bool(payload["degenerate_direction_path"]):
                warnings.append(
                    f"- `{predict_type}` has a degenerate direction path "
                    f"(Up rate {float(payload['up_rate']):.1%})."
                )
            if abs(float(payload["horizon_delta_pct"])) >= 30.0:
                warnings.append(
                    f"- `{predict_type}` cumulative horizon move is large "
                    f"({float(payload['horizon_delta_pct']):+.2f}%)."
                )
            if int(payload["constraint_applied_count"]) > 0:
                warnings.append(
                    f"- `{predict_type}` had {int(payload['constraint_applied_count'])} "
                    "guardrail-constrained forecast rows "
                    f"({float(payload.get('constraint_rate', 0.0)):.1%})."
                )
            if bool(payload.get("hit_lower_band")) or bool(payload.get("hit_upper_band")):
                warnings.append(
                    f"- `{predict_type}` touched the dynamic cumulative band "
                    f"(lower={int(payload.get('hit_lower_band_count', 0))}, "
                    f"upper={int(payload.get('hit_upper_band_count', 0))})."
                )
            if bool(payload.get("flat_path")):
                warnings.append(f"- `{predict_type}` has a flat forecast path.")
            if bool(payload.get("monotonic_path")):
                warnings.append(f"- `{predict_type}` has a monotonic forecast path.")
        if not warnings:
            return "No forecast stability warnings were triggered."
        return "\n".join(["### Forecast Warnings", "", *warnings])

    @staticmethod
    def _build_forecast_endpoint_table(frame: pd.DataFrame) -> str:
        lines = [
            "| Type | First date | First close | Last date | Last close | Last horizon return | Last direction |",
            "|---|---|---:|---|---:|---:|---|",
        ]
        for predict_type, group in frame.groupby("predict_type"):
            ordered = group.sort_values("forecast_step")
            first = ordered.iloc[0]
            last = ordered.iloc[-1]
            lines.append(
                "| "
                f"{predict_type} | "
                f"{first['forecast_date']} | "
                f"{float(first['predicted_close']):.4f} | "
                f"{last['forecast_date']} | "
                f"{float(last['predicted_close']):.4f} | "
                f"{float(last['horizon_return_from_last_observed']) * 100.0:+.2f}% | "
                f"{last['predicted_direction_label']} |"
            )
        return "\n".join(lines)

    @classmethod
    def _score_regression_asset(
        cls,
        *,
        asset_payload: dict[str, Any],
        manifest_path: Path,
    ) -> tuple[float, float, float, float, int]:
        return (
            cls._metric_or_inf(asset_payload.get("test_metrics"), "mae", "rmse"),
            cls._metric_or_inf(asset_payload.get("validation_metrics"), "mae", "rmse"),
            cls._metric_or_inf(asset_payload.get("test_metrics"), "rmse"),
            cls._metric_or_inf(asset_payload.get("validation_metrics"), "rmse"),
            -cls._trained_at_token(manifest_path),
        )

    @staticmethod
    def _metric_or_inf(
        metrics_payload: Any,
        *preferred_keys: str,
    ) -> float:
        metrics = metrics_payload or {}
        for key in preferred_keys:
            value = metrics.get(key)
            if value is not None:
                return abs(float(value))
        return float("inf")

    @staticmethod
    def _trained_at_token(manifest_path: Path) -> int:
        token = manifest_path.parent.name.split("=", 1)[-1]
        normalized = token.replace("T", "").replace("Z", "")
        try:
            return int(normalized)
        except ValueError:
            return 0

    @staticmethod
    def _build_future_predict_row(
        *,
        source: str,
        symbol: str,
        target_column: str,
        extraction_date: date,
        generated_at_utc: str,
        generated_at_token: str,
        predict_type: str,
        model_family: str,
        model_name: str,
        model_local_path: str,
        training_manifest_local_path: str | None,
        training_generated_at_utc: str | None,
        lookback: int,
        horizon_days: int,
        forecast_step: int,
        forecast_date: pd.Timestamp,
        predicted_close: float,
        predicted_scaled: float | None,
        predicted_direction: int,
        raw_model_predicted_close: float | None,
        prediction_constraint_applied: bool,
        prediction_constraint_method: str | None,
        prediction_return_cap: float | None,
        predicted_step_return: float,
        horizon_return_from_last_observed: float,
        price_proxy_return: float | None,
        hit_lower_band: bool,
        hit_upper_band: bool,
        dynamic_cumulative_return_cap: float,
        last_observed_date: pd.Timestamp,
        last_observed_close: float,
        input_window_start_date: pd.Timestamp,
        input_window_end_date: pd.Timestamp,
        input_window_end_close: float,
        input_window_end_origin: str,
        observed_points_in_window: int,
        predicted_points_in_window: int,
        is_price_proxy: bool,
        price_proxy_method: str | None,
    ) -> dict[str, Any]:
        return {
            "symbol": symbol.upper(),
            "source": source,
            "target_column": target_column,
            "extraction_date": extraction_date.isoformat(),
            "generated_at_utc": generated_at_utc,
            "generated_at_token": generated_at_token,
            "predict_type": predict_type,
            "model_family": model_family,
            "model_name": model_name,
            "model_local_path": model_local_path,
            "training_manifest_local_path": training_manifest_local_path,
            "training_generated_at_utc": training_generated_at_utc,
            "lookback": int(lookback),
            "horizon_days": int(horizon_days),
            "forecast_step": int(forecast_step),
            "forecast_date": forecast_date.strftime("%Y-%m-%d"),
            "predicted_close": float(predicted_close),
            "predicted_scaled": (
                float(predicted_scaled) if predicted_scaled is not None else None
            ),
            "raw_model_predicted_close": (
                float(raw_model_predicted_close)
                if raw_model_predicted_close is not None
                else None
            ),
            "predicted_direction": int(predicted_direction),
            "predicted_direction_label": "up" if int(predicted_direction) == 1 else "down",
            "prediction_constraint_applied": bool(prediction_constraint_applied),
            "prediction_constraint_method": prediction_constraint_method,
            "prediction_return_cap": (
                float(prediction_return_cap) if prediction_return_cap is not None else None
            ),
            "predicted_step_return": float(predicted_step_return),
            "horizon_return_from_last_observed": float(horizon_return_from_last_observed),
            "price_proxy_return": (
                float(price_proxy_return) if price_proxy_return is not None else None
            ),
            "hit_lower_band": bool(hit_lower_band),
            "hit_upper_band": bool(hit_upper_band),
            "dynamic_cumulative_return_cap": float(dynamic_cumulative_return_cap),
            "last_observed_date": last_observed_date.strftime("%Y-%m-%d"),
            "last_observed_close": float(last_observed_close),
            "input_window_start_date": input_window_start_date.strftime("%Y-%m-%d"),
            "input_window_end_date": input_window_end_date.strftime("%Y-%m-%d"),
            "input_window_end_close": float(input_window_end_close),
            "input_window_end_origin": input_window_end_origin,
            "observed_points_in_window": int(observed_points_in_window),
            "predicted_points_in_window": int(predicted_points_in_window),
            "recursive_forecast": True,
            "is_price_proxy": bool(is_price_proxy),
            "price_proxy_method": price_proxy_method,
        }

    @staticmethod
    def _scale_array(
        values: np.ndarray,
        *,
        min_offset: float,
        scale: float,
    ) -> np.ndarray:
        return values * scale + min_offset

    @staticmethod
    def _inverse_scale_array(
        values: np.ndarray,
        *,
        min_offset: float,
        scale: float,
    ) -> np.ndarray:
        if scale == 0:
            raise ValueError("Cannot inverse scale values because scale is zero.")
        return (values - min_offset) / scale

    @staticmethod
    def _build_future_predict_relative_path(
        *,
        source: str,
        symbol: str,
        lookback: int,
        horizon_days: int,
        extraction_date: date,
        generated_at_token: str,
    ) -> Path:
        return (
            Path("future_predict")
            / f"source={source}"
            / f"symbol={symbol.upper()}"
            / f"lookback={lookback}"
            / f"horizon_days={horizon_days}"
            / f"extraction_date={extraction_date.isoformat()}"
            / f"generated_at={generated_at_token}"
            / "future_predict.parquet"
        )

    @staticmethod
    def _build_unified_report_relative_path(
        *,
        extraction_date: date,
        generated_at_token: str,
    ) -> Path:
        return (
            Path("future_predict")
            / f"extraction_date={extraction_date.isoformat()}"
            / f"generated_at={generated_at_token}"
            / "unified_forecast_report.md"
        )

    @staticmethod
    def _build_manifest_relative_path(
        *,
        extraction_date: date,
        generated_at_token: str,
    ) -> Path:
        return (
            Path("manifests")
            / f"extraction_date={extraction_date.isoformat()}"
            / f"generated_at={generated_at_token}"
            / "future_predict_manifest.json"
        )

    @staticmethod
    def _to_path_safe_timestamp(generated_at_utc: str) -> str:
        parsed = datetime.fromisoformat(generated_at_utc)
        return parsed.strftime("%Y%m%dT%H%M%SZ")


def main() -> int:
    print(
        "GenerateForecastBatchUseCase is a library module. "
        "Use `python scripts/generate_forecast.py --help` to run forecast generation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
