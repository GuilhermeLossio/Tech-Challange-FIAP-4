from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import json
import os
import sys
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd

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
    local_path: str
    s3_uri: str | None


@dataclass(frozen=True)
class ForecastBatchResult:
    source: str
    generated_at_utc: str
    manifest_local_path: str
    manifest_s3_uri: str | None
    assets: tuple[ForecastAssetArtifact, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "generated_at_utc": self.generated_at_utc,
            "manifest_local_path": self.manifest_local_path,
            "manifest_s3_uri": self.manifest_s3_uri,
            "asset_count": len(self.assets),
            "assets": [asdict(asset) for asset in self.assets],
        }


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
                combined_rows.extend(
                    self._build_normal_rows(
                        raw_frame=raw_frame,
                        request=request,
                        symbol=symbol,
                        generated_at_utc=generated_at_utc,
                        generated_at_token=generated_at_token,
                        model=normal_model,
                        scaler_metadata=scaler_metadata,
                        model_metadata=normal_metadata,
                    )
                )
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
                quantum_model = self._build_quantum_predictor(quantum_payload)
                combined_rows.extend(
                    self._build_quantum_rows(
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
                )
                predict_types.append("quant")

            future_predict_frame = pd.DataFrame(combined_rows)
            future_predict_frame = future_predict_frame.sort_values(
                by=["forecast_step", "predict_type"],
            ).reset_index(drop=True)

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

            forecast_dates = pd.to_datetime(future_predict_frame["forecast_date"])
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
                    local_path=str(local_path),
                    s3_uri=s3_uri,
                )
            )

        manifest_payload = {
            "source": request.source,
            "generated_at_utc": generated_at_utc,
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
        manifests_root = (
            self._models_root_dir
            / "manifests"
            / f"extraction_date={extraction_date.isoformat()}"
        )
        if manifests_root.exists():
            candidate_paths = sorted(
                manifests_root.glob("trained_at=*/keras_training_manifest.json"),
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
                    "training_manifest_local_path": str(manifest_path),
                    "training_generated_at_utc": str(payload.get("generated_at_utc")),
                }

        fallback_path = self._models_root_dir / expected_model_name
        if fallback_path.exists():
            return {
                "model_local_path": str(fallback_path),
                "training_manifest_local_path": None,
                "training_generated_at_utc": None,
            }

        raise FileNotFoundError(
            f"Could not find a trained Keras model for symbol {symbol!r}. "
            f"Expected manifest under {manifests_root} or fallback model {fallback_path}."
        )

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
        *,
        raw_frame: pd.DataFrame,
        request: ForecastBatchRequest,
        symbol: str,
        generated_at_utc: str,
        generated_at_token: str,
        model: Any,
        scaler_metadata: dict[str, float],
        model_metadata: dict[str, str | None],
    ) -> list[dict[str, Any]]:
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
        for step, forecast_date in enumerate(forecast_dates, start=1):
            input_window_start_date = pd.Timestamp(window_dates[0])
            input_window_end_date = pd.Timestamp(window_dates[-1])
            input_window_end_close = float(raw_window[-1])

            prediction_input = scaled_window.reshape(1, request.lookback, 1)
            predicted_scaled = float(model.predict(prediction_input, verbose=0).reshape(-1)[0])
            predicted_close = float(
                self._inverse_scale_array(
                    np.asarray([predicted_scaled], dtype=np.float32),
                    min_offset=scaler_metadata["min_offset"],
                    scale=scaler_metadata["scale"],
                )[0]
            )
            predicted_direction = int(predicted_close > input_window_end_close)

            observed_points_in_window = max(request.lookback - (step - 1), 0)
            predicted_points_in_window = min(step - 1, request.lookback)
            rows.append(
                self._build_future_predict_row(
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
            )

            raw_window = np.concatenate(
                [raw_window[1:], np.asarray([predicted_close], dtype=np.float32)]
            )
            scaled_window = np.concatenate(
                [scaled_window[1:], np.asarray([predicted_scaled], dtype=np.float32)]
            )
            window_dates = window_dates[1:] + [forecast_date]

        return rows

    def _build_quantum_rows(
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
    ) -> list[dict[str, Any]]:
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
        for step, forecast_date in enumerate(forecast_dates, start=1):
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
            proxy_return = self._compute_quantum_price_proxy_return(raw_window)
            predicted_close = float(
                input_window_end_close
                * (1.0 + proxy_return if predicted_direction == 1 else 1.0 - proxy_return)
            )

            observed_points_in_window = max(request.lookback - (step - 1), 0)
            predicted_points_in_window = min(step - 1, request.lookback)
            rows.append(
                self._build_future_predict_row(
                    source=request.source,
                    symbol=symbol,
                    target_column=request.target_column,
                    extraction_date=request.extraction_date,
                    generated_at_utc=generated_at_utc,
                    generated_at_token=generated_at_token,
                    predict_type="quant",
                    model_family="quantum_vqc_direction_classifier",
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
                        "directional_return_proxy_mean_abs_10d_from_quant_signal"
                    ),
                )
            )

            raw_window = np.concatenate(
                [raw_window[1:], np.asarray([predicted_close], dtype=np.float64)]
            )
            window_dates = window_dates[1:] + [forecast_date]

        return rows

    def _build_quantum_predictor(self, model_payload: dict[str, Any]) -> VQC:
        runtime_context = self._build_quantum_runtime_context(model_payload)
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

    def _build_quantum_runtime_context(self, model_payload: dict[str, Any]) -> dict[str, Any]:
        execution_mode = str(model_payload.get("execution_mode", "local"))
        backend_name = str(model_payload.get("backend_name", "fake_manila"))

        if execution_mode == "local" or backend_name.startswith("fake_"):
            backend = FakeManilaV2()
            sampler = Sampler(mode=backend, options={"simulator": {"seed_simulator": 42}})
            sampler.options.default_shots = 1024
            pass_manager = generate_preset_pass_manager(
                backend=backend,
                optimization_level=1,
            )
            return {
                "backend": backend,
                "sampler": sampler,
                "pass_manager": pass_manager,
            }

        service = self._build_cloud_service()
        backend = service.backend(backend_name)
        sampler = Sampler(mode=backend)
        sampler.options.default_shots = 1024
        pass_manager = generate_preset_pass_manager(
            backend=backend,
            optimization_level=1,
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

    def _compute_quantum_price_proxy_return(self, raw_window: np.ndarray) -> float:
        feature_engineer = GenerateFeatureDatasetUseCase
        daily_returns = feature_engineer._compute_daily_returns(raw_window)  # type: ignore[attr-defined]
        if len(daily_returns) == 0:
            return 0.01

        effective_periods = min(10, len(daily_returns))
        proxy_return = float(np.mean(np.abs(daily_returns[-effective_periods:])))
        return min(max(proxy_return, 0.0025), 0.15)

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
            "predicted_direction": int(predicted_direction),
            "predicted_direction_label": "up" if int(predicted_direction) == 1 else "down",
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
