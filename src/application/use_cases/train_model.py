from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from html import escape
from pathlib import Path
import os
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # pyright: ignore[reportMissingModuleSource]
    from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input, LSTM, BatchNormalization  # pyright: ignore[reportMissingModuleSource]
    from tensorflow.keras.models import Model, Sequential  # type: ignore
    from tensorflow.keras.optimizers import Adam  # pyright: ignore[reportMissingModuleSource]
    from tensorflow.keras.regularizers import l2  # pyright: ignore[reportMissingModuleSource]
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    tf = None
    EarlyStopping = ModelCheckpoint = ReduceLROnPlateau = None
    Concatenate = Dense = Dropout = Input = LSTM = BatchNormalization = Model = Sequential = Adam = None
    l2 = None
    _TENSORFLOW_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _TENSORFLOW_IMPORT_ERROR = None

# Allows direct execution from IDEs that run the file instead of the package module.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.application.use_cases._dataset_loading import load_preferred_training_frame
from src.infrastructure.storage.local_model_store import LocalModelStore
from src.infrastructure.storage.s3_raw_store import S3RawStore


class TrainingInterruptedError(RuntimeError):
    def __init__(self, symbol: str, checkpoint_path: Path) -> None:
        super().__init__(
            "Training interrupted by user while fitting "
            f"{symbol.upper()}. Latest checkpoint: {checkpoint_path}"
        )
        self.symbol = symbol.upper()
        self.checkpoint_path = checkpoint_path


@dataclass(frozen=True)
class KerasTrainingRequest:
    symbols: tuple[str, ...]
    extraction_date: date
    source: str = "yfinance"
    target_column: str = "close"
    lookback: int = 60
    epochs: int = 100
    batch_size: int = 32
    patience: int = 20                  # ← era 10; mais tolerância ao plateau ruidoso
    learning_rate: float = 0.0005       # ← era 0.001; mais estável para retornos pequenos
    seed: int = 42
    verbose: int = 1
    model_name_prefix: str = "lstm"
    prediction_target_mode: str = "price"
    feature_input_mode: str = "sequence_price"
    l2_reg: float = 1e-4               # ← novo: regularização L2 nas camadas Dense
    clip_norm: float = 1.0             # ← novo: gradient clipping para estabilidade


@dataclass(frozen=True)
class SplitMetrics:
    sample_count: int
    loss_mse: float | None
    mae_scaled: float | None
    mae: float | None
    rmse: float | None
    mape: float | None


@dataclass(frozen=True)
class KerasModelArtifact:
    symbol: str
    row_count: int
    train_count: int
    validation_count: int
    test_count: int
    feature_count: int
    feature_input_mode: str
    sequence_input_kind: str
    sequence_length: int
    engineered_feature_columns: tuple[str, ...]
    epochs_ran: int
    best_epoch: int
    immutable_model_local_path: str
    published_model_local_path: str | None
    model_local_path: str
    history_local_path: str
    report_local_path: str
    loss_chart_local_path: str
    metrics_chart_local_path: str
    model_s3_uri: str | None
    history_s3_uri: str | None
    report_s3_uri: str | None
    loss_chart_s3_uri: str | None
    metrics_chart_s3_uri: str | None
    train_metrics: SplitMetrics
    validation_metrics: SplitMetrics
    test_metrics: SplitMetrics


@dataclass(frozen=True)
class KerasTrainingResult:
    source: str
    generated_at_utc: str
    manifest_local_path: str
    manifest_s3_uri: str | None
    assets: tuple[KerasModelArtifact, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "generated_at_utc": self.generated_at_utc,
            "manifest_local_path": self.manifest_local_path,
            "manifest_s3_uri": self.manifest_s3_uri,
            "asset_count": len(self.assets),
            "assets": [asdict(asset) for asset in self.assets],
        }


class KerasTrainingService:
    def __init__(
        self,
        processed_root_dir: Path,
        local_store: LocalModelStore,
        s3_store: S3RawStore | None = None,
    ) -> None:
        self._processed_root_dir = processed_root_dir
        self._local_store = local_store
        self._s3_store = s3_store

    def train(self, request: KerasTrainingRequest) -> KerasTrainingResult:
        self._ensure_tensorflow_available()
        self._validate_request(request)

        generated_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        trained_at_token = self._to_path_safe_timestamp(generated_at_utc)
        artifacts: list[KerasModelArtifact] = []

        for symbol in request.symbols:
            tf.keras.backend.clear_session()
            frame, scaler_metadata, dataset_kind = self._load_training_frame(
                source=request.source,
                symbol=symbol,
                extraction_date=request.extraction_date,
                lookback=request.lookback,
                target_column=request.target_column,
            )
            dataset = self._build_training_dataset(
                frame=frame,
                request=request,
                scaler_metadata=scaler_metadata,
                dataset_kind=dataset_kind,
            )
            self._set_random_seed(request.seed)

            model_relative_path = self._build_model_relative_path(
                source=request.source,
                symbol=symbol,
                lookback=request.lookback,
                extraction_date=request.extraction_date,
                trained_at_token=trained_at_token,
                model_name_prefix=request.model_name_prefix,
            )
            model_path = self._local_store.prepare_path(model_relative_path)

            model = self._build_model(
                sequence_input_shape=dataset["sequence_input_shape"],
                engineered_feature_count=dataset["engineered_feature_count"],
                learning_rate=request.learning_rate,
                feature_input_mode=request.feature_input_mode,
                l2_reg=request.l2_reg,
                clip_norm=request.clip_norm,
            )

            monitor_metric = "val_loss" if dataset["validation_count"] > 0 else "loss"

            # ── Callbacks ──────────────────────────────────────────────────────
            callbacks = [
                EarlyStopping(
                    monitor=monitor_metric,
                    patience=request.patience,
                    restore_best_weights=True,
                    min_delta=1e-6,         # ← novo: ignora melhorias irrelevantes
                ),
                ModelCheckpoint(
                    filepath=str(model_path),
                    monitor=monitor_metric,
                    save_best_only=True,
                ),
                # ← novo: reduz LR pela metade se ficar 7 epochs sem melhorar
                ReduceLROnPlateau(
                    monitor=monitor_metric,
                    factor=0.5,
                    patience=7,
                    min_lr=1e-6,
                    verbose=1,
                ),
            ]

            # ── Shuffle só faz sentido para retornos (não para preço sequencial) ──
            should_shuffle = request.prediction_target_mode == "return"

            fit_kwargs: dict[str, Any] = {
                "x": dataset["X_train_model"],
                "y": dataset["y_train_scaled"],
                "epochs": request.epochs,
                "batch_size": request.batch_size,
                "callbacks": callbacks,
                "verbose": request.verbose,
                "shuffle": should_shuffle,  # ← era sempre False
            }
            if dataset["validation_count"] > 0:
                fit_kwargs["validation_data"] = (
                    dataset["X_validation_model"],
                    dataset["y_validation_scaled"],
                )

            try:
                history = model.fit(**fit_kwargs)
            except KeyboardInterrupt as exc:
                if not model_path.exists():
                    try:
                        model.save(model_path)
                    except Exception:
                        pass
                raise TrainingInterruptedError(
                    symbol=symbol,
                    checkpoint_path=model_path,
                ) from exc

            if not model_path.exists():
                model.save(model_path)
            model = tf.keras.models.load_model(model_path)
            published_model_path = self._publish_latest_model_alias(
                model_path=model_path,
                symbol=symbol,
                model_name_prefix=request.model_name_prefix,
            )
            model_s3_uri = None
            if self._s3_store is not None:
                model_s3_uri = self._s3_store.upload_file(
                    local_path=model_path,
                    relative_path=model_relative_path,
                )

            history_payload = self._build_history_payload(
                history=history.history,
                monitor_metric=monitor_metric,
            )
            history_relative_path = self._build_history_relative_path(
                source=request.source,
                symbol=symbol,
                lookback=request.lookback,
                extraction_date=request.extraction_date,
                trained_at_token=trained_at_token,
            )
            history_path = self._local_store.write_json(
                history_payload,
                history_relative_path,
            )
            history_s3_uri = None
            if self._s3_store is not None:
                history_s3_uri = self._s3_store.upload_file(
                    local_path=history_path,
                    relative_path=history_relative_path,
                )

            train_metrics = self._evaluate_split(
                model=model,
                X=dataset["X_train_model"],
                y_scaled=dataset["y_train_scaled"],
                y_raw=dataset["y_train_raw"],
                current_raw=dataset["current_train_raw"],
                scaler_metadata=scaler_metadata,
                prediction_target_mode=request.prediction_target_mode,
            )
            validation_metrics = self._evaluate_split(
                model=model,
                X=dataset["X_validation_model"],
                y_scaled=dataset["y_validation_scaled"],
                y_raw=dataset["y_validation_raw"],
                current_raw=dataset["current_validation_raw"],
                scaler_metadata=scaler_metadata,
                prediction_target_mode=request.prediction_target_mode,
            )
            test_metrics = self._evaluate_split(
                model=model,
                X=dataset["X_test_model"],
                y_scaled=dataset["y_test_scaled"],
                y_raw=dataset["y_test_raw"],
                current_raw=dataset["current_test_raw"],
                scaler_metadata=scaler_metadata,
                prediction_target_mode=request.prediction_target_mode,
            )
            loss_chart_relative_path = self._build_run_artifact_relative_path(
                source=request.source,
                symbol=symbol,
                lookback=request.lookback,
                extraction_date=request.extraction_date,
                trained_at_token=trained_at_token,
                filename="training_loss.svg",
            )
            loss_chart_path = self._local_store.prepare_path(loss_chart_relative_path)
            self._write_line_chart_svg(
                destination=loss_chart_path,
                title=f"{symbol.upper()} Training Loss",
                series={
                    "loss": history_payload["history"].get("loss", []),
                    "val_loss": history_payload["history"].get("val_loss", []),
                },
                y_label="MSE loss",
            )
            loss_chart_s3_uri = None
            if self._s3_store is not None:
                loss_chart_s3_uri = self._s3_store.upload_file(
                    local_path=loss_chart_path,
                    relative_path=loss_chart_relative_path,
                )

            metrics_chart_relative_path = self._build_run_artifact_relative_path(
                source=request.source,
                symbol=symbol,
                lookback=request.lookback,
                extraction_date=request.extraction_date,
                trained_at_token=trained_at_token,
                filename="training_metrics.svg",
            )
            metrics_chart_path = self._local_store.prepare_path(metrics_chart_relative_path)
            self._write_keras_metrics_svg(
                destination=metrics_chart_path,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
            )
            metrics_chart_s3_uri = None
            if self._s3_store is not None:
                metrics_chart_s3_uri = self._s3_store.upload_file(
                    local_path=metrics_chart_path,
                    relative_path=metrics_chart_relative_path,
                )

            report_relative_path = self._build_run_artifact_relative_path(
                source=request.source,
                symbol=symbol,
                lookback=request.lookback,
                extraction_date=request.extraction_date,
                trained_at_token=trained_at_token,
                filename="training_report.md",
            )
            report_path = self._local_store.prepare_path(report_relative_path)
            self._write_training_report(
                destination=report_path,
                symbol=symbol,
                request=request,
                generated_at_utc=generated_at_utc,
                dataset=dataset,
                history_payload=history_payload,
                model_path=model_path,
                published_model_path=published_model_path,
                history_path=history_path,
                loss_chart_path=loss_chart_path,
                metrics_chart_path=metrics_chart_path,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
            )
            report_s3_uri = None
            if self._s3_store is not None:
                report_s3_uri = self._s3_store.upload_file(
                    local_path=report_path,
                    relative_path=report_relative_path,
                )

            artifacts.append(
                KerasModelArtifact(
                    symbol=symbol.upper(),
                    row_count=dataset["row_count"],
                    train_count=dataset["train_count"],
                    validation_count=dataset["validation_count"],
                    test_count=dataset["test_count"],
                    feature_count=dataset["model_feature_count"],
                    feature_input_mode=request.feature_input_mode,
                    sequence_input_kind=dataset["sequence_input_kind"],
                    sequence_length=dataset["sequence_length"],
                    engineered_feature_columns=tuple(dataset["engineered_feature_columns"]),
                    epochs_ran=history_payload["epochs_ran"],
                    best_epoch=history_payload["best_epoch"],
                    immutable_model_local_path=str(model_path),
                    published_model_local_path=str(published_model_path),
                    model_local_path=str(model_path),
                    history_local_path=str(history_path),
                    report_local_path=str(report_path),
                    loss_chart_local_path=str(loss_chart_path),
                    metrics_chart_local_path=str(metrics_chart_path),
                    model_s3_uri=model_s3_uri,
                    history_s3_uri=history_s3_uri,
                    report_s3_uri=report_s3_uri,
                    loss_chart_s3_uri=loss_chart_s3_uri,
                    metrics_chart_s3_uri=metrics_chart_s3_uri,
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                    test_metrics=test_metrics,
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
                "epochs": request.epochs,
                "batch_size": request.batch_size,
                "patience": request.patience,
                "learning_rate": request.learning_rate,
                "seed": request.seed,
                "verbose": request.verbose,
                "model_name_prefix": request.model_name_prefix,
                "prediction_target_mode": request.prediction_target_mode,
                "feature_input_mode": request.feature_input_mode,
                "l2_reg": request.l2_reg,
                "clip_norm": request.clip_norm,
            },
            "asset_count": len(artifacts),
            "assets": [asdict(artifact) for artifact in artifacts],
        }
        manifest_relative_path = self._build_manifest_relative_path(
            extraction_date=request.extraction_date,
            trained_at_token=trained_at_token,
        )
        manifest_local_path = self._local_store.write_json(
            manifest_payload,
            manifest_relative_path,
        )
        manifest_s3_uri = None
        if self._s3_store is not None:
            manifest_s3_uri = self._s3_store.upload_file(
                local_path=manifest_local_path,
                relative_path=manifest_relative_path,
            )

        return KerasTrainingResult(
            source=request.source,
            generated_at_utc=generated_at_utc,
            manifest_local_path=str(manifest_local_path),
            manifest_s3_uri=manifest_s3_uri,
            assets=tuple(artifacts),
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _ensure_tensorflow_available(self) -> None:
        if _TENSORFLOW_IMPORT_ERROR is not None:
            raise RuntimeError(
                "TensorFlow is required for Keras training. "
                "Install the project dependencies again after adding TensorFlow."
            ) from _TENSORFLOW_IMPORT_ERROR

    def _validate_request(self, request: KerasTrainingRequest) -> None:
        if not request.symbols:
            raise ValueError("At least one symbol must be provided.")
        if request.lookback <= 0:
            raise ValueError("lookback must be greater than zero.")
        if request.epochs <= 0:
            raise ValueError("epochs must be greater than zero.")
        if request.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if request.patience < 0:
            raise ValueError("patience must be non-negative.")
        if request.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")
        if request.verbose not in {0, 1, 2}:
            raise ValueError("verbose must be one of: 0, 1, 2.")
        if not request.model_name_prefix.strip():
            raise ValueError("model_name_prefix must not be blank.")
        if request.prediction_target_mode not in {"price", "return"}:
            raise ValueError("prediction_target_mode must be either 'price' or 'return'.")
        if request.feature_input_mode not in {"sequence_price", "technical_returns"}:
            raise ValueError(
                "feature_input_mode must be either 'sequence_price' or "
                "'technical_returns'."
            )
        if (
            request.feature_input_mode == "technical_returns"
            and request.prediction_target_mode != "return"
        ):
            raise ValueError(
                "feature_input_mode='technical_returns' requires "
                "prediction_target_mode='return'."
            )
        if request.l2_reg < 0:
            raise ValueError("l2_reg must be non-negative.")
        if request.clip_norm <= 0:
            raise ValueError("clip_norm must be greater than zero.")

    def _load_training_frame(
        self,
        *,
        source: str,
        symbol: str,
        extraction_date: date,
        lookback: int,
        target_column: str,
    ) -> tuple[pd.DataFrame, dict[str, float], str]:
        frame, scaler_metadata, dataset_kind = load_preferred_training_frame(
            processed_root_dir=self._processed_root_dir,
            source=source,
            symbol=symbol,
            extraction_date=extraction_date,
            lookback=lookback,
            target_column=target_column,
        )
        return frame, scaler_metadata, dataset_kind

    def _build_training_dataset(
        self,
        *,
        frame: pd.DataFrame,
        request: KerasTrainingRequest,
        scaler_metadata: dict[str, float],
        dataset_kind: str,
    ) -> dict[str, Any]:
        if request.feature_input_mode == "technical_returns" and dataset_kind != "feature":
            raise ValueError(
                "feature_input_mode='technical_returns' requires a fresh "
                "feature_manifest.json. Run scripts/generate_features.py after "
                "scripts/generate_refined.py."
            )
        feature_columns = [
            f"{request.target_column}_t_minus_{lag}"
            for lag in range(request.lookback, 0, -1)
        ]
        missing_columns = [column for column in feature_columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                "Refined dataset is missing expected feature columns: "
                f"{missing_columns}"
            )

        target_raw_column = f"y_{request.target_column}"
        if target_raw_column not in frame.columns:
            raise ValueError(
                f"Refined dataset is missing expected target column {target_raw_column!r}."
            )
        if "split" not in frame.columns:
            raise ValueError("Refined dataset is missing the `split` column.")
        if "y_scaled" not in frame.columns:
            raise ValueError("Refined dataset is missing the `y_scaled` column.")

        ordered = frame.copy()
        if "target_date" in ordered.columns:
            ordered["target_date"] = pd.to_datetime(ordered["target_date"])
            ordered = ordered.sort_values("target_date").reset_index(drop=True)

        scaled_sequence = ordered.loc[:, feature_columns].to_numpy(dtype=np.float32)
        raw_sequence = self._inverse_scale(
            scaled_sequence,
            min_offset=scaler_metadata["min_offset"],
            scale=scaler_metadata["scale"],
        ).astype(np.float32)
        X_price = scaled_sequence.reshape(-1, request.lookback, 1)
        X_returns = self._build_sequence_return_tensor(raw_sequence)
        engineered_feature_columns = self._select_engineered_feature_columns(
            ordered=ordered,
            feature_input_mode=request.feature_input_mode,
        )
        X_engineered = (
            ordered.loc[:, engineered_feature_columns].to_numpy(dtype=np.float32)
            if engineered_feature_columns
            else np.empty((len(ordered.index), 0), dtype=np.float32)
        )
        X_model: Any
        if request.feature_input_mode == "technical_returns":
            X_model = [X_returns, X_engineered]
        else:
            X_model = X_price
        current_raw = self._inverse_scale(
            ordered[feature_columns[-1]].to_numpy(dtype=np.float32),
            min_offset=scaler_metadata["min_offset"],
            scale=scaler_metadata["scale"],
        ).astype(np.float32)
        target_return = self._build_target_return_array(
            ordered=ordered,
            target_raw_column=target_raw_column,
            current_raw=current_raw,
        )
        y_scaled = (
            target_return
            if request.prediction_target_mode == "return"
            else ordered["y_scaled"].to_numpy(dtype=np.float32)
        )
        y_raw = ordered[target_raw_column].to_numpy(dtype=np.float32)
        split = ordered["split"].astype(str).str.lower()

        train_mask = (split == "train").to_numpy()
        validation_mask = (split == "validation").to_numpy()
        test_mask = (split == "test").to_numpy()

        if not train_mask.any():
            raise ValueError("Refined dataset does not contain any train rows.")
        if not test_mask.any():
            raise ValueError("Refined dataset does not contain any test rows.")

        return {
            "row_count": int(len(ordered.index)),
            "X_train": X_price[train_mask],
            "X_train_model": self._slice_model_inputs(X_model, train_mask),
            "y_train_scaled": y_scaled[train_mask],
            "y_train_raw": y_raw[train_mask],
            "current_train_raw": current_raw[train_mask],
            "X_validation": X_price[validation_mask],
            "X_validation_model": self._slice_model_inputs(X_model, validation_mask),
            "y_validation_scaled": y_scaled[validation_mask],
            "y_validation_raw": y_raw[validation_mask],
            "current_validation_raw": current_raw[validation_mask],
            "X_test": X_price[test_mask],
            "X_test_model": self._slice_model_inputs(X_model, test_mask),
            "y_test_scaled": y_scaled[test_mask],
            "y_test_raw": y_raw[test_mask],
            "current_test_raw": current_raw[test_mask],
            "train_count": int(train_mask.sum()),
            "validation_count": int(validation_mask.sum()),
            "test_count": int(test_mask.sum()),
            "sequence_input_shape": (
                (request.lookback - 1, 1)
                if request.feature_input_mode == "technical_returns"
                else (request.lookback, 1)
            ),
            "sequence_input_kind": (
                "returns"
                if request.feature_input_mode == "technical_returns"
                else "scaled_price"
            ),
            "sequence_length": (
                request.lookback - 1
                if request.feature_input_mode == "technical_returns"
                else request.lookback
            ),
            "engineered_feature_columns": tuple(engineered_feature_columns),
            "engineered_feature_count": len(engineered_feature_columns),
            "model_feature_count": (
                len(engineered_feature_columns) + 1
                if request.feature_input_mode == "technical_returns"
                else 1
            ),
        }

    @staticmethod
    def _slice_model_inputs(X_model: Any, mask: np.ndarray) -> Any:
        if isinstance(X_model, list):
            return [part[mask] for part in X_model]
        return X_model[mask]

    @staticmethod
    def _build_sequence_return_tensor(raw_sequence: np.ndarray) -> np.ndarray:
        previous = raw_sequence[:, :-1]
        current = raw_sequence[:, 1:]
        returns = np.zeros_like(current, dtype=np.float32)
        non_zero_mask = np.abs(previous) > 1e-8
        returns[non_zero_mask] = current[non_zero_mask] / previous[non_zero_mask] - 1.0
        return returns.reshape(raw_sequence.shape[0], raw_sequence.shape[1] - 1, 1)

    @staticmethod
    def _select_engineered_feature_columns(
        *,
        ordered: pd.DataFrame,
        feature_input_mode: str,
    ) -> list[str]:
        if feature_input_mode != "technical_returns":
            return []
        excluded = {
            "feature_current_price",
            "feature_window_mean",
            "feature_window_std",
            "feature_window_min",
            "feature_window_max",
            "feature_window_range",
        }
        columns = [
            column
            for column in ordered.columns
            if column.startswith("feature_") and column not in excluded
        ]
        if not columns:
            raise ValueError(
                "feature_input_mode='technical_returns' requires a feature dataset "
                "with engineered feature_* columns. Run scripts/generate_features.py "
                "after scripts/generate_refined.py."
            )
        return columns

    @staticmethod
    def _build_target_return_array(
        *,
        ordered: pd.DataFrame,
        target_raw_column: str,
        current_raw: np.ndarray,
    ) -> np.ndarray:
        if "target_return_1d" in ordered.columns:
            return ordered["target_return_1d"].to_numpy(dtype=np.float32)
        target_value = ordered[target_raw_column].to_numpy(dtype=np.float32)
        returns = np.zeros(len(ordered.index), dtype=np.float32)
        non_zero_mask = np.abs(current_raw) > 1e-8
        returns[non_zero_mask] = target_value[non_zero_mask] / current_raw[non_zero_mask] - 1.0
        return returns

    def _set_random_seed(self, seed: int) -> None:
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        tf.random.set_seed(seed)

    def _build_model(
        self,
        *,
        sequence_input_shape: tuple[int, int],
        engineered_feature_count: int,
        learning_rate: float,
        feature_input_mode: str,
        l2_reg: float,
        clip_norm: float,
    ) -> Any:
        """
        technical_returns  →  LSTM(128→64) + Dense(64) branch → merge → Dense(64→32) → output
        sequence_price     →  LSTM(128→64) → Dense(32) → output

        Mudanças em relação à versão anterior:
        - Capacidade dobrada: LSTM 64→128, 32→64; Dense 32→64 no merge
        - BatchNormalization antes de cada bloco Dense para estabilizar gradientes
        - L2 regularization nas camadas Dense para reduzir overfitting
        - Gradient clipping via clipnorm no Adam
        - learning_rate configurável (default 0.0005 ao invés de 0.001)
        """
        reg = l2(l2_reg)

        if feature_input_mode == "technical_returns":
            # ── Branch sequencial (retornos diários) ──
            sequence_input = Input(shape=sequence_input_shape, name="sequence_returns")
            x = LSTM(128, return_sequences=True)(sequence_input)      # ← era 64
            x = Dropout(0.2)(x)
            x = LSTM(64, return_sequences=False)(x)                   # ← era 32
            x = Dropout(0.2)(x)

            # ── Branch features técnicas ──
            feature_input = Input(
                shape=(engineered_feature_count,),
                name="engineered_features",
            )
            f = BatchNormalization()(feature_input)                    # ← novo
            f = Dense(64, activation="relu", kernel_regularizer=reg)(f)  # ← era 32
            f = Dropout(0.1)(f)

            # ── Merge ──
            merged = Concatenate()([x, f])
            merged = BatchNormalization()(merged)                      # ← novo
            merged = Dense(64, activation="relu", kernel_regularizer=reg)(merged)  # ← era 32
            merged = Dropout(0.2)(merged)                              # ← novo dropout
            merged = Dense(32, activation="relu", kernel_regularizer=reg)(merged)  # ← nova camada
            output = Dense(1)(merged)

            model = Model(inputs=[sequence_input, feature_input], outputs=output)

        else:
            # ── Modo preço sequencial ──
            model = Sequential(
                [
                    Input(shape=sequence_input_shape),
                    LSTM(128, return_sequences=True),
                    Dropout(0.2),
                    LSTM(64, return_sequences=False),
                    Dropout(0.2),
                    BatchNormalization(),                               # ← novo
                    Dense(64, activation="relu", kernel_regularizer=reg),  # ← era 32
                    Dropout(0.2),                                       # ← novo dropout
                    Dense(32, activation="relu", kernel_regularizer=reg),  # ← nova camada
                    Dense(1),
                ]
            )

        model.compile(
            optimizer=Adam(
                learning_rate=learning_rate,
                clipnorm=clip_norm,            # ← novo: evita explosão de gradiente
            ),
            loss="mse",
            metrics=["mae"],
        )
        return model

    def _build_history_payload(
        self,
        *,
        history: dict[str, list[float]],
        monitor_metric: str,
    ) -> dict[str, Any]:
        serialized_history = {
            key: [float(value) for value in values]
            for key, values in history.items()
        }
        monitored_values = serialized_history.get(monitor_metric, [])
        best_epoch = 0
        if monitored_values:
            best_epoch = int(np.argmin(monitored_values)) + 1

        return {
            "monitor_metric": monitor_metric,
            "best_epoch": best_epoch,
            "epochs_ran": len(serialized_history.get("loss", [])),
            "history": serialized_history,
        }

    def _evaluate_split(
        self,
        *,
        model: Any,
        X: Any,
        y_scaled: np.ndarray,
        y_raw: np.ndarray,
        current_raw: np.ndarray,
        scaler_metadata: dict[str, float],
        prediction_target_mode: str,
    ) -> SplitMetrics:
        sample_count = len(X[0]) if isinstance(X, list) else len(X)
        if sample_count == 0:
            return SplitMetrics(
                sample_count=0,
                loss_mse=None,
                mae_scaled=None,
                mae=None,
                rmse=None,
                mape=None,
            )

        evaluation = model.evaluate(X, y_scaled, verbose=0)
        predictions_scaled = model.predict(X, verbose=0).reshape(-1)
        if prediction_target_mode == "return":
            predictions_raw = current_raw * (1.0 + predictions_scaled)
        else:
            predictions_raw = self._inverse_scale(
                predictions_scaled,
                min_offset=scaler_metadata["min_offset"],
                scale=scaler_metadata["scale"],
            )

        mae = float(np.mean(np.abs(y_raw - predictions_raw)))
        rmse = float(np.sqrt(np.mean(np.square(y_raw - predictions_raw))))
        non_zero_mask = np.abs(y_raw) > 1e-8
        mape = None
        if np.any(non_zero_mask):
            mape = float(
                np.mean(
                    np.abs(
                        (y_raw[non_zero_mask] - predictions_raw[non_zero_mask])
                        / y_raw[non_zero_mask]
                    )
                )
                * 100
            )

        return SplitMetrics(
            sample_count=int(sample_count),
            loss_mse=float(evaluation[0]),
            mae_scaled=float(evaluation[1]) if len(evaluation) > 1 else None,
            mae=mae,
            rmse=rmse,
            mape=mape,
        )

    def _write_training_report(
        self,
        *,
        destination: Path,
        symbol: str,
        request: KerasTrainingRequest,
        generated_at_utc: str,
        dataset: dict[str, Any],
        history_payload: dict[str, Any],
        model_path: Path,
        published_model_path: Path | None,
        history_path: Path,
        loss_chart_path: Path,
        metrics_chart_path: Path,
        train_metrics: SplitMetrics,
        validation_metrics: SplitMetrics,
        test_metrics: SplitMetrics,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = "\n".join(
            self._format_regression_metric_row(name, metrics)
            for name, metrics in (
                ("Train", train_metrics),
                ("Validation", validation_metrics),
                ("Test", test_metrics),
            )
        )
        content = f"""# Relatorio de Treinamento - {symbol.upper()}

## Execucao

| Campo | Valor |
| --- | --- |
| Gerado em UTC | `{generated_at_utc}` |
| Source | `{request.source}` |
| Data de extracao | `{request.extraction_date.isoformat()}` |
| Coluna alvo | `{request.target_column}` |
| Lookback | `{request.lookback}` |
| Epocas solicitadas | `{request.epochs}` |
| Epocas executadas | `{history_payload["epochs_ran"]}` |
| Melhor epoca | `{history_payload["best_epoch"]}` |
| Batch size | `{request.batch_size}` |
| Paciencia do early stopping | `{request.patience}` |
| Learning rate inicial | `{request.learning_rate}` |
| L2 regularization | `{request.l2_reg}` |
| Gradient clip norm | `{request.clip_norm}` |
| Seed | `{request.seed}` |
| Prediction target mode | `{request.prediction_target_mode}` |
| Feature input mode | `{request.feature_input_mode}` |
| Sequence input kind | `{dataset["sequence_input_kind"]}` |
| Sequence length | `{dataset["sequence_length"]}` |
| Engineered feature count | `{dataset["engineered_feature_count"]}` |
| Shuffle no treino | `{"sim (modo retorno)" if request.prediction_target_mode == "return" else "nao (modo preco)"}` |

## Dataset

| Split | Linhas |
| --- | ---: |
| Train | {dataset["train_count"]} |
| Validation | {dataset["validation_count"]} |
| Test | {dataset["test_count"]} |
| Total | {dataset["row_count"]} |

## Metricas

| Split | Samples | Loss MSE | MAE scaled | MAE | RMSE | MAPE % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

## Graficos

![Training loss]({loss_chart_path.name})

![Training metrics]({metrics_chart_path.name})

## Artefatos

| Artefato | Caminho |
| --- | --- |
| Modelo imutavel | `{model_path}` |
| Modelo latest publicado | `{published_model_path}` |
| History JSON | `{history_path}` |

## Notas Tecnicas

- Este relatorio descreve apenas o comportamento de treinamento. A qualidade real deve ser avaliada nas previsoes geradas contra baselines.
- Em `sequence_price`, o LSTM recebe uma janela univariada de preco escalado.
- Em `technical_returns`, o LSTM recebe retornos sequenciais e features tecnicas; colunas absolutas de preco nao entram no modelo.
- Grandes diferencas entre validacao e teste indicam generalizacao temporal fraca, nao evidencia pronta para producao.
- O ReduceLROnPlateau reduz o learning rate automaticamente quando val_loss estagna, permitindo convergencia mais fina sem treinar manualmente com LR menor.
- Shuffle ativado para modo retorno: retornos diarios tem baixa autocorrelacao, e a ordem temporal nao e essencial para o sinal; o shuffle reduz vies de sequencia.
"""
        destination.write_text(content, encoding="utf-8")

    @staticmethod
    def _format_regression_metric_row(name: str, metrics: SplitMetrics) -> str:
        return (
            f"| {name} | {metrics.sample_count} | "
            f"{KerasTrainingService._format_optional_float(metrics.loss_mse)} | "
            f"{KerasTrainingService._format_optional_float(metrics.mae_scaled)} | "
            f"{KerasTrainingService._format_optional_float(metrics.mae)} | "
            f"{KerasTrainingService._format_optional_float(metrics.rmse)} | "
            f"{KerasTrainingService._format_optional_float(metrics.mape)} |"
        )

    @staticmethod
    def _format_optional_float(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.6g}"

    @staticmethod
    def _write_line_chart_svg(
        *,
        destination: Path,
        title: str,
        series: dict[str, list[float]],
        y_label: str,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        width, height = 840, 360
        left, right, top, bottom = 64, 28, 44, 54
        plot_width = width - left - right
        plot_height = height - top - bottom
        colors = {"loss": "#1f77b4", "val_loss": "#d62728"}
        values = [
            float(value)
            for values_for_series in series.values()
            for value in values_for_series
            if value is not None and np.isfinite(value)
        ]
        max_len = max((len(values_for_series) for values_for_series in series.values()), default=0)
        if not values or max_len == 0:
            destination.write_text(
                KerasTrainingService._empty_svg(title, "No training history available."),
                encoding="utf-8",
            )
            return

        y_min = min(values)
        y_max = max(values)
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0
        padding = (y_max - y_min) * 0.08
        y_min -= padding
        y_max += padding

        def x_for(index: int) -> float:
            denominator = max(max_len - 1, 1)
            return left + (index / denominator) * plot_width

        def y_for(value: float) -> float:
            return top + ((y_max - value) / (y_max - y_min)) * plot_height

        polylines: list[str] = []
        legend: list[str] = []
        for index, (name, raw_values) in enumerate(series.items()):
            clean_values = [
                float(value)
                for value in raw_values
                if value is not None and np.isfinite(value)
            ]
            if not clean_values:
                continue
            points = " ".join(
                f"{x_for(point_index):.1f},{y_for(value):.1f}"
                for point_index, value in enumerate(clean_values)
            )
            color = colors.get(name, "#2ca02c")
            polylines.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" '
                'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />'
            )
            legend_y = 72 + index * 22
            legend.append(
                f'<rect x="690" y="{legend_y - 10}" width="12" height="12" fill="{color}" />'
                f'<text x="708" y="{legend_y}" font-size="13">{escape(name)}</text>'
            )

        y_ticks = []
        for tick in range(5):
            value = y_min + ((y_max - y_min) * tick / 4)
            y = y_for(value)
            y_ticks.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
                'stroke="#e5e7eb" />'
                f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
                f'font-size="11">{value:.4g}</text>'
            )

        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff" />
<text x="{left}" y="26" font-size="20" font-weight="700">{escape(title)}</text>
<text x="{left}" y="{height - 16}" font-size="12">Epoch</text>
<text x="18" y="{top + plot_height / 2:.1f}" font-size="12" transform="rotate(-90 18 {top + plot_height / 2:.1f})">{escape(y_label)}</text>
{''.join(y_ticks)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111827" />
<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111827" />
{''.join(polylines)}
{''.join(legend)}
</svg>
"""
        destination.write_text(svg, encoding="utf-8")

    @staticmethod
    def _write_keras_metrics_svg(
        *,
        destination: Path,
        train_metrics: SplitMetrics,
        validation_metrics: SplitMetrics,
        test_metrics: SplitMetrics,
    ) -> None:
        bars = [
            ("Train MAE", train_metrics.mae, "#1f77b4"),
            ("Validation MAE", validation_metrics.mae, "#ff7f0e"),
            ("Test MAE", test_metrics.mae, "#d62728"),
            ("Train RMSE", train_metrics.rmse, "#4f46e5"),
            ("Validation RMSE", validation_metrics.rmse, "#059669"),
            ("Test RMSE", test_metrics.rmse, "#b91c1c"),
        ]
        KerasTrainingService._write_bar_chart_svg(
            destination=destination,
            title="Regression Metrics by Split",
            bars=bars,
            y_label="Original target scale",
        )

    @staticmethod
    def _write_bar_chart_svg(
        *,
        destination: Path,
        title: str,
        bars: list[tuple[str, float | None, str]],
        y_label: str,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        values = [float(value) for _, value, _ in bars if value is not None and np.isfinite(value)]
        if not values:
            destination.write_text(
                KerasTrainingService._empty_svg(title, "No metric values available."),
                encoding="utf-8",
            )
            return
        width, height = 900, 380
        left, right, top, bottom = 72, 24, 46, 104
        plot_width = width - left - right
        plot_height = height - top - bottom
        y_max = max(values) * 1.12
        y_max = y_max if y_max > 0 else 1.0
        bar_gap = 18
        bar_width = (plot_width - bar_gap * (len(bars) - 1)) / max(len(bars), 1)
        svg_bars: list[str] = []
        for index, (label, value, color) in enumerate(bars):
            numeric_value = 0.0 if value is None or not np.isfinite(value) else float(value)
            x = left + index * (bar_width + bar_gap)
            bar_height = (numeric_value / y_max) * plot_height
            y = top + plot_height - bar_height
            svg_bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" fill="{color}" />'
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" '
                f'text-anchor="middle" font-size="11">{numeric_value:.4g}</text>'
                f'<text x="{x + bar_width / 2:.1f}" y="{height - 72}" '
                f'text-anchor="end" font-size="11" transform="rotate(-35 {x + bar_width / 2:.1f} {height - 72})">{escape(label)}</text>'
            )
        y_ticks = []
        for tick in range(5):
            value = y_max * tick / 4
            y = top + plot_height - (value / y_max) * plot_height
            y_ticks.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb" />'
                f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{value:.4g}</text>'
            )
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff" />
<text x="{left}" y="28" font-size="20" font-weight="700">{escape(title)}</text>
<text x="18" y="{top + plot_height / 2:.1f}" font-size="12" transform="rotate(-90 18 {top + plot_height / 2:.1f})">{escape(y_label)}</text>
{''.join(y_ticks)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" />
<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" stroke="#111827" />
{''.join(svg_bars)}
</svg>
"""
        destination.write_text(svg, encoding="utf-8")

    @staticmethod
    def _inverse_scale(
        values: np.ndarray,
        *,
        min_offset: float,
        scale: float,
    ) -> np.ndarray:
        if scale == 0:
            raise ValueError("Cannot inverse scale predictions because scale is zero.")
        return (values - min_offset) / scale

    @staticmethod
    def _empty_svg(title: str, message: str) -> str:
        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="840" height="260" viewBox="0 0 840 260">
<rect width="100%" height="100%" fill="#ffffff" />
<text x="42" y="42" font-size="20" font-weight="700">{escape(title)}</text>
<text x="42" y="92" font-size="14">{escape(message)}</text>
</svg>
"""

    @staticmethod
    def _to_path_safe_timestamp(generated_at_utc: str) -> str:
        parsed = datetime.fromisoformat(generated_at_utc)
        return parsed.strftime("%Y%m%dT%H%M%SZ")

    @staticmethod
    def _build_model_relative_path(
        *,
        source: str,
        symbol: str,
        lookback: int,
        extraction_date: date,
        trained_at_token: str,
        model_name_prefix: str,
    ) -> Path:
        return (
            Path("training_runs")
            / f"source={source}"
            / f"symbol={symbol.upper()}"
            / f"lookback={lookback}"
            / f"extraction_date={extraction_date.isoformat()}"
            / f"trained_at={trained_at_token}"
            / f"{model_name_prefix}_{symbol.lower()}.keras"
        )

    def _publish_latest_model_alias(
        self,
        *,
        model_path: Path,
        symbol: str,
        model_name_prefix: str,
    ) -> Path:
        published_relative_path = self._build_published_model_relative_path(
            symbol=symbol,
            model_name_prefix=model_name_prefix,
        )
        published_path = self._local_store.prepare_path(published_relative_path)
        shutil.copy2(model_path, published_path)
        return published_path

    @staticmethod
    def _build_published_model_relative_path(
        *,
        symbol: str,
        model_name_prefix: str,
    ) -> Path:
        return Path(f"{model_name_prefix}_{symbol.lower()}.keras")

    @staticmethod
    def _build_history_relative_path(
        *,
        source: str,
        symbol: str,
        lookback: int,
        extraction_date: date,
        trained_at_token: str,
    ) -> Path:
        return (
            Path("training_runs")
            / f"source={source}"
            / f"symbol={symbol.upper()}"
            / f"lookback={lookback}"
            / f"extraction_date={extraction_date.isoformat()}"
            / f"trained_at={trained_at_token}"
            / "history.json"
        )

    @staticmethod
    def _build_run_artifact_relative_path(
        *,
        source: str,
        symbol: str,
        lookback: int,
        extraction_date: date,
        trained_at_token: str,
        filename: str,
    ) -> Path:
        return (
            Path("training_runs")
            / f"source={source}"
            / f"symbol={symbol.upper()}"
            / f"lookback={lookback}"
            / f"extraction_date={extraction_date.isoformat()}"
            / f"trained_at={trained_at_token}"
            / filename
        )

    @staticmethod
    def _build_manifest_relative_path(
        *,
        extraction_date: date,
        trained_at_token: str,
    ) -> Path:
        return (
            Path("manifests")
            / f"extraction_date={extraction_date.isoformat()}"
            / f"trained_at={trained_at_token}"
            / "keras_training_manifest.json"
        )


def main() -> int:
    print(
        "KerasTrainingService is a library module. "
        "Use `python scripts/train_keras.py --help` to run model training."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())