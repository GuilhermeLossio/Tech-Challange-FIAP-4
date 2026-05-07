from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # pyright: ignore[reportMissingModuleSource]
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM # pyright: ignore[reportMissingModuleSource]
    from tensorflow.keras.models import Sequential # type: ignore
    from tensorflow.keras.optimizers import Adam # pyright: ignore[reportMissingModuleSource]
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    tf = None
    EarlyStopping = ModelCheckpoint = Dense = Dropout = Input = LSTM = Sequential = Adam = None
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
    patience: int = 10
    learning_rate: float = 0.001
    seed: int = 42
    verbose: int = 1
    model_name_prefix: str = "lstm"


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
    epochs_ran: int
    best_epoch: int
    immutable_model_local_path: str
    published_model_local_path: str | None
    model_local_path: str
    history_local_path: str
    model_s3_uri: str | None
    history_s3_uri: str | None
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
            frame, scaler_metadata = self._load_training_frame(
                source=request.source,
                symbol=symbol,
                extraction_date=request.extraction_date,
                lookback=request.lookback,
                target_column=request.target_column,
            )
            dataset = self._build_training_dataset(frame=frame, request=request)
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
                input_shape=(request.lookback, 1),
                learning_rate=request.learning_rate,
            )

            monitor_metric = "val_loss" if dataset["validation_count"] > 0 else "loss"
            callbacks = [
                EarlyStopping(
                    monitor=monitor_metric,
                    patience=request.patience,
                    restore_best_weights=True,
                ),
                ModelCheckpoint(
                    filepath=str(model_path),
                    monitor=monitor_metric,
                    save_best_only=True,
                ),
            ]

            fit_kwargs: dict[str, Any] = {
                "x": dataset["X_train"],
                "y": dataset["y_train_scaled"],
                "epochs": request.epochs,
                "batch_size": request.batch_size,
                "callbacks": callbacks,
                "verbose": request.verbose,
                "shuffle": False,
            }
            if dataset["validation_count"] > 0:
                fit_kwargs["validation_data"] = (
                    dataset["X_validation"],
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
                X=dataset["X_train"],
                y_scaled=dataset["y_train_scaled"],
                y_raw=dataset["y_train_raw"],
                scaler_metadata=scaler_metadata,
            )
            validation_metrics = self._evaluate_split(
                model=model,
                X=dataset["X_validation"],
                y_scaled=dataset["y_validation_scaled"],
                y_raw=dataset["y_validation_raw"],
                scaler_metadata=scaler_metadata,
            )
            test_metrics = self._evaluate_split(
                model=model,
                X=dataset["X_test"],
                y_scaled=dataset["y_test_scaled"],
                y_raw=dataset["y_test_raw"],
                scaler_metadata=scaler_metadata,
            )

            artifacts.append(
                KerasModelArtifact(
                    symbol=symbol.upper(),
                    row_count=dataset["row_count"],
                    train_count=dataset["train_count"],
                    validation_count=dataset["validation_count"],
                    test_count=dataset["test_count"],
                    feature_count=1,
                    epochs_ran=history_payload["epochs_ran"],
                    best_epoch=history_payload["best_epoch"],
                    immutable_model_local_path=str(model_path),
                    published_model_local_path=str(published_model_path),
                    model_local_path=str(model_path),
                    history_local_path=str(history_path),
                    model_s3_uri=model_s3_uri,
                    history_s3_uri=history_s3_uri,
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

    def _load_training_frame(
        self,
        *,
        source: str,
        symbol: str,
        extraction_date: date,
        lookback: int,
        target_column: str,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        frame, scaler_metadata, _ = load_preferred_training_frame(
            processed_root_dir=self._processed_root_dir,
            source=source,
            symbol=symbol,
            extraction_date=extraction_date,
            lookback=lookback,
            target_column=target_column,
        )
        return frame, scaler_metadata

    def _build_training_dataset(
        self,
        *,
        frame: pd.DataFrame,
        request: KerasTrainingRequest,
    ) -> dict[str, Any]:
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

        X = ordered.loc[:, feature_columns].to_numpy(dtype=np.float32).reshape(
            -1, request.lookback, 1
        )
        y_scaled = ordered["y_scaled"].to_numpy(dtype=np.float32)
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
            "X_train": X[train_mask],
            "y_train_scaled": y_scaled[train_mask],
            "y_train_raw": y_raw[train_mask],
            "X_validation": X[validation_mask],
            "y_validation_scaled": y_scaled[validation_mask],
            "y_validation_raw": y_raw[validation_mask],
            "X_test": X[test_mask],
            "y_test_scaled": y_scaled[test_mask],
            "y_test_raw": y_raw[test_mask],
            "train_count": int(train_mask.sum()),
            "validation_count": int(validation_mask.sum()),
            "test_count": int(test_mask.sum()),
        }

    def _set_random_seed(self, seed: int) -> None:
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        tf.random.set_seed(seed)

    def _build_model(
        self,
        *,
        input_shape: tuple[int, int],
        learning_rate: float,
    ) -> Sequential:
        model = Sequential(
            [
                Input(shape=input_shape),
                LSTM(128, return_sequences=True),
                Dropout(0.2),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation="relu"),
                Dense(1),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
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
        model: Sequential,
        X: np.ndarray,
        y_scaled: np.ndarray,
        y_raw: np.ndarray,
        scaler_metadata: dict[str, float],
    ) -> SplitMetrics:
        if len(X) == 0:
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
            sample_count=int(len(X)),
            loss_mse=float(evaluation[0]),
            mae_scaled=float(evaluation[1]) if len(evaluation) > 1 else None,
            mae=mae,
            rmse=rmse,
            mape=mape,
        )

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
