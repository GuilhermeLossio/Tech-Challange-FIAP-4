from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import os
import shutil
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    from qiskit.circuit.library import real_amplitudes, zz_feature_map
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    from qiskit_machine_learning.algorithms import VQC
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    COBYLA = SPSA = FakeManilaV2 = QiskitRuntimeService = Sampler = VQC = None
    generate_preset_pass_manager = real_amplitudes = zz_feature_map = None
    _QISKIT_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _QISKIT_IMPORT_ERROR = None

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.application.use_cases._dataset_loading import load_preferred_training_frame
from src.infrastructure.config.settings import load_env_file
from src.infrastructure.storage.local_model_store import LocalModelStore
from src.infrastructure.storage.s3_raw_store import S3RawStore


class QuantumTrainingInterruptedError(RuntimeError):
    def __init__(self, symbol: str) -> None:
        super().__init__(
            "Quantum training interrupted by user while fitting "
            f"{symbol.upper()}."
        )
        self.symbol = symbol.upper()


@dataclass(frozen=True)
class QuantumTrainingRequest:
    symbols: tuple[str, ...]
    extraction_date: date
    source: str = "yfinance"
    target_column: str = "close"
    lookback: int = 60
    execution_mode: str = "local"
    backend_name: str | None = None
    num_qubits: int = 2
    feature_map_reps: int = 1
    ansatz_reps: int = 1
    shots: int = 1024
    optimization_level: int = 1
    optimizer_name: str = "cobyla"
    optimizer_maxiter: int = 30
    max_train_samples: int = 64
    max_validation_samples: int = 32
    max_test_samples: int = 32
    seed: int = 42
    model_name_prefix: str = "quantum_vqc"


@dataclass(frozen=True)
class QuantumClassificationMetrics:
    sample_count: int
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    positive_rate: float | None
    confusion_matrix: dict[str, int]


@dataclass(frozen=True)
class QuantumModelArtifact:
    symbol: str
    execution_mode: str
    backend_name: str
    original_counts: dict[str, int]
    sampled_counts: dict[str, int]
    num_qubits: int
    optimizer_name: str
    optimizer_maxiter: int
    function_evaluations: int | None
    objective_value: float | None
    explained_variance_ratio: tuple[float, ...]
    model_local_path: str
    preprocessor_local_path: str
    training_details_local_path: str
    model_s3_uri: str | None
    preprocessor_s3_uri: str | None
    training_details_s3_uri: str | None
    train_metrics: QuantumClassificationMetrics
    validation_metrics: QuantumClassificationMetrics
    test_metrics: QuantumClassificationMetrics


@dataclass(frozen=True)
class QuantumTrainingResult:
    source: str
    generated_at_utc: str
    manifest_local_path: str
    manifest_s3_uri: str | None
    assets: tuple[QuantumModelArtifact, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "generated_at_utc": self.generated_at_utc,
            "manifest_local_path": self.manifest_local_path,
            "manifest_s3_uri": self.manifest_s3_uri,
            "asset_count": len(self.assets),
            "assets": [asdict(asset) for asset in self.assets],
        }


class TrainQuantumModelUseCase:
    """Train a small hybrid quantum classifier on refined market windows."""

    def __init__(
        self,
        processed_root_dir: Path,
        local_store: LocalModelStore,
        s3_store: S3RawStore | None = None,
    ) -> None:
        self._processed_root_dir = processed_root_dir
        self._local_store = local_store
        self._s3_store = s3_store

    def train(self, request: QuantumTrainingRequest) -> QuantumTrainingResult:
        self._ensure_quantum_dependencies_available()
        self._validate_request(request)

        generated_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        trained_at_token = self._to_path_safe_timestamp(generated_at_utc)
        artifacts: list[QuantumModelArtifact] = []

        for symbol in request.symbols:
            frame = self._load_training_frame(
                source=request.source,
                symbol=symbol,
                extraction_date=request.extraction_date,
                lookback=request.lookback,
                target_column=request.target_column,
            )
            dataset = self._build_direction_dataset(frame=frame, request=request)
            sampled_dataset = self._sample_dataset(dataset=dataset, request=request)
            transformed_dataset, preprocessing_artifact = self._build_quantum_features(
                dataset=sampled_dataset,
                request=request,
            )
            runtime_context = self._build_runtime_context(
                request=request,
                min_num_qubits=transformed_dataset["num_qubits"],
            )
            model, training_details = self._fit_quantum_model(
                dataset=transformed_dataset,
                request=request,
                runtime_context=runtime_context,
                symbol=symbol,
            )

            model_relative_path = self._build_model_relative_path(
                source=request.source,
                symbol=symbol,
                lookback=request.lookback,
                extraction_date=request.extraction_date,
                trained_at_token=trained_at_token,
                model_name_prefix=request.model_name_prefix,
            )
            model_payload = {
                "model_family": "variational_quantum_classifier",
                "task_type": "next_day_direction_classification",
                "symbol": symbol.upper(),
                "source": request.source,
                "execution_mode": request.execution_mode,
                "backend_name": runtime_context["backend_name"],
                "num_qubits": transformed_dataset["num_qubits"],
                "feature_map": {
                    "name": "zz_feature_map",
                    "reps": request.feature_map_reps,
                    "entanglement": "linear",
                },
                "ansatz": {
                    "name": "real_amplitudes",
                    "reps": request.ansatz_reps,
                },
                "optimizer": training_details["optimizer"],
                "weights": training_details["weights"],
                "class_labels": [0, 1],
                "preprocessing_summary": preprocessing_artifact["summary"],
            }
            model_path = self._local_store.write_json(model_payload, model_relative_path)
            self._publish_latest_model_alias(
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

            preprocessor_relative_path = self._build_preprocessor_relative_path(
                source=request.source,
                symbol=symbol,
                lookback=request.lookback,
                extraction_date=request.extraction_date,
                trained_at_token=trained_at_token,
            )
            preprocessor_path = self._local_store.prepare_path(preprocessor_relative_path)
            joblib.dump(preprocessing_artifact["bundle"], preprocessor_path)
            preprocessor_s3_uri = None
            if self._s3_store is not None:
                preprocessor_s3_uri = self._s3_store.upload_file(
                    local_path=preprocessor_path,
                    relative_path=preprocessor_relative_path,
                )

            details_relative_path = self._build_training_details_relative_path(
                source=request.source,
                symbol=symbol,
                lookback=request.lookback,
                extraction_date=request.extraction_date,
                trained_at_token=trained_at_token,
            )
            details_path = self._local_store.write_json(training_details, details_relative_path)
            training_details_s3_uri = None
            if self._s3_store is not None:
                training_details_s3_uri = self._s3_store.upload_file(
                    local_path=details_path,
                    relative_path=details_relative_path,
                )

            train_metrics = self._evaluate_classifier(
                model=model,
                X=transformed_dataset["X_train"],
                y=transformed_dataset["y_train"],
            )
            validation_metrics = self._evaluate_classifier(
                model=model,
                X=transformed_dataset["X_validation"],
                y=transformed_dataset["y_validation"],
            )
            test_metrics = self._evaluate_classifier(
                model=model,
                X=transformed_dataset["X_test"],
                y=transformed_dataset["y_test"],
            )

            artifacts.append(
                QuantumModelArtifact(
                    symbol=symbol.upper(),
                    execution_mode=request.execution_mode,
                    backend_name=runtime_context["backend_name"],
                    original_counts=sampled_dataset["original_counts"],
                    sampled_counts=sampled_dataset["sampled_counts"],
                    num_qubits=transformed_dataset["num_qubits"],
                    optimizer_name=training_details["optimizer"]["name"],
                    optimizer_maxiter=training_details["optimizer"]["effective_maxiter"],
                    function_evaluations=training_details["optimizer"]["function_evaluations"],
                    objective_value=training_details["optimizer"]["objective_value"],
                    explained_variance_ratio=tuple(
                        float(value)
                        for value in preprocessing_artifact["summary"][
                            "explained_variance_ratio"
                        ]
                    ),
                    model_local_path=str(model_path),
                    preprocessor_local_path=str(preprocessor_path),
                    training_details_local_path=str(details_path),
                    model_s3_uri=model_s3_uri,
                    preprocessor_s3_uri=preprocessor_s3_uri,
                    training_details_s3_uri=training_details_s3_uri,
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
                "execution_mode": request.execution_mode,
                "backend_name": request.backend_name,
                "num_qubits": request.num_qubits,
                "feature_map_reps": request.feature_map_reps,
                "ansatz_reps": request.ansatz_reps,
                "shots": request.shots,
                "optimization_level": request.optimization_level,
                "optimizer_name": request.optimizer_name,
                "optimizer_maxiter": request.optimizer_maxiter,
                "max_train_samples": request.max_train_samples,
                "max_validation_samples": request.max_validation_samples,
                "max_test_samples": request.max_test_samples,
                "seed": request.seed,
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

        return QuantumTrainingResult(
            source=request.source,
            generated_at_utc=generated_at_utc,
            manifest_local_path=str(manifest_local_path),
            manifest_s3_uri=manifest_s3_uri,
            assets=tuple(artifacts),
        )

    def list_cloud_backends(self, limit: int = 15) -> list[dict[str, Any]]:
        self._ensure_quantum_dependencies_available()
        service = self._build_cloud_service()
        backends = service.backends(operational=True, simulator=False)
        rows: list[dict[str, Any]] = []
        for backend in backends[:limit]:
            rows.append(
                {
                    "name": getattr(backend, "name", str(backend)),
                    "num_qubits": getattr(backend, "num_qubits", None),
                    "operational": True,
                }
            )
        return rows

    def _ensure_quantum_dependencies_available(self) -> None:
        if _QISKIT_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Quantum training requires Qiskit, IBM Runtime, and Qiskit Machine Learning. "
                "Install the project dependencies again after adding the quantum packages."
            ) from _QISKIT_IMPORT_ERROR

    def _validate_request(self, request: QuantumTrainingRequest) -> None:
        if not request.symbols:
            raise ValueError("At least one symbol must be provided.")
        if request.lookback <= 0:
            raise ValueError("lookback must be greater than zero.")
        if request.execution_mode not in {"local", "cloud"}:
            raise ValueError("execution_mode must be either 'local' or 'cloud'.")
        if request.num_qubits <= 0:
            raise ValueError("num_qubits must be greater than zero.")
        if request.feature_map_reps <= 0:
            raise ValueError("feature_map_reps must be greater than zero.")
        if request.ansatz_reps <= 0:
            raise ValueError("ansatz_reps must be greater than zero.")
        if request.shots <= 0:
            raise ValueError("shots must be greater than zero.")
        if request.optimization_level not in {0, 1, 2, 3}:
            raise ValueError("optimization_level must be one of: 0, 1, 2, 3.")
        if request.optimizer_name not in {"cobyla", "spsa"}:
            raise ValueError("optimizer_name must be either 'cobyla' or 'spsa'.")
        if request.optimizer_maxiter <= 0:
            raise ValueError("optimizer_maxiter must be greater than zero.")
        if request.max_train_samples <= 0:
            raise ValueError("max_train_samples must be greater than zero.")
        if request.max_validation_samples <= 0:
            raise ValueError("max_validation_samples must be greater than zero.")
        if request.max_test_samples <= 0:
            raise ValueError("max_test_samples must be greater than zero.")
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
    ) -> pd.DataFrame:
        frame, _, _ = load_preferred_training_frame(
            processed_root_dir=self._processed_root_dir,
            source=source,
            symbol=symbol,
            extraction_date=extraction_date,
            lookback=lookback,
            target_column=target_column,
        )
        if f"y_{target_column}" not in frame.columns:
            raise ValueError(
                f"Refined dataset for symbol {symbol!r} does not contain "
                f"`y_{target_column}`."
            )
        return frame

    def _build_direction_dataset(
        self,
        *,
        frame: pd.DataFrame,
        request: QuantumTrainingRequest,
    ) -> dict[str, Any]:
        engineered_feature_columns = [
            column for column in frame.columns
            if column.startswith("feature_")
        ]
        if engineered_feature_columns:
            feature_columns = engineered_feature_columns
        else:
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

        ordered = frame.copy()
        if "target_date" in ordered.columns:
            ordered["target_date"] = pd.to_datetime(ordered["target_date"])
            ordered = ordered.sort_values("target_date").reset_index(drop=True)

        current_price_column = f"{request.target_column}_t_minus_1"
        next_price_column = "y_scaled"
        if current_price_column not in ordered.columns:
            raise ValueError(
                f"Refined dataset is missing the current price column {current_price_column!r}."
            )
        if next_price_column not in ordered.columns:
            raise ValueError("Refined dataset is missing the `y_scaled` column.")
        if "split" not in ordered.columns:
            raise ValueError("Refined dataset is missing the `split` column.")

        X = ordered.loc[:, feature_columns].to_numpy(dtype=np.float64)
        if "target_direction" in ordered.columns:
            # When the feature-generation stage ran first, the label is already
            # persisted as an explicit column in the feature dataset.
            y = ordered["target_direction"].astype(int).to_numpy()
        else:
            # The binary label represents next-day direction:
            # 1 means the next close is above the latest point in the window, 0 otherwise.
            y = (ordered[next_price_column] > ordered[current_price_column]).astype(int).to_numpy()
        split = ordered["split"].astype(str).str.lower().to_numpy()

        train_mask = split == "train"
        validation_mask = split == "validation"
        test_mask = split == "test"

        if train_mask.sum() < 4:
            raise ValueError(
                "Quantum training needs at least four train rows after refinement."
            )
        if np.unique(y[train_mask]).size < 2:
            raise ValueError(
                "Quantum direction classification needs both classes in the train split."
            )

        original_counts = {
            "train": int(train_mask.sum()),
            "validation": int(validation_mask.sum()),
            "test": int(test_mask.sum()),
        }

        return {
            "feature_columns": feature_columns,
            "X_train": X[train_mask],
            "y_train": y[train_mask],
            "X_validation": X[validation_mask],
            "y_validation": y[validation_mask],
            "X_test": X[test_mask],
            "y_test": y[test_mask],
            "original_counts": original_counts,
        }

    def _sample_dataset(
        self,
        *,
        dataset: dict[str, Any],
        request: QuantumTrainingRequest,
    ) -> dict[str, Any]:
        X_train, y_train = self._limit_samples(
            X=dataset["X_train"],
            y=dataset["y_train"],
            max_samples=request.max_train_samples,
            seed=request.seed,
        )
        X_validation, y_validation = self._limit_samples(
            X=dataset["X_validation"],
            y=dataset["y_validation"],
            max_samples=request.max_validation_samples,
            seed=request.seed + 1,
        )
        X_test, y_test = self._limit_samples(
            X=dataset["X_test"],
            y=dataset["y_test"],
            max_samples=request.max_test_samples,
            seed=request.seed + 2,
        )

        if np.unique(y_train).size < 2:
            raise ValueError(
                "The sampled train split kept only one class. "
                "Increase max_train_samples or change the random seed."
            )

        sampled_counts = {
            "train": int(len(X_train)),
            "validation": int(len(X_validation)),
            "test": int(len(X_test)),
        }

        return {
            "feature_columns": dataset["feature_columns"],
            "X_train": X_train,
            "y_train": y_train,
            "X_validation": X_validation,
            "y_validation": y_validation,
            "X_test": X_test,
            "y_test": y_test,
            "original_counts": dataset["original_counts"],
            "sampled_counts": sampled_counts,
        }

    def _limit_samples(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        max_samples: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(X) == 0 or len(X) <= max_samples:
            return X, y

        indices = np.arange(len(X))
        stratify = y if np.unique(y).size > 1 else None
        selected_indices, _ = train_test_split(
            indices,
            train_size=max_samples,
            random_state=seed,
            stratify=stratify,
        )
        selected_indices = np.sort(selected_indices)
        return X[selected_indices], y[selected_indices]

    def _build_quantum_features(
        self,
        *,
        dataset: dict[str, Any],
        request: QuantumTrainingRequest,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Quantum circuits can only carry a small number of meaningful features.
        # We therefore compress the original 60-lag window with PCA before angle encoding.
        standard_scaler = StandardScaler()
        X_train_standard = standard_scaler.fit_transform(dataset["X_train"])

        num_qubits = min(
            request.num_qubits,
            X_train_standard.shape[0],
            X_train_standard.shape[1],
        )
        if num_qubits <= 0:
            raise ValueError("Could not determine a valid number of qubits from the train set.")

        pca = PCA(n_components=num_qubits, random_state=request.seed)
        X_train_reduced = pca.fit_transform(X_train_standard)

        # Angle-encoding feature maps work best when classical values are mapped into
        # a bounded rotation range. We use [0, pi] to generate stable gate parameters.
        angle_scaler = MinMaxScaler(feature_range=(0.0, float(np.pi)))
        X_train_quantum = angle_scaler.fit_transform(X_train_reduced)

        def transform_split(values: np.ndarray) -> np.ndarray:
            if len(values) == 0:
                return np.empty((0, num_qubits), dtype=np.float64)
            values_standard = standard_scaler.transform(values)
            values_reduced = pca.transform(values_standard)
            return angle_scaler.transform(values_reduced)

        transformed_dataset = {
            "X_train": X_train_quantum,
            "y_train": dataset["y_train"],
            "X_validation": transform_split(dataset["X_validation"]),
            "y_validation": dataset["y_validation"],
            "X_test": transform_split(dataset["X_test"]),
            "y_test": dataset["y_test"],
            "num_qubits": num_qubits,
            "feature_columns": dataset["feature_columns"],
            "original_counts": dataset["original_counts"],
            "sampled_counts": dataset["sampled_counts"],
        }

        preprocessing_bundle = {
            "standard_scaler": standard_scaler,
            "pca": pca,
            "angle_scaler": angle_scaler,
            "feature_columns": dataset["feature_columns"],
            "num_qubits": num_qubits,
        }
        preprocessing_summary = {
            "feature_columns": dataset["feature_columns"],
            "num_qubits": num_qubits,
            "explained_variance_ratio": [
                float(value) for value in pca.explained_variance_ratio_
            ],
            "angle_range": [0.0, float(np.pi)],
        }

        return transformed_dataset, {
            "bundle": preprocessing_bundle,
            "summary": preprocessing_summary,
        }

    def _build_runtime_context(
        self,
        *,
        request: QuantumTrainingRequest,
        min_num_qubits: int,
    ) -> dict[str, Any]:
        if request.execution_mode == "local":
            backend = FakeManilaV2()
            sampler = Sampler(
                mode=backend,
                options={"simulator": {"seed_simulator": request.seed}},
            )
            sampler.options.default_shots = request.shots
            pass_manager = generate_preset_pass_manager(
                backend=backend,
                optimization_level=request.optimization_level,
            )
            return {
                "backend_name": getattr(backend, "name", str(backend)),
                "backend": backend,
                "sampler": sampler,
                "pass_manager": pass_manager,
            }

        service = self._build_cloud_service()
        if request.backend_name:
            backend = service.backend(request.backend_name)
        else:
            backend = service.least_busy(
                operational=True,
                simulator=False,
                min_num_qubits=min_num_qubits,
            )
        sampler = Sampler(mode=backend)
        sampler.options.default_shots = request.shots
        pass_manager = generate_preset_pass_manager(
            backend=backend,
            optimization_level=request.optimization_level,
        )
        return {
            "backend_name": getattr(backend, "name", str(backend)),
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

    def _fit_quantum_model(
        self,
        *,
        dataset: dict[str, Any],
        request: QuantumTrainingRequest,
        runtime_context: dict[str, Any],
        symbol: str,
    ) -> tuple[VQC, dict[str, Any]]:
        # The feature map encodes classical angles into a quantum state.
        # The ansatz adds trainable gates whose parameters are tuned by a classical optimizer.
        feature_map = zz_feature_map(
            feature_dimension=dataset["num_qubits"],
            reps=request.feature_map_reps,
            entanglement="linear",
        )
        ansatz = real_amplitudes(
            num_qubits=dataset["num_qubits"],
            reps=request.ansatz_reps,
        )
        optimizer, effective_maxiter = self._build_optimizer(
            request=request,
            ansatz_num_parameters=ansatz.num_parameters,
        )

        model = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            sampler=runtime_context["sampler"],
            pass_manager=runtime_context["pass_manager"],
            optimizer=optimizer,
        )

        try:
            model.fit(dataset["X_train"], dataset["y_train"])
        except KeyboardInterrupt as exc:
            raise QuantumTrainingInterruptedError(symbol=symbol) from exc

        fit_result = getattr(model, "fit_result", None)
        training_details = {
            "backend": {
                "execution_mode": request.execution_mode,
                "backend_name": runtime_context["backend_name"],
                "shots": request.shots,
                "optimization_level": request.optimization_level,
            },
            "optimizer": {
                "name": request.optimizer_name,
                "requested_maxiter": request.optimizer_maxiter,
                "effective_maxiter": effective_maxiter,
                "objective_value": (
                    float(getattr(fit_result, "fun"))
                    if fit_result is not None and getattr(fit_result, "fun", None) is not None
                    else None
                ),
                "function_evaluations": (
                    int(getattr(fit_result, "nfev"))
                    if fit_result is not None and getattr(fit_result, "nfev", None) is not None
                    else None
                ),
                "iterations": (
                    int(getattr(fit_result, "nit"))
                    if fit_result is not None and getattr(fit_result, "nit", None) is not None
                    else None
                ),
            },
            "weights": [float(value) for value in getattr(model, "weights", [])],
            "hybrid_training_explanation": (
                "The classical optimizer updates circuit weights. "
                "For each candidate weight vector, the Sampler primitive executes "
                "the parameterized circuit and returns measurement statistics. "
                "Those statistics are converted into a loss value, and the optimizer "
                "uses that loss to propose the next weight update."
            ),
        }
        return model, training_details

    def _build_optimizer(
        self,
        *,
        request: QuantumTrainingRequest,
        ansatz_num_parameters: int,
    ) -> tuple[Any, int]:
        if request.optimizer_name == "cobyla":
            effective_maxiter = max(request.optimizer_maxiter, ansatz_num_parameters + 2)
            return COBYLA(maxiter=effective_maxiter), effective_maxiter

        return SPSA(maxiter=request.optimizer_maxiter), request.optimizer_maxiter

    def _evaluate_classifier(
        self,
        *,
        model: VQC,
        X: np.ndarray,
        y: np.ndarray,
    ) -> QuantumClassificationMetrics:
        if len(X) == 0:
            return QuantumClassificationMetrics(
                sample_count=0,
                accuracy=None,
                precision=None,
                recall=None,
                f1=None,
                positive_rate=None,
                confusion_matrix={"tn": 0, "fp": 0, "fn": 0, "tp": 0},
            )

        predictions = np.asarray(model.predict(X)).reshape(-1).astype(int)
        matrix = confusion_matrix(y, predictions, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()

        return QuantumClassificationMetrics(
            sample_count=int(len(X)),
            accuracy=float(accuracy_score(y, predictions)),
            precision=float(precision_score(y, predictions, zero_division=0)),
            recall=float(recall_score(y, predictions, zero_division=0)),
            f1=float(f1_score(y, predictions, zero_division=0)),
            positive_rate=float(np.mean(predictions)),
            confusion_matrix={
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        )

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
            Path("quantum_training_runs")
            / f"source={source}"
            / f"symbol={symbol.upper()}"
            / f"lookback={lookback}"
            / f"extraction_date={extraction_date.isoformat()}"
            / f"trained_at={trained_at_token}"
            / f"{model_name_prefix}_{symbol.lower()}.json"
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
        return Path(f"{model_name_prefix}_{symbol.lower()}.json")

    @staticmethod
    def _build_preprocessor_relative_path(
        *,
        source: str,
        symbol: str,
        lookback: int,
        extraction_date: date,
        trained_at_token: str,
    ) -> Path:
        return (
            Path("quantum_training_runs")
            / f"source={source}"
            / f"symbol={symbol.upper()}"
            / f"lookback={lookback}"
            / f"extraction_date={extraction_date.isoformat()}"
            / f"trained_at={trained_at_token}"
            / "preprocessor.joblib"
        )

    @staticmethod
    def _build_training_details_relative_path(
        *,
        source: str,
        symbol: str,
        lookback: int,
        extraction_date: date,
        trained_at_token: str,
    ) -> Path:
        return (
            Path("quantum_training_runs")
            / f"source={source}"
            / f"symbol={symbol.upper()}"
            / f"lookback={lookback}"
            / f"extraction_date={extraction_date.isoformat()}"
            / f"trained_at={trained_at_token}"
            / "training_details.json"
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
            / "quantum_training_manifest.json"
        )


def main() -> int:
    print(
        "TrainQuantumModelUseCase is a library module. "
        "Use `python scripts/train_model_quantum.py --help` to run quantum training."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
