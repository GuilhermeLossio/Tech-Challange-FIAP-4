from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import json
from typing import Any


@dataclass(frozen=True)
class KerasTrainingSummary:
    symbol: str
    extraction_date: str
    trained_at: str
    generated_at_utc: str | None
    epochs_ran: int
    best_epoch: int
    row_count: int
    train_count: int
    validation_count: int
    test_count: int
    train_metrics: dict[str, Any]
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    history: dict[str, list[float]]
    history_monitor_metric: str | None
    model_local_path: str
    history_local_path: str | None


@dataclass(frozen=True)
class QuantumTrainingSummary:
    symbol: str
    extraction_date: str
    trained_at: str
    generated_at_utc: str | None
    backend_name: str
    execution_mode: str
    num_qubits: int
    optimizer_name: str
    optimizer_maxiter: int
    function_evaluations: int | None
    objective_value: float | None
    train_metrics: dict[str, Any]
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    model_local_path: str
    training_details_local_path: str | None


@dataclass(frozen=True)
class TrainingDashboardSummary:
    symbol: str
    extraction_date: str
    available_extraction_dates: tuple[str, ...]
    keras: KerasTrainingSummary | None
    quantum: QuantumTrainingSummary | None


class TrainingCatalogService:
    def __init__(
        self,
        *,
        models_root_dir: Path,
        source: str = "yfinance",
        target_column: str = "close",
        lookback: int = 60,
    ) -> None:
        self._models_root_dir = models_root_dir
        self._source = source
        self._target_column = target_column
        self._lookback = lookback

    def list_available_extraction_dates(self) -> tuple[str, ...]:
        manifests_root = self._models_root_dir / "manifests"
        candidates: set[str] = set()
        if manifests_root.exists():
            for partition in manifests_root.glob("extraction_date=*"):
                token = partition.name.split("=", 1)[-1]
                try:
                    datetime.strptime(token, "%Y-%m-%d")
                except ValueError:
                    continue
                candidates.add(token)
        return tuple(sorted(candidates))

    def get_latest_extraction_date(self) -> str:
        dates = self.list_available_extraction_dates()
        if not dates:
            raise FileNotFoundError(
                "No training manifests were found under models/manifests."
            )
        return dates[-1]

    def get_dashboard_summary(
        self,
        *,
        symbol: str,
        extraction_date: str | None = None,
    ) -> TrainingDashboardSummary:
        available_extraction_dates = self.list_available_extraction_dates()
        if not available_extraction_dates:
            raise FileNotFoundError(
                "No training manifests were found under models/manifests."
            )

        selected_extraction_date = extraction_date or available_extraction_dates[-1]
        if selected_extraction_date not in available_extraction_dates:
            eligible = [
                candidate
                for candidate in available_extraction_dates
                if candidate <= selected_extraction_date
            ]
            if not eligible:
                raise FileNotFoundError(
                    "No training extraction_date is available on or before "
                    f"{selected_extraction_date}."
                )
            selected_extraction_date = eligible[-1]

        return TrainingDashboardSummary(
            symbol=symbol.upper(),
            extraction_date=selected_extraction_date,
            available_extraction_dates=available_extraction_dates,
            keras=self._load_keras_summary(
                symbol=symbol.upper(),
                extraction_date=selected_extraction_date,
            ),
            quantum=self._load_quantum_summary(
                symbol=symbol.upper(),
                extraction_date=selected_extraction_date,
            ),
        )

    def _load_keras_summary(
        self,
        *,
        symbol: str,
        extraction_date: str,
    ) -> KerasTrainingSummary | None:
        manifests_root = self._models_root_dir / "manifests" / f"extraction_date={extraction_date}"
        if not manifests_root.exists():
            return None

        best_candidate: tuple[
            tuple[float, float, float, float, int],
            Path,
            dict[str, Any],
            dict[str, Any],
        ] | None = None
        for manifest_path in manifests_root.glob("trained_at=*/keras_training_manifest.json"):
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            request_payload = payload.get("request", {})
            if (
                request_payload.get("source") != self._source
                or request_payload.get("target_column") != self._target_column
                or int(request_payload.get("lookback", self._lookback)) != self._lookback
            ):
                continue

            asset = next(
                (
                    item
                    for item in payload.get("assets", [])
                    if str(item.get("symbol", "")).upper() == symbol.upper()
                ),
                None,
            )
            if asset is None:
                continue

            candidate_score = self._score_regression_asset(
                asset_payload=asset,
                manifest_path=manifest_path,
            )
            if best_candidate is None or candidate_score < best_candidate[0]:
                best_candidate = (
                    candidate_score,
                    manifest_path,
                    payload,
                    asset,
                )

        if best_candidate is None:
            return None

        _, manifest_path, payload, asset = best_candidate
        history_payload = self._load_json_if_exists(
            Path(str(asset.get("history_local_path", "")))
        )
        return KerasTrainingSummary(
            symbol=symbol.upper(),
            extraction_date=extraction_date,
            trained_at=manifest_path.parent.name.split("=", 1)[-1],
            generated_at_utc=payload.get("generated_at_utc"),
            epochs_ran=int(asset.get("epochs_ran", 0)),
            best_epoch=int(asset.get("best_epoch", 0)),
            row_count=int(asset.get("row_count", 0)),
            train_count=int(asset.get("train_count", 0)),
            validation_count=int(asset.get("validation_count", 0)),
            test_count=int(asset.get("test_count", 0)),
            train_metrics=dict(asset.get("train_metrics", {})),
            validation_metrics=dict(asset.get("validation_metrics", {})),
            test_metrics=dict(asset.get("test_metrics", {})),
            history=dict((history_payload or {}).get("history", {})),
            history_monitor_metric=(history_payload or {}).get("monitor_metric"),
            model_local_path=str(asset.get("model_local_path", "")),
            history_local_path=(
                str(asset.get("history_local_path", ""))
                if asset.get("history_local_path")
                else None
            ),
        )
        

    def _load_quantum_summary(
        self,
        *,
        symbol: str,
        extraction_date: str,
    ) -> QuantumTrainingSummary | None:
        manifests_root = self._models_root_dir / "manifests" / f"extraction_date={extraction_date}"
        if not manifests_root.exists():
            return None

        for manifest_path in sorted(
            manifests_root.glob("trained_at=*/quantum_training_manifest.json"),
            reverse=True,
        ):
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            request_payload = payload.get("request", {})
            if (
                request_payload.get("source") != self._source
                or request_payload.get("target_column") != self._target_column
                or int(request_payload.get("lookback", self._lookback)) != self._lookback
            ):
                continue

            asset = next(
                (
                    item
                    for item in payload.get("assets", [])
                    if str(item.get("symbol", "")).upper() == symbol.upper()
                ),
                None,
            )
            if asset is None:
                continue

            return QuantumTrainingSummary(
                symbol=symbol.upper(),
                extraction_date=extraction_date,
                trained_at=manifest_path.parent.name.split("=", 1)[-1],
                generated_at_utc=payload.get("generated_at_utc"),
                backend_name=str(asset.get("backend_name", "unknown")),
                execution_mode=str(asset.get("execution_mode", "unknown")),
                num_qubits=int(asset.get("num_qubits", 0)),
                optimizer_name=str(asset.get("optimizer_name", "")),
                optimizer_maxiter=int(asset.get("optimizer_maxiter", 0)),
                function_evaluations=(
                    int(asset["function_evaluations"])
                    if asset.get("function_evaluations") is not None
                    else None
                ),
                objective_value=(
                    float(asset["objective_value"])
                    if asset.get("objective_value") is not None
                    else None
                ),
                train_metrics=dict(asset.get("train_metrics", {})),
                validation_metrics=dict(asset.get("validation_metrics", {})),
                test_metrics=dict(asset.get("test_metrics", {})),
                model_local_path=str(asset.get("model_local_path", "")),
                training_details_local_path=(
                    str(asset.get("training_details_local_path", ""))
                    if asset.get("training_details_local_path")
                    else None
                ),
            )
        return None

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
    def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
        if not path or not str(path).strip():
            return None
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
