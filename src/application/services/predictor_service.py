from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import json
import math
import os
from typing import Any

import numpy as np

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src.application.services.forecast_guardrails import (
    apply_standard_forecast_guardrail,
    clamp_interval_to_guardrail_band,
)
from src.application.services.model_promotion_registry import ModelPromotionRegistry

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    tf = None
    _TENSORFLOW_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _TENSORFLOW_IMPORT_ERROR = None


DEFAULT_SYMBOLS = ("NVDA", "AMD", "TSM", "ASML", "QCOM")


@dataclass(frozen=True)
class ModelResolution:
    symbol: str
    extraction_date: date
    model_name: str
    model_local_path: Path
    training_manifest_local_path: Path | None
    training_generated_at_utc: str | None
    interval_width: float | None


@dataclass(frozen=True)
class StandardPredictionResult:
    symbol: str
    predicted_close: float
    lower_bound: float
    upper_bound: float
    confidence: float
    currency: str
    model: str
    timestamp: str
    extraction_date: str
    target_column: str
    predict_type: str
    input_mode: str
    prices_provided_count: int
    requested_reference_date: str | None
    resolved_window_start_date: str | None
    resolved_window_end_date: str | None
    raw_model_predicted_close: float | None = None
    prediction_constraint_applied: bool = False
    prediction_constraint_method: str | None = None
    prediction_return_cap: float | None = None


class StandardPredictorService:
    def __init__(
        self,
        *,
        raw_root_dir: Path,
        processed_root_dir: Path,
        models_root_dir: Path,
        source: str = "yfinance",
        target_column: str = "close",
        lookback: int = 60,
        model_name_prefix: str = "lstm",
    ) -> None:
        self._raw_root_dir = raw_root_dir.resolve()
        self._processed_root_dir = processed_root_dir.resolve()
        self._models_root_dir = models_root_dir.resolve()
        self._source = source
        self._target_column = target_column
        self._lookback = lookback
        self._model_name_prefix = model_name_prefix
        self._model_cache: dict[str, Any] = {}
        self._promotion_registry = ModelPromotionRegistry(
            models_root_dir=self._models_root_dir,
        )

    def get_supported_symbols(self) -> tuple[str, ...]:
        promoted_symbols = self._promotion_registry.list_promoted_symbols()
        if promoted_symbols:
            return promoted_symbols

        symbols: set[str] = set()
        manifests_root = self._models_root_dir / "manifests"
        if manifests_root.exists():
            for manifest_path in manifests_root.glob(
                "extraction_date=*/trained_at=*/keras_training_manifest.json"
            ):
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                request_payload = payload.get("request", {})
                if (
                    request_payload.get("source") != self._source
                    or request_payload.get("target_column") != self._target_column
                    or int(request_payload.get("lookback", self._lookback)) != self._lookback
                ):
                    continue
                for asset in payload.get("assets", []):
                    symbol = str(asset.get("symbol", "")).strip().upper()
                    if symbol:
                        symbols.add(symbol)

        if not symbols:
            for model_path in self._models_root_dir.glob(f"{self._model_name_prefix}_*.keras"):
                suffix = model_path.stem.replace(f"{self._model_name_prefix}_", "", 1)
                if suffix:
                    symbols.add(suffix.upper())

        if not symbols:
            return DEFAULT_SYMBOLS
        return tuple(sorted(symbols))

    def get_latest_extraction_date(self) -> date:
        candidates = self._list_available_extraction_dates()
        if not candidates:
            raise FileNotFoundError(
                "No Keras training manifests were found. "
                "Run `python scripts/train_keras.py` first."
            )
        return max(candidates)

    def describe_registry(self) -> dict[str, Any]:
        latest_extraction_date = self.get_latest_extraction_date()
        supported_symbols = self.get_supported_symbols()
        default_symbol = "NVDA" if "NVDA" in supported_symbols else supported_symbols[0]
        resolution = self._resolve_serving_model_resolution(
            symbol=default_symbol,
            requested_extraction_date=None,
        )
        return {
            "latest_extraction_date": latest_extraction_date.isoformat(),
            "default_serving_extraction_date": resolution.extraction_date.isoformat(),
            "supported_symbols": list(supported_symbols),
            "default_model_name": resolution.model_name,
            "default_model_path": str(resolution.model_local_path),
            "online_quantum_inference_enabled": False,
            "promotion_policy_enabled": self._promotion_registry.exists(),
            "promotion_policy_path": (
                str(self._promotion_registry.policy_path)
                if self._promotion_registry.exists()
                else None
            ),
        }

    def resolve_serving_extraction_date(
        self,
        *,
        symbol: str,
        requested_extraction_date: date | None = None,
    ) -> date:
        return self._resolve_serving_model_resolution(
            symbol=symbol.strip().upper(),
            requested_extraction_date=requested_extraction_date,
        ).extraction_date

    def predict(
        self,
        *,
        symbol: str,
        prices: list[float] | None,
        extraction_date: date | None = None,
        reference_date: date | None = None,
    ) -> StandardPredictionResult:
        self._ensure_tensorflow_available()

        normalized_symbol = symbol.strip().upper()
        resolution = self._resolve_serving_model_resolution(
            symbol=normalized_symbol,
            requested_extraction_date=extraction_date,
        )
        selected_extraction_date = resolution.extraction_date
        input_mode = "client_prices"
        requested_reference_date = reference_date.isoformat() if reference_date else None
        resolved_window_start_date: str | None = None
        resolved_window_end_date: str | None = None

        if prices is None:
            (
                prices,
                resolved_window_start_date,
                resolved_window_end_date,
            ) = self._load_historical_price_window(
                symbol=normalized_symbol,
                extraction_date=selected_extraction_date,
                reference_date=reference_date,
            )
            input_mode = "historical_auto_window"
        else:
            if reference_date is not None:
                raise ValueError(
                    "`reference_date` can only be used when `prices` is omitted."
                )
            if len(prices) != self._lookback:
                raise ValueError(
                    f"`prices` must contain exactly {self._lookback} closing prices."
                )
            if any(not math.isfinite(float(value)) for value in prices):
                raise ValueError("`prices` must contain only finite numeric values.")

        scaler_metadata = self._load_scaler_metadata(
            symbol=normalized_symbol,
            extraction_date=selected_extraction_date,
        )
        model = self._load_model(resolution.model_local_path)

        raw_window = np.asarray(prices, dtype=np.float32)
        scaled_window = self._scale_array(
            raw_window,
            min_offset=scaler_metadata["min_offset"],
            scale=scaler_metadata["scale"],
        ).astype(np.float32)
        prediction_input = scaled_window.reshape(1, self._lookback, 1)
        predicted_scaled = float(model.predict(prediction_input, verbose=0).reshape(-1)[0])
        predicted_close = float(
            self._inverse_scale_array(
                np.asarray([predicted_scaled], dtype=np.float32),
                min_offset=scaler_metadata["min_offset"],
                scale=scaler_metadata["scale"],
            )[0]
        )
        guardrail = apply_standard_forecast_guardrail(
            raw_model_close=predicted_close,
            current_close=float(raw_window[-1]),
            recent_window=raw_window.astype(np.float64),
        )
        predicted_close = guardrail.constrained_close

        interval_width = resolution.interval_width or max(abs(predicted_close) * 0.05, 1.0)
        lower_bound, upper_bound = clamp_interval_to_guardrail_band(
            predicted_close=predicted_close,
            interval_width=interval_width,
            reference_close=float(raw_window[-1]),
            return_cap=guardrail.return_cap,
        )

        return StandardPredictionResult(
            symbol=normalized_symbol,
            predicted_close=predicted_close,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence=0.95,
            currency="USD",
            model=resolution.model_name,
            timestamp=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            extraction_date=selected_extraction_date.isoformat(),
            target_column=self._target_column,
            predict_type="normal",
            input_mode=input_mode,
            prices_provided_count=len(prices),
            requested_reference_date=requested_reference_date,
            resolved_window_start_date=resolved_window_start_date,
            resolved_window_end_date=resolved_window_end_date,
            raw_model_predicted_close=guardrail.raw_model_close,
            prediction_constraint_applied=guardrail.applied,
            prediction_constraint_method=guardrail.method,
            prediction_return_cap=guardrail.return_cap,
        )

    def _ensure_tensorflow_available(self) -> None:
        if _TENSORFLOW_IMPORT_ERROR is not None or tf is None:
            raise RuntimeError(
                "TensorFlow is required to serve `POST /predict`. "
                "Install the project dependencies from requirements.txt."
            ) from _TENSORFLOW_IMPORT_ERROR

    def _load_scaler_metadata(
        self,
        *,
        symbol: str,
        extraction_date: date,
    ) -> dict[str, float]:
        manifest_path = (
            self._processed_root_dir
            / "manifests"
            / f"extraction_date={extraction_date.isoformat()}"
            / "refined_manifest.json"
        )
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Refined manifest not found for {extraction_date.isoformat()}: {manifest_path}"
            )

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        request_payload = payload.get("request", {})
        if request_payload.get("source") != self._source:
            raise ValueError(
                f"Refined manifest source {request_payload.get('source')!r} "
                f"does not match API source {self._source!r}."
            )
        if request_payload.get("target_column") != self._target_column:
            raise ValueError(
                f"Refined manifest target_column {request_payload.get('target_column')!r} "
                f"does not match API target_column {self._target_column!r}."
            )

        asset_payload = next(
            (
                asset
                for asset in payload.get("assets", [])
                if str(asset.get("symbol", "")).upper() == symbol.upper()
                and int(asset.get("feature_count", self._lookback)) == self._lookback
            ),
            None,
        )
        if asset_payload is None:
            raise FileNotFoundError(
                f"Scaler metadata not found for symbol {symbol!r} in {manifest_path}."
            )

        return {
            "min_offset": float(asset_payload["scaler_min_offset"]),
            "scale": float(asset_payload["scaler_scale"]),
        }

    def _load_historical_price_window(
        self,
        *,
        symbol: str,
        extraction_date: date,
        reference_date: date | None,
    ) -> tuple[list[float], str, str]:
        raw_path = (
            self._raw_root_dir
            / "market_data"
            / f"source={self._source}"
            / f"symbol={symbol.upper()}"
            / f"extraction_date={extraction_date.isoformat()}"
            / "ohlcv.csv"
        )
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw market data not found for symbol={symbol!r} and "
                f"extraction_date={extraction_date.isoformat()}: {raw_path}"
            )

        import pandas as pd

        frame = pd.read_csv(raw_path, parse_dates=["date"])
        if "date" not in frame.columns or self._target_column not in frame.columns:
            raise ValueError(
                f"Raw market data at {raw_path} does not contain the required columns."
            )

        working = frame.loc[:, ["date", self._target_column]].dropna().copy()
        working["date"] = pd.to_datetime(working["date"])
        working = working.sort_values("date").reset_index(drop=True)

        if reference_date is not None:
            working = working.loc[
                working["date"] <= pd.Timestamp(reference_date)
            ].reset_index(drop=True)
            if working.empty:
                raise ValueError(
                    f"No historical rows are available for symbol={symbol!r} "
                    f"up to reference_date={reference_date.isoformat()}."
                )

        if len(working.index) < self._lookback:
            raise ValueError(
                f"Not enough historical rows to build a {self._lookback}-day window "
                f"for symbol={symbol!r}."
            )

        window = working.tail(self._lookback).reset_index(drop=True)
        return (
            window[self._target_column].astype(float).tolist(),
            window["date"].iloc[0].strftime("%Y-%m-%d"),
            window["date"].iloc[-1].strftime("%Y-%m-%d"),
        )

    def _resolve_serving_model_resolution(
        self,
        *,
        symbol: str,
        requested_extraction_date: date | None,
    ) -> ModelResolution:
        promoted_resolution = self._find_promoted_model_resolution(
            symbol=symbol,
            requested_extraction_date=requested_extraction_date,
        )
        if promoted_resolution is not None:
            return promoted_resolution
        if self._promotion_registry.exists():
            if requested_extraction_date is None:
                requested_window = " for the current serving default"
            else:
                requested_window = (
                    " on or before "
                    f"extraction_date={requested_extraction_date.isoformat()}"
                )
            raise FileNotFoundError(
                f"No approved serving artifact was found for symbol {symbol!r}"
                f"{requested_window} in {self._promotion_registry.policy_path}."
            )

        resolution = self._find_best_manifest_model_resolution(
            symbol=symbol,
            requested_extraction_date=requested_extraction_date,
        )
        if resolution is not None:
            return resolution

        selected_extraction_date = self._resolve_effective_extraction_date(
            requested_extraction_date
        )
        return self._resolve_model_resolution(
            symbol=symbol,
            extraction_date=selected_extraction_date,
        )

    def _resolve_effective_extraction_date(
        self,
        requested_extraction_date: date | None,
    ) -> date:
        candidates = self._list_available_extraction_dates()
        if not candidates:
            raise FileNotFoundError(
                "No Keras training manifests were found. "
                "Run `python scripts/train_keras.py` first."
            )

        candidates = sorted(set(candidates))
        if requested_extraction_date is None:
            return candidates[-1]

        eligible = [candidate for candidate in candidates if candidate <= requested_extraction_date]
        if not eligible:
            raise FileNotFoundError(
                "No trained extraction_date is available on or before "
                f"{requested_extraction_date.isoformat()}."
            )
        return eligible[-1]

    def _find_promoted_model_resolution(
        self,
        *,
        symbol: str,
        requested_extraction_date: date | None,
    ) -> ModelResolution | None:
        expected_model_name = f"{self._model_name_prefix}_{symbol.lower()}.keras"
        for promotion in self._promotion_registry.list_candidates(
            symbol=symbol,
            requested_extraction_date=requested_extraction_date,
        ):
            manifest_path = promotion.manifest_local_path
            model_local_path = promotion.model_local_path
            if not manifest_path.exists() or not model_local_path.exists():
                continue
            if model_local_path.name.lower() != expected_model_name.lower():
                continue

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            request_payload = payload.get("request", {})
            if (
                request_payload.get("source") != self._source
                or request_payload.get("target_column") != self._target_column
                or int(request_payload.get("lookback", self._lookback)) != self._lookback
            ):
                continue

            asset_payload = next(
                (
                    asset
                    for asset in payload.get("assets", [])
                    if str(asset.get("symbol", "")).upper() == symbol.upper()
                ),
                None,
            )
            if asset_payload is None:
                continue

            immutable_model_path = self._resolve_manifest_model_path(
                asset_payload=asset_payload,
                expected_model_name=expected_model_name,
                require_immutable_path=True,
            )
            if immutable_model_path is None:
                continue
            if immutable_model_path.resolve() != model_local_path.resolve():
                continue

            return ModelResolution(
                symbol=symbol.upper(),
                extraction_date=promotion.extraction_date,
                model_name=model_local_path.stem,
                model_local_path=model_local_path,
                training_manifest_local_path=manifest_path,
                training_generated_at_utc=str(payload.get("generated_at_utc")),
                interval_width=self._resolve_interval_width(asset_payload),
            )
        return None

    def _find_best_manifest_model_resolution(
        self,
        *,
        symbol: str,
        requested_extraction_date: date | None,
    ) -> ModelResolution | None:
        best_candidate: tuple[tuple[float, float, float, float, int], ModelResolution] | None = None
        expected_model_name = f"{self._model_name_prefix}_{symbol.lower()}.keras"
        manifests_root = self._models_root_dir / "manifests"
        if not manifests_root.exists():
            return None

        for extraction_date in self._list_available_extraction_dates():
            if (
                requested_extraction_date is not None
                and extraction_date > requested_extraction_date
            ):
                continue

            extraction_root = manifests_root / f"extraction_date={extraction_date.isoformat()}"
            for manifest_path in extraction_root.glob("trained_at=*/keras_training_manifest.json"):
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                request_payload = payload.get("request", {})
                if (
                    request_payload.get("source") != self._source
                    or request_payload.get("target_column") != self._target_column
                    or int(request_payload.get("lookback", self._lookback)) != self._lookback
                ):
                    continue

                asset_payload = next(
                    (
                        asset
                        for asset in payload.get("assets", [])
                        if str(asset.get("symbol", "")).upper() == symbol.upper()
                    ),
                    None,
                )
                if asset_payload is None:
                    continue
                model_local_path = self._resolve_manifest_model_path(
                    asset_payload=asset_payload,
                    expected_model_name=expected_model_name,
                    require_immutable_path=True,
                )
                if model_local_path is None:
                    continue

                resolution = ModelResolution(
                    symbol=symbol.upper(),
                    extraction_date=extraction_date,
                    model_name=model_local_path.stem,
                    model_local_path=model_local_path,
                    training_manifest_local_path=manifest_path,
                    training_generated_at_utc=str(payload.get("generated_at_utc")),
                    interval_width=self._resolve_interval_width(asset_payload),
                )
                candidate_score = self._score_regression_asset(
                    asset_payload=asset_payload,
                    manifest_path=manifest_path,
                )
                if best_candidate is None or candidate_score < best_candidate[0]:
                    best_candidate = (candidate_score, resolution)

        if best_candidate is None:
            return None
        return best_candidate[1]

    def _resolve_model_resolution(
        self,
        *,
        symbol: str,
        extraction_date: date,
    ) -> ModelResolution:
        expected_model_name = f"{self._model_name_prefix}_{symbol.lower()}.keras"
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
                    request_payload.get("source") != self._source
                    or request_payload.get("target_column") != self._target_column
                    or int(request_payload.get("lookback", self._lookback)) != self._lookback
                ):
                    continue

                asset_payload = next(
                    (
                        asset
                        for asset in payload.get("assets", [])
                        if str(asset.get("symbol", "")).upper() == symbol.upper()
                    ),
                    None,
                )
                if asset_payload is None:
                    continue
                model_local_path = self._resolve_manifest_model_path(
                    asset_payload=asset_payload,
                    expected_model_name=expected_model_name,
                    require_immutable_path=False,
                )
                if model_local_path is None:
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
                model_local_path = self._resolve_manifest_model_path(
                    asset_payload=asset_payload,
                    expected_model_name=expected_model_name,
                    require_immutable_path=False,
                )
                if model_local_path is None:
                    raise FileNotFoundError(
                        f"Manifest candidate for symbol {symbol!r} "
                        f"and extraction_date={extraction_date.isoformat()} "
                        "does not reference a usable model artifact."
                    )
                interval_width = self._resolve_interval_width(asset_payload)
                return ModelResolution(
                    symbol=symbol.upper(),
                    extraction_date=extraction_date,
                    model_name=model_local_path.stem,
                    model_local_path=model_local_path,
                    training_manifest_local_path=manifest_path,
                    training_generated_at_utc=str(payload.get("generated_at_utc")),
                    interval_width=interval_width,
                )

        fallback_path = self._models_root_dir / expected_model_name
        if fallback_path.exists():
            return ModelResolution(
                symbol=symbol.upper(),
                extraction_date=extraction_date,
                model_name=fallback_path.stem,
                model_local_path=fallback_path,
                training_manifest_local_path=None,
                training_generated_at_utc=None,
                interval_width=None,
            )

        raise FileNotFoundError(
            f"Could not find a trained Keras model for symbol {symbol!r} "
            f"and extraction_date={extraction_date.isoformat()}."
        )

    def _resolve_manifest_model_path(
        self,
        *,
        asset_payload: dict[str, Any],
        expected_model_name: str,
        require_immutable_path: bool,
    ) -> Path | None:
        immutable_recorded_path = self._resolve_recorded_artifact_path(
            asset_payload.get("immutable_model_local_path")
        )
        if (
            immutable_recorded_path.name.lower() == expected_model_name.lower()
            and immutable_recorded_path.exists()
        ):
            return immutable_recorded_path

        recorded_path = self._resolve_recorded_artifact_path(
            asset_payload.get("model_local_path")
        )
        if recorded_path.name.lower() != expected_model_name.lower():
            return None

        history_local_path = self._resolve_recorded_artifact_path(
            asset_payload.get("history_local_path")
        )
        stable_candidate = history_local_path.parent / expected_model_name
        if str(history_local_path).strip() and stable_candidate.exists():
            return stable_candidate.resolve()

        if require_immutable_path and self._is_published_alias_path(recorded_path):
            return None
        if recorded_path.exists():
            return recorded_path.resolve()
        return None

    def _resolve_recorded_artifact_path(self, raw_path: object) -> Path:
        token = str(raw_path).strip() if raw_path is not None else ""
        if not token:
            return Path()
        path = Path(token).expanduser()
        if not path.is_absolute():
            path = self._models_root_dir / path
        return path.resolve()

    def _is_published_alias_path(self, model_path: Path) -> bool:
        if not model_path.parts:
            return False
        return model_path.parent == self._models_root_dir

    def _list_available_extraction_dates(self) -> list[date]:
        manifests_root = self._models_root_dir / "manifests"
        candidates: list[date] = []
        if manifests_root.exists():
            for partition in manifests_root.glob("extraction_date=*"):
                token = partition.name.split("=", 1)[-1]
                try:
                    candidates.append(datetime.strptime(token, "%Y-%m-%d").date())
                except ValueError:
                    continue
        return sorted(set(candidates))

    def _load_model(self, model_path: Path) -> Any:
        cache_key = str(model_path.resolve())
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        model = tf.keras.models.load_model(str(model_path), compile=False)  # type: ignore[union-attr]
        self._model_cache[cache_key] = model
        return model

    @staticmethod
    def _resolve_interval_width(asset_payload: dict[str, Any]) -> float | None:
        for section in ("test_metrics", "validation_metrics", "train_metrics"):
            metrics = asset_payload.get(section) or {}
            mae = metrics.get("mae")
            if mae is not None:
                return abs(float(mae))
            rmse = metrics.get("rmse")
            if rmse is not None:
                return abs(float(rmse))
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
