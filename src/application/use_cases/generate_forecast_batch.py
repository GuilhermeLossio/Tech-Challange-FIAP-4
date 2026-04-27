from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import json
import sys
from typing import Any

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    tf = None
    _TENSORFLOW_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _TENSORFLOW_IMPORT_ERROR = None

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.infrastructure.storage.local_processed_store import LocalProcessedStore
from src.infrastructure.storage.s3_raw_store import S3RawStore


@dataclass(frozen=True)
class ForecastBatchRequest:
    symbols: tuple[str, ...]
    extraction_date: date
    source: str = "yfinance"
    target_column: str = "close"
    lookback: int = 60
    horizon_days: int = 30
    model_name_prefix: str = "lstm"
    upload_to_s3: bool = True


@dataclass(frozen=True)
class ForecastAssetArtifact:
    symbol: str
    row_count: int
    forecast_start_date: str
    forecast_end_date: str
    last_observed_date: str
    last_observed_close: float
    model_local_path: str
    training_manifest_local_path: str | None
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
    Generate recursive multi-step forecasts from the latest observed market window.

    The output is a flat parquet dataset with one row per future business day so it
    can be queried directly from Athena and reused later by an API layer.
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
        self._ensure_tensorflow_available()
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
            model_metadata = self._resolve_model_metadata(
                symbol=symbol,
                extraction_date=request.extraction_date,
                source=request.source,
                target_column=request.target_column,
                lookback=request.lookback,
                model_name_prefix=request.model_name_prefix,
            )
            model = tf.keras.models.load_model(  # type: ignore[union-attr]
                model_metadata["model_local_path"],
                compile=False,
            )
            forecast_frame = self._build_forecast_frame(
                raw_frame=raw_frame,
                request=request,
                symbol=symbol,
                generated_at_utc=generated_at_utc,
                generated_at_token=generated_at_token,
                model=model,
                scaler_metadata=scaler_metadata,
                model_metadata=model_metadata,
            )
            relative_path = self._build_forecast_relative_path(
                source=request.source,
                symbol=symbol,
                lookback=request.lookback,
                horizon_days=request.horizon_days,
                extraction_date=request.extraction_date,
                generated_at_token=generated_at_token,
            )
            local_path = self._local_store.write_frame(forecast_frame, relative_path)
            s3_uri = None
            if request.upload_to_s3 and self._s3_store is not None:
                s3_uri = self._s3_store.upload_dataframe(
                    frame=forecast_frame,
                    relative_path=relative_path,
                )

            artifacts.append(
                ForecastAssetArtifact(
                    symbol=symbol.upper(),
                    row_count=int(len(forecast_frame.index)),
                    forecast_start_date=str(forecast_frame["forecast_date"].iloc[0]),
                    forecast_end_date=str(forecast_frame["forecast_date"].iloc[-1]),
                    last_observed_date=str(forecast_frame["last_observed_date"].iloc[0]),
                    last_observed_close=float(forecast_frame["last_observed_close"].iloc[0]),
                    model_local_path=str(model_metadata["model_local_path"]),
                    training_manifest_local_path=model_metadata["training_manifest_local_path"],
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
                "TensorFlow is required for forecast generation. "
                "Install the project dependencies again after adding TensorFlow."
            ) from _TENSORFLOW_IMPORT_ERROR

    def _validate_request(self, request: ForecastBatchRequest) -> None:
        if not request.symbols:
            raise ValueError("At least one symbol must be provided.")
        if request.lookback <= 0:
            raise ValueError("lookback must be greater than zero.")
        if request.horizon_days <= 0:
            raise ValueError("horizon_days must be greater than zero.")
        if not request.model_name_prefix.strip():
            raise ValueError("model_name_prefix must not be blank.")

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

    def _resolve_model_metadata(
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
            f"Could not find a trained model for symbol {symbol!r}. "
            f"Expected manifest under {manifests_root} or fallback model {fallback_path}."
        )

    def _build_forecast_frame(
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
    ) -> pd.DataFrame:
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

            observed_points_in_window = max(request.lookback - (step - 1), 0)
            predicted_points_in_window = min(step - 1, request.lookback)
            rows.append(
                {
                    "symbol": symbol.upper(),
                    "source": request.source,
                    "target_column": request.target_column,
                    "extraction_date": request.extraction_date.isoformat(),
                    "generated_at_utc": generated_at_utc,
                    "generated_at_token": generated_at_token,
                    "model_name": Path(str(model_metadata["model_local_path"])).stem,
                    "model_local_path": model_metadata["model_local_path"],
                    "training_manifest_local_path": model_metadata["training_manifest_local_path"],
                    "training_generated_at_utc": model_metadata["training_generated_at_utc"],
                    "lookback": request.lookback,
                    "horizon_days": request.horizon_days,
                    "forecast_step": step,
                    "forecast_date": forecast_date.strftime("%Y-%m-%d"),
                    "predicted_close": predicted_close,
                    "predicted_scaled": predicted_scaled,
                    "last_observed_date": last_observed_date.strftime("%Y-%m-%d"),
                    "last_observed_close": last_observed_close,
                    "input_window_start_date": input_window_start_date.strftime("%Y-%m-%d"),
                    "input_window_end_date": input_window_end_date.strftime("%Y-%m-%d"),
                    "input_window_end_close": input_window_end_close,
                    "input_window_end_origin": "observed" if step == 1 else "predicted",
                    "observed_points_in_window": observed_points_in_window,
                    "predicted_points_in_window": predicted_points_in_window,
                    "recursive_forecast": True,
                }
            )

            raw_window = np.concatenate(
                [raw_window[1:], np.asarray([predicted_close], dtype=np.float32)]
            )
            scaled_window = np.concatenate(
                [scaled_window[1:], np.asarray([predicted_scaled], dtype=np.float32)]
            )
            window_dates = window_dates[1:] + [forecast_date]

        return pd.DataFrame(rows)

    @staticmethod
    def _scale_array(
        values: np.ndarray,
        *,
        min_offset: float,
        scale: float,
    ) -> np.ndarray:
        if scale == 0:
            raise ValueError("Cannot scale values because scale is zero.")
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
    def _build_forecast_relative_path(
        *,
        source: str,
        symbol: str,
        lookback: int,
        horizon_days: int,
        extraction_date: date,
        generated_at_token: str,
    ) -> Path:
        return (
            Path("forecast_data")
            / f"source={source}"
            / f"symbol={symbol.upper()}"
            / f"lookback={lookback}"
            / f"horizon_days={horizon_days}"
            / f"extraction_date={extraction_date.isoformat()}"
            / f"generated_at={generated_at_token}"
            / "forecast.parquet"
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
            / "forecast_manifest.json"
        )

    @staticmethod
    def _to_path_safe_timestamp(generated_at_utc: str) -> str:
        parsed = datetime.fromisoformat(generated_at_utc)
        return parsed.strftime("%Y%m%dT%H%M%SZ")


def main() -> int:
    print(
        "GenerateForecastBatchUseCase is a library module. "
        "Use `python scripts/generate_forecast.py --help` to run batch forecasting."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
