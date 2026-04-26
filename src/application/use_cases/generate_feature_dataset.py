from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.application.use_cases._dataset_loading import load_refined_frame_with_scaler
from src.infrastructure.storage.local_processed_store import LocalProcessedStore
from src.infrastructure.storage.s3_raw_store import S3RawStore


@dataclass(frozen=True)
class FeatureDatasetRequest:
    symbols: tuple[str, ...]
    extraction_date: date
    source: str = "yfinance"
    target_column: str = "close"
    lookback: int = 60
    upload_to_s3: bool = True


@dataclass(frozen=True)
class FeatureAssetArtifact:
    symbol: str
    lookback: int
    row_count: int
    sequence_feature_count: int
    engineered_feature_count: int
    engineered_feature_names: tuple[str, ...]
    local_path: str
    s3_uri: str | None
    scaler_min_offset: float
    scaler_scale: float
    data_min: float
    data_max: float


@dataclass(frozen=True)
class FeatureDatasetResult:
    source: str
    generated_at_utc: str
    manifest_local_path: str
    manifest_s3_uri: str | None
    assets: tuple[FeatureAssetArtifact, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "generated_at_utc": self.generated_at_utc,
            "manifest_local_path": self.manifest_local_path,
            "manifest_s3_uri": self.manifest_s3_uri,
            "asset_count": len(self.assets),
            "assets": [asdict(asset) for asset in self.assets],
        }


class GenerateFeatureDatasetUseCase:
    """
    Build engineered window-level features from the refined sliding-window dataset.

    The generated dataset keeps the original lag columns so sequence models can still
    train as before, while adding compact features that are easier to explain and
    more suitable for the quantum classifier.
    """

    def __init__(
        self,
        processed_root_dir: Path,
        local_store: LocalProcessedStore,
        s3_store: S3RawStore | None = None,
    ) -> None:
        self._processed_root_dir = processed_root_dir
        self._local_store = local_store
        self._s3_store = s3_store

    def generate(self, request: FeatureDatasetRequest) -> FeatureDatasetResult:
        self._validate_request(request)

        generated_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        artifacts: list[FeatureAssetArtifact] = []

        for symbol in request.symbols:
            refined_frame, scaler_metadata = load_refined_frame_with_scaler(
                processed_root_dir=self._processed_root_dir,
                source=request.source,
                symbol=symbol,
                extraction_date=request.extraction_date,
                lookback=request.lookback,
                target_column=request.target_column,
            )
            feature_frame, engineered_feature_names = self._build_feature_frame(
                frame=refined_frame,
                request=request,
                scaler_metadata=scaler_metadata,
            )
            relative_path = self._build_feature_relative_path(
                source=request.source,
                symbol=symbol,
                lookback=request.lookback,
                extraction_date=request.extraction_date,
            )
            local_path = self._local_store.write_frame(feature_frame, relative_path)
            s3_uri = None
            if request.upload_to_s3 and self._s3_store is not None:
                s3_uri = self._s3_store.upload_dataframe(
                    frame=feature_frame,
                    relative_path=relative_path,
                )

            artifacts.append(
                FeatureAssetArtifact(
                    symbol=symbol.upper(),
                    lookback=request.lookback,
                    row_count=int(len(feature_frame.index)),
                    sequence_feature_count=request.lookback,
                    engineered_feature_count=len(engineered_feature_names),
                    engineered_feature_names=tuple(engineered_feature_names),
                    local_path=str(local_path),
                    s3_uri=s3_uri,
                    scaler_min_offset=float(scaler_metadata["min_offset"]),
                    scaler_scale=float(scaler_metadata["scale"]),
                    data_min=float(scaler_metadata["data_min"]),
                    data_max=float(scaler_metadata["data_max"]),
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
                "upload_to_s3": request.upload_to_s3,
            },
            "asset_count": len(artifacts),
            "assets": [asdict(artifact) for artifact in artifacts],
        }
        manifest_relative_path = self._build_manifest_relative_path(request.extraction_date)
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

        return FeatureDatasetResult(
            source=request.source,
            generated_at_utc=generated_at_utc,
            manifest_local_path=str(manifest_local_path),
            manifest_s3_uri=manifest_s3_uri,
            assets=tuple(artifacts),
        )

    def _validate_request(self, request: FeatureDatasetRequest) -> None:
        if not request.symbols:
            raise ValueError("At least one symbol must be provided.")
        if request.lookback <= 1:
            raise ValueError("lookback must be greater than one for feature engineering.")

    def _build_feature_frame(
        self,
        *,
        frame: pd.DataFrame,
        request: FeatureDatasetRequest,
        scaler_metadata: dict[str, float],
    ) -> tuple[pd.DataFrame, list[str]]:
        lag_columns = [
            f"{request.target_column}_t_minus_{lag}"
            for lag in range(request.lookback, 0, -1)
        ]
        missing_columns = [column for column in lag_columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                "Refined dataset is missing expected sequence columns: "
                f"{missing_columns}"
            )

        target_raw_column = f"y_{request.target_column}"
        if target_raw_column not in frame.columns:
            raise ValueError(
                f"Refined dataset is missing expected target column {target_raw_column!r}."
            )

        ordered = frame.copy()
        if "target_date" in ordered.columns:
            ordered["target_date"] = pd.to_datetime(ordered["target_date"])
            ordered = ordered.sort_values("target_date").reset_index(drop=True)

        scaled_windows = ordered.loc[:, lag_columns].to_numpy(dtype=np.float64)
        raw_windows = self._inverse_scale_array(
            scaled_windows,
            min_offset=float(scaler_metadata["min_offset"]),
            scale=float(scaler_metadata["scale"]),
        )
        target_values = ordered[target_raw_column].to_numpy(dtype=np.float64)

        feature_rows: list[dict[str, float | int]] = []
        for window, target_value in zip(raw_windows, target_values, strict=False):
            current_price = float(window[-1])
            daily_returns = self._compute_daily_returns(window)

            sma_5 = self._compute_sma(window, 5)
            sma_10 = self._compute_sma(window, 10)
            sma_20 = self._compute_sma(window, 20)

            feature_rows.append(
                {
                    "feature_current_price": current_price,
                    "feature_window_mean": float(np.mean(window)),
                    "feature_window_std": float(np.std(window)),
                    "feature_window_min": float(np.min(window)),
                    "feature_window_max": float(np.max(window)),
                    "feature_window_range": float(np.max(window) - np.min(window)),
                    "feature_return_1d": self._compute_window_return(window, 1),
                    "feature_return_5d": self._compute_window_return(window, 5),
                    "feature_return_10d": self._compute_window_return(window, 10),
                    "feature_return_20d": self._compute_window_return(window, 20),
                    "feature_sma_gap_5d": self._compute_gap_ratio(current_price, sma_5),
                    "feature_sma_gap_10d": self._compute_gap_ratio(current_price, sma_10),
                    "feature_sma_gap_20d": self._compute_gap_ratio(current_price, sma_20),
                    "feature_ema_gap_5d": self._compute_gap_ratio(
                        current_price,
                        self._compute_ema(window, 5),
                    ),
                    "feature_ema_gap_10d": self._compute_gap_ratio(
                        current_price,
                        self._compute_ema(window, 10),
                    ),
                    "feature_volatility_5d": self._compute_volatility(daily_returns, 5),
                    "feature_volatility_10d": self._compute_volatility(daily_returns, 10),
                    "feature_trend_slope_10d": self._compute_trend_slope(window, 10),
                    "feature_trend_slope_20d": self._compute_trend_slope(window, 20),
                    "feature_up_day_ratio_5d": self._compute_up_day_ratio(daily_returns, 5),
                    "feature_up_day_ratio_10d": self._compute_up_day_ratio(daily_returns, 10),
                    "feature_position_in_window": self._compute_position_in_window(window),
                    "feature_window_max_drawdown": self._compute_max_drawdown(window),
                    "target_return_1d": self._compute_gap_ratio(target_value, current_price),
                    "target_direction": int(target_value > current_price),
                }
            )

        feature_frame = ordered.copy()
        engineered_frame = pd.DataFrame(feature_rows, index=feature_frame.index)
        feature_frame = pd.concat([feature_frame, engineered_frame], axis=1)
        engineered_feature_names = [
            column for column in engineered_frame.columns if column.startswith("feature_")
        ]
        return feature_frame, engineered_feature_names

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
    def _compute_daily_returns(window: np.ndarray) -> np.ndarray:
        if len(window) <= 1:
            return np.empty(0, dtype=np.float64)
        previous = window[:-1]
        current = window[1:]
        returns = np.zeros(len(previous), dtype=np.float64)
        non_zero_mask = np.abs(previous) > 1e-8
        returns[non_zero_mask] = current[non_zero_mask] / previous[non_zero_mask] - 1.0
        return returns

    @staticmethod
    def _compute_window_return(window: np.ndarray, periods: int) -> float:
        if len(window) <= 1:
            return 0.0
        effective_periods = min(periods, len(window) - 1)
        baseline = float(window[-(effective_periods + 1)])
        current = float(window[-1])
        if abs(baseline) <= 1e-8:
            return 0.0
        return current / baseline - 1.0

    @staticmethod
    def _compute_sma(window: np.ndarray, periods: int) -> float:
        effective_periods = min(periods, len(window))
        return float(np.mean(window[-effective_periods:]))

    @staticmethod
    def _compute_ema(window: np.ndarray, periods: int) -> float:
        effective_periods = min(periods, len(window))
        segment = window[-effective_periods:]
        alpha = 2.0 / (effective_periods + 1.0)
        ema_value = float(segment[0])
        for value in segment[1:]:
            ema_value = alpha * float(value) + (1.0 - alpha) * ema_value
        return ema_value

    @staticmethod
    def _compute_gap_ratio(current: float, reference: float) -> float:
        if abs(reference) <= 1e-8:
            return 0.0
        return current / reference - 1.0

    @staticmethod
    def _compute_volatility(daily_returns: np.ndarray, periods: int) -> float:
        if len(daily_returns) == 0:
            return 0.0
        effective_periods = min(periods, len(daily_returns))
        return float(np.std(daily_returns[-effective_periods:]))

    @staticmethod
    def _compute_trend_slope(window: np.ndarray, periods: int) -> float:
        effective_periods = min(periods, len(window))
        segment = window[-effective_periods:]
        if len(segment) <= 1:
            return 0.0
        x_axis = np.arange(len(segment), dtype=np.float64)
        slope = float(np.polyfit(x_axis, segment, 1)[0])
        scale = max(abs(float(np.mean(segment))), 1e-8)
        return slope / scale

    @staticmethod
    def _compute_up_day_ratio(daily_returns: np.ndarray, periods: int) -> float:
        if len(daily_returns) == 0:
            return 0.0
        effective_periods = min(periods, len(daily_returns))
        segment = daily_returns[-effective_periods:]
        return float(np.mean(segment > 0))

    @staticmethod
    def _compute_position_in_window(window: np.ndarray) -> float:
        lower = float(np.min(window))
        upper = float(np.max(window))
        if abs(upper - lower) <= 1e-8:
            return 0.5
        return (float(window[-1]) - lower) / (upper - lower)

    @staticmethod
    def _compute_max_drawdown(window: np.ndarray) -> float:
        running_max = np.maximum.accumulate(window)
        drawdowns = np.zeros(len(window), dtype=np.float64)
        non_zero_mask = np.abs(running_max) > 1e-8
        drawdowns[non_zero_mask] = window[non_zero_mask] / running_max[non_zero_mask] - 1.0
        return float(np.min(drawdowns))

    @staticmethod
    def _build_feature_relative_path(
        *,
        source: str,
        symbol: str,
        lookback: int,
        extraction_date: date,
    ) -> Path:
        return (
            Path("feature_data")
            / f"source={source}"
            / f"symbol={symbol.upper()}"
            / f"lookback={lookback}"
            / f"extraction_date={extraction_date.isoformat()}"
            / "features.parquet"
        )

    @staticmethod
    def _build_manifest_relative_path(extraction_date: date) -> Path:
        return (
            Path("manifests")
            / f"extraction_date={extraction_date.isoformat()}"
            / "feature_manifest.json"
        )
