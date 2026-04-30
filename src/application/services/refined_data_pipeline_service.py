from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Allows direct execution from IDEs that run the file instead of the package module.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.infrastructure.storage.local_processed_store import LocalProcessedStore
from src.infrastructure.storage.s3_raw_store import S3RawStore


DEFAULT_HISTORY_YEARS = 10


@dataclass(frozen=True)
class RefinedDatasetRequest:
    symbols: tuple[str, ...]
    extraction_date: date
    source: str = "yfinance"
    target_column: str = "close"
    lookback: int = 60
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    history_years: int | None = DEFAULT_HISTORY_YEARS
    upload_to_s3: bool = True


@dataclass(frozen=True)
class RefinedAssetArtifact:
    symbol: str
    row_count: int
    feature_count: int
    split_counts: dict[str, int]
    local_path: str
    s3_uri: str | None
    scaler_min_offset: float
    scaler_scale: float
    data_min: float
    data_max: float
    history_years: int | None
    history_start_date: str
    history_end_date: str
    scaler_fit_start_date: str
    scaler_fit_end_date: str


@dataclass(frozen=True)
class RefinedDatasetResult:
    source: str
    generated_at_utc: str
    manifest_local_path: str
    manifest_s3_uri: str | None
    assets: tuple[RefinedAssetArtifact, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "generated_at_utc": self.generated_at_utc,
            "manifest_local_path": self.manifest_local_path,
            "manifest_s3_uri": self.manifest_s3_uri,
            "asset_count": len(self.assets),
            "assets": [asdict(asset) for asset in self.assets],
        }


class RefinedDataPipelineService:
    def __init__(
        self,
        raw_root_dir: Path,
        local_store: LocalProcessedStore,
        s3_store: S3RawStore | None = None,
    ) -> None:
        self._raw_root_dir = raw_root_dir
        self._local_store = local_store
        self._s3_store = s3_store

    def generate(self, request: RefinedDatasetRequest) -> RefinedDatasetResult:
        self._validate_request(request)

        artifacts: list[RefinedAssetArtifact] = []

        for symbol in request.symbols:
            raw_frame = self._load_raw_frame(
                source=request.source,
                symbol=symbol,
                extraction_date=request.extraction_date,
            )
            refined_frame, scaler_metadata = self._build_refined_frame(
                raw_frame=raw_frame,
                request=request,
                symbol=symbol,
            )

            relative_path = self._build_refined_relative_path(
                source=request.source,
                symbol=symbol,
                extraction_date=request.extraction_date,
                lookback=request.lookback,
            )
            local_path = self._local_store.write_frame(refined_frame, relative_path)

            s3_uri = None
            if request.upload_to_s3 and self._s3_store is not None:
                s3_uri = self._s3_store.upload_dataframe(
                    frame=refined_frame,
                    relative_path=relative_path,
                )

            split_counts = {
                str(split): int(count)
                for split, count in refined_frame["split"].value_counts(sort=False).items()
            }

            artifacts.append(
                RefinedAssetArtifact(
                    symbol=symbol.upper(),
                    row_count=len(refined_frame.index),
                    feature_count=request.lookback,
                    split_counts=split_counts,
                    local_path=str(local_path),
                    s3_uri=s3_uri,
                    scaler_min_offset=float(scaler_metadata["min_offset"]),
                    scaler_scale=float(scaler_metadata["scale"]),
                    data_min=float(scaler_metadata["data_min"]),
                    data_max=float(scaler_metadata["data_max"]),
                    history_years=request.history_years,
                    history_start_date=str(scaler_metadata["history_start_date"]),
                    history_end_date=str(scaler_metadata["history_end_date"]),
                    scaler_fit_start_date=str(scaler_metadata["scaler_fit_start_date"]),
                    scaler_fit_end_date=str(scaler_metadata["scaler_fit_end_date"]),
                )
            )

        generated_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        manifest_payload = {
            "source": request.source,
            "generated_at_utc": generated_at_utc,
            "request": {
                "symbols": list(request.symbols),
                "extraction_date": request.extraction_date.isoformat(),
                "source": request.source,
                "target_column": request.target_column,
                "lookback": request.lookback,
                "train_ratio": request.train_ratio,
                "validation_ratio": request.validation_ratio,
                "history_years": request.history_years,
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
            manifest_s3_uri = self._s3_store.upload_dataframe(
                frame=self._build_manifest_frame(
                    request=request,
                    generated_at_utc=generated_at_utc,
                    artifacts=artifacts,
                ),
                relative_path=self._to_parquet_relative_path(manifest_relative_path),
            )

        return RefinedDatasetResult(
            source=request.source,
            generated_at_utc=generated_at_utc,
            manifest_local_path=str(manifest_local_path),
            manifest_s3_uri=manifest_s3_uri,
            assets=tuple(artifacts),
        )

    def _validate_request(self, request: RefinedDatasetRequest) -> None:
        if not request.symbols:
            raise ValueError("At least one symbol must be provided.")
        if request.lookback <= 0:
            raise ValueError("lookback must be greater than zero.")
        if not 0 < request.train_ratio < 1:
            raise ValueError("train_ratio must be between zero and one.")
        if not 0 <= request.validation_ratio < 1:
            raise ValueError("validation_ratio must be between zero and one.")
        if request.train_ratio + request.validation_ratio >= 1:
            raise ValueError("train_ratio + validation_ratio must be less than one.")
        if request.history_years is not None and request.history_years <= 0:
            raise ValueError("history_years must be greater than zero when provided.")

    def _load_raw_frame(
        self,
        *,
        source: str,
        symbol: str,
        extraction_date: date,
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
        return frame

    def _build_refined_frame(
        self,
        *,
        raw_frame: pd.DataFrame,
        request: RefinedDatasetRequest,
        symbol: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if request.target_column not in raw_frame.columns:
            raise ValueError(
                f"Raw input for symbol {symbol!r} does not contain "
                f"target column {request.target_column!r}."
            )

        working = raw_frame.loc[:, ["date", request.target_column]].copy()
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
        working[request.target_column] = pd.to_numeric(
            working[request.target_column],
            errors="coerce",
        )
        working = working.dropna(subset=["date", request.target_column])
        working = working.sort_values("date").reset_index(drop=True)
        if working["date"].duplicated().any():
            raise ValueError(
                f"Raw input for symbol {symbol!r} contains duplicate trading dates."
            )

        if request.history_years is not None:
            cutoff_date = working["date"].max() - pd.DateOffset(years=request.history_years)
            working = working.loc[working["date"] >= cutoff_date].reset_index(drop=True)

        if len(working.index) <= request.lookback:
            raise ValueError(
                f"Not enough rows to build refined sequences for symbol {symbol!r}. "
                f"Need more than {request.lookback} rows."
            )

        total_sequences = len(working.index) - request.lookback
        train_end = int(total_sequences * request.train_ratio)
        validation_end = int(total_sequences * (request.train_ratio + request.validation_ratio))
        if train_end <= 0:
            raise ValueError(
                f"Not enough rows to allocate a non-empty train split for symbol {symbol!r}."
            )

        scaler_fit_end_index = request.lookback + train_end - 1
        scaler_fit_frame = working.iloc[: scaler_fit_end_index + 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(scaler_fit_frame[[request.target_column]])
        scaled_values = scaler.transform(working[[request.target_column]]).reshape(-1)

        rows: list[dict[str, Any]] = []
        for sequence_index in range(total_sequences):
            target_index = sequence_index + request.lookback
            window = scaled_values[sequence_index:target_index]
            row = {
                "symbol": symbol.upper(),
                "source": request.source,
                "target_column": request.target_column,
                "extraction_date": request.extraction_date.isoformat(),
                "lookback": request.lookback,
                "split": self._resolve_split(
                    sequence_index=sequence_index,
                    train_end=train_end,
                    validation_end=validation_end,
                ),
                "window_start_date": working["date"].iloc[sequence_index].strftime("%Y-%m-%d"),
                "window_end_date": working["date"].iloc[target_index - 1].strftime("%Y-%m-%d"),
                "target_date": working["date"].iloc[target_index].strftime("%Y-%m-%d"),
                "y_scaled": float(scaled_values[target_index]),
                f"y_{request.target_column}": float(
                    working[request.target_column].iloc[target_index]
                ),
            }
            for position, value in enumerate(window):
                lag = request.lookback - position
                row[f"{request.target_column}_t_minus_{lag}"] = float(value)
            rows.append(row)

        refined_frame = pd.DataFrame(rows)
        scaler_metadata = {
            "min_offset": float(scaler.min_[0]),
            "scale": float(scaler.scale_[0]),
            "data_min": float(scaler.data_min_[0]),
            "data_max": float(scaler.data_max_[0]),
            "history_start_date": working["date"].iloc[0].strftime("%Y-%m-%d"),
            "history_end_date": working["date"].iloc[-1].strftime("%Y-%m-%d"),
            "scaler_fit_start_date": scaler_fit_frame["date"].iloc[0].strftime("%Y-%m-%d"),
            "scaler_fit_end_date": scaler_fit_frame["date"].iloc[-1].strftime("%Y-%m-%d"),
        }
        return refined_frame, scaler_metadata

    @staticmethod
    def _resolve_split(
        *,
        sequence_index: int,
        train_end: int,
        validation_end: int,
    ) -> str:
        if sequence_index < train_end:
            return "train"
        if sequence_index < validation_end:
            return "validation"
        return "test"

    @staticmethod
    def _build_refined_relative_path(
        *,
        source: str,
        symbol: str,
        extraction_date: date,
        lookback: int,
    ) -> Path:
        return (
            Path("refined_data")
            / f"source={source}"
            / f"symbol={symbol.upper()}"
            / f"lookback={lookback}"
            / f"extraction_date={extraction_date.isoformat()}"
            / "refined.parquet"
        )

    @staticmethod
    def _build_manifest_relative_path(extraction_date: date) -> Path:
        return (
            Path("manifests")
            / f"extraction_date={extraction_date.isoformat()}"
            / "refined_manifest.json"
        )

    @staticmethod
    def _to_parquet_relative_path(relative_path: Path) -> Path:
        return relative_path.with_suffix(".parquet")

    def _build_manifest_frame(
        self,
        *,
        request: RefinedDatasetRequest,
        generated_at_utc: str,
        artifacts: list[RefinedAssetArtifact],
    ) -> pd.DataFrame:
        rows = []
        for artifact in artifacts:
            rows.append(
                {
                    "source": request.source,
                    "generated_at_utc": generated_at_utc,
                    "extraction_date": request.extraction_date.isoformat(),
                    "target_column": request.target_column,
                    "lookback": request.lookback,
                    "train_ratio": request.train_ratio,
                    "validation_ratio": request.validation_ratio,
                    "history_years": request.history_years,
                    "upload_to_s3": request.upload_to_s3,
                    "symbol": artifact.symbol,
                    "row_count": artifact.row_count,
                    "feature_count": artifact.feature_count,
                    "train_count": artifact.split_counts.get("train", 0),
                    "validation_count": artifact.split_counts.get("validation", 0),
                    "test_count": artifact.split_counts.get("test", 0),
                    "local_path": artifact.local_path,
                    "s3_uri": artifact.s3_uri,
                    "scaler_min_offset": artifact.scaler_min_offset,
                    "scaler_scale": artifact.scaler_scale,
                    "data_min": artifact.data_min,
                    "data_max": artifact.data_max,
                    "history_start_date": artifact.history_start_date,
                    "history_end_date": artifact.history_end_date,
                    "scaler_fit_start_date": artifact.scaler_fit_start_date,
                    "scaler_fit_end_date": artifact.scaler_fit_end_date,
                }
            )
        return pd.DataFrame(rows)


def main() -> int:
    print(
        "RefinedDataPipelineService is a library module. "
        "Use `python scripts/generate_refined.py --skip-s3` for local generation "
        "or `python scripts/generate_refined.py` for local + S3 upload."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
