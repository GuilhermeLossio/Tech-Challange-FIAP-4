from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import sys

import pandas as pd

# Allows direct execution from IDEs that run the file instead of the package module.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.domain.interfaces.i_stock_repository import IStockRepository
from src.infrastructure.storage.local_raw_store import LocalRawStore
from src.infrastructure.storage.s3_raw_store import S3RawStore


@dataclass(frozen=True)
class RawIngestionRequest:
    symbols: tuple[str, ...]
    start_date: date
    end_date: date
    extraction_date: date
    upload_to_s3: bool = True


@dataclass(frozen=True)
class RawAssetArtifact:
    symbol: str
    row_count: int
    local_path: str
    s3_uri: str | None


@dataclass(frozen=True)
class RawIngestionResult:
    source: str
    generated_at_utc: str
    manifest_local_path: str
    manifest_s3_uri: str | None
    assets: tuple[RawAssetArtifact, ...]

    def to_manifest(self) -> dict:
        return {
            "source": self.source,
            "generated_at_utc": self.generated_at_utc,
            "manifest_local_path": self.manifest_local_path,
            "manifest_s3_uri": self.manifest_s3_uri,
            "asset_count": len(self.assets),
            "assets": [asdict(asset) for asset in self.assets],
        }


class RawDataPipelineService:
    def __init__(
        self,
        stock_repository: IStockRepository,
        local_store: LocalRawStore,
        s3_store: S3RawStore | None = None,
    ) -> None:
        self._stock_repository = stock_repository
        self._local_store = local_store
        self._s3_store = s3_store

    def generate(self, request: RawIngestionRequest) -> RawIngestionResult:
        if not request.symbols:
            raise ValueError("At least one symbol must be provided.")

        artifacts: list[RawAssetArtifact] = []

        for symbol in request.symbols:
            frame = self._stock_repository.fetch(
                symbol=symbol,
                start_date=request.start_date,
                end_date=request.end_date,
            )
            relative_path = self._build_asset_relative_path(
                source=self._resolve_source_name(),
                symbol=symbol,
                extraction_date=request.extraction_date,
            )
            local_path = self._local_store.write_frame(frame, relative_path)

            s3_uri = None
            if request.upload_to_s3 and self._s3_store is not None:
                s3_uri = self._s3_store.upload_dataframe(
                    frame=frame,
                    relative_path=self._to_parquet_relative_path(relative_path),
                )

            artifacts.append(
                RawAssetArtifact(
                    symbol=symbol.upper(),
                    row_count=len(frame.index),
                    local_path=str(local_path),
                    s3_uri=s3_uri,
                )
            )

        generated_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        manifest_relative_path = self._build_manifest_relative_path(request.extraction_date)

        manifest_payload = {
            "source": self._resolve_source_name(),
            "generated_at_utc": generated_at_utc,
            "request": {
                "symbols": list(request.symbols),
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat(),
                "extraction_date": request.extraction_date.isoformat(),
                "upload_to_s3": request.upload_to_s3,
            },
            "asset_count": len(artifacts),
            "assets": [asdict(artifact) for artifact in artifacts],
        }

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

        return RawIngestionResult(
            source=self._resolve_source_name(),
            generated_at_utc=generated_at_utc,
            manifest_local_path=str(manifest_local_path),
            manifest_s3_uri=manifest_s3_uri,
            assets=tuple(artifacts),
        )

    def _resolve_source_name(self) -> str:
        return getattr(self._stock_repository, "source_name", "market_data")

    @staticmethod
    def _build_asset_relative_path(
        *,
        source: str,
        symbol: str,
        extraction_date: date,
    ) -> Path:
        return (
            Path("market_data")
            / f"source={source}"
            / f"symbol={symbol.upper()}"
            / f"extraction_date={extraction_date.isoformat()}"
            / "ohlcv.csv"
        )

    @staticmethod
    def _build_manifest_relative_path(extraction_date: date) -> Path:
        return (
            Path("manifests")
            / f"extraction_date={extraction_date.isoformat()}"
            / "raw_manifest.json"
        )

    @staticmethod
    def _to_parquet_relative_path(relative_path: Path) -> Path:
        return relative_path.with_suffix(".parquet")

    def _build_manifest_frame(
        self,
        *,
        request: RawIngestionRequest,
        generated_at_utc: str,
        artifacts: list[RawAssetArtifact],
    ) -> pd.DataFrame:
        rows = []
        for artifact in artifacts:
            rows.append(
                {
                    "source": self._resolve_source_name(),
                    "generated_at_utc": generated_at_utc,
                    "start_date": request.start_date.isoformat(),
                    "end_date": request.end_date.isoformat(),
                    "extraction_date": request.extraction_date.isoformat(),
                    "upload_to_s3": request.upload_to_s3,
                    "asset_count": len(artifacts),
                    "symbol": artifact.symbol,
                    "row_count": artifact.row_count,
                    "local_path": artifact.local_path,
                    "s3_uri": artifact.s3_uri,
                }
            )
        return pd.DataFrame(rows)


def main() -> int:
    print(
        "RawDataPipelineService is a library module. "
        "Use `python scripts/generate_raw.py --skip-s3` for local generation "
        "or `python scripts/generate_raw.py` for local + S3 upload."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
