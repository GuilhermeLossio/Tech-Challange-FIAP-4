from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FuturePredictionQueryResult:
    symbol: str
    source: str
    extraction_date: str
    generated_at: str
    generated_at_utc: str
    lookback: int
    horizon_days: int
    available_predict_types: tuple[str, ...]
    available_forecast_start_date: str
    available_forecast_end_date: str
    returned_forecast_start_date: str
    returned_forecast_end_date: str
    last_observed_date: str
    last_observed_close: float
    row_count: int
    rows: tuple[dict[str, Any], ...]
    local_path: str


class FuturePredictionService:
    def __init__(
        self,
        *,
        processed_root_dir: Path,
        source: str = "yfinance",
        lookback: int = 60,
        horizon_days: int = 30,
    ) -> None:
        self._processed_root_dir = processed_root_dir
        self._source = source
        self._lookback = lookback
        self._horizon_days = horizon_days

    def has_materialized_forecasts(self) -> bool:
        root = self._processed_root_dir / "future_predict" / f"source={self._source}"
        return root.exists() and any(root.glob("symbol=*"))

    def list_symbols(self) -> tuple[str, ...]:
        root = self._processed_root_dir / "future_predict" / f"source={self._source}"
        if not root.exists():
            return tuple()

        symbols = [
            entry.name.split("=", 1)[-1].upper()
            for entry in root.glob("symbol=*")
            if entry.is_dir()
        ]
        return tuple(sorted(set(symbols)))

    def list_available_extraction_dates(
        self,
        *,
        symbol: str,
        lookback: int | None = None,
        horizon_days: int | None = None,
    ) -> tuple[date, ...]:
        symbol_root = self._build_symbol_root(
            symbol=symbol,
            lookback=lookback or self._lookback,
            horizon_days=horizon_days or self._horizon_days,
        )
        if not symbol_root.exists():
            return tuple()

        candidates: list[date] = []
        for partition in symbol_root.glob("extraction_date=*"):
            token = partition.name.split("=", 1)[-1]
            try:
                candidates.append(datetime.strptime(token, "%Y-%m-%d").date())
            except ValueError:
                continue
        return tuple(sorted(set(candidates)))

    def list_available_horizon_days(
        self,
        *,
        symbol: str,
        lookback: int | None = None,
    ) -> tuple[int, ...]:
        symbol_root = (
            self._processed_root_dir
            / "future_predict"
            / f"source={self._source}"
            / f"symbol={symbol.strip().upper()}"
            / f"lookback={lookback or self._lookback}"
        )
        if not symbol_root.exists():
            return tuple()

        horizons: list[int] = []
        for partition in symbol_root.glob("horizon_days=*"):
            token = partition.name.split("=", 1)[-1]
            try:
                horizons.append(int(token))
            except ValueError:
                continue
        return tuple(sorted(set(horizons)))

    def resolve_latest_horizon_days(
        self,
        *,
        symbol: str,
        lookback: int | None = None,
        requested_extraction_date: date | None = None,
    ) -> int | None:
        symbol_root = (
            self._processed_root_dir
            / "future_predict"
            / f"source={self._source}"
            / f"symbol={symbol.strip().upper()}"
            / f"lookback={lookback or self._lookback}"
        )
        if not symbol_root.exists():
            return None

        candidates: list[tuple[datetime, date, int]] = []
        for horizon_partition in symbol_root.glob("horizon_days=*"):
            try:
                horizon_days = int(horizon_partition.name.split("=", 1)[-1])
            except ValueError:
                continue
            for extraction_partition in horizon_partition.glob("extraction_date=*"):
                token = extraction_partition.name.split("=", 1)[-1]
                try:
                    extraction_date = datetime.strptime(token, "%Y-%m-%d").date()
                except ValueError:
                    continue
                if (
                    requested_extraction_date is not None
                    and extraction_date > requested_extraction_date
                ):
                    continue
                generated_partitions = sorted(extraction_partition.glob("generated_at=*"))
                for generated_partition in generated_partitions:
                    if (generated_partition / "future_predict.parquet").exists():
                        generated_at = self._parse_generated_at_partition(
                            generated_partition,
                        )
                        candidates.append(
                            (generated_at, extraction_date, horizon_days)
                        )

        if not candidates:
            return None
        return max(candidates)[2]

    def resolve_effective_extraction_date(
        self,
        *,
        symbol: str,
        requested_extraction_date: date | None = None,
        lookback: int | None = None,
        horizon_days: int | None = None,
    ) -> date:
        candidates = self.list_available_extraction_dates(
            symbol=symbol,
            lookback=lookback,
            horizon_days=horizon_days,
        )
        if not candidates:
            symbol_root = self._build_symbol_root(
                symbol=symbol,
                lookback=lookback or self._lookback,
                horizon_days=horizon_days or self._horizon_days,
            )
            raise FileNotFoundError(
                f"No extraction_date partitions were found under {symbol_root}."
            )

        if requested_extraction_date is None:
            return candidates[-1]

        eligible = [candidate for candidate in candidates if candidate <= requested_extraction_date]
        if not eligible:
            symbol_root = self._build_symbol_root(
                symbol=symbol,
                lookback=lookback or self._lookback,
                horizon_days=horizon_days or self._horizon_days,
            )
            raise FileNotFoundError(
                "No materialized future prediction extraction_date is available on or before "
                f"{requested_extraction_date.isoformat()} under {symbol_root}."
            )
        return eligible[-1]

    def load_forecasts(
        self,
        *,
        symbol: str,
        extraction_date: date | None = None,
        predict_type: str = "all",
        forecast_date_from: date | None = None,
        forecast_date_to: date | None = None,
        lookback: int | None = None,
        horizon_days: int | None = None,
        limit: int | None = None,
    ) -> FuturePredictionQueryResult:
        selected_lookback = lookback or self._lookback
        selected_horizon_days = horizon_days or self._horizon_days
        if (
            forecast_date_from is not None
            and forecast_date_to is not None
            and forecast_date_from > forecast_date_to
        ):
            raise ValueError("`forecast_date_from` must be less than or equal to `forecast_date_to`.")

        dataset_path = self._resolve_dataset_path(
            symbol=symbol,
            extraction_date=extraction_date,
            lookback=selected_lookback,
            horizon_days=selected_horizon_days,
        )
        frame = pd.read_parquet(dataset_path)
        ordered = frame.sort_values(
            by=["forecast_step", "predict_type"],
        ).reset_index(drop=True)
        ordered["forecast_date"] = pd.to_datetime(ordered["forecast_date"])

        available_forecast_start_date = ordered["forecast_date"].min().strftime("%Y-%m-%d")
        available_forecast_end_date = ordered["forecast_date"].max().strftime("%Y-%m-%d")

        if predict_type != "all":
            ordered = ordered.loc[
                ordered["predict_type"].astype(str).str.lower() == predict_type.lower()
            ].reset_index(drop=True)

        if forecast_date_from is not None:
            ordered = ordered.loc[
                ordered["forecast_date"] >= pd.Timestamp(forecast_date_from)
            ].reset_index(drop=True)

        if forecast_date_to is not None:
            ordered = ordered.loc[
                ordered["forecast_date"] <= pd.Timestamp(forecast_date_to)
            ].reset_index(drop=True)

        if limit is not None:
            ordered = ordered.head(limit).reset_index(drop=True)

        if ordered.empty:
            raise FileNotFoundError(
                f"No materialized future predictions were found for symbol={symbol!r} "
                f"with predict_type={predict_type!r} in the requested date interval."
            )

        available_predict_types = tuple(
            sorted(frame["predict_type"].astype(str).str.lower().unique().tolist())
        )
        first_row = ordered.iloc[0]
        returned_forecast_start_date = ordered["forecast_date"].min().strftime("%Y-%m-%d")
        returned_forecast_end_date = ordered["forecast_date"].max().strftime("%Y-%m-%d")

        serialized_rows = ordered.assign(
            forecast_date=ordered["forecast_date"].dt.strftime("%Y-%m-%d")
        )

        return FuturePredictionQueryResult(
            symbol=str(first_row["symbol"]).upper(),
            source=str(first_row["source"]),
            extraction_date=str(first_row["extraction_date"]),
            generated_at=str(first_row["generated_at_token"]),
            generated_at_utc=str(first_row["generated_at_utc"]),
            lookback=int(first_row["lookback"]),
            horizon_days=int(first_row["horizon_days"]),
            available_predict_types=available_predict_types,
            available_forecast_start_date=available_forecast_start_date,
            available_forecast_end_date=available_forecast_end_date,
            returned_forecast_start_date=returned_forecast_start_date,
            returned_forecast_end_date=returned_forecast_end_date,
            last_observed_date=str(first_row["last_observed_date"]),
            last_observed_close=float(first_row["last_observed_close"]),
            row_count=int(len(serialized_rows.index)),
            rows=tuple(serialized_rows.to_dict(orient="records")),
            local_path=str(dataset_path),
        )

    def _resolve_dataset_path(
        self,
        *,
        symbol: str,
        extraction_date: date | None,
        lookback: int,
        horizon_days: int,
    ) -> Path:
        symbol_root = self._build_symbol_root(
            symbol=symbol,
            lookback=lookback,
            horizon_days=horizon_days,
        )
        if not symbol_root.exists():
            raise FileNotFoundError(
                f"Materialized future predictions were not found under {symbol_root}. "
                "Run `python scripts/generate_forecast.py` first."
            )

        if extraction_date is None:
            latest_dataset_path = self._resolve_latest_dataset_path(
                symbol_root=symbol_root,
            )
            if latest_dataset_path is not None:
                return latest_dataset_path

            selected_extraction_date = self.resolve_effective_extraction_date(
                symbol=symbol,
                requested_extraction_date=None,
                lookback=lookback,
                horizon_days=horizon_days,
            )
        else:
            selected_extraction_date = extraction_date

        extraction_root = symbol_root / f"extraction_date={selected_extraction_date.isoformat()}"
        if not extraction_root.exists():
            raise FileNotFoundError(
                f"Materialized future predictions for extraction_date="
                f"{selected_extraction_date.isoformat()} were not found under {symbol_root}."
            )

        generated_partitions = sorted(extraction_root.glob("generated_at=*"))
        if not generated_partitions:
            raise FileNotFoundError(
                f"No generated_at partitions were found under {extraction_root}."
            )

        latest_partition = generated_partitions[-1]
        dataset_path = latest_partition / "future_predict.parquet"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Future prediction parquet not found: {dataset_path}")
        return dataset_path

    def _resolve_latest_dataset_path(self, *, symbol_root: Path) -> Path | None:
        candidates: list[tuple[datetime, Path]] = []
        for generated_partition in symbol_root.glob("extraction_date=*/generated_at=*"):
            dataset_path = generated_partition / "future_predict.parquet"
            if dataset_path.exists():
                candidates.append(
                    (
                        self._parse_generated_at_partition(generated_partition),
                        dataset_path,
                    )
                )
        if not candidates:
            return None
        return max(candidates)[1]

    @staticmethod
    def _parse_generated_at_partition(generated_partition: Path) -> datetime:
        token = generated_partition.name.split("=", 1)[-1]
        for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S"):
            try:
                return datetime.strptime(token, fmt)
            except ValueError:
                continue
        return datetime.min

    def _build_symbol_root(
        self,
        *,
        symbol: str,
        lookback: int,
        horizon_days: int,
    ) -> Path:
        return (
            self._processed_root_dir
            / "future_predict"
            / f"source={self._source}"
            / f"symbol={symbol.strip().upper()}"
            / f"lookback={lookback}"
            / f"horizon_days={horizon_days}"
        )
