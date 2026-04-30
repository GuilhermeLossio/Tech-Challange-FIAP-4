from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class HistoricalPriceWindow:
    symbol: str
    source: str
    extraction_date: str
    start_date: str
    end_date: str
    row_count: int
    rows: tuple[dict[str, Any], ...]


class HistoricalMarketService:
    def __init__(
        self,
        *,
        raw_root_dir: Path,
        source: str = "yfinance",
    ) -> None:
        self._raw_root_dir = raw_root_dir
        self._source = source

    def load_close_history(
        self,
        *,
        symbol: str,
        extraction_date: date | None = None,
        as_of_date: date | None = None,
        trading_days: int = 126,
    ) -> HistoricalPriceWindow:
        if trading_days <= 0:
            raise ValueError("`trading_days` must be greater than zero.")

        dataset_path, resolved_extraction_date = self._resolve_dataset_path(
            symbol=symbol,
            extraction_date=extraction_date,
        )
        frame = pd.read_csv(dataset_path, usecols=["date", "close"])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["date", "close"]).sort_values("date")

        if as_of_date is not None:
            frame = frame.loc[frame["date"] <= pd.Timestamp(as_of_date)]

        if frame.empty:
            raise FileNotFoundError(
                f"No historical closes were found for symbol={symbol!r} "
                f"up to {as_of_date.isoformat() if as_of_date else 'the latest date'}."
            )

        selected = frame.tail(trading_days).copy()
        serialized = selected.assign(date=selected["date"].dt.strftime("%Y-%m-%d"))
        rows = tuple(serialized.to_dict(orient="records"))

        return HistoricalPriceWindow(
            symbol=symbol.strip().upper(),
            source=self._source,
            extraction_date=resolved_extraction_date.isoformat(),
            start_date=str(rows[0]["date"]),
            end_date=str(rows[-1]["date"]),
            row_count=len(rows),
            rows=rows,
        )

    def _resolve_dataset_path(
        self,
        *,
        symbol: str,
        extraction_date: date | None,
    ) -> tuple[Path, date]:
        symbol_root = (
            self._raw_root_dir
            / "market_data"
            / f"source={self._source}"
            / f"symbol={symbol.strip().upper()}"
        )
        if not symbol_root.exists():
            raise FileNotFoundError(
                f"Historical market data was not found under {symbol_root}. "
                "Run `python scripts/generate_raw.py` first."
            )

        candidates: list[date] = []
        for partition in symbol_root.glob("extraction_date=*"):
            token = partition.name.split("=", 1)[-1]
            try:
                candidates.append(datetime.strptime(token, "%Y-%m-%d").date())
            except ValueError:
                continue

        if not candidates:
            raise FileNotFoundError(
                f"No extraction_date partitions were found under {symbol_root}."
            )

        if extraction_date is None:
            resolved_extraction_date = max(candidates)
        else:
            eligible = [candidate for candidate in candidates if candidate <= extraction_date]
            if not eligible:
                raise FileNotFoundError(
                    "No raw extraction_date is available on or before "
                    f"{extraction_date.isoformat()} for symbol={symbol!r}."
                )
            resolved_extraction_date = max(eligible)

        dataset_path = symbol_root / f"extraction_date={resolved_extraction_date.isoformat()}" / "ohlcv.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Historical OHLCV file not found: {dataset_path}")

        return dataset_path, resolved_extraction_date
