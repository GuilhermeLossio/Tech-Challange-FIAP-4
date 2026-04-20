from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd

from src.domain.interfaces.i_stock_repository import IStockRepository


class YFinanceRepository(IStockRepository):
    source_name = "yfinance"

    def fetch(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "yfinance is required to generate raw market data. "
                "Install the dependencies from requirements.txt."
            ) from exc

        if end_date < start_date:
            raise ValueError("end_date must be greater than or equal to start_date")

        # Yahoo treats `end` as exclusive, so add one day to keep the requested date.
        frame = yf.download(
            tickers=symbol,
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if frame.empty:
            raise ValueError(f"No market data returned for symbol {symbol!r}.")

        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)

        normalized = frame.reset_index().rename(columns=self._rename_columns)
        if "date" not in normalized.columns:
            raise ValueError("The downloaded data did not contain a date column.")

        extracted_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

        normalized["date"] = pd.to_datetime(normalized["date"]).dt.strftime("%Y-%m-%d")
        normalized.insert(0, "symbol", symbol.upper())
        normalized.insert(1, "source", self.source_name)
        normalized["extracted_at_utc"] = extracted_at

        ordered_columns = [
            "symbol",
            "source",
            "date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "extracted_at_utc",
        ]
        existing_columns = [column for column in ordered_columns if column in normalized.columns]
        remaining_columns = [
            column for column in normalized.columns if column not in existing_columns
        ]
        return normalized.loc[:, existing_columns + remaining_columns]

    @staticmethod
    def _rename_columns(column_name: object) -> str:
        raw_name = str(column_name).strip().lower().replace(" ", "_")
        if raw_name == "datetime":
            return "date"
        return raw_name
