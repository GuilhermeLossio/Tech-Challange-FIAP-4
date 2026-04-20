from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class IStockRepository(ABC):
    @abstractmethod
    def fetch(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Return raw daily OHLCV rows for the requested symbol and date range."""
