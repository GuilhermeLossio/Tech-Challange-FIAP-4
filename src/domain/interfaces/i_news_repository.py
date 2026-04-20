from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from src.domain.entities.news_signal import NewsSignal


class INewsRepository(ABC):
    @abstractmethod
    def fetch(self, symbol: str, date: date | str) -> list[NewsSignal]:
        """Return the news items available for a symbol on the requested date."""
