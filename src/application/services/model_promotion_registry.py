from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import json


@dataclass(frozen=True)
class ApprovedModelPromotion:
    symbol: str
    extraction_date: date
    trained_at_token: str
    manifest_local_path: Path
    model_local_path: Path
    approved_at_utc: str | None
    approved_by: str | None
    reason: str | None


class ModelPromotionRegistry:
    def __init__(
        self,
        *,
        models_root_dir: Path,
        policy_filename: str = "serving_promotions.json",
    ) -> None:
        self._models_root_dir = models_root_dir.resolve()
        self._policy_path = self._models_root_dir / policy_filename

    @property
    def policy_path(self) -> Path:
        return self._policy_path

    def exists(self) -> bool:
        return self._policy_path.exists()

    def load(self) -> tuple[ApprovedModelPromotion, ...]:
        if not self._policy_path.exists():
            return tuple()

        payload = json.loads(self._policy_path.read_text(encoding="utf-8"))
        promotions: list[ApprovedModelPromotion] = []
        for item in payload.get("approvals", []):
            symbol = str(item.get("symbol", "")).strip().upper()
            extraction_date_token = str(item.get("extraction_date", "")).strip()
            trained_at_token = str(item.get("trained_at", "")).strip()
            manifest_local_path = self._resolve_configured_path(
                relative_path=item.get("manifest_relative_path"),
                absolute_path=item.get("manifest_local_path"),
            )
            model_local_path = self._resolve_configured_path(
                relative_path=item.get("model_relative_path"),
                absolute_path=item.get("model_local_path"),
            )

            if (
                not symbol
                or not extraction_date_token
                or not trained_at_token
                or manifest_local_path is None
                or model_local_path is None
            ):
                continue

            try:
                extraction_date = date.fromisoformat(extraction_date_token)
            except ValueError:
                continue

            promotions.append(
                ApprovedModelPromotion(
                    symbol=symbol,
                    extraction_date=extraction_date,
                    trained_at_token=trained_at_token,
                    manifest_local_path=manifest_local_path,
                    model_local_path=model_local_path,
                    approved_at_utc=self._optional_string(item.get("approved_at_utc")),
                    approved_by=self._optional_string(item.get("approved_by")),
                    reason=self._optional_string(item.get("reason")),
                )
            )

        return tuple(promotions)

    def list_candidates(
        self,
        *,
        symbol: str,
        requested_extraction_date: date | None = None,
    ) -> tuple[ApprovedModelPromotion, ...]:
        normalized_symbol = symbol.strip().upper()
        candidates = [
            promotion
            for promotion in self.load()
            if promotion.symbol == normalized_symbol
            and (
                requested_extraction_date is None
                or promotion.extraction_date <= requested_extraction_date
            )
        ]
        return tuple(
            sorted(
                candidates,
                key=lambda item: (
                    item.extraction_date,
                    self._trained_at_token_value(item.trained_at_token),
                ),
                reverse=True,
            )
        )

    def list_promoted_symbols(self) -> tuple[str, ...]:
        symbols = sorted({promotion.symbol for promotion in self.load()})
        return tuple(symbols)

    def _resolve_configured_path(
        self,
        *,
        relative_path: object,
        absolute_path: object,
    ) -> Path | None:
        relative_token = self._optional_string(relative_path)
        if relative_token:
            return (self._models_root_dir / Path(relative_token)).resolve()

        absolute_token = self._optional_string(absolute_path)
        if absolute_token:
            return Path(absolute_token).expanduser().resolve()

        return None

    @staticmethod
    def _optional_string(value: object) -> str | None:
        token = str(value).strip() if value is not None else ""
        return token or None

    @staticmethod
    def _trained_at_token_value(trained_at_token: str) -> int:
        normalized = trained_at_token.replace("T", "").replace("Z", "")
        try:
            return int(normalized)
        except ValueError:
            return 0
