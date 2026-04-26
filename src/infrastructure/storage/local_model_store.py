from __future__ import annotations

from pathlib import Path
import json


class LocalModelStore:
    def __init__(self, root_dir: Path) -> None:
        self._root_dir = root_dir

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    def prepare_path(self, relative_path: Path) -> Path:
        destination = self._root_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def write_json(self, payload: dict, relative_path: Path) -> Path:
        destination = self.prepare_path(relative_path)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return destination
