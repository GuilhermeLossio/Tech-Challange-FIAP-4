from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


class LocalRawStore:
    def __init__(self, root_dir: Path) -> None:
        self._root_dir = root_dir

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    def write_frame(self, frame: pd.DataFrame, relative_path: Path) -> Path:
        destination = self._root_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
        return destination

    def write_json(self, payload: dict, relative_path: Path) -> Path:
        destination = self._root_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return destination
