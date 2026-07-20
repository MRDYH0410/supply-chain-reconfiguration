from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import csv
from pathlib import Path


@dataclass
class CSVLogger:
    path: Path
    fieldnames: List[str]
    _writer: Any = field(init=False, default=None)
    _fh: Any = field(init=False, default=None)

    def __post_init__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
        self._writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
