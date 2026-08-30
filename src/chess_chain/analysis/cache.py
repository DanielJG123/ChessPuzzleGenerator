"""Small JSON key-value cache on disk for analysis results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["DiskKeyValueCache"]


class DiskKeyValueCache:
    """Stores one JSON document per key under a directory.

    Keys are sha256 hex digests of a canonical JSON representation, so callers
    decide exactly which inputs participate in the cache identity.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    @staticmethod
    def key_from(mapping: dict[str, Any]) -> str:
        canonical = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def has(self, key: str) -> bool:
        return self._path(key).is_file()

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        temp = self._path(key).with_suffix(".tmp")
        temp.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        temp.replace(self._path(key))
