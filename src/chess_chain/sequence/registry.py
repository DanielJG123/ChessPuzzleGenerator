"""Local registry of generated / manufactured sequences (stage 6).

The registry persists product ids, sequence fingerprints and the puzzle ids
each product consumed, so ``forbid_puzzle_reuse`` can be enforced across
manufactured products while remaining configurable (enforcing it consumes the
corpus much faster than merely requiring unique sequence fingerprints).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["SequenceRegistry", "load_registry"]


@dataclass
class SequenceRegistry:
    """A JSON-backed registry of manufactured products."""

    path: Path
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> SequenceRegistry:
        if path.is_file():
            try:
                entries = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                entries = {}
        else:
            entries = {}
        return cls(path=path, entries=dict(entries))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".part")
        temp.write_text(json.dumps(self.entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(self.path)

    def register(
        self,
        *,
        product_id: str,
        fingerprint: str,
        puzzle_ids: tuple[str, ...],
        request_summary: dict[str, object] | None = None,
    ) -> None:
        self.entries[product_id] = {
            "fingerprint": fingerprint,
            "puzzle_ids": list(puzzle_ids),
            "request": request_summary or {},
            "created_at": datetime.now(UTC).isoformat(),
        }
        self.save()

    def is_registered(self, product_id: str) -> bool:
        return product_id in self.entries

    def has_fingerprint(self, fingerprint: str) -> bool:
        return any(str(entry.get("fingerprint")) == fingerprint for entry in self.entries.values())

    def used_puzzle_ids(self) -> set[str]:
        used: set[str] = set()
        for entry in self.entries.values():
            used.update(str(item) for item in (entry.get("puzzle_ids") or []))
        return used

    def reused_puzzles(self, puzzle_ids: tuple[str, ...]) -> tuple[str, ...]:
        used = self.used_puzzle_ids()
        return tuple(puzzle_id for puzzle_id in puzzle_ids if puzzle_id in used)

    def to_dict(self) -> dict[str, object]:
        return dict(self.entries)


def load_registry(path: Path) -> SequenceRegistry:
    return SequenceRegistry.load(path)
