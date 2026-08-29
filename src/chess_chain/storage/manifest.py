"""Download manifest persistence and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DownloadManifest:
    source_url: str
    retrieved_at: str
    content_length: int | None
    etag: str | None
    last_modified: str | None
    local_filename: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DownloadManifest:
        return cls(
            source_url=str(data["source_url"]),
            retrieved_at=str(data["retrieved_at"]),
            content_length=(
                int(data["content_length"]) if data.get("content_length") is not None else None
            ),
            etag=str(data["etag"]) if data.get("etag") is not None else None,
            last_modified=(
                str(data["last_modified"]) if data.get("last_modified") is not None else None
            ),
            local_filename=str(data["local_filename"]),
            sha256=str(data["sha256"]),
        )


def manifest_path_for(target: Path) -> Path:
    return target.with_suffix(target.suffix + ".manifest.json")


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, manifest: DownloadManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".part")
    temp_path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def load_manifest(path: Path) -> DownloadManifest | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    return DownloadManifest.from_dict(data)


def is_download_complete(target: Path, manifest_path: Path | None = None) -> bool:
    """Return True only when the file and manifest exist and the hash matches."""
    manifest_file = manifest_path or manifest_path_for(target)
    manifest = load_manifest(manifest_file)
    if manifest is None or not target.is_file():
        return False
    if manifest.content_length is not None and target.stat().st_size != manifest.content_length:
        return False
    return sha256_file(target) == manifest.sha256


class IncompleteDownloadError(RuntimeError):
    """Raised when a partial download cannot be treated as complete."""
