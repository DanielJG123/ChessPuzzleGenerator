"""Storage helpers: manifests, parquet schemas, DuckDB catalog and queries."""

from __future__ import annotations

from chess_chain.storage.catalog import (
    activate_dataset,
    discover_dataset,
    ensure_catalog,
    open_catalog,
    refresh_views,
)
from chess_chain.storage.manifest import (
    DownloadManifest,
    IncompleteDownloadError,
    is_download_complete,
    load_manifest,
    manifest_path_for,
    sha256_file,
    utc_now_iso,
    write_manifest,
)
from chess_chain.storage.repository import PuzzleRepository
from chess_chain.storage.schema import PUZZLE_SCHEMA, QUARANTINE_SCHEMA

__all__ = [
    "DownloadManifest",
    "IncompleteDownloadError",
    "PUZZLE_SCHEMA",
    "QUARANTINE_SCHEMA",
    "PuzzleRepository",
    "activate_dataset",
    "discover_dataset",
    "ensure_catalog",
    "is_download_complete",
    "load_manifest",
    "manifest_path_for",
    "open_catalog",
    "refresh_views",
    "sha256_file",
    "utc_now_iso",
    "write_manifest",
]
