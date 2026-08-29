"""DuckDB catalog and views over processed Parquet datasets."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from chess_chain.ingest.records import IngestSummary, load_ingest_manifest

logger = logging.getLogger(__name__)

_CREATE_DATASETS = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_name VARCHAR,
    transform_version INTEGER,
    source_path VARCHAR,
    source_fingerprint VARCHAR,
    puzzle_rows BIGINT,
    quarantine_rows BIGINT,
    created_at TIMESTAMP,
    dataset_dir VARCHAR,
    is_active BOOLEAN
)
"""


def open_catalog(processed_dir: Path) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) the catalog DuckDB database."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(processed_dir / "catalog.duckdb"))
    connection.execute(_CREATE_DATASETS)
    return connection


def discover_dataset(processed_dir: Path) -> Path | None:
    """Return the newest published dataset directory, if any."""
    candidates = [path.parent for path in processed_dir.glob("*_v*/_SUCCESS")]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def activate_dataset(connection: duckdb.DuckDBPyConnection, summary: IngestSummary) -> None:
    """Register ``summary`` as the active dataset and refresh the views."""
    dataset_dir = summary.dataset_dir
    connection.execute("UPDATE datasets SET is_active = FALSE")
    connection.execute("DELETE FROM datasets WHERE dataset_dir = ?", [str(dataset_dir)])
    connection.execute(
        """
        INSERT INTO datasets (
            dataset_name, transform_version, source_path, source_fingerprint,
            puzzle_rows, quarantine_rows, created_at, dataset_dir, is_active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, TRUE)
        """,
        [
            summary.dataset_name,
            summary.transform_version,
            summary.source_path,
            summary.source_fingerprint,
            summary.puzzle_rows,
            summary.quarantine_rows,
            summary.created_at,
            str(dataset_dir),
        ],
    )
    refresh_views(connection, dataset_dir)


def refresh_views(connection: duckdb.DuckDBPyConnection, dataset_dir: Path) -> None:
    """Point the ``puzzles`` and ``quarantined`` views at ``dataset_dir``."""

    def _sql_literal(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    puzzles_glob = str(dataset_dir / "puzzles" / "*" / "*.parquet")
    quarantine_glob = str(dataset_dir / "quarantine" / "*.parquet")
    connection.execute(
        "CREATE OR REPLACE VIEW puzzles AS "
        f"SELECT * FROM read_parquet({_sql_literal(puzzles_glob)}, hive_partitioning = true)"
    )
    connection.execute(
        "CREATE OR REPLACE VIEW quarantined AS "
        f"SELECT * FROM read_parquet({_sql_literal(quarantine_glob)})"
    )
    logger.info("Catalog views point at %s", dataset_dir)


def ensure_catalog(processed_dir: Path) -> duckdb.DuckDBPyConnection:
    """Open the catalog and activate the newest discovered dataset if any."""
    connection = open_catalog(processed_dir)
    dataset_dir = discover_dataset(processed_dir)
    if dataset_dir is not None:
        manifest = load_ingest_manifest(dataset_dir)
        if manifest is not None:
            summary = IngestSummary(
                dataset_dir=dataset_dir,
                dataset_name=str(manifest.get("dataset_name", "lichess_puzzles")),
                transform_version=int(str(manifest.get("transform_version", 0))),
                source_path=str(manifest.get("source_path", "")),
                source_fingerprint=str(manifest.get("source_fingerprint", "")),
                puzzle_rows=int(str(manifest.get("puzzle_rows", 0))),
                quarantine_rows=int(str(manifest.get("quarantine_rows", 0))),
                created_at=str(manifest.get("created_at", "")),
                elapsed_seconds=0.0,
                skipped=False,
            )
            activate_dataset(connection, summary)
    return connection
