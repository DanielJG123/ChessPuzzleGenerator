"""Similarity candidate index.

The index is built once from the active DuckDB dataset and persisted (Parquet
rows + a JSON manifest carrying a version fingerprint). Candidate retrieval
never materialises all pairs: rows are filtered by rating band and
product-quality constraints, grouped by piece count and material signature,
and only the shortlist receives the exact per-endpoint piece-assignment
analysis. An occupancy hash map answers distance-zero lookups directly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import pyarrow as pa
import pyarrow.parquet as pq

from chess_chain.domain.pieces import COUNT_NAMES
from chess_chain.similarity.schema import INDEX_SCHEMA
from chess_chain.storage.repository import PuzzleRepository

__all__ = [
    "IndexedPuzzle",
    "SimilarityIndex",
    "build_index",
    "expected_fingerprint",
    "index_is_fresh",
    "load_index",
    "save_index",
]

INDEX_PARQUET = "index.parquet"
INDEX_MANIFEST = "index.manifest.json"


@dataclass(frozen=True)
class IndexedPuzzle:
    """One candidate next puzzle row, sufficient for retrieval and shortlist
    scoring (the start board is reconstructed lazily from ``start_fen``)."""

    puzzle_id: str
    source: str
    transform_version: int
    rating: int
    rating_deviation: int
    popularity: int
    nb_plays: int
    start_fen: str
    presented_piece_count: int
    start_occupancy: int
    start_castling: int
    start_ep_square: chess.Square | None
    start_counts: tuple[int, ...]
    duplicate_queens: bool
    unusual_material: bool

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> IndexedPuzzle:
        """Build from a row produced by ``PuzzleRepository.stream_similarity_rows``."""
        (
            puzzle_id,
            source,
            transform_version,
            rating,
            rating_deviation,
            popularity,
            nb_plays,
            start_fen,
            presented_piece_count,
            start_occupancy,
            start_castling,
            start_ep_square,
            *count_values,
            duplicate_queens,
            unusual_material,
        ) = row
        counts = tuple(int(value) for value in count_values)
        if len(counts) != len(COUNT_NAMES):
            raise ValueError(f"expected {len(COUNT_NAMES)} counts, got {len(counts)}")
        return cls(
            puzzle_id=str(puzzle_id),
            source=str(source),
            transform_version=int(transform_version),
            rating=int(rating),
            rating_deviation=int(rating_deviation),
            popularity=int(popularity),
            nb_plays=int(nb_plays),
            start_fen=str(start_fen),
            presented_piece_count=int(presented_piece_count),
            start_occupancy=int(start_occupancy),
            start_castling=int(start_castling),
            start_ep_square=(
                chess.Square(int(start_ep_square)) if start_ep_square is not None else None
            ),
            start_counts=counts,
            duplicate_queens=bool(duplicate_queens),
            unusual_material=bool(unusual_material),
        )

    def board(self) -> chess.Board:
        return chess.Board(self.start_fen)


@dataclass(frozen=True)
class SimilarityIndex:
    """The in-memory candidate index over the active dataset."""

    version: int
    transform_version: int
    source_fingerprint: str
    dataset_name: str
    rows: tuple[IndexedPuzzle, ...]
    #: occupancy mask -> row indices (distance-zero lookup).
    occupancy_map: dict[int, tuple[int, ...]] = field(default_factory=dict)
    #: puzzle_id -> row index.
    row_index: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_rows(
        cls,
        *,
        version: int,
        transform_version: int,
        source_fingerprint: str,
        dataset_name: str,
        rows: tuple[IndexedPuzzle, ...],
    ) -> SimilarityIndex:
        occupancy_map: dict[int, list[int]] = {}
        row_index: dict[str, int] = {}
        for index, row in enumerate(rows):
            occupancy_map.setdefault(row.start_occupancy, []).append(index)
            row_index[row.puzzle_id] = index
        return cls(
            version=version,
            transform_version=transform_version,
            source_fingerprint=source_fingerprint,
            dataset_name=dataset_name,
            rows=rows,
            occupancy_map={mask: tuple(indices) for mask, indices in occupancy_map.items()},
            row_index=row_index,
        )

    def fingerprint(self) -> str:
        return expected_fingerprint(
            version=self.version,
            transform_version=self.transform_version,
            source_fingerprint=self.source_fingerprint,
            dataset_name=self.dataset_name,
            row_count=len(self.rows),
        )


def expected_fingerprint(
    *,
    version: int,
    transform_version: int,
    source_fingerprint: str,
    dataset_name: str,
    row_count: int,
) -> str:
    """Version fingerprint that makes stale indexes detectable."""
    payload = {
        "similarity_version": version,
        "transform_version": transform_version,
        "source_fingerprint": source_fingerprint,
        "dataset_name": dataset_name,
        "row_count": row_count,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_index(
    repo: PuzzleRepository,
    *,
    version: int,
    where: str = "",
) -> SimilarityIndex:
    """Build the index from the active dataset (streamed, bounded batches)."""
    name, source_fingerprint, transform_version = repo.active_dataset_fingerprint()
    indexed = [IndexedPuzzle.from_row(row) for row in repo.stream_similarity_rows(where=where)]
    return SimilarityIndex.from_rows(
        version=version,
        transform_version=transform_version,
        source_fingerprint=source_fingerprint,
        dataset_name=name,
        rows=tuple(indexed),
    )


def save_index(directory: Path, index: SimilarityIndex) -> None:
    """Persist index rows (Parquet) and a versioned manifest (JSON)."""
    directory.mkdir(parents=True, exist_ok=True)
    records = [_indexed_to_record(row) for row in index.rows]
    table = pa.Table.from_pylist(records, schema=INDEX_SCHEMA)
    with pq.ParquetWriter(directory / INDEX_PARQUET, INDEX_SCHEMA, compression="zstd") as writer:
        writer.write_table(table)
    manifest = {
        "fingerprint": index.fingerprint(),
        "similarity_version": index.version,
        "transform_version": index.transform_version,
        "source_fingerprint": index.source_fingerprint,
        "dataset_name": index.dataset_name,
        "row_count": len(index.rows),
    }
    manifest_path = directory / INDEX_MANIFEST
    temp_path = manifest_path.with_suffix(manifest_path.suffix + ".part")
    temp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(manifest_path)


def load_index(directory: Path) -> SimilarityIndex | None:
    """Load a persisted index, or None when no valid index exists."""
    parquet_path = directory / INDEX_PARQUET
    manifest_path = directory / INDEX_MANIFEST
    if not parquet_path.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = pq.read_table(parquet_path).to_pylist()
        rows = tuple(IndexedPuzzle.from_row(_record_to_row(record)) for record in records)
        return SimilarityIndex.from_rows(
            version=int(manifest.get("similarity_version", 0)),
            transform_version=int(manifest.get("transform_version", 0)),
            source_fingerprint=str(manifest.get("source_fingerprint", "")),
            dataset_name=str(manifest.get("dataset_name", "")),
            rows=rows,
        )
    except (ValueError, OSError, KeyError):
        return None


def index_is_fresh(directory: Path, index: SimilarityIndex) -> bool:
    """True when the persisted manifest fingerprint matches ``index`` (same
    algorithm version, source dataset and row count)."""
    manifest_path = directory / INDEX_MANIFEST
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return False
    return str(manifest.get("fingerprint", "")) == index.fingerprint()


def _indexed_to_record(row: IndexedPuzzle) -> dict[str, Any]:
    return {
        "puzzle_id": row.puzzle_id,
        "source": row.source,
        "transform_version": row.transform_version,
        "rating": row.rating,
        "rating_deviation": row.rating_deviation,
        "popularity": row.popularity,
        "nb_plays": row.nb_plays,
        "start_fen": row.start_fen,
        "presented_piece_count": row.presented_piece_count,
        "start_occupancy": row.start_occupancy,
        "start_castling": row.start_castling,
        "start_ep_square": row.start_ep_square,
        **{
            f"start_{name}": count
            for name, count in zip(COUNT_NAMES, row.start_counts, strict=True)
        },
        "duplicate_queens": row.duplicate_queens,
        "unusual_material": row.unusual_material,
    }


def _record_to_row(record: dict[str, Any]) -> tuple[Any, ...]:
    """Reverse of ``_indexed_to_record`` with the column order used by
    ``IndexedPuzzle.from_row``."""
    return (
        record["puzzle_id"],
        record["source"],
        record["transform_version"],
        record["rating"],
        record["rating_deviation"],
        record["popularity"],
        record["nb_plays"],
        record["start_fen"],
        record["presented_piece_count"],
        record["start_occupancy"],
        record["start_castling"],
        record["start_ep_square"],
        *[record[f"start_{name}"] for name in COUNT_NAMES],
        record["duplicate_queens"],
        record["unusual_material"],
    )
