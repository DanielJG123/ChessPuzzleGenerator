"""Tests for the similarity index: building, persistence and stale detection."""

from __future__ import annotations

from pathlib import Path

from chess_chain.config import Settings
from chess_chain.ingest.pipeline import ingest_puzzles
from chess_chain.similarity.index import (
    IndexedPuzzle,
    SimilarityIndex,
    build_index,
    expected_fingerprint,
    index_is_fresh,
    load_index,
    save_index,
)
from chess_chain.storage.catalog import ensure_catalog
from chess_chain.storage.repository import PuzzleRepository
from tests.conftest import MATE_ROWS, SAMPLE_ROWS, write_csv_zst


def _ingest(project_root: Path) -> tuple[Settings, PuzzleRepository]:
    source = write_csv_zst(project_root / "in.csv.zst", MATE_ROWS + SAMPLE_ROWS[:2])
    settings = Settings(
        project_root=project_root,
        data_raw_dir=project_root / "data" / "raw",
        data_processed_dir=project_root / "data" / "processed",
        ingest_batch_size=2,
        parquet_row_group_rows=2,
    )
    processed = settings.data_processed_dir
    assert processed is not None
    ingest_puzzles(input_path=source, output_dir=processed, settings=settings)
    connection = ensure_catalog(processed)
    return settings, PuzzleRepository(connection)


def test_build_index_groups_by_piece_count_and_signature(project_root: Path) -> None:
    _, repo = _ingest(project_root)
    try:
        index = build_index(repo, version=1)
        assert len(index.rows) == 3
        assert len(index.occupancy_map) <= 3
        # Every row's start occupancy is present in the exact-lookup hash map.
        for row in index.rows:
            assert row.start_occupancy in index.occupancy_map
        assert "MATE1" in index.row_index
        assert index.transform_version == 1
        assert index.dataset_name
    finally:
        repo.close()


def test_index_persist_round_trip_and_fingerprint(project_root: Path, tmp_path: Path) -> None:
    _, repo = _ingest(project_root)
    try:
        index = build_index(repo, version=1)
        save_index(tmp_path, index)
        assert (tmp_path / "index.parquet").is_file()
        assert (tmp_path / "index.manifest.json").is_file()

        loaded = load_index(tmp_path)
        assert loaded is not None
        assert loaded.fingerprint() == index.fingerprint()
        assert [row.puzzle_id for row in loaded.rows] == [row.puzzle_id for row in index.rows]
        assert index_is_fresh(tmp_path, index) is True
    finally:
        repo.close()


def test_stale_index_detected_when_version_changes(tmp_path: Path) -> None:
    rows = (
        IndexedPuzzle(
            puzzle_id="P1",
            source="t",
            transform_version=1,
            rating=1500,
            rating_deviation=50,
            popularity=10,
            nb_plays=5,
            start_fen="4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            presented_piece_count=2,
            start_occupancy=int(
                __import__("chess").Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1").occupied
            ),
            start_castling=0,
            start_ep_square=None,
            start_counts=(0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1),
            duplicate_queens=False,
            unusual_material=False,
        ),
    )
    v1 = SimilarityIndex.from_rows(
        version=1,
        transform_version=1,
        source_fingerprint="fp",
        dataset_name="d",
        rows=rows,
    )
    save_index(tmp_path, v1)
    assert index_is_fresh(tmp_path, v1) is True

    # Same dataset identity but a bumped similarity algorithm version.
    v2 = SimilarityIndex.from_rows(
        version=2,
        transform_version=1,
        source_fingerprint="fp",
        dataset_name="d",
        rows=rows,
    )
    assert index_is_fresh(tmp_path, v2) is False

    # Bumped dataset transform version also invalidates.
    v3 = SimilarityIndex.from_rows(
        version=1,
        transform_version=2,
        source_fingerprint="fp",
        dataset_name="d",
        rows=rows,
    )
    assert index_is_fresh(tmp_path, v3) is False
    assert v1.fingerprint() != v3.fingerprint()


def test_expected_fingerprint_is_stable_and_sensitive() -> None:
    base = expected_fingerprint(
        version=1, transform_version=1, source_fingerprint="fp", dataset_name="d", row_count=3
    )
    assert (
        expected_fingerprint(
            version=1, transform_version=1, source_fingerprint="fp", dataset_name="d", row_count=3
        )
        == base
    )
    assert (
        expected_fingerprint(
            version=2, transform_version=1, source_fingerprint="fp", dataset_name="d", row_count=3
        )
        != base
    )
