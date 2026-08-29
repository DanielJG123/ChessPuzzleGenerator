"""Integration tests for the streaming ingest pipeline."""

from __future__ import annotations

from pathlib import Path

from chess_chain.config import Settings
from chess_chain.domain.pieces import MaterialSignature
from chess_chain.ingest.pipeline import ingest_puzzles
from chess_chain.storage.catalog import ensure_catalog
from chess_chain.storage.repository import PuzzleRepository
from tests.conftest import (
    BAD_ROWS,
    CAPTURE_ROWS,
    CHECK_ROWS,
    MATE_ROWS,
    PROMOTION_ROWS,
    SAMPLE_ROWS,
    write_csv_zst,
)


def _ingest_settings(project_root: Path) -> Settings:
    return Settings(
        project_root=project_root,
        data_raw_dir=project_root / "data" / "raw",
        data_processed_dir=project_root / "data" / "processed",
        ingest_batch_size=3,
        parquet_row_group_rows=2,
        rating_bucket_width=100,
    )


def _mixed_rows() -> list[str]:
    valid = SAMPLE_ROWS + MATE_ROWS + CHECK_ROWS + CAPTURE_ROWS + PROMOTION_ROWS
    return valid + [row for row, _ in BAD_ROWS]


def test_stream_to_parquet_and_duckdb_queries(project_root: Path) -> None:
    source = write_csv_zst(project_root / "in.csv.zst", _mixed_rows())
    settings = _ingest_settings(project_root)
    processed = settings.data_processed_dir
    assert processed is not None

    summary = ingest_puzzles(input_path=source, output_dir=processed, settings=settings)
    assert not summary.skipped
    assert summary.puzzle_rows == 8
    assert summary.quarantine_rows == len(BAD_ROWS)

    dataset_dir = summary.dataset_dir
    assert (dataset_dir / "_SUCCESS").is_file()
    assert (dataset_dir / "manifest.json").is_file()
    quarantine_files = list((dataset_dir / "quarantine").glob("part-*.parquet"))
    assert quarantine_files, "expected quarantine parquet files"
    partition_files = list((dataset_dir / "puzzles").glob("*/part-*.parquet"))
    assert partition_files, "expected hive-partitioned parquet files"

    con = ensure_catalog(processed)
    try:
        repo = PuzzleRepository(con)
        assert repo.count() == 8
        assert con.execute("SELECT count(*) FROM quarantined").fetchone()[0] == len(BAD_ROWS)

        # rating_bucket is a hive partition column.
        columns = [row[0] for row in con.execute("DESCRIBE puzzles").fetchall()]
        assert "rating_bucket" in columns

        assert repo.checkmate_status("checkmate").height == 2  # MATE1 + 00sHx
        assert repo.checkmate_status("check").height == 3  # CHECK1 + 00sJ9 + PROM1
        assert repo.rating_range(min_rating=1490, max_rating=1510).height == 4
        assert repo.play_count(min_plays=100).height >= 1  # 00sJ9 has 325 plays
        assert repo.theme("capture").height == 1
        assert repo.line_length(min_plies=3, max_plies=3).height >= 1

        promo = (
            repo.query("SELECT puzzle_id FROM puzzles WHERE has_promotion").to_series().to_list()
        )
        assert promo == ["PROM1"]

        # CHECK1 starts with exactly rook+king vs king; several other rows are supersets.
        sig = MaterialSignature.from_counts(wr=1, wk=1, bk=1)
        assert repo.material(sig, position="start", mode="equal").height == 1
        assert repo.material(sig, position="start", mode="superset").height == 6
    finally:
        con.close()


def test_quarantine_reasons_are_recorded(project_root: Path) -> None:
    source = write_csv_zst(project_root / "in.csv.zst", _mixed_rows())
    settings = _ingest_settings(project_root)
    processed = settings.data_processed_dir
    assert processed is not None
    ingest_puzzles(input_path=source, output_dir=processed, settings=settings)

    con = ensure_catalog(processed)
    try:
        reasons = con.execute("SELECT reason FROM quarantined").fetchall()
        reason_text = " ".join(row[0] for row in reasons)
        for _, expected in BAD_ROWS:
            assert expected in reason_text
    finally:
        con.close()
