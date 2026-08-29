"""Tests for the DuckDB stats command layer."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from chess_chain.config import Settings
from chess_chain.ingest.pipeline import ingest_puzzles
from chess_chain.reporting.stats import compute_stats
from chess_chain.storage.catalog import ensure_catalog
from tests.conftest import BAD_ROWS, MATE_ROWS, SAMPLE_ROWS, write_csv_zst


def test_compute_stats_reports_distributions(project_root: Path) -> None:
    rows = SAMPLE_ROWS + MATE_ROWS + [row for row, _ in BAD_ROWS]
    source = write_csv_zst(project_root / "in.csv.zst", rows)
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

    con = ensure_catalog(processed)
    try:
        stats = compute_stats(con)
        assert stats["counts"]["puzzle_rows"][0] == len(SAMPLE_ROWS) + 1
        assert stats["quarantined"]["quarantine_rows"][0] == len(BAD_ROWS)
        outcomes = stats["by_outcome"].filter(pl.col("outcome") == "checkmate")
        assert outcomes["n"][0] >= 1
        # Distributions are computed in SQL, not by loading the dataset.
        assert stats["by_rating_bucket"].height >= 1
        assert stats["by_piece_count"].height >= 1
        assert stats["by_theme"].height >= 1
        assert stats["quarantine_reasons"].height >= 1
        assert stats["flags"]["checkmates"][0] >= 1
    finally:
        con.close()
