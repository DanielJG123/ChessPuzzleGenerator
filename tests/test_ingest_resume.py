"""Idempotency, resume and partial-output tests for the ingest pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from chess_chain.config import Settings
from chess_chain.ingest.pipeline import ingest_puzzles
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


def test_ingest_is_idempotent_and_force_rebuilds(project_root: Path) -> None:
    source = write_csv_zst(project_root / "in.csv.zst", _mixed_rows())
    settings = _ingest_settings(project_root)
    processed = settings.data_processed_dir
    assert processed is not None

    first = ingest_puzzles(input_path=source, output_dir=processed, settings=settings)
    second = ingest_puzzles(input_path=source, output_dir=processed, settings=settings)
    assert second.skipped
    assert second.dataset_dir == first.dataset_dir

    forced = ingest_puzzles(input_path=source, output_dir=processed, settings=settings, force=True)
    assert not forced.skipped
    assert forced.puzzle_rows == first.puzzle_rows


def test_partial_outputs_are_unmistakable_and_resume_works(project_root: Path, monkeypatch) -> None:
    import chess_chain.ingest.pipeline as pipeline

    source = write_csv_zst(project_root / "in.csv.zst", _mixed_rows())
    settings = _ingest_settings(project_root)
    processed = settings.data_processed_dir
    assert processed is not None

    state = {"fail_after": 5}
    real_derive = pipeline.derive_core

    def flaky_derive(source):
        if state["fail_after"] > 0:
            state["fail_after"] -= 1
            if state["fail_after"] == 0:
                raise RuntimeError("simulated crash")
        return real_derive(source)

    monkeypatch.setattr(pipeline, "derive_core", flaky_derive)

    with pytest.raises(RuntimeError, match="simulated crash"):
        ingest_puzzles(input_path=source, output_dir=processed, settings=settings)

    # Partial output is unmistakable: only a .staging-* dir exists, no final dataset.
    staging_dirs = list(processed.glob(".staging-*"))
    assert len(staging_dirs) == 1
    assert not (staging_dirs[0] / "_SUCCESS").exists()
    assert not any(processed.glob("*_v*/_SUCCESS"))

    # Resume: no more failures; the dataset completes.
    monkeypatch.setattr(pipeline, "derive_core", real_derive)
    summary = ingest_puzzles(
        input_path=source, output_dir=processed, settings=settings, resume=True
    )
    assert not summary.skipped
    assert summary.puzzle_rows == 8
    assert summary.quarantine_rows == len(BAD_ROWS)
    assert (summary.dataset_dir / "_SUCCESS").is_file()
    assert not list(processed.glob(".staging-*"))


def test_source_change_invalidates_dataset(project_root: Path) -> None:
    source = write_csv_zst(project_root / "in.csv.zst", _mixed_rows())
    settings = _ingest_settings(project_root)
    processed = settings.data_processed_dir
    assert processed is not None

    first = ingest_puzzles(input_path=source, output_dir=processed, settings=settings)
    # Changing the source content changes its fingerprint, so a rebuild happens.
    write_csv_zst(source, _mixed_rows() + MATE_ROWS)
    second = ingest_puzzles(input_path=source, output_dir=processed, settings=settings)
    assert not second.skipped
    assert second.dataset_dir == first.dataset_dir  # stable name, safely replaced
    assert second.puzzle_rows == first.puzzle_rows + 1
