"""Idempotency, resume and partial-output tests for the ingest pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from chess_chain.config import Settings
from chess_chain.ingest.pipeline import ingest_puzzles
from tests.conftest import (
    BAD_ROWS,
    CAPTURE_ROWS,
    CASTLING_ROWS,
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
    valid = SAMPLE_ROWS + MATE_ROWS + CHECK_ROWS + CAPTURE_ROWS + PROMOTION_ROWS + CASTLING_ROWS
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

    # Crash after the 5th record is converted. derive_core failures are now
    # quarantined per-row, so the fatal crash is simulated one stage later
    # (inside core_to_record, outside the per-row quarantine boundary).
    state = {"fail_after": 5}
    real_to_record = pipeline.core_to_record

    def flaky_record(core, **kwargs):
        if state["fail_after"] > 0:
            state["fail_after"] -= 1
            if state["fail_after"] == 0:
                raise RuntimeError("simulated crash")
        return real_to_record(core, **kwargs)

    monkeypatch.setattr(pipeline, "core_to_record", flaky_record)

    with pytest.raises(RuntimeError, match="simulated crash"):
        ingest_puzzles(input_path=source, output_dir=processed, settings=settings)

    # Partial output is unmistakable: only a .staging-* dir exists, no final dataset.
    staging_dirs = list(processed.glob(".staging-*"))
    assert len(staging_dirs) == 1
    assert not (staging_dirs[0] / "_SUCCESS").exists()
    assert not any(processed.glob("*_v*/_SUCCESS"))

    # Resume: no more failures; the dataset completes.
    monkeypatch.setattr(pipeline, "core_to_record", real_to_record)
    summary = ingest_puzzles(
        input_path=source, output_dir=processed, settings=settings, resume=True
    )
    assert not summary.skipped
    assert summary.puzzle_rows == 10
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
