"""Streaming, partitioned Parquet ingestion orchestration.

Reads a ``.csv`` / ``.csv.zst`` Lichess puzzle export row by row, replays each
line with python-chess, writes valid rows to hive-partitioned Parquet
(partitioned by rating bucket) and quarantines invalid rows. Memory stays
bounded: no full decompression and no full-dataset DataFrame in memory.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from chess_chain.chess.replay import derive_core
from chess_chain.config import Settings
from chess_chain.domain.errors import IngestRejectError
from chess_chain.domain.source import source_from_fields
from chess_chain.ingest.reader import iter_csv_fields
from chess_chain.ingest.records import (
    IngestSummary,
    core_to_record,
    load_ingest_manifest,
    quarantine_record,
    source_fingerprint,
    write_ingest_manifest,
)
from chess_chain.storage.schema import PUZZLE_SCHEMA, QUARANTINE_SCHEMA

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "lichess_puzzles"

__all__ = ["DEFAULT_DATASET_NAME", "PartitionedParquetWriter", "ingest_puzzles"]


class PartitionedParquetWriter:
    """Buffers records and writes one Parquet file per hive partition.

    Records are buffered per partition value and flushed as row groups so that
    row groups stay reasonably sized and the number of files stays equal to the
    number of distinct partition values (avoiding thousands of tiny files).
    """

    def __init__(
        self,
        root: Path,
        schema: pa.Schema,
        partition_col: str,
        row_group_rows: int,
    ) -> None:
        self.root = root
        self.schema = schema
        self.partition_col = partition_col
        self.row_group_rows = row_group_rows
        self._buffers: dict[object, list[dict[str, object]]] = {}
        self._writers: dict[object, pq.ParquetWriter] = {}

    def add(self, partition_value: object, record: dict[str, object]) -> None:
        bucket = self._buffers.setdefault(partition_value, [])
        bucket.append(record)
        if len(bucket) >= self.row_group_rows:
            self._flush(partition_value)

    def flush_all(self) -> None:
        """Flush every buffered partition to disk (persists all added records)."""
        for partition_value in list(self._buffers):
            self._flush(partition_value)

    def _flush(self, partition_value: object) -> None:
        records = self._buffers.pop(partition_value)
        if not records:
            return
        table = pa.Table.from_pylist(records, schema=self.schema)
        writer = self._writer_for(partition_value)
        writer.write_table(table)

    def _writer_for(self, partition_value: object) -> pq.ParquetWriter:
        writer = self._writers.get(partition_value)
        if writer is None:
            partition_dir = self.root / f"{self.partition_col}={partition_value}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            part_index = len(list(partition_dir.glob("part-*.parquet")))
            path = partition_dir / f"part-{part_index:05d}.parquet"
            writer = pq.ParquetWriter(path, self.schema, compression="zstd")
            self._writers[partition_value] = writer
        return writer

    def close(self) -> None:
        for partition_value in list(self._buffers):
            self._flush(partition_value)
        if not self._writers:
            # Guarantee the dataset is readable even when empty: write one empty
            # part file so readers can glob the hive-partitioned directory.
            partition_dir = self.root / f"{self.partition_col}=0"
            partition_dir.mkdir(parents=True, exist_ok=True)
            path = partition_dir / "part-00000.parquet"
            with pq.ParquetWriter(path, self.schema, compression="zstd") as writer:
                writer.write_table(pa.Table.from_pylist([], schema=self.schema))
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()


class PartFileWriter:
    """Streams records into incrementing ``part-NNNNN.parquet`` files.

    Existing part files are preserved on resume; new records go into the next
    part file, so previously persisted records are never duplicated or lost.
    """

    def __init__(self, directory: Path, schema: pa.Schema, row_group_rows: int) -> None:
        self.directory = directory
        self._schema = schema
        self._row_group_rows = row_group_rows
        directory.mkdir(parents=True, exist_ok=True)
        existing = list(directory.glob("part-*.parquet"))
        self._part_index = len(existing)
        self._buffer: list[dict[str, object]] = []
        self._writer: pq.ParquetWriter | None = None

    def add(self, record: dict[str, object]) -> None:
        self._buffer.append(record)
        if len(self._buffer) >= self._row_group_rows:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        if self._writer is None:
            path = self.directory / f"part-{self._part_index:05d}.parquet"
            self._part_index += 1
            self._writer = pq.ParquetWriter(path, self._schema, compression="zstd")
        assert self._writer is not None
        table = pa.Table.from_pylist(self._buffer, schema=self._schema)
        self._writer.write_table(table)
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        elif not any(self.directory.glob("part-*.parquet")):
            # Guarantee readers can glob this directory even when no records
            # were written (e.g. zero quarantined rows).
            path = self.directory / "part-00000.parquet"
            with pq.ParquetWriter(path, self._schema, compression="zstd") as writer:
                writer.write_table(pa.Table.from_pylist([], schema=self._schema))


@dataclass(frozen=True)
class _Checkpoint:
    rows_processed: int
    puzzle_rows: int
    quarantine_rows: int


def _write_checkpoint(staging_dir: Path, checkpoint: _Checkpoint) -> None:
    path = staging_dir / "checkpoint.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(
            {
                "rows_processed": checkpoint.rows_processed,
                "puzzle_rows": checkpoint.puzzle_rows,
                "quarantine_rows": checkpoint.quarantine_rows,
            }
        ),
        encoding="utf-8",
    )
    os.replace(temp, path)


def _read_checkpoint(staging_dir: Path) -> _Checkpoint:
    path = staging_dir / "checkpoint.json"
    if not path.is_file():
        return _Checkpoint(rows_processed=0, puzzle_rows=0, quarantine_rows=0)
    data = json.loads(path.read_text(encoding="utf-8"))
    return _Checkpoint(
        rows_processed=int(data.get("rows_processed", 0)),
        puzzle_rows=int(data.get("puzzle_rows", 0)),
        quarantine_rows=int(data.get("quarantine_rows", 0)),
    )


def _find_staging_dir(output_dir: Path) -> Path | None:
    candidates = [path for path in output_dir.glob(".staging-*") if path.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _publish_staging(staging_dir: Path, final_dir: Path) -> None:
    """Move a complete staging directory into place, replacing any old output."""
    if final_dir.exists():
        backup = final_dir.with_name(final_dir.name + ".old")
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        os.rename(final_dir, backup)
        try:
            os.rename(staging_dir, final_dir)
        except Exception:
            os.rename(backup, final_dir)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    else:
        os.rename(staging_dir, final_dir)


def ingest_puzzles(
    *,
    input_path: Path,
    output_dir: Path,
    settings: Settings,
    force: bool = False,
    resume: bool = False,
    dataset_name: str = DEFAULT_DATASET_NAME,
) -> IngestSummary:
    """Stream ``input_path`` into a partitioned Parquet dataset.

    Idempotency: if the final dataset already exists with the same
    transformation version and source fingerprint, the run is skipped unless
    ``force`` is set. Partial outputs live in ``.staging-*`` directories that
    are never mistaken for final outputs; ``resume`` continues from the last
    checkpoint at a safe batch boundary.
    """
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input file does not exist: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    fingerprint = source_fingerprint(input_path)
    source_name = input_path.name
    transform_version = settings.transform_version
    final_dir = output_dir / f"{dataset_name}_v{transform_version}"
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat()

    existing_manifest = load_ingest_manifest(final_dir)
    success_marker = final_dir / "_SUCCESS"
    already_current = (
        not force
        and success_marker.is_file()
        and existing_manifest is not None
        and existing_manifest.get("transform_version") == transform_version
        and existing_manifest.get("source_fingerprint") == fingerprint
    )
    if already_current:
        assert existing_manifest is not None
        logger.info("Dataset already up to date: %s (use --force to rebuild)", final_dir)
        return IngestSummary(
            dataset_dir=final_dir,
            dataset_name=dataset_name,
            transform_version=transform_version,
            source_path=str(input_path),
            source_fingerprint=fingerprint,
            puzzle_rows=int(str(existing_manifest["puzzle_rows"])),
            quarantine_rows=int(str(existing_manifest["quarantine_rows"])),
            created_at=str(existing_manifest.get("created_at", created_at)),
            elapsed_seconds=0.0,
            skipped=True,
        )

    staging_dir = output_dir / f".staging-{uuid.uuid4().hex}"
    skip_rows = 0
    puzzle_rows = 0
    quarantine_rows = 0
    if resume:
        previous = _find_staging_dir(output_dir)
        if previous is not None:
            checkpoint = _read_checkpoint(previous)
            skip_rows = checkpoint.rows_processed
            puzzle_rows = checkpoint.puzzle_rows
            quarantine_rows = checkpoint.quarantine_rows
            staging_dir = previous
            logger.info(
                "Resuming from row %d in %s (%d puzzles, %d quarantined already persisted)",
                skip_rows,
                staging_dir,
                puzzle_rows,
                quarantine_rows,
            )
        else:
            logger.info("No staging directory found; starting a fresh run")
    elif not force:
        stale = _find_staging_dir(output_dir)
        if stale is not None:
            logger.warning("Removing stale partial output %s", stale)
            shutil.rmtree(stale, ignore_errors=True)

    staging_dir.mkdir(parents=True, exist_ok=True)

    batch_size = max(1, settings.ingest_batch_size)
    row_group_rows = max(1, settings.parquet_row_group_rows)
    bucket_width = max(1, settings.rating_bucket_width)

    puzzle_writer = PartitionedParquetWriter(
        staging_dir / "puzzles",
        schema=PUZZLE_SCHEMA,
        partition_col="rating_bucket",
        row_group_rows=row_group_rows,
    )
    quarantine_writer = PartFileWriter(
        staging_dir / "quarantine",
        schema=QUARANTINE_SCHEMA,
        row_group_rows=max(1, row_group_rows // 4),
    )

    rows_processed = skip_rows
    last_checkpoint = time.monotonic()

    try:
        for line_number, fields in iter_csv_fields(input_path, skip_rows=skip_rows):
            try:
                source = source_from_fields(fields, line_number)
            except IngestRejectError as exc:
                quarantine_writer.add(
                    quarantine_record(
                        source_name=source_name,
                        line_number=line_number,
                        reason=str(exc),
                        fields=fields,
                    )
                )
                quarantine_rows += 1
                rows_processed += 1
                continue

            try:
                core = derive_core(source)
            except IngestRejectError as exc:
                quarantine_writer.add(
                    quarantine_record(
                        source_name=source_name,
                        line_number=line_number,
                        reason=str(exc),
                        fields=fields,
                    )
                )
                quarantine_rows += 1
                rows_processed += 1
                continue

            record = core_to_record(
                core,
                source_name=source_name,
                transform_version=transform_version,
            )
            bucket = (source.rating // bucket_width) * bucket_width
            puzzle_writer.add(bucket, record)
            puzzle_rows += 1
            rows_processed += 1

            if rows_processed % batch_size == 0:
                puzzle_writer.flush_all()
                quarantine_writer.flush()
                _write_checkpoint(
                    staging_dir,
                    _Checkpoint(rows_processed, puzzle_rows, quarantine_rows),
                )
                now = time.monotonic()
                elapsed = max(now - last_checkpoint, 1e-9)
                rate = batch_size / elapsed
                logger.info(
                    "processed=%d valid=%d quarantined=%d (%.0f rows/s)",
                    rows_processed,
                    puzzle_rows,
                    quarantine_rows,
                    rate,
                )
                last_checkpoint = now

        puzzle_writer.close()
        quarantine_writer.close()
        _write_checkpoint(
            staging_dir,
            _Checkpoint(rows_processed, puzzle_rows, quarantine_rows),
        )
        (staging_dir / "_SUCCESS").write_text("ok\n", encoding="utf-8")

        summary = IngestSummary(
            dataset_dir=final_dir,
            dataset_name=dataset_name,
            transform_version=transform_version,
            source_path=str(input_path),
            source_fingerprint=fingerprint,
            puzzle_rows=puzzle_rows,
            quarantine_rows=quarantine_rows,
            created_at=created_at,
            elapsed_seconds=time.monotonic() - started,
            skipped=False,
        )
        write_ingest_manifest(staging_dir, summary.to_manifest_dict())
        _publish_staging(staging_dir, final_dir)
    except Exception:
        puzzle_writer.close()
        quarantine_writer.close()
        _write_checkpoint(
            staging_dir,
            _Checkpoint(rows_processed, puzzle_rows, quarantine_rows),
        )
        logger.error(
            "Ingestion interrupted after %d rows; partial output kept at %s",
            rows_processed,
            staging_dir,
        )
        raise

    logger.info(
        "Ingest complete: %d puzzles, %d quarantined -> %s",
        puzzle_rows,
        quarantine_rows,
        final_dir,
    )
    return summary
