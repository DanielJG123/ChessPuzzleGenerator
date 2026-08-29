"""Streamed sample extraction tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import zstandard as zstd

from chess_chain.ingest.sample import PUZZLE_HEADER, stream_sample_rows
from chess_chain.storage.manifest import (
    DownloadManifest,
    manifest_path_for,
    sha256_file,
    write_manifest,
)
from tests.conftest import SAMPLE_ROWS


def _mark_complete(path: Path) -> None:
    manifest = DownloadManifest(
        source_url="fixture",
        retrieved_at="2025-01-01T00:00:00+00:00",
        content_length=path.stat().st_size,
        etag=None,
        last_modified=None,
        local_filename=path.name,
        sha256=sha256_file(path),
    )
    write_manifest(manifest_path_for(path), manifest)


def test_stream_sample_rows_from_tiny_zst(tiny_puzzle_zst: Path, project_root: Path) -> None:
    _mark_complete(tiny_puzzle_zst)
    output = project_root / "data" / "samples" / "sample.csv"
    written = stream_sample_rows(tiny_puzzle_zst, output, rows=3)

    text = output.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert written == 3
    assert lines[0] == PUZZLE_HEADER.strip()
    assert lines[1] == SAMPLE_ROWS[0].strip()
    assert lines[3] == SAMPLE_ROWS[2].strip()


def test_stream_sample_rows_rejects_short_source(project_root: Path) -> None:
    csv_text = PUZZLE_HEADER + SAMPLE_ROWS[0]
    source = project_root / "data" / "raw" / "one_row.csv.zst"
    source.write_bytes(zstd.ZstdCompressor().compress(csv_text.encode("utf-8")))
    _mark_complete(source)
    output = project_root / "data" / "samples" / "too_many.csv"

    with pytest.raises(ValueError, match="only 1 data rows"):
        stream_sample_rows(source, output, rows=3)
