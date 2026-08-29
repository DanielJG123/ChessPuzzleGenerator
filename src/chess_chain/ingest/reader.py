"""Streaming CSV readers for raw puzzle exports.

Inputs may be plain ``.csv`` or zstandard-compressed ``.csv.zst`` files. The
decompressed stream is never materialised in memory: rows are read lazily.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import zstandard as zstd

__all__ = ["open_text_stream", "iter_csv_fields"]


def open_text_stream(path: Path) -> io.TextIOWrapper:
    """Open a ``.csv`` or ``.csv.zst`` file as a streaming text wrapper."""
    raw: Any = path.open("rb")
    if path.suffix.lower() == ".zst":
        raw = zstd.ZstdDecompressor().stream_reader(raw)
    return io.TextIOWrapper(raw, encoding="utf-8", newline="")


def iter_csv_fields(
    path: Path,
    *,
    skip_header: bool = True,
    skip_rows: int = 0,
) -> Iterator[tuple[int, list[str]]]:
    """Yield ``(line_number, fields)`` for each data row.

    Line numbers refer to positions in the decompressed text starting at 1.
    ``skip_rows`` discards that many data rows (used for resume) while
    preserving original line numbers for quarantine reporting.
    """
    text = open_text_stream(path)
    with text:
        reader = csv.reader(text)
        skipped = 0
        for line_number, fields in enumerate(reader, start=1):
            if not fields:
                continue
            if skip_header and fields[0].strip().lower() == "puzzleid":
                continue
            if skipped < skip_rows:
                skipped += 1
                continue
            yield line_number, fields
