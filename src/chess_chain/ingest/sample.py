"""Streamed extraction of development samples from compressed puzzle exports."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import zstandard as zstd

from chess_chain.domain.source import PUZZLE_HEADER

logger = logging.getLogger(__name__)


def stream_sample_rows(
    source_path: Path,
    output_path: Path,
    *,
    rows: int,
    chunk_size: int = 1024 * 1024,
) -> int:
    """Write the header and first *rows* data lines from a .csv.zst source."""
    if rows < 1:
        raise ValueError("rows must be at least 1")

    source_path = source_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written_rows = 0
    decompressor = zstd.ZstdDecompressor()
    pending = ""

    with (
        source_path.open("rb") as compressed,
        output_path.open("w", encoding="utf-8") as out,
        decompressor.stream_reader(compressed) as reader,
    ):
        buffer = io.BufferedReader(reader)
        header_read = False

        while written_rows < rows:
            chunk = buffer.read(chunk_size)
            if not chunk:
                break
            pending += chunk.decode("utf-8")

            if not header_read:
                newline_index = pending.find("\n")
                if newline_index == -1:
                    continue
                header_line = pending[: newline_index + 1]
                pending = pending[newline_index + 1 :]
                if header_line != PUZZLE_HEADER:
                    logger.warning(
                        "Source header differs from expected Lichess export; copying as-is"
                    )
                out.write(header_line)
                header_read = True

            while written_rows < rows:
                newline_index = pending.find("\n")
                if newline_index == -1:
                    break
                out.write(pending[: newline_index + 1])
                pending = pending[newline_index + 1 :]
                written_rows += 1

    if written_rows < rows:
        raise ValueError(
            f"Source {source_path} contains only {written_rows} data rows; requested {rows}"
        )

    logger.info("Wrote %s sample rows to %s", written_rows, output_path)
    return written_rows
