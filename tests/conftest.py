"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import zstandard as zstd

from chess_chain.config import Settings
from chess_chain.ingest.sample import PUZZLE_HEADER

SAMPLE_ROWS = [
    (
        "00sHx,q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17,"
        "e8d7 a2e6 d7d8 f7f8,1760,80,83,72,mate mateIn2 middlegame short,"
        "https://lichess.org/yyznGmXs/black#34,"
        "Italian_Game Italian_Game_Classical_Variation,\n"
    ),
    (
        "00sJ9,r3r1k1/p4ppp/2p2n2/1p6/3P1qb1/2NQR3/PPB2PP1/R1B3K1 w - - 5 18,"
        "e3g3 e8e1 g1h2 e1c1 a1c1 f4h6 h2g1 h6c1,2671,105,87,325,"
        "advantage attraction fork middlegame sacrifice veryLong,"
        "https://lichess.org/gyFeQsOE#35,"
        "French_Defense French_Defense_Exchange_Variation,1607774862751\n"
    ),
    (
        "00sJb,Q1b2r1k/p2np2p/5bp1/q7/5P2/4B3/PPP3PP/2KR1B1R w - - 1 17,"
        "d1d7 a5e1 d7d1 e1e3 c1b1 e3b6,2235,76,97,64,advantage fork long,"
        "https://lichess.org/kiuvTFoE#33,"
        "Sicilian_Defense Sicilian_Defense_Dragon_Variation,\n"
    ),
    (
        "00sO1,1k1r4/pp3pp1/2p1p3/4b3/P3n1P1/8/KPP2PN1/3rBR1R b - - 2 31,"
        "b8c7 e1a5 b7b6 f1d1,998,85,94,293,"
        "advantage discoveredAttack master middlegame short,"
        "https://lichess.org/vsfFkG0s/black#62,,\n"
    ),
]


@dataclass
class MockHttpClient:
    """In-memory HTTP client for offline download tests."""

    url: str
    payload: bytes
    etag: str = '"test-etag"'
    last_modified: str = "Sat, 01 Jan 2025 00:00:00 GMT"
    accept_ranges: bool = True
    ignore_range: bool = False
    fail_after_bytes: int | None = None
    head_calls: int = 0
    get_calls: list[dict[str, str]] = field(default_factory=list)

    def head(self, url: str) -> dict[str, str]:
        self.head_calls += 1
        headers = {
            "content-length": str(len(self.payload)),
            "etag": self.etag,
            "last-modified": self.last_modified,
        }
        if self.accept_ranges:
            headers["accept-ranges"] = "bytes"
        return headers

    def stream_get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        chunk_size: int = 1024 * 1024,
    ) -> tuple[int, dict[str, str], Iterator[bytes]]:
        self.get_calls.append(dict(headers or {}))
        range_header = (headers or {}).get("Range")
        start = 0
        if range_header:
            start = int(range_header.removeprefix("bytes=").split("-", 1)[0])
            if self.ignore_range:
                start = 0
                body = self.payload
                status = 200
            else:
                body = self.payload[start:]
                status = 206
        else:
            body = self.payload
            status = 200

        if start > len(self.payload):
            return 416, {}, iter(())

        def _chunks() -> Iterator[bytes]:
            sent = 0
            offset = 0
            while offset < len(body):
                end = min(offset + chunk_size, len(body))
                chunk = body[offset:end]
                if self.fail_after_bytes is not None and sent + len(chunk) > self.fail_after_bytes:
                    partial = chunk[: self.fail_after_bytes - sent]
                    if partial:
                        yield partial
                    raise ConnectionError("simulated interruption")
                yield chunk
                sent += len(chunk)
                offset = end

        return status, {}, _chunks()


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "samples").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def settings(project_root: Path) -> Settings:
    return Settings(
        project_root=project_root,
        data_raw_dir=project_root / "data" / "raw",
        data_samples_dir=project_root / "data" / "samples",
        download_chunk_size=16,
        download_space_margin_bytes=0,
    )


@pytest.fixture
def tiny_puzzle_zst(project_root: Path) -> Path:
    csv_text = PUZZLE_HEADER + "".join(SAMPLE_ROWS)
    target = project_root / "data" / "raw" / "tiny_puzzles.csv.zst"
    compressor = zstd.ZstdCompressor()
    target.write_bytes(compressor.compress(csv_text.encode("utf-8")))
    return target
