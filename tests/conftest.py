"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import chess
import pytest
import zstandard as zstd

from chess_chain.analysis.engine import EngineAnalysisResult, EngineMoveInfo
from chess_chain.config import Settings
from chess_chain.ingest.sample import PUZZLE_HEADER


def engine_fen(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])


class FakeEngineAdapter:
    """Deterministic UCI engine substitute for offline deep-analysis tests.

    ``results`` maps an engine FEN to an ordered list of ``(uci, score_cp)``
    tuples (best first); every ``analyze`` call is recorded in ``calls``.
    """

    def __init__(
        self,
        results: dict[str, list[tuple[str, int]]],
        engine_name: str = "FakeStockfish",
        engine_version: str = "0.1.0",
        budget: str = '{"fake": true}',
    ) -> None:
        self.results = results
        self.engine_name = engine_name
        self.engine_version = engine_version
        self.budget = budget
        self.calls: list[str] = []

    def engine_identity(self) -> tuple[str, str]:
        return (self.engine_name, self.engine_version)

    def analyze(
        self,
        board: chess.Board,
        *,
        multipv: int | None = None,
        mate_value_cp: int = 10_000,
    ) -> EngineAnalysisResult:
        fen = engine_fen(board)
        self.calls.append(fen)
        moves = tuple(
            EngineMoveInfo(uci=uci, score_cp=score_cp, score_mate=None, rank=rank)
            for rank, (uci, score_cp) in enumerate(self.results.get(fen, []))
        )
        return EngineAnalysisResult(
            fen=fen,
            moves=moves,
            engine_name=self.engine_name,
            engine_version=self.engine_version,
            budget=self.budget,
        )

    def close(self) -> None:
        self.calls = []


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

# Crafted rows exercising captures, promotions, check and mate.
CAPTURE_ROWS = [
    (
        "CAPT1,6k1/5ppp/8/8/8/8/8/6RK b - - 0 1,g8f8 g1g7,"
        "1500,80,10,50,capture,https://lichess.org/xyz,\n"
    ),
]

PROMOTION_ROWS = [
    (
        "PROM1,4k3/4P3/8/8/8/8/8/4K3 b - - 0 1,e8d7 e7e8q,"
        "1500,80,10,50,promotion,https://lichess.org/xyz,\n"
    ),
]

MATE_ROWS = [
    (
        "MATE1,6k1/6pp/8/8/8/8/8/5R1K b - - 0 1,g8h8 f1f8,"
        "1500,80,10,50,mate,https://lichess.org/xyz,\n"
    ),
]

CHECK_ROWS = [
    (
        "CHECK1,7k/8/8/8/8/8/8/R6K b - - 0 1,h8h7 a1a7,"
        "1500,80,10,50,advantage,https://lichess.org/xyz,\n"
    ),
]

# Castling: one UCI moves king *and* rook, so the castled rook must be tracked.
# ``090As`` is the real Lichess export row whose final ``e8g8`` previously
# crashed ingest with ``KeyError: (False, 61)`` because the rook identity was
# left on h8 while the real board had it on f8 delivering check.
CASTLING_ROWS = [
    (
        "090As,4k2r/2qn2p1/1r2p3/p1P4p/Q3p1n1/B6P/P3B1P1/R4R1K w k - 1 26,"
        "h3g4 h5g4 h1g1 c7h2 g1f2 e8g8,2789,76,97,64,advantage castling long,"
        "https://lichess.org/x/black#34,\n"
    ),
    (
        "CASTL2,4k3/8/8/8/8/8/8/R3K3 b Q - 0 1,e8d8 e1c1,"
        "1500,80,10,50,advantage,https://lichess.org/xyz,\n"
    ),
]

# Rows that must be quarantined, with the reason substring to expect.
BAD_ROWS: list[tuple[str, str]] = [
    (
        "BAD1,not-a-fen,e2e4,1500,80,10,50,mate,https://lichess.org/xyz,\n",
        "malformed FEN",
    ),
    (
        "BAD2,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,e2e5 e7e5,"
        "1500,80,10,50,mate,https://lichess.org/xyz,\n",
        "illegal preceding move",
    ),
    (
        "BAD3,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,e2e4 e7e5 q6q6,"
        "1500,80,10,50,mate,https://lichess.org/xyz,\n",
        "malformed solution move",
    ),
    (
        "BAD4,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,e2e4,"
        "1500,80,10,50,mate,https://lichess.org/xyz,\n",
        "no solution moves",
    ),
    (
        "BAD5,x,y,z\n",
        "expected at least 10 fields",
    ),
    (
        "BAD6,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,e2e4 e7e5,"
        "abc,80,10,50,mate,https://lichess.org/xyz,\n",
        "non-numeric rating",
    ),
]


def write_csv_zst(path: Path, rows: list[str], header: str = PUZZLE_HEADER) -> Path:
    """Write ``rows`` (already including trailing newlines) as a .csv.zst file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (header + "".join(rows)).encode("utf-8")
    path.write_bytes(zstd.ZstdCompressor().compress(payload))
    return path


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
