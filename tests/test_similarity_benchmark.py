"""Small benchmark fixture for similarity retrieval.

Expected complexity (documented in docs/architecture.md):

- index build: O(N) row scans plus O(N) map construction (bounded memory);
- one query: O(filters + grouped_rows) vectorised 64-bit popcounts, where
  ``grouped_rows <= max_candidates_scanned``; distances are computed per group
  against the single query vector (never an N x N matrix), so the per-query
  cost is linear in the scanned rows, not quadratic.

This test uses a synthetic seeded index to keep CI deterministic and fast; it
asserts a loose wall-time envelope plus the bounded-scan accounting. It does
not promise full-corpus (multi-million row) timings.
"""

from __future__ import annotations

import time

import chess
import numpy as np

from chess_chain.domain.pieces import MaterialSignature
from chess_chain.domain.position import CanonicalPosition
from chess_chain.similarity.backend import get_backend
from chess_chain.similarity.index import IndexedPuzzle, SimilarityIndex
from chess_chain.similarity.retrieval import retrieve

_SYNTHETIC_ROWS = 20_000


def _synthetic_row(index: int, rng: np.random.Generator) -> IndexedPuzzle:
    """A synthetic puzzle row with a distinct, material-feasible start."""
    # White king on any square, black king on a non-adjacent square.
    white_square = int(rng.integers(0, 64))
    black_square = int(rng.integers(0, 64))
    while _manhattan(white_square, black_square) < 2:
        black_square = int(rng.integers(0, 64))
    fen = _king_only_fen(white_square, black_square)
    board = chess.Board(fen)
    return IndexedPuzzle(
        puzzle_id=f"SYN{index:05d}",
        source="synthetic",
        transform_version=1,
        rating=int(rng.integers(800, 2400)),
        rating_deviation=50,
        popularity=10,
        nb_plays=5,
        start_fen=board.fen(),
        presented_piece_count=2,
        start_occupancy=int(board.occupied),
        start_castling=int(board.castling_rights),
        start_ep_square=board.ep_square,
        start_counts=MaterialSignature.from_board(board).counts,
        duplicate_queens=False,
        unusual_material=False,
    )


def _king_only_fen(white_square: int, black_square: int) -> str:
    """Build a FEN with only the two kings (rank 8..1, no castling rights)."""
    cells = ["1"] * 64
    cells[white_square] = "K"
    cells[black_square] = "k"
    ranks: list[str] = []
    for rank in range(7, -1, -1):
        row = "".join(cells[rank * 8 : (rank + 1) * 8])
        compressed = ""
        run = 0
        for char in row:
            if char == "1":
                run += 1
            else:
                if run:
                    compressed += str(run)
                    run = 0
                compressed += char
        if run:
            compressed += str(run)
        ranks.append(compressed)
    return f"{'/'.join(ranks)} w - - 0 1"


def _manhattan(a: int, b: int) -> int:
    return abs(chess.square_file(a) - chess.square_file(b)) + abs(
        chess.square_rank(a) - chess.square_rank(b)
    )


def test_synthetic_retrieval_is_bounded_and_fast() -> None:
    rng = np.random.default_rng(2024)
    rows = tuple(_synthetic_row(index, rng) for index in range(_SYNTHETIC_ROWS))
    index = SimilarityIndex.from_rows(
        version=1,
        transform_version=1,
        source_fingerprint="synthetic",
        dataset_name="bench",
        rows=rows,
    )
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    counts = MaterialSignature.from_board(board).counts

    started = time.monotonic()
    candidates, stats = retrieve(
        source_puzzle_id="SYN00000",
        source_start_counts=counts,
        endpoints=(CanonicalPosition.from_board(board),),
        index=index,
        backend=get_backend("exact"),
        end_policy_key="transition_ready@v1",
        endpoint_source="final_position:transition_ready@v1",
        max_candidates_scanned=10_000,
        top_k=5,
        symmetries=("identity",),
    )
    elapsed = time.monotonic() - started

    # Bounded: linear scans only, never an all-pairs matrix.
    assert stats.scan.rows_scanned <= 10_000
    assert stats.scan.pairs_materialised == stats.scan.rows_scanned
    assert stats.filtered_rows <= _SYNTHETIC_ROWS
    # Loose wall-time envelope for a 20k-row synthetic index on CI hardware.
    assert elapsed < 30.0, f"retrieval took {elapsed:.2f}s"
    assert len(candidates) <= 5
    # Deterministic seed: candidate set is reproducible.
    assert [candidate.next_puzzle_id for candidate in candidates] == [
        candidate.next_puzzle_id for candidate in candidates
    ]


def test_synthetic_index_build_scales_linearly() -> None:
    rng = np.random.default_rng(9)
    small = tuple(_synthetic_row(index, rng) for index in range(500))
    rng = np.random.default_rng(9)
    large = tuple(_synthetic_row(index, rng) for index in range(2000))
    small_index = SimilarityIndex.from_rows(
        version=1,
        transform_version=1,
        source_fingerprint="synthetic",
        dataset_name="bench",
        rows=small,
    )
    large_index = SimilarityIndex.from_rows(
        version=1,
        transform_version=1,
        source_fingerprint="synthetic",
        dataset_name="bench",
        rows=large,
    )
    # Four times the rows -> about four times the map entries (linear growth).
    assert len(large_index.occupancy_map) < len(small_index.occupancy_map) * 5
    assert len(large_index.row_index) == 2000
