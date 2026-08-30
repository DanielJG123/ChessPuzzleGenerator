"""Tests for similarity retrieval: ranking, exact lookups and bounded scans."""

from __future__ import annotations

import chess

from chess_chain.domain.pieces import MaterialSignature
from chess_chain.domain.position import CanonicalPosition
from chess_chain.similarity.backend import get_backend
from chess_chain.similarity.index import IndexedPuzzle, SimilarityIndex
from chess_chain.similarity.retrieval import retrieve


def _pos(fen: str) -> CanonicalPosition:
    return CanonicalPosition.from_board(chess.Board(fen))


def _row(pid: str, fen: str, rating: int = 1500) -> IndexedPuzzle:
    board = chess.Board(fen)
    return IndexedPuzzle(
        puzzle_id=pid,
        source="t",
        transform_version=1,
        rating=rating,
        rating_deviation=50,
        popularity=10,
        nb_plays=5,
        start_fen=board.fen(),
        presented_piece_count=len(board.piece_map()),
        start_occupancy=int(board.occupied),
        start_castling=int(board.castling_rights),
        start_ep_square=board.ep_square,
        start_counts=MaterialSignature.from_board(board).counts,
        duplicate_queens=False,
        unusual_material=False,
    )


def _index(*rows: IndexedPuzzle) -> SimilarityIndex:
    return SimilarityIndex.from_rows(
        version=1,
        transform_version=1,
        source_fingerprint="fp",
        dataset_name="d",
        rows=tuple(rows),
    )


def test_retrieval_ranks_exact_match_first() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    counts = MaterialSignature.from_board(board).counts
    index = _index(
        _row("NEAR", "4k3/8/8/8/8/8/8/5K2 w - - 0 1"),
        _row("EXACT", "4k3/8/8/8/8/8/8/4K3 w - - 0 1"),
        _row("FAR", "4k3/8/8/8/8/8/8/7K w - - 0 1"),
    )
    candidates, stats = retrieve(
        source_puzzle_id="S1",
        source_start_counts=counts,
        endpoints=(_pos(board.fen()),),
        index=index,
        backend=get_backend("exact"),
        end_policy_key="transition_ready@v1",
        endpoint_source="final_position:transition_ready@v1",
        top_k=3,
        symmetries=("identity",),
    )
    assert [candidate.next_puzzle_id for candidate in candidates] == [
        "EXACT",
        "NEAR",
        "FAR",
    ]
    assert candidates[0].d_occ_best == 0.0
    assert all(comparison.exact_journey for comparison in candidates[0].per_endpoint)
    assert stats.scan.exact_lookup_hits >= 1


def test_rating_band_filters_targets() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    counts = MaterialSignature.from_board(board).counts
    index = _index(
        _row("LOW", board.fen(), rating=900),
        _row("HIGH", board.fen(), rating=1600),
    )
    candidates, stats = retrieve(
        source_puzzle_id="S1",
        source_start_counts=counts,
        endpoints=(_pos(board.fen()),),
        index=index,
        backend=get_backend("exact"),
        end_policy_key="transition_ready@v1",
        endpoint_source="final_position:transition_ready@v1",
        rating_min=1500,
        rating_max=1800,
        top_k=5,
        symmetries=("identity",),
    )
    assert [candidate.next_puzzle_id for candidate in candidates] == ["HIGH"]


def test_retrieval_never_materialises_all_pairs() -> None:
    # A single large group plus a hard scan budget: the retriever must cap the
    # rows scanned and report linear (not quadratic) work.
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    counts = MaterialSignature.from_board(board).counts
    rows = [
        _row(f"P{i:04d}", "4k3/8/8/8/8/8/8/5K2 w - - 0 1" if i % 3 == 0 else board.fen())
        for i in range(3000)
    ]
    index = _index(*rows)
    candidates, stats = retrieve(
        source_puzzle_id="S1",
        source_start_counts=counts,
        endpoints=(_pos(board.fen()),),
        index=index,
        backend=get_backend("exact"),
        end_policy_key="transition_ready@v1",
        endpoint_source="final_position:transition_ready@v1",
        max_candidates_scanned=50,
        top_k=5,
        symmetries=("identity",),
    )
    assert stats.scan.rows_scanned <= 50
    # Every scanned row is compared against the single query vector (linear).
    assert stats.scan.pairs_materialised == stats.scan.rows_scanned
    assert stats.scan.rows_scanned < 3000
    assert stats.scan.truncated is True
    assert candidates


def test_accelerated_backend_returns_compatible_rankings() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    counts = MaterialSignature.from_board(board).counts
    index = _index(
        _row("A", "4k3/8/8/8/8/8/8/4K3 w - - 0 1"),
        _row("B", "4k3/8/8/8/8/8/8/5K2 w - - 0 1"),
        _row("C", "4k3/8/8/8/8/8/8/6K1 w - - 0 1"),
    )
    exact_candidates, _ = retrieve(
        source_puzzle_id="S1",
        source_start_counts=counts,
        endpoints=(_pos(board.fen()),),
        index=index,
        backend=get_backend("exact"),
        end_policy_key="transition_ready@v1",
        endpoint_source="final_position:transition_ready@v1",
        top_k=3,
        symmetries=("identity",),
    )
    accelerated_candidates, _ = retrieve(
        source_puzzle_id="S1",
        source_start_counts=counts,
        endpoints=(_pos(board.fen()),),
        index=index,
        backend=get_backend("accelerated"),
        end_policy_key="transition_ready@v1",
        endpoint_source="final_position:transition_ready@v1",
        top_k=3,
        symmetries=("identity",),
    )
    assert [c.next_puzzle_id for c in exact_candidates] == [
        c.next_puzzle_id for c in accelerated_candidates
    ]
    assert [c.d_occ_best for c in exact_candidates] == [
        c.d_occ_best for c in accelerated_candidates
    ]


def test_multi_endpoint_query_uses_every_endpoint_mask() -> None:
    # Two accepted endpoints with different occupancy: the retrieval must query
    # both masks (bounded) and the resulting candidate must carry per-endpoint
    # journeys for both.
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    other = chess.Board("4k3/8/8/8/8/8/8/5K2 w - - 0 1")
    counts = MaterialSignature.from_board(board).counts
    index = _index(
        _row("TARGET", board.fen()),
        _row("OTHER", "4k3/8/8/8/8/8/8/5K2 w - - 0 1"),
    )
    candidates, stats = retrieve(
        source_puzzle_id="S1",
        source_start_counts=counts,
        endpoints=(_pos(board.fen()), _pos(other.fen())),
        index=index,
        backend=get_backend("exact"),
        end_policy_key="transition_ready@v1",
        endpoint_source="variation_analysis:transition_ready@v1",
        top_k=2,
        symmetries=("identity",),
    )
    assert len(candidates) == 2
    target = next(candidate for candidate in candidates if candidate.next_puzzle_id == "TARGET")
    assert len(target.per_endpoint) == 2
    assert {comparison.d_occ for comparison in target.per_endpoint} == {0, 2}
    assert target.d_occ_mean == 1.0
    # Bounded: no all-pairs materialisation happened.
    assert stats.scan.pairs_materialised == stats.scan.rows_scanned
    # Both endpoint masks were queried (2 exact-lookup hits, one per mask).
    assert stats.scan.exact_lookup_hits == 2
