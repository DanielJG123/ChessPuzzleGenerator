"""Tests for layered position distance, inventory and piece assignment."""

from __future__ import annotations

import itertools

import chess
import pytest

from chess_chain.domain.pieces import MaterialSignature
from chess_chain.domain.position import CanonicalPosition
from chess_chain.similarity.assignment import (
    CostWeights,
    assign_pieces,
    tray_counts,
)
from chess_chain.similarity.candidate import build_candidate
from chess_chain.similarity.distance import (
    available_counts,
    compare_endpoint,
)

COST = CostWeights(move=1.0, add=3.0, remove=2.0)


def _pos(fen: str) -> CanonicalPosition:
    return CanonicalPosition.from_board(chess.Board(fen))


def _counts(board: chess.Board) -> tuple[int, ...]:
    return MaterialSignature.from_board(board).counts


def test_known_hamming_distances() -> None:
    # White king one square over: two occupancy bits differ.
    endpoint = _pos("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    start = _pos("4k3/8/8/8/8/8/8/5K2 w - - 0 1")
    comparison = compare_endpoint(
        endpoint,
        start,
        endpoint_index=0,
        source_start_counts=_counts(endpoint.board),
        symmetry="identity",
        cost_weights=COST,
    )
    assert comparison.d_occ == 2
    assert comparison.d_placement == 2


def test_same_occupancy_but_swapped_piece_identities() -> None:
    # Same occupied squares, but a white knight and a white bishop trade places.
    endpoint = _pos("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    start = _pos("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    endpoint.board.set_piece_at(chess.A1, chess.Piece(chess.KNIGHT, chess.WHITE))
    endpoint.board.set_piece_at(chess.B1, chess.Piece(chess.BISHOP, chess.WHITE))
    start.board.set_piece_at(chess.A1, chess.Piece(chess.BISHOP, chess.WHITE))
    start.board.set_piece_at(chess.B1, chess.Piece(chess.KNIGHT, chess.WHITE))
    endpoint = _pos(endpoint.board.fen())
    start = _pos(start.board.fen())
    comparison = compare_endpoint(
        endpoint,
        start,
        endpoint_index=0,
        source_start_counts=_counts(endpoint.board),
        symmetry="identity",
        cost_weights=COST,
    )
    assert comparison.d_occ == 0
    # One swapped pair contributes two bits to each of the two affected
    # bitboards (knight and bishop), so the placement distance is 4.
    assert comparison.d_placement == 4
    assert comparison.occupancy_journey is True
    assert comparison.exact_journey is False
    assert comparison.exact_position_match is False
    assert comparison.assignment.moves == 4  # both kings + the exchanged pair
    assert comparison.assignment.swaps == 1


def test_one_piece_relative_move_cost() -> None:
    endpoint = _pos("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    start = _pos("4k3/8/8/8/8/8/8/5K2 w - - 0 1")
    comparison = compare_endpoint(
        endpoint,
        start,
        endpoint_index=0,
        source_start_counts=_counts(endpoint.board),
        symmetry="identity",
        cost_weights=COST,
    )
    # Both kings matched; white king e1->f1 costs move(1) + manhattan(1); the
    # black king stays (move(1)); total 3 with 2 moves and no swaps.
    assert comparison.assignment.total_cost == 3.0
    assert comparison.assignment.moves == 2
    assert comparison.assignment.swaps == 0
    assert comparison.assignment.displacement_vectors == ((1, 0), (0, 0))


def test_capture_and_off_board_replacement() -> None:
    # The stage started with two white rooks; one was captured (in the tray).
    # The next start needs both rooks again: one moves on board, one is added
    # from the tray.
    endpoint = _pos("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    start = _pos("4k3/8/8/8/8/8/8/2K2R2 w - - 0 1")
    source_start = _pos("4k3/8/8/8/8/8/8/R3K2R w - - 0 1")
    start_counts = _counts(source_start.board)
    assert tray_counts(start_counts, _counts(endpoint.board))[3] == 1  # wr in tray
    comparison = compare_endpoint(
        endpoint,
        start,
        endpoint_index=0,
        source_start_counts=start_counts,
        symmetry="identity",
        cost_weights=COST,
    )
    assert comparison.assignment.feasible is True
    # Three target squares must be filled: two on-board pieces move, and the
    # third white rook comes from the tray (the exact move/add split is the
    # minimum-cost matching, which may prefer remove+add over a long move).
    assert comparison.assignment.moves + comparison.assignment.additions == 3
    assert comparison.assignment.additions == 1
    assert comparison.inventory_valid is True
    tray_pairs = [pair for pair in comparison.assignment.matched if pair.from_tray]
    assert len(tray_pairs) == 1
    assert tray_pairs[0].piece_type == chess.ROOK


def test_insufficient_physical_inventory_is_rejected() -> None:
    # The source stage started with a single white knight; the next start needs
    # two white knights and no other white knight is physically available.
    endpoint = _pos("4k3/8/8/8/8/8/8/NR2K3 w - - 0 1")
    start = _pos("4k3/8/8/8/8/8/8/NNK4R w - - 0 1")
    source_start = _pos("4k3/8/8/8/8/8/8/NR2K2R w - - 0 1")
    comparison = compare_endpoint(
        endpoint,
        start,
        endpoint_index=0,
        source_start_counts=_counts(source_start.board),
        symmetry="identity",
        cost_weights=COST,
    )
    assert comparison.inventory_valid is False
    assert comparison.assignment.feasible is False
    assert comparison.assignment.note is not None
    assert "knight" in comparison.assignment.note


def test_assignment_matches_brute_force_minimum() -> None:
    # Enumerate every injective assignment of sources to targets (leftover
    # sources are removals) and compare against the Hungarian optimum.
    rng_boards = [
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "4k3/8/8/8/8/8/8/4K3 w - - 0 1"),
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "4k3/8/8/8/8/8/8/5K2 w - - 0 1"),
        ("4k3/8/8/8/8/8/8/1K1N4 w - - 0 1", "4k3/8/8/8/8/8/8/2N1K3 w - - 0 1"),
        ("4k3/8/8/8/8/8/8/NNB1K3 w - - 0 1", "4k3/8/8/8/8/8/8/K1B1NN2 w - - 0 1"),
    ]
    for endpoint_fen, start_fen in rng_boards:
        endpoint = _pos(endpoint_fen)
        start = _pos(start_fen)
        source_start = _pos(endpoint_fen)
        start_counts = _counts(source_start.board)
        endpoint_counts = _counts(endpoint.board)
        result = assign_pieces(endpoint.board, start.board, start_counts, endpoint_counts, COST)
        brute = _brute_force_cost(endpoint.board, start.board, start_counts, endpoint_counts)
        assert result.feasible is True
        assert result.total_cost == pytest.approx(brute)


def _brute_force_cost(
    endpoint: chess.Board,
    start: chess.Board,
    start_counts: tuple[int, ...],
    endpoint_counts: tuple[int, ...],
) -> float:
    from chess_chain.domain.pieces import PIECE_ORDER

    tray = tray_counts(start_counts, endpoint_counts)
    sources: list[tuple[chess.Square | None, chess.PieceType, chess.Color]] = []
    for index, (colour, piece_type) in enumerate(PIECE_ORDER):
        for square in _bit_squares(int(endpoint.pieces(piece_type, colour))):
            sources.append((square, piece_type, colour))
        for _ in range(tray[index]):
            sources.append((None, piece_type, colour))
    targets: list[tuple[chess.Square, chess.PieceType, chess.Color]] = []
    for colour, piece_type in PIECE_ORDER:
        for square in _bit_squares(int(start.pieces(piece_type, colour))):
            targets.append((square, piece_type, colour))
    if len(targets) > len(sources):
        return float("inf")

    best = float("inf")
    for source_order in itertools.permutations(range(len(sources)), len(targets)):
        total = 0.0
        for source_index, target in zip(source_order, targets, strict=True):
            source_square, piece_type, colour = sources[source_index]
            if (piece_type, colour) != (target[1], target[2]):
                total = float("inf")
                break
            if source_square is None:
                total += COST.add
            else:
                total += COST.move + _manhattan(source_square, target[0])
        if total == float("inf"):
            continue
        total += (len(sources) - len(targets)) * COST.remove
        best = min(best, total)
    return best


def _bit_squares(mask: int) -> list[int]:
    result: list[int] = []
    while mask:
        lsb = mask & -mask
        result.append(lsb.bit_length() - 1)
        mask ^= lsb
    return result


def _manhattan(a: int, b: int) -> int:
    return abs(chess.square_file(a) - chess.square_file(b)) + abs(
        chess.square_rank(a) - chess.square_rank(b)
    )


def test_available_counts_is_elementwise_maximum() -> None:
    start = (8, 2, 2, 2, 1, 1, 8, 2, 2, 2, 1, 1)
    endpoint = (7, 2, 2, 2, 1, 1, 8, 2, 1, 2, 1, 1)
    assert available_counts(start, endpoint) == (8, 2, 2, 2, 1, 1, 8, 2, 2, 2, 1, 1)


def test_exact_endpoint_start_match() -> None:
    board = _pos("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    comparison = compare_endpoint(
        board,
        board,
        endpoint_index=0,
        source_start_counts=_counts(board.board),
        symmetry="identity",
        cost_weights=COST,
    )
    assert comparison.exact_position_match is True
    assert comparison.exact_journey is True
    assert comparison.occupancy_journey is True
    assert comparison.inventory_valid is True
    assert comparison.castling_annotation_required is False


def test_visually_equal_boards_with_different_castling_rights_not_exact() -> None:
    endpoint = _pos("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
    start = _pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    comparison = compare_endpoint(
        endpoint,
        start,
        endpoint_index=0,
        source_start_counts=_counts(endpoint.board),
        symmetry="identity",
        cost_weights=COST,
    )
    assert comparison.d_occ == 0
    assert comparison.d_placement == 0
    assert comparison.exact_position_match is False
    # The endpoint lost its rights during the stage, so the transition must
    # re-establish the non-visible rights of the next start.
    assert comparison.castling_annotation_required is True
    assert "re-established" in (comparison.rights_note or "")
    # Reversed direction: the start needs no rights, so no annotation.
    reversed_comparison = compare_endpoint(
        start,
        endpoint,
        endpoint_index=0,
        source_start_counts=_counts(start.board),
        symmetry="identity",
        cost_weights=COST,
    )
    assert reversed_comparison.castling_annotation_required is False


def test_visually_equal_boards_with_different_ep_square_not_exact() -> None:
    endpoint = _pos("4k3/8/8/3p4/8/8/8/4K3 w - - 0 1")
    start = _pos("4k3/8/8/3p4/8/8/8/4K3 w - d6 0 1")
    comparison = compare_endpoint(
        endpoint,
        start,
        endpoint_index=0,
        source_start_counts=_counts(endpoint.board),
        symmetry="identity",
        cost_weights=COST,
    )
    assert comparison.d_occ == 0
    assert comparison.d_placement == 0
    assert comparison.exact_position_match is False
    assert comparison.ep_annotation_required is True


def test_multiple_endpoints_with_different_occupancy_scored_separately() -> None:
    board = _pos("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    other = _pos("4k3/8/8/8/8/8/8/5K2 w - - 0 1")
    start_counts = _counts(board.board)
    candidate = build_candidate(
        source_puzzle_id="S1",
        next_puzzle_id="N1",
        source_stage_end_policy="transition_ready@v1",
        endpoint_source="final_position:transition_ready@v1",
        endpoints=(board, other),
        source_start_counts=start_counts,
        next_start=board,
        symmetries=("identity",),
        cost_weights=COST,
    )
    assert len(candidate.per_endpoint) == 2
    assert {comparison.d_occ for comparison in candidate.per_endpoint} == {0, 2}
    assert candidate.d_occ_best == 0.0
    assert candidate.d_occ_worst == 2.0
    assert candidate.d_occ_mean == 1.0
    # Different-occupancy endpoints are retained and scored, never rejected.
    assert not any("occupancy" in reason for reason in candidate.reasons_for_rejection)


def test_castling_annotation_preserved_when_endpoint_keeps_rights() -> None:
    endpoint = _pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    start = _pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    comparison = compare_endpoint(
        endpoint,
        start,
        endpoint_index=0,
        source_start_counts=_counts(endpoint.board),
        symmetry="identity",
        cost_weights=COST,
    )
    assert comparison.castling_annotation_required is False
    assert comparison.exact_position_match is True
