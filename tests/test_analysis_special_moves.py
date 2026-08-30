"""Tests for special-move metadata in per-ply analysis (castling, ep, promotion)."""

from __future__ import annotations

import chess

from chess_chain.analysis.cheap import analyze_cheap_puzzle
from chess_chain.domain.position import PositionFingerprint


def _analyze(fen: str, moves: tuple[str, ...], solver: str):
    plies = analyze_cheap_puzzle(
        puzzle_id="SPECIAL", start_fen=fen, solution_uci=moves, solver_colour=solver
    )
    return plies


def test_castling_replay_four_square_occupancy_delta() -> None:
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    plies = _analyze(fen, ("e1g1",), "white")
    ply = plies[0]
    assert ply.is_castling is True
    # King e1->g1 and rook h1->f1: four squares change.
    assert sorted(ply.squares_removed) == [chess.E1, chess.H1]
    assert sorted(ply.squares_added) == [chess.F1, chess.G1]
    assert ply.occupancy_after != ply.occupancy_before
    before = set(chess.scan_reversed(ply.occupancy_before))
    after = set(chess.scan_reversed(ply.occupancy_after))
    assert before - after == {chess.E1, chess.H1}
    assert after - before == {chess.F1, chess.G1}


def test_castling_rights_distinguish_identical_arrangements() -> None:
    with_castle = chess.Board("7k/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    without_castle = chess.Board("7k/8/8/8/8/8/8/R3K2R w - - 0 1")
    assert PositionFingerprint.from_board(with_castle) != PositionFingerprint.from_board(
        without_castle
    )
    # The piece arrangement is identical; only the castling rights differ.
    assert with_castle.fen().split()[0] == without_castle.fen().split()[0]
    # Castling is only legal when the rights exist.
    assert any(move.uci() == "e1g1" for move in with_castle.legal_moves)
    assert not any(move.uci() == "e1g1" for move in without_castle.legal_moves)


def test_en_passant_removes_pawn_from_non_destination_square() -> None:
    fen = "8/8/8/3pP3/8/8/8/4K2k w - d6 0 1"
    plies = _analyze(fen, ("e5d6",), "white")
    ply = plies[0]
    assert ply.is_en_passant is True
    # The captured pawn is removed from d5, not from the destination d6.
    assert sorted(ply.squares_removed) == [chess.D5, chess.E5]
    assert sorted(ply.squares_added) == [chess.D6]
    after = set(chess.scan_reversed(ply.occupancy_after))
    assert chess.D5 not in after
    assert chess.D6 in after


def test_promotion_inventory_metadata() -> None:
    fen = "6k1/4P3/8/8/8/8/8/4K3 w - - 0 1"
    plies = _analyze(fen, ("e7e8q",), "white")
    ply = plies[0]
    assert ply.is_promotion is True
    assert ply.promotion_piece == chess.QUEEN
    assert sorted(ply.squares_removed) == [chess.E7]
    assert sorted(ply.squares_added) == [chess.E8]


def test_ordinary_move_toggles_two_squares() -> None:
    fen = "6k1/8/8/8/8/8/8/6RK w - - 0 1"
    plies = _analyze(fen, ("g1g7",), "white")
    ply = plies[0]
    assert sorted(ply.squares_removed) == [chess.G1]
    assert sorted(ply.squares_added) == [chess.G7]


def test_capture_removes_two_squares() -> None:
    # White rook captures the black pawn on h7: the occupancy delta is exactly
    # removed={h1} (the destination stays occupied by the capturing rook), not
    # "two squares toggled".
    fen = "6kr/7p/8/8/8/8/8/6KR w - - 0 1"
    plies = _analyze(fen, ("h1h7",), "white")
    ply = plies[0]
    assert ply.is_capture is True
    assert sorted(ply.squares_removed) == [chess.H1]
    assert sorted(ply.squares_added) == []


def test_recapture_flag() -> None:
    # White rook captures the h7 pawn; the black rook then recaptures on h7.
    fen = "6kr/7p/8/8/8/8/8/6KR w - - 0 1"
    plies = _analyze(fen, ("h1h7", "h8h7"), "white")
    assert plies[0].is_capture is True
    assert plies[0].is_recapture is False
    assert plies[1].is_capture is True
    assert plies[1].is_recapture is True
