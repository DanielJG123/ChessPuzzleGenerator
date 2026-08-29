"""Property tests for bitboard/FEN round trips."""

from __future__ import annotations

import chess
from hypothesis import given
from hypothesis import strategies as st

from chess_chain.domain.pieces import PIECE_ORDER
from chess_chain.domain.position import PositionFingerprint


@given(board=st.builds(chess.Board))
def test_fingerprint_round_trip_equals_board_bits(board: chess.Board) -> None:
    fingerprint = PositionFingerprint.from_board(board)
    for index, (colour, piece_type) in enumerate(PIECE_ORDER):
        assert fingerprint.bitboards[index] == int(board.pieces(piece_type, colour))
    assert fingerprint.occupied == int(board.occupied)


@given(board=st.builds(chess.Board))
def test_fingerprint_reconstructs_board(board: chess.Board) -> None:
    """The fingerprint bitboards reconstruct an equivalent chess position."""
    fingerprint = PositionFingerprint.from_board(board)
    reconstructed = chess.Board()
    reconstructed.clear()
    reconstructed.turn = fingerprint.turn
    reconstructed.castling_rights = fingerprint.castling
    if fingerprint.ep_square is not None:
        reconstructed.ep_square = fingerprint.ep_square
    for index, (colour, piece_type) in enumerate(PIECE_ORDER):
        for square in chess.scan_reversed(fingerprint.bitboards[index]):
            reconstructed.set_piece_at(square, chess.Piece(piece_type, colour))
    assert PositionFingerprint.from_board(reconstructed) == fingerprint


@given(board=st.builds(chess.Board))
def test_fingerprint_ignores_move_counters(board: chess.Board) -> None:
    moved = board.copy()
    moved.fullmove_number = board.fullmove_number + 1
    moved.halfmove_clock = board.halfmove_clock + 2
    assert PositionFingerprint.from_board(board) == PositionFingerprint.from_board(moved)


@given(board=st.builds(chess.Board))
def test_scan_reversed_occupancy_consistency(board: chess.Board) -> None:
    fp = PositionFingerprint.from_board(board)
    squares = set(chess.scan_reversed(fp.occupied))
    assert squares == set(board.piece_map())
