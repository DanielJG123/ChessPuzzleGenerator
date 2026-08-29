"""Tests for domain structures (identities, fingerprints, material)."""

from __future__ import annotations

import chess
import pytest

from chess_chain.domain.pieces import (
    COUNT_NAMES,
    MaterialSignature,
    StagePieceId,
)
from chess_chain.domain.position import PositionFingerprint


def test_stage_piece_id_key_round_trip() -> None:
    identity = StagePieceId(chess.WHITE, chess.ROOK, chess.A8)
    assert identity.key() == "wra8"
    assert StagePieceId.from_key("wra8") == identity
    black_knight = StagePieceId(chess.BLACK, chess.KNIGHT, chess.B6)
    assert black_knight.key() == "bnb6"
    assert StagePieceId.from_key("bnb6") == black_knight


def test_stage_piece_id_rejects_invalid_keys() -> None:
    with pytest.raises(ValueError):
        StagePieceId.from_key("bogus")
    with pytest.raises(ValueError):
        StagePieceId.from_key("wpa9")


def test_material_signature_counts() -> None:
    board = chess.Board()
    material = MaterialSignature.from_board(board)
    assert material.piece_count() == 32
    assert material.counts_by_colour(chess.WHITE)[chess.PAWN] == 8
    assert material.material_score(chess.WHITE) == 39
    assert material.material_score(chess.BLACK) == 39
    assert not material.duplicate_queens()
    assert not material.exceeds_standard_set()


def test_material_signature_flags_duplicate_queens() -> None:
    # A position with two white queens.
    board = chess.Board("8/8/8/8/8/8/8/1QK4Q w - - 0 1")
    material = MaterialSignature.from_board(board)
    assert material.duplicate_queens()
    assert material.exceeds_standard_set()


def test_material_signature_from_counts_kwargs() -> None:
    signature = MaterialSignature.from_counts(wp=8, bp=8, wn=2, bn=2, wb=2, bb=2, wr=2, br=2)
    assert signature.piece_count() == 28
    assert len(COUNT_NAMES) == 12


def test_start_position_fingerprint_bitboards() -> None:
    board = chess.Board()
    fp = PositionFingerprint.from_board(board)
    # Standard start: white pawns on rank 2, white king e1, etc.
    assert fp.bitboards[0] == 0xFF00  # white pawns, rank 2
    assert fp.bitboards[1] == 0x42  # white knights b1/g1
    assert fp.bitboards[2] == 0x24  # white bishops c1/f1
    assert fp.bitboards[3] == 0x81  # white rooks a1/h1
    assert fp.bitboards[4] == 0x8  # white queen d1
    assert fp.bitboards[5] == 0x10  # white king e1
    assert fp.occupied == 0xFFFF00000000FFFF  # ranks 1-2 (white) and 7-8 (black)
    assert fp.turn is True
    # python-chess represents castling rights as the rook-square bitboard.
    assert fp.castling == 0x8100000000000081
    assert fp.ep_square is None


def test_fingerprint_key_is_stable_and_deterministic() -> None:
    board = chess.Board()
    fp = PositionFingerprint.from_board(board)
    assert fp.to_key() == PositionFingerprint.from_board(board.copy()).to_key()


def test_fingerprint_distinguishes_castling_rights() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    other = chess.Board("4k3/8/8/8/8/8/8/R3K2R w - - 0 1")
    assert PositionFingerprint.from_board(board).castling == 0x81  # rook squares a1/h1
    assert PositionFingerprint.from_board(other).castling == 0
    assert PositionFingerprint.from_board(board).to_key() != (
        PositionFingerprint.from_board(other).to_key()
    )
