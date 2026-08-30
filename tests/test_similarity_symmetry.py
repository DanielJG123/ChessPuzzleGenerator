"""Tests for whole-board symmetry transformations and validation."""

from __future__ import annotations

import chess

from chess_chain.domain.position import CanonicalPosition
from chess_chain.similarity.symmetry import (
    SYMMETRY_FILE_REFLECTION,
    SYMMETRY_IDENTITY,
    SYMMETRY_RANK_REFLECTION,
    SYMMETRY_ROT180,
    apply_occupancy_symmetry,
    apply_symmetry,
    swaps_colour_and_turn,
)


def _pos(fen: str) -> CanonicalPosition:
    return CanonicalPosition.from_board(chess.Board(fen))


def _square(board: chess.Board, piece_type: int, colour: int) -> chess.Square | None:
    mask = int(board.pieces(piece_type, colour))
    return chess.Square((mask & -mask).bit_length() - 1) if mask else None


def test_identity_is_noop() -> None:
    position = _pos("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert apply_symmetry(position, SYMMETRY_IDENTITY) is position


def test_file_reflection_mirrors_files_and_preserves_colour() -> None:
    position = _pos("7k/8/8/8/8/8/8/R5K1 w - - 0 1")
    reflected = apply_symmetry(position, SYMMETRY_FILE_REFLECTION)
    assert reflected is not None
    # White rook a1 -> h1; white king g1 -> b1; black king h8 -> a8.
    assert _square(reflected.board, chess.ROOK, chess.WHITE) == chess.H1
    assert _square(reflected.board, chess.KING, chess.WHITE) == chess.B1
    assert _square(reflected.board, chess.KING, chess.BLACK) == chess.A8
    assert reflected.fingerprint.turn == position.fingerprint.turn
    assert swaps_colour_and_turn(SYMMETRY_FILE_REFLECTION) is False
    # Occupancy mirrors under the same transform.
    assert (
        apply_occupancy_symmetry(position.fingerprint.occupied, SYMMETRY_FILE_REFLECTION)
        == reflected.fingerprint.occupied
    )


def test_rank_reflection_swaps_colours_and_turn() -> None:
    position = _pos("7k/8/8/8/8/8/8/R5K1 w - - 0 1")
    reflected = apply_symmetry(position, SYMMETRY_RANK_REFLECTION)
    assert reflected is not None
    # White rook a1 -> black rook a8; white king g1 -> black king g8.
    assert _square(reflected.board, chess.ROOK, chess.BLACK) == chess.A8
    assert _square(reflected.board, chess.KING, chess.BLACK) == chess.G8
    assert _square(reflected.board, chess.KING, chess.WHITE) == chess.H1
    assert reflected.fingerprint.turn is False  # side to move exchanges
    assert swaps_colour_and_turn(SYMMETRY_RANK_REFLECTION) is True


def test_rot180_swaps_colours_and_turn() -> None:
    position = _pos("7k/8/8/8/8/8/8/R5K1 w - - 0 1")
    rotated = apply_symmetry(position, SYMMETRY_ROT180)
    assert rotated is not None
    # White rook a1 -> black rook h8; white king g1 -> black king b8.
    assert _square(rotated.board, chess.ROOK, chess.BLACK) == chess.H8
    assert _square(rotated.board, chess.KING, chess.BLACK) == chess.B8
    assert rotated.fingerprint.turn is False
    assert swaps_colour_and_turn(SYMMETRY_ROT180) is True


def test_rank_reflection_preserves_castling_with_colour_exchange() -> None:
    position = _pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    rank = apply_symmetry(position, SYMMETRY_RANK_REFLECTION)
    assert rank is not None
    # White rights become black rights; the mask is mirrored vertically.
    assert rank.fingerprint.castling == int(chess.flip_vertical(position.fingerprint.castling))


def test_file_and_rot180_reject_castling_positions() -> None:
    # File reflection maps the king e1 -> d1 and rot180 maps e1 -> d8: standard
    # chess cannot keep castling rights with the king off e1/e8, so python-chess
    # rejects the reconstructed positions and the transformations are skipped.
    position = _pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert apply_symmetry(position, SYMMETRY_FILE_REFLECTION) is None
    assert apply_symmetry(position, SYMMETRY_ROT180) is None
    # Without rights the same board transforms fine.
    rights_free = _pos("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
    assert apply_symmetry(rights_free, SYMMETRY_FILE_REFLECTION) is not None
    assert apply_symmetry(rights_free, SYMMETRY_ROT180) is not None


def test_invalid_transformed_position_is_rejected() -> None:
    # A position with far too many white queens is structurally invalid under
    # every non-identity symmetry and must be rejected by the python-chess
    # validation.
    position = _pos("QQQQQQQQ/QQQQQQQQ/8/8/8/8/8/k6K w - - 0 1")
    for symmetry in (SYMMETRY_FILE_REFLECTION, SYMMETRY_RANK_REFLECTION, SYMMETRY_ROT180):
        assert apply_symmetry(position, symmetry) is None


def test_all_symmetries_round_trip_via_fingerprint() -> None:
    # Rank reflection and rot180 are involutions on a rights-free position:
    # applying the symmetry twice returns an identical fingerprint.
    position = _pos("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w - - 0 1")
    for symmetry in (SYMMETRY_RANK_REFLECTION, SYMMETRY_ROT180):
        once = apply_symmetry(position, symmetry)
        assert once is not None
        twice = apply_symmetry(once, symmetry)
        assert twice is not None
        assert twice.fingerprint == position.fingerprint


def test_ep_square_follows_the_transformation() -> None:
    # Black pawn just doubled from d7: white to move, ep target d6.
    position = _pos("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    file_reflected = apply_symmetry(position, SYMMETRY_FILE_REFLECTION)
    assert file_reflected is not None
    assert file_reflected.fingerprint.ep_square == chess.D6 ^ 7  # d-file <-> e-file
    rank_reflected = apply_symmetry(position, SYMMETRY_RANK_REFLECTION)
    assert rank_reflected is not None
    assert rank_reflected.fingerprint.ep_square == chess.D6 ^ 56
    rotated = apply_symmetry(position, SYMMETRY_ROT180)
    assert rotated is not None
    assert rotated.fingerprint.ep_square == chess.D6 ^ 63


def test_unknown_symmetry_raises() -> None:
    position = _pos("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    try:
        apply_symmetry(position, "diagonal_flip")  # type: ignore[arg-type]
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown symmetry")


def test_fingerprint_of_reflection_is_consistent() -> None:
    position = _pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1")
    reflected = apply_symmetry(position, SYMMETRY_FILE_REFLECTION)
    assert reflected is not None
    # Reconstructing directly from python-chess's own transform must agree.
    expected_bb = tuple(
        int(chess.flip_horizontal(bitboard)) for bitboard in position.fingerprint.bitboards
    )
    assert reflected.fingerprint.bitboards == expected_bb
