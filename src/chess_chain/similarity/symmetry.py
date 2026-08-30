"""Whole-board geometric transformations for similarity comparison.

A conservative set of board symmetries is supported:

- ``identity`` - no transformation;
- ``file_reflection`` - horizontal mirror (a-file <-> h-file); colours and
  side to move are preserved;
- ``rank_reflection`` - vertical mirror (rank 1 <-> rank 8); colours and side
  to move are exchanged because the white and black home ranks swap;
- ``rot180`` - 180-degree rotation; colours and side to move are exchanged.

Every transformed candidate is *reconstructed* as a new python-chess board and
validated with ``Board.is_valid()``. Symmetries that do not preserve a legal
standard-chess position (``is_valid()`` is false) return ``None`` and are
skipped for that candidate: geometric symmetry does not imply chess legality.
Castling rights (python-chess 1.11 stores the rook-home-square bitmask) and the
en-passant square are carried through the same square transformation.
"""

from __future__ import annotations

from collections.abc import Callable

import chess

from chess_chain.domain.pieces import PIECE_ORDER
from chess_chain.domain.position import CanonicalPosition, PositionFingerprint

SYMMETRY_IDENTITY = "identity"
SYMMETRY_FILE_REFLECTION = "file_reflection"
SYMMETRY_RANK_REFLECTION = "rank_reflection"
SYMMETRY_ROT180 = "rot180"

SYMMETRIES: tuple[str, ...] = (
    SYMMETRY_IDENTITY,
    SYMMETRY_FILE_REFLECTION,
    SYMMETRY_RANK_REFLECTION,
    SYMMETRY_ROT180,
)

__all__ = [
    "SYMMETRIES",
    "SYMMETRY_FILE_REFLECTION",
    "SYMMETRY_IDENTITY",
    "SYMMETRY_RANK_REFLECTION",
    "SYMMETRY_ROT180",
    "apply_occupancy_symmetry",
    "apply_symmetry",
    "swaps_colour_and_turn",
]


def _mask_mapper(symmetry: str) -> Callable[[int], int]:
    """Return a bitboard transform for one symmetry."""
    if symmetry == SYMMETRY_IDENTITY:
        return lambda mask: mask
    if symmetry == SYMMETRY_FILE_REFLECTION:
        return chess.flip_horizontal
    if symmetry == SYMMETRY_RANK_REFLECTION:
        return chess.flip_vertical
    if symmetry == SYMMETRY_ROT180:
        return lambda mask: chess.flip_horizontal(chess.flip_vertical(mask))
    raise ValueError(f"unknown symmetry {symmetry!r}; expected one of {SYMMETRIES}")


def _square_mapper(symmetry: str) -> Callable[[chess.Square], chess.Square]:
    transform_mask = _mask_mapper(symmetry)

    def _map_square(square: chess.Square) -> chess.Square:
        bit = transform_mask(1 << square)
        return chess.Square((bit & -bit).bit_length() - 1)

    return _map_square


def apply_occupancy_symmetry(mask: int, symmetry: str) -> int:
    """Transform a 64-bit occupancy mask under ``symmetry``."""
    return int(_mask_mapper(symmetry)(int(mask)))


def swaps_colour_and_turn(symmetry: str) -> bool:
    """Return True when the symmetry exchanges white and black (and thus the
    side to move)."""
    return symmetry in (SYMMETRY_RANK_REFLECTION, SYMMETRY_ROT180)


def apply_symmetry(position: CanonicalPosition, symmetry: str) -> CanonicalPosition | None:
    """Transform ``position`` under ``symmetry`` and return a new validated
    position, or ``None`` when the transformed position is not a legal
    standard-chess position (``Board.is_valid()`` is false) or does not
    faithfully round-trip (castling / en-passant squares mismatch).
    """
    if symmetry not in SYMMETRIES:
        raise ValueError(f"unknown symmetry {symmetry!r}; expected one of {SYMMETRIES}")
    if symmetry == SYMMETRY_IDENTITY:
        return position

    transform_mask = _mask_mapper(symmetry)
    map_square = _square_mapper(symmetry)
    swap_colour = swaps_colour_and_turn(symmetry)

    source = position.fingerprint
    bitboards: list[int] = []
    for colour, piece_type in PIECE_ORDER:
        source_colour = chess.Color(1 - int(colour)) if swap_colour else colour
        source_index = PIECE_ORDER.index((source_colour, piece_type))
        bitboards.append(int(transform_mask(source.bitboards[source_index])))
    turn = (not source.turn) if swap_colour else source.turn
    castling = int(transform_mask(source.castling))
    ep_square: chess.Square | None = (
        map_square(source.ep_square) if source.ep_square is not None else None
    )

    reconstructed = _board_from_bitboards(
        bitboards=tuple(bitboards),
        turn=turn,
        castling=castling,
        ep_square=ep_square,
    )
    if reconstructed is None:
        return None
    if not reconstructed.is_valid():
        return None
    fingerprint = PositionFingerprint.from_board(reconstructed)
    if fingerprint.occupied != int(transform_mask(source.occupied)):
        return None
    # Castling rights are carried through the transformation, but standard chess
    # only admits a right when the king and rook sit on their home squares: a
    # mirrored king (e.g. file reflection maps e1 -> d1) legitimately loses the
    # rights python-chess cannot represent, so we accept any *subset*.
    if fingerprint.castling & ~castling:
        return None
    if fingerprint.ep_square != ep_square:
        return None
    return CanonicalPosition.from_board(reconstructed)


def _board_from_bitboards(
    *,
    bitboards: tuple[int, ...],
    turn: bool,
    castling: int,
    ep_square: chess.Square | None,
) -> chess.Board | None:
    """Build a FEN from transformed components and parse it with python-chess.

    Building the FEN and parsing it exercises the parser's own validation of
    piece placement, castling rights and the en-passant square. Returns None on
    parse errors (malformed transformed position).
    """
    try:
        rows: list[str] = []
        for rank in range(7, -1, -1):
            row: list[str] = []
            empty = 0
            for file in range(8):
                square = chess.square(file, rank)
                piece = _piece_at(bitboards, square)
                if piece is None:
                    empty += 1
                    continue
                if empty:
                    row.append(str(empty))
                    empty = 0
                row.append(piece.symbol())
            if empty:
                row.append(str(empty))
            rows.append("".join(row))
        castling_field = _castling_letters(castling)
        ep_field = "-" if ep_square is None else chess.square_name(ep_square)
        fen = f"{'/'.join(rows)} {'w' if turn else 'b'} {castling_field} {ep_field} 0 1"
        return chess.Board(fen)
    except ValueError:
        return None


def _piece_at(bitboards: tuple[int, ...], square: chess.Square) -> chess.Piece | None:
    bit = 1 << square
    for index, (colour, piece_type) in enumerate(PIECE_ORDER):
        if bitboards[index] & bit:
            return chess.Piece(piece_type, colour)
    return None


def _castling_letters(mask: int) -> str:
    """Standard-chess castling letters from a python-chess 1.11 rights mask."""
    letters: list[str] = []
    if mask & (1 << chess.H1):
        letters.append("K")
    if mask & (1 << chess.A1):
        letters.append("Q")
    if mask & (1 << chess.H8):
        letters.append("k")
    if mask & (1 << chess.A8):
        letters.append("q")
    return "".join(letters) or "-"
