"""Canonical position representation and fingerprints."""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

from chess_chain.domain.pieces import PIECE_ORDER


@dataclass(frozen=True)
class PositionFingerprint:
    """Chess-position identity ignoring fullmove/halfmove counters.

    Holds the 12 colour/piece bitboards, the 64-bit occupancy mask, side to
    move, castling rights and the en-passant square. Two positions with equal
    fingerprints are the same chess position for equality and replay purposes.
    Move counters are ignored, as in standard chess position identity; any
    additional relaxation (for example ignoring castling or en-passant state)
    must be an explicitly named comparison fingerprint, not this type.
    """

    bitboards: tuple[int, ...]  # 12 values in PIECE_ORDER
    occupied: int
    turn: bool
    castling: int
    ep_square: chess.Square | None

    def __post_init__(self) -> None:
        if len(self.bitboards) != len(PIECE_ORDER):
            raise ValueError(f"expected {len(PIECE_ORDER)} bitboards, got {len(self.bitboards)}")
        if not (0 <= self.occupied < 1 << 64):
            raise ValueError("occupancy mask out of 64-bit range")

    @classmethod
    def from_board(cls, board: chess.Board) -> PositionFingerprint:
        bitboards = tuple(
            int(board.pieces(piece_type, colour)) for colour, piece_type in PIECE_ORDER
        )
        return cls(
            bitboards=bitboards,
            occupied=int(board.occupied),
            turn=bool(board.turn),
            castling=int(board.castling_rights),
            ep_square=board.ep_square,
        )

    def to_key(self) -> str:
        """Return a stable serialised key for storage and equality joins."""
        parts = [f"{value:x}" for value in (*self.bitboards, self.occupied)]
        parts.append("1" if self.turn else "0")
        parts.append(str(self.castling))
        parts.append("-" if self.ep_square is None else str(self.ep_square))
        return ":".join(parts)


@dataclass(frozen=True, eq=False)
class CanonicalPosition:
    """A board snapshot with a counter-insensitive fingerprint for equality.

    The embedded ``board`` is used for replay; equality and hashing are defined
    by the fingerprint only.
    """

    fen: str
    fingerprint: PositionFingerprint
    board: chess.Board = field(repr=False, compare=False)

    @classmethod
    def from_board(cls, board: chess.Board) -> CanonicalPosition:
        return cls(
            fen=board.fen(),
            fingerprint=PositionFingerprint.from_board(board),
            board=board.copy(),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CanonicalPosition):
            return NotImplemented
        return self.fingerprint == other.fingerprint

    def __hash__(self) -> int:
        return hash(self.fingerprint)
