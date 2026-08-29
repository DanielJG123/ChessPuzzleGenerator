"""Move-level events derived during puzzle replay."""

from __future__ import annotations

from dataclasses import dataclass

import chess

from chess_chain.domain.pieces import StagePieceId


@dataclass(frozen=True)
class MoveEvent:
    """One ply of the intended solution, with stage-local piece identities."""

    ply_index: int
    side: chess.Color
    uci: str
    san: str
    moving_piece_id: StagePieceId
    moving_piece_type: chess.PieceType
    origin: chess.Square
    destination: chess.Square
    captured_piece_id: StagePieceId | None
    captured_piece_type: chess.PieceType | None
    capture_square: chess.Square | None
    promotion: chess.PieceType | None
    is_check: bool
    is_mate: bool
