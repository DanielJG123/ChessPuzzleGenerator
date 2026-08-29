"""Puzzle core: the replayed, derived form of a source puzzle."""

from __future__ import annotations

from dataclasses import dataclass

import chess

from chess_chain.domain.moves import MoveEvent
from chess_chain.domain.pieces import StagePieceId
from chess_chain.domain.position import CanonicalPosition
from chess_chain.domain.source import SourcePuzzle


@dataclass(frozen=True)
class PuzzleCore:
    """The derived, replay-verified form of one source puzzle.

    ``presented_start`` is the position the player actually sees: the raw FEN
    after the preceding move (``Moves[0]``). ``solution_uci`` / ``solution_san``
    are the intended solution ``Moves[1:]``. ``final_position`` is the board
    after all solution moves.
    """

    source: SourcePuzzle
    presented_start: CanonicalPosition
    solution_uci: tuple[str, ...]
    solution_san: tuple[str, ...]
    final_position: CanonicalPosition
    solver_colour: chess.Color
    outcome: str
    is_check: bool
    is_checkmate: bool
    is_stalemate: bool
    events: tuple[MoveEvent, ...]
    moved_ids: frozenset[StagePieceId]
    unmoved_ids: frozenset[StagePieceId]
    captured_ids: tuple[StagePieceId, ...]
    last_moved_id: StagePieceId
    final_checking_ids: frozenset[StagePieceId]
    promotion_events: tuple[str, ...]
    has_promotion: bool
    duplicate_queens: bool
    unusual_material: bool
