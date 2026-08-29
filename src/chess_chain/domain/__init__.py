"""Domain types for chess-chain puzzle data.

Product constraints (see ``docs/architecture.md``):

1. The player sees the first chess position and every written transition
   instruction from the start.
2. There are no concealed codes, digital reveals, or staged instructions.
3. A solved endpoint ``F_i`` plus a visible transition ``T_i`` produces the
   next start ``S_(i+1)``: ``T_i(F_i) = S_(i+1)``.
4. The player moves both colours.
5. The mechanical checker recognises occupied squares only.
6. The final puzzle ends in checkmate.
7. The project must eventually generate many unique sequences at target
   difficulty levels.
"""

from __future__ import annotations

from chess_chain.domain.errors import IngestRejectError
from chess_chain.domain.moves import MoveEvent
from chess_chain.domain.pieces import (
    COUNT_NAMES,
    PIECE_ORDER,
    PIECE_VALUES,
    STANDARD_SET_LIMITS,
    STANDARD_SET_PIECE_LIMIT,
    MaterialSignature,
    StagePieceId,
)
from chess_chain.domain.position import CanonicalPosition, PositionFingerprint
from chess_chain.domain.puzzle import PuzzleCore
from chess_chain.domain.source import PUZZLE_HEADER, SourcePuzzle, source_from_fields

__all__ = [
    "COUNT_NAMES",
    "CanonicalPosition",
    "IngestRejectError",
    "MaterialSignature",
    "MoveEvent",
    "PIECE_ORDER",
    "PIECE_VALUES",
    "PUZZLE_HEADER",
    "PositionFingerprint",
    "PuzzleCore",
    "SourcePuzzle",
    "StagePieceId",
    "STANDARD_SET_LIMITS",
    "STANDARD_SET_PIECE_LIMIT",
    "source_from_fields",
]
