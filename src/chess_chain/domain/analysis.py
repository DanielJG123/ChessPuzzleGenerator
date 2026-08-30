"""Stage 3 analysis value types (per-ply records, forcedness categories)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import chess

__all__ = ["ForcednessCategory", "PlyAnalysis"]


class ForcednessCategory(StrEnum):
    """Category describing how forced the intended move is at one ply.

    ``literal_forced`` and ``invalid_source_line`` are structural; the rest
    depend on configured engine thresholds.
    """

    LITERAL_FORCED = "literal_forced"
    UNIQUE_BEST = "unique_best"
    PRACTICALLY_FORCED = "practically_forced"
    AMBIGUOUS = "ambiguous"
    INVALID_SOURCE_LINE = "invalid_source_line"


@dataclass(frozen=True)
class PlyAnalysis:
    """All recorded facts for one ply of the intended solution.

    Occupancy deltas are computed exactly (castling touches four squares,
    en passant removes a pawn from a non-destination square, captures remove
    two). Engine-derived fields are ``None`` in the cheap tier.
    """

    ply_index: int
    side: chess.Color
    is_solver_ply: bool
    uci: str
    san: str
    legal_move_count: int
    evasion_count: int | None
    in_check: bool
    only_legal_move: bool
    is_capture: bool
    is_recapture: bool
    is_check: bool
    is_forcing_check: bool
    is_mate: bool
    is_castling: bool
    is_en_passant: bool
    is_promotion: bool
    promotion_piece: chess.PieceType | None
    occupancy_before: int
    occupancy_after: int
    squares_removed: tuple[chess.Square, ...]
    squares_added: tuple[chess.Square, ...]
    transposing_move_count: int
    engine_rank: int | None
    engine_score_cp: int | None
    engine_score_mate: int | None
    second_best_gap_cp: int | None
    plausible_count: int | None
    plausible_moves: tuple[str, ...]
    forcedness: ForcednessCategory | None
