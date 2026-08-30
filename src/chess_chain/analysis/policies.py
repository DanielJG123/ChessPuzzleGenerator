"""Versioned stage-end policies and explicit predicates.

A stage ends when the player may open the box / move on. The hardware model
only mechanically validates the *final* puzzle of a complete sequence, so
intermediate stages need a reliable visible stopping rule; these policies
describe what that rule is. No single policy is chosen globally in code.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

__all__ = [
    "CHECKMATE_POLICY",
    "EXPLICIT_PREDICATE_QUEEN_CAPTURED",
    "FIXED_PLIES_POLICY_4",
    "STAGE_END_POLICIES",
    "StageEndPolicy",
    "TRANSITION_READY_POLICY",
    "evaluate_predicate",
    "get_policy",
]

CHECKMATE = "CHECKMATE"
TRANSITION_READY = "TRANSITION_READY"
EXPLICIT_PREDICATE = "EXPLICIT_PREDICATE"
FIXED_PLIES = "FIXED_PLIES"


@dataclass(frozen=True)
class StageEndPolicy:
    """A named, versioned policy describing when a stage ends."""

    name: str
    version: int
    kind: str
    # Formal predicate key (EXPLICIT_PREDICATE only), e.g. "absent:black_queen".
    predicate: str | None = None
    # Rendered wording printed to the player (may leak tactical information).
    wording: str | None = None
    # Half-move count for FIXED_PLIES.
    fixed_plies: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in (CHECKMATE, TRANSITION_READY, EXPLICIT_PREDICATE, FIXED_PLIES):
            raise ValueError(f"unknown stage-end kind {self.kind!r}")
        if self.kind == FIXED_PLIES and (self.fixed_plies is None or self.fixed_plies < 1):
            raise ValueError("FIXED_PLIES requires fixed_plies >= 1")
        if self.kind == EXPLICIT_PREDICATE and not self.predicate:
            raise ValueError("EXPLICIT_PREDICATE requires a formal predicate")

    def key(self) -> str:
        return f"{self.name}@v{self.version}"


CHECKMATE_POLICY = StageEndPolicy(
    name="checkmate",
    version=1,
    kind=CHECKMATE,
    wording="when the position is checkmate",
)

TRANSITION_READY_POLICY = StageEndPolicy(
    name="transition_ready",
    version=1,
    kind=TRANSITION_READY,
    wording="when the next transition instruction becomes executable",
)

EXPLICIT_PREDICATE_QUEEN_CAPTURED = StageEndPolicy(
    name="explicit_black_queen_captured",
    version=1,
    kind=EXPLICIT_PREDICATE,
    predicate="absent:black_queen",
    wording="when the black queen has been captured",
)

FIXED_PLIES_POLICY_4 = StageEndPolicy(
    name="fixed_plies_4",
    version=1,
    kind=FIXED_PLIES,
    fixed_plies=4,
    wording="after exactly 4 half-moves",
)

STAGE_END_POLICIES: dict[str, StageEndPolicy] = {
    CHECKMATE_POLICY.name: CHECKMATE_POLICY,
    TRANSITION_READY_POLICY.name: TRANSITION_READY_POLICY,
    EXPLICIT_PREDICATE_QUEEN_CAPTURED.name: EXPLICIT_PREDICATE_QUEEN_CAPTURED,
    FIXED_PLIES_POLICY_4.name: FIXED_PLIES_POLICY_4,
}


def get_policy(name: str, version: int = 1) -> StageEndPolicy:
    policy = STAGE_END_POLICIES.get(name)
    if policy is None:
        raise KeyError(f"unknown stage-end policy {name!r}")
    if policy.version != version:
        raise KeyError(
            f"stage-end policy {name!r} is at version {policy.version}, requested {version}"
        )
    return policy


def evaluate_predicate(board: chess.Board, predicate: str) -> bool:
    """Evaluate a formal predicate against a board.

    Supported forms (extend explicitly; never rely on eval()):
      - ``checkmate``
      - ``absent:black_queen`` / ``absent:white_queen``
      - ``absent:<fen_symbol>`` (e.g. ``absent:q``)
      - ``present:<fen_symbol>:<square>`` (e.g. ``present:q:a4``)
    """
    kind, _, arg = predicate.partition(":")
    if kind == "checkmate":
        return board.is_checkmate()
    if kind == "absent":
        if arg == "black_queen":
            return not board.pieces(chess.QUEEN, chess.BLACK)
        if arg == "white_queen":
            return not board.pieces(chess.QUEEN, chess.WHITE)
        piece = chess.Piece.from_symbol(arg)
        return not board.pieces(piece.piece_type, piece.color)
    if kind == "present":
        symbol, _, square = arg.partition(":")
        piece = chess.Piece.from_symbol(symbol)
        return board.piece_at(chess.parse_square(square)) == piece
    raise ValueError(f"unsupported predicate {predicate!r}")
