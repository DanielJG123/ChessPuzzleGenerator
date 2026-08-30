"""Minimum-cost physical piece assignment between a stage endpoint and the next
start position.

The physical model treats pieces of the same colour and piece type as
interchangeable objects (stage-local identities from stage 2 end at the stage
boundary). The pieces available for a transition are the endpoint's on-board
pieces plus the tray: everything that was on the presented start board but is
not on the endpoint board (captured pieces, set-aside promoted pawns). This is
exactly ``max(0, start_counts - endpoint_counts)`` per colour/type.

Assignment decomposes per (colour, type): a physical white knight can only ever
become a white knight on the next board. Each sub-problem is solved exactly with
``scipy.optimize.linear_sum_assignment``; unmatched on-board pieces become
removals. Additions, replacements (a tray piece placed on a square vacated by a
removed piece) and swaps (two matched pieces exchanging squares) are derived
from the matching and reported as explainable counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np
from scipy.optimize import linear_sum_assignment

from chess_chain.domain.pieces import PIECE_ORDER

__all__ = [
    "AssignedPair",
    "AssignmentResult",
    "CostWeights",
    "PieceSource",
    "assign_pieces",
    "default_cost_weights",
    "tray_counts",
]


@dataclass(frozen=True)
class CostWeights:
    """Per-instruction cost weights used inside the assignment (decomposable).

    - ``move``: base cost for moving an on-board piece to a target square;
      the Manhattan displacement (file + rank delta) is added on top.
    - ``add``: cost for placing a tray piece onto the board.
    - ``remove``: cost for taking an on-board piece off the board.
    """

    move: float = 1.0
    add: float = 3.0
    remove: float = 2.0

    def as_dict(self) -> dict[str, float]:
        return {"move": self.move, "add": self.add, "remove": self.remove}


def default_cost_weights() -> CostWeights:
    return CostWeights()


@dataclass(frozen=True)
class PieceSource:
    """One physical piece available for the transition."""

    colour: chess.Color
    piece_type: chess.PieceType
    square: chess.Square | None  # None means the piece starts off board (tray)
    from_tray: bool = False


@dataclass(frozen=True)
class AssignedPair:
    """One matched source piece -> target square."""

    colour: chess.Color
    piece_type: chess.PieceType
    source_square: chess.Square | None
    target_square: chess.Square
    cost: float
    from_tray: bool = False

    def source_key(self) -> str:
        symbol = chess.piece_symbol(self.piece_type)
        if self.source_square is None:
            return f"tray:{'w' if self.colour == chess.WHITE else 'b'}{symbol}"
        return (
            f"{'w' if self.colour == chess.WHITE else 'b'}"
            f"{symbol}{chess.square_name(self.source_square)}"
        )

    def displacement(self) -> tuple[int, int]:
        if self.source_square is None:
            return (0, 0)
        return (
            chess.square_file(self.target_square) - chess.square_file(self.source_square),
            chess.square_rank(self.target_square) - chess.square_rank(self.source_square),
        )


@dataclass(frozen=True)
class AssignmentResult:
    """The outcome of assigning physical pieces to the next start's squares."""

    feasible: bool
    total_cost: float
    matched: tuple[AssignedPair, ...]
    moves: int
    additions: int
    removals: int
    replacements: int
    swaps: int
    displacement_vectors: tuple[tuple[int, int], ...]
    note: str | None = None

    def moves_only(self) -> bool:
        """True when every source piece stays on board and no tray piece is
        needed (a pure rearrangement, possibly with swaps)."""
        return self.feasible and self.additions == 0 and self.removals == 0


def tray_counts(start_counts: tuple[int, ...], endpoint_counts: tuple[int, ...]) -> tuple[int, ...]:
    """Pieces off the board after the stage: everything the stage started with
    that is no longer on the endpoint board (captures and set-aside promoted
    pawns)."""
    return tuple(max(0, s - e) for s, e in zip(start_counts, endpoint_counts, strict=True))


def assign_pieces(
    endpoint_board: chess.Board,
    start_board: chess.Board,
    start_counts: tuple[int, ...],
    endpoint_counts: tuple[int, ...],
    weights: CostWeights | None = None,
) -> AssignmentResult:
    """Compute the minimum-cost physical assignment endpoint -> start.

    ``start_board`` is the *transformed* next-start position (symmetry already
    applied). ``start_counts`` is the presented-start material of the source
    puzzle; ``endpoint_counts`` is the material of the endpoint board.
    """
    cost_weights = weights or default_cost_weights()
    tray = tray_counts(start_counts, endpoint_counts)
    sources: list[PieceSource] = []
    targets: list[AssignedPair] = []

    for index, (colour, piece_type) in enumerate(PIECE_ORDER):
        for square in _bit_squares(int(endpoint_board.pieces(piece_type, colour))):
            sources.append(PieceSource(colour=colour, piece_type=piece_type, square=square))
        for _ in range(tray[index]):
            sources.append(
                PieceSource(colour=colour, piece_type=piece_type, square=None, from_tray=True)
            )
        for square in _bit_squares(int(start_board.pieces(piece_type, colour))):
            targets.append(
                AssignedPair(
                    colour=colour,
                    piece_type=piece_type,
                    source_square=None,
                    target_square=square,
                    cost=0.0,
                )
            )

    def _infeasible(note: str) -> AssignmentResult:
        return AssignmentResult(
            feasible=False,
            total_cost=float("inf"),
            matched=(),
            moves=0,
            additions=0,
            removals=0,
            replacements=0,
            swaps=0,
            displacement_vectors=(),
            note=note,
        )

    if len(targets) > len(sources):
        return _infeasible(
            f"next start needs {len(targets)} pieces but only {len(sources)} "
            "are physically available"
        )

    matched: list[AssignedPair] = []
    removals = 0
    for colour, piece_type in PIECE_ORDER:
        group_sources = [
            source
            for source in sources
            if source.colour == colour and source.piece_type == piece_type
        ]
        group_targets = [
            target
            for target in targets
            if target.colour == colour and target.piece_type == piece_type
        ]
        if len(group_targets) > len(group_sources):
            return _infeasible(
                f"insufficient physical inventory of "
                f"{'white' if colour == chess.WHITE else 'black'} "
                f"{chess.piece_name(piece_type)} "
                f"(need {len(group_targets)}, have {len(group_sources)})"
            )
        if not group_sources or not group_targets:
            removals += len(group_sources) - len(group_targets)
            continue
        costs = _cost_matrix(group_sources, group_targets, cost_weights)
        row_indices, column_indices = linear_sum_assignment(costs)
        for row, column in zip(row_indices, column_indices, strict=True):
            source = group_sources[int(row)]
            target = group_targets[int(column)]
            matched.append(
                AssignedPair(
                    colour=source.colour,
                    piece_type=source.piece_type,
                    source_square=source.square,
                    target_square=target.target_square,
                    cost=float(costs[int(row), int(column)]),
                    from_tray=source.from_tray,
                )
            )

    on_board_moves = [pair for pair in matched if not pair.from_tray]
    additions = len(matched) - len(on_board_moves)
    swaps = _count_swaps(on_board_moves)
    removed_squares = _removed_squares(endpoint_board, matched)
    replacements = sum(
        1 for pair in matched if pair.from_tray and pair.target_square in removed_squares
    )
    displacement_vectors = tuple(pair.displacement() for pair in on_board_moves)
    total_cost = sum(pair.cost for pair in matched) + removals * cost_weights.remove
    return AssignmentResult(
        feasible=True,
        total_cost=total_cost,
        matched=tuple(matched),
        moves=len(on_board_moves),
        additions=additions,
        removals=removals,
        replacements=replacements,
        swaps=swaps,
        displacement_vectors=displacement_vectors,
    )


def _cost_matrix(
    sources: list[PieceSource],
    targets: list[AssignedPair],
    weights: CostWeights,
) -> np.ndarray:
    matrix = np.zeros((len(sources), len(targets)), dtype=np.float64)
    for i, source in enumerate(sources):
        for j, target in enumerate(targets):
            if source.from_tray:
                matrix[i, j] = weights.add
            else:
                assert source.square is not None
                matrix[i, j] = weights.move + _manhattan(source.square, target.target_square)
    return matrix


def _manhattan(a: chess.Square, b: chess.Square) -> int:
    return abs(chess.square_file(a) - chess.square_file(b)) + abs(
        chess.square_rank(a) - chess.square_rank(b)
    )


def _bit_squares(mask: int) -> list[chess.Square]:
    result: list[chess.Square] = []
    while mask:
        lsb = mask & -mask
        result.append(chess.Square(lsb.bit_length() - 1))
        mask ^= lsb
    return result


def _count_swaps(moves: list[AssignedPair]) -> int:
    """Count 2-cycles among matched on-board moves (two pieces exchange squares)."""
    destination_of: dict[chess.Square, chess.Square] = {}
    for pair in moves:
        if pair.source_square is not None:
            destination_of[pair.source_square] = pair.target_square
    swaps = 0
    seen: set[tuple[chess.Square, chess.Square]] = set()
    for source, destination in destination_of.items():
        if source == destination:
            continue
        if destination_of.get(destination) == source and (destination, source) not in seen:
            seen.add((source, destination))
            swaps += 1
    return swaps


def _removed_squares(endpoint_board: chess.Board, matched: list[AssignedPair]) -> set[chess.Square]:
    """Squares occupied at the endpoint by a piece that is *not* placed on the
    next board (the piece is removed to the tray)."""
    matched_sources = {pair.source_square for pair in matched if pair.source_square is not None}
    return {
        square
        for square in _bit_squares(int(endpoint_board.occupied))
        if square not in matched_sources
    }
