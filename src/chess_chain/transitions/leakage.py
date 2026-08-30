"""Clue and leakage accounting for transition programs.

Two leakage families are implemented:

1. Geometric no-wrap relative-vector leakage. For a ``(dx, dy)`` offset on an
   otherwise uniform 8x8 board the number of possible origin squares is

       N = (8 - abs(dx)) * (8 - abs(dy))

   and the simple geometric estimate is ``log2(64 / N)`` bits. An offset of
   six files and six ranks therefore has four possible origins and leaks four
   bits of square information.

2. Conditional leakage from a plausible-endpoint distribution. ``before`` is
   the candidate final-square distribution of an anchor piece across plausible
   (legal and/or near-best) lines; ``after`` is the subset of lines where the
   instruction is on-board, unambiguous and executable. Leakage is
   ``H(before) - H(after)`` using weighted entropy when counts are available
   (bounded legal search) and log-cardinality otherwise; the label records
   which was used.

The epistemic-anchor rule is deliberately *not* "referenced pieces must have
moved": an unmoved queen is a strong anchor when many plausible pre-solution
lines would have moved it, and a moved piece is weak when its destination is
obvious. Uncertainty comes from the plausible distribution, never from whether
the piece moved.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import chess

from chess_chain.domain.pieces import StagePieceId
from chess_chain.transitions.ast import (
    AbsoluteSquare,
    Add,
    CapturedPieceBeganOnSquare,
    Conditional,
    InDirection,
    Move,
    NearestTo,
    OccupantOfSquare,
    RelativeOffset,
    RelativeToAnchor,
    Remove,
    Replace,
    Selector,
    SquareOf,
    Swap,
    TransitionProgram,
    VacatedBy,
)
from chess_chain.transitions.context import (
    EndpointContext,
    build_context_from_boards,
    piece_squares_after,
)
from chess_chain.transitions.executor import _State, resolve_selector

__all__ = [
    "ClueReport",
    "PlausibleEndpointSet",
    "PlausibleLines",
    "anchor_uncertainty_bits",
    "compute_clue_report",
    "conditional_leakage",
    "derive_plausible_lines",
    "plausible_endpoint_set",
    "relative_vector_leak_bits",
    "vector_origin_count",
]

#: Bits for naming one square outright.
SQUARE_BITS = math.log2(64)
#: Bits for naming one captured piece's colour/type (12 combinations).
CAPTURED_PIECE_BITS = math.log2(12)


def vector_origin_count(dx: int, dy: int) -> int:
    """Number of source squares that keep a ``(dx, dy)`` offset on the board."""
    files = 8 - abs(dx)
    ranks = 8 - abs(dy)
    if files <= 0 or ranks <= 0:
        return 0
    return files * ranks


def relative_vector_leak_bits(dx: int, dy: int) -> float:
    """``log2(64 / N)`` for a no-wrap relative vector; ``inf`` when the vector
    has no on-board origin."""
    origins = vector_origin_count(dx, dy)
    if origins <= 0:
        return float("inf")
    return math.log2(64.0 / origins)


@dataclass(frozen=True)
class PlausibleEndpointSet:
    """A weighted candidate-endpoint distribution for one physical piece."""

    entries: tuple[tuple[chess.Square | None, float], ...]
    labelled: str  # "weighted_entropy" or "log_cardinality"

    def cardinality(self) -> int:
        return len(self.entries)

    def entropy(self) -> float:
        total = sum(weight for _, weight in self.entries)
        if total <= 0:
            return 0.0
        probabilities = [weight / total for _, weight in self.entries]
        return -sum(p * math.log2(p) for p in probabilities if p > 0)


def anchor_uncertainty_bits(before: PlausibleEndpointSet) -> float:
    """Entropy or log-cardinality of the anchor's plausible distribution."""
    if before.labelled == "log_cardinality":
        return math.log2(before.cardinality())
    return before.entropy()


@dataclass(frozen=True)
class PlausibleLines:
    """Bounded legal (and/or near-best) candidate lines from the presented
    start, with per-line final squares for every physical piece."""

    depth: int
    node_cap: int
    lines: tuple[tuple[str, ...], ...]
    squares: tuple[dict[StagePieceId, chess.Square | None], ...]


def derive_plausible_lines(
    context: EndpointContext,
    *,
    depth: int = 2,
    node_cap: int = 2000,
) -> PlausibleLines:
    """Enumerate legal lines of exactly ``depth`` plies from the presented
    start (bounded by ``node_cap``) and record each physical piece's final
    square (``None`` = captured) per line. Weights are the line counts, so the
    resulting distributions are labelled ``weighted_entropy``."""
    start = context.presented_start.board
    pieces = frozenset(context.piece_square)
    lines: list[tuple[str, ...]] = []
    squares: list[dict[StagePieceId, chess.Square | None]] = []
    nodes = 0

    def visit(board: chess.Board, line: tuple[str, ...]) -> None:
        nonlocal nodes
        if nodes >= node_cap:
            return
        nodes += 1
        if len(line) == depth:
            lines.append(line)
            squares.append(piece_squares_after(start, line, pieces))
            return
        for move in board.legal_moves:
            if nodes >= node_cap:
                return
            board.push(move)
            visit(board, line + (move.uci(),))
            board.pop()

    visit(start.copy(), ())
    return PlausibleLines(
        depth=depth, node_cap=node_cap, lines=tuple(lines), squares=tuple(squares)
    )


def plausible_endpoint_set(plausible: PlausibleLines, piece: StagePieceId) -> PlausibleEndpointSet:
    """The plausible final-square distribution of ``piece`` across the lines."""
    counts: Counter[chess.Square | None] = Counter()
    for line_squares in plausible.squares:
        counts[line_squares.get(piece)] += 1
    if not counts:
        return PlausibleEndpointSet(entries=(), labelled="weighted_entropy")
    return PlausibleEndpointSet(
        entries=tuple((square, float(count)) for square, count in counts.items()),
        labelled="weighted_entropy",
    )


def conditional_leakage(
    program: TransitionProgram,
    context: EndpointContext,
    plausible: PlausibleLines,
    piece: StagePieceId,
) -> tuple[float, PlausibleEndpointSet, PlausibleEndpointSet]:
    """``H(before) - H(after)`` for ``piece``.

    ``before`` is the plausible final-square distribution; ``after`` keeps only
    the lines whose full program execution is on-board, unambiguous and
    executable (``status == ok``) on that line's own context.
    """
    from chess_chain.transitions.executor import execute_program

    before = plausible_endpoint_set(plausible, piece)
    after_counts: Counter[chess.Square | None] = Counter()
    for line, line_squares in zip(plausible.lines, plausible.squares, strict=True):
        if piece not in line_squares:
            continue
        line_context = build_context_from_boards(
            puzzle_id=context.puzzle_id,
            presented_fen=context.presented_start.fen,
            endpoint_fen=_fen_after(context.presented_start.fen, line),
            solution_uci=line,
            solver_colour=context.solver_colour,
            policy=context.stage_end_policy,
        )
        if execute_program(program, line_context).ok:
            after_counts[line_squares.get(piece)] += 1
    if not after_counts:
        return float("inf"), before, PlausibleEndpointSet(entries=(), labelled="weighted_entropy")
    after = PlausibleEndpointSet(
        entries=tuple((square, float(count)) for square, count in after_counts.items()),
        labelled="weighted_entropy",
    )
    return anchor_uncertainty_bits(before) - anchor_uncertainty_bits(after), before, after


def _fen_after(fen: str, moves: tuple[str, ...]) -> str:
    board = chess.Board(fen)
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
    return board.fen()


@dataclass
class _ActionCounts:
    absolute_source_squares: int = 0
    absolute_destination_squares: int = 0
    occupied_square_disclosures: int = 0
    vector_leakage_bits: float = 0.0


@dataclass(frozen=True)
class ClueReport:
    """The combined clue report for a transition program and its guard.

    Absolute-square disclosures are counted (source selectors and destination
    expressions); geometric vector leakage and conditional leakage are measured
    in bits; guard disclosures are folded in rather than rating the move
    instructions in isolation.
    """

    absolute_source_squares: int = 0
    absolute_destination_squares: int = 0
    occupied_square_disclosures: int = 0
    vector_leakage_bits: float = 0.0
    conditional_leakage_bits: float = 0.0
    anchor_uncertainty_bits: float = 0.0
    weakest_anchor_uncertainty_bits: float | None = None
    independent_anchors: int = 0
    guard_goal_leakage: bool = False
    guard_named_captured_pieces: int = 0
    guard_check_disclosures: int = 0
    guard_promotion_disclosures: int = 0
    guard_occupied_square_disclosures: int = 0
    guard_line_length_revealed: int | None = None
    distinct_next_positions_across_plausible: int = 0
    total_clue_bits: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "absolute_source_squares": self.absolute_source_squares,
            "absolute_destination_squares": self.absolute_destination_squares,
            "occupied_square_disclosures": self.occupied_square_disclosures,
            "vector_leakage_bits": round(self.vector_leakage_bits, 4),
            "conditional_leakage_bits": round(self.conditional_leakage_bits, 4),
            "anchor_uncertainty_bits": round(self.anchor_uncertainty_bits, 4),
            "weakest_anchor_uncertainty_bits": (
                round(self.weakest_anchor_uncertainty_bits, 4)
                if self.weakest_anchor_uncertainty_bits is not None
                else None
            ),
            "independent_anchors": self.independent_anchors,
            "guard_goal_leakage": self.guard_goal_leakage,
            "guard_named_captured_pieces": self.guard_named_captured_pieces,
            "guard_check_disclosures": self.guard_check_disclosures,
            "guard_promotion_disclosures": self.guard_promotion_disclosures,
            "guard_occupied_square_disclosures": self.guard_occupied_square_disclosures,
            "guard_line_length_revealed": self.guard_line_length_revealed,
            "distinct_next_positions_across_plausible": (
                self.distinct_next_positions_across_plausible
            ),
            "total_clue_bits": round(self.total_clue_bits, 4),
        }


def compute_clue_report(
    program: TransitionProgram,
    context: EndpointContext,
    *,
    plausible: PlausibleLines,
    guard_disclosures: object,
    distinct_next_positions: int = 0,
) -> ClueReport:
    """Compute the clue report for ``program`` (see module docstring)."""
    from chess_chain.transitions.guards import GuardDisclosures

    counts = _ActionCounts()

    def count_selector(selector: Selector) -> None:
        # ``PieceBeganOnSquare`` names a square that was public in the initial
        # position; only endpoint-dependent selectors count as disclosures.
        if isinstance(selector, (OccupantOfSquare, CapturedPieceBeganOnSquare)):
            counts.absolute_source_squares += 1
            counts.occupied_square_disclosures += 1

    def count_destination(destination: object) -> None:
        if isinstance(destination, AbsoluteSquare):
            counts.absolute_destination_squares += 1
        elif isinstance(destination, (RelativeOffset, RelativeToAnchor)):
            counts.vector_leakage_bits += relative_vector_leak_bits(destination.dx, destination.dy)

    def walk_selector(selector: Selector) -> None:
        count_selector(selector)
        if isinstance(selector, (NearestTo, InDirection, RelativeToAnchor)):
            walk_selector(selector.anchor)

    def walk_destination(destination: object) -> None:
        count_destination(destination)
        if isinstance(destination, (RelativeOffset, SquareOf, VacatedBy)):
            walk_selector(destination.piece)
        elif isinstance(destination, RelativeToAnchor):
            walk_selector(destination.anchor)

    def walk_action(action: object) -> None:
        if isinstance(action, Move):
            walk_selector(action.piece)
            walk_destination(action.destination)
        elif isinstance(action, Add):
            walk_selector(action.source)
            walk_destination(action.destination)
        elif isinstance(action, Remove):
            walk_selector(action.piece)
        elif isinstance(action, Swap):
            walk_selector(action.a)
            walk_selector(action.b)
        elif isinstance(action, Replace):
            walk_selector(action.occupant)
            walk_selector(action.replacement)
        elif isinstance(action, Conditional):
            for branch in action.branches:
                for sub in branch.actions:
                    walk_action(sub)

    for action in program.actions:
        walk_action(action)

    # Anchor uncertainty: measure the plausible-endpoint entropy of every
    # referenced piece (never "did it move?").
    state = _State(context)
    anchor_bits = 0.0
    weakest_anchor: float | None = None
    independent_anchors = 0
    seen: set[StagePieceId] = set()

    def record_anchor(selector: Selector) -> None:
        nonlocal anchor_bits, weakest_anchor, independent_anchors
        try:
            pieces = resolve_selector(state, selector)
        except Exception:
            return
        if len(pieces) != 1:
            return
        piece = pieces[0]
        if piece in seen:
            return
        seen.add(piece)
        bits = anchor_uncertainty_bits(plausible_endpoint_set(plausible, piece))
        anchor_bits += bits
        weakest_anchor = bits if weakest_anchor is None else min(weakest_anchor, bits)
        independent_anchors += 1

    def walk_anchor_selectors(selector: Selector) -> None:
        record_anchor(selector)
        if isinstance(selector, (NearestTo, InDirection, RelativeToAnchor)):
            walk_anchor_selectors(selector.anchor)

    for action in program.actions:
        if isinstance(action, Move):
            walk_anchor_selectors(action.piece)
            if isinstance(action.destination, RelativeOffset):
                walk_anchor_selectors(action.destination.piece)
        elif isinstance(action, Add):
            walk_anchor_selectors(action.source)
        elif isinstance(action, Swap):
            walk_anchor_selectors(action.a)
            walk_anchor_selectors(action.b)
        elif isinstance(action, (Remove, Replace)):
            walk_anchor_selectors(action.piece if isinstance(action, Remove) else action.occupant)

    # Conditional leakage for relative-offset moves: how much of the moved
    # piece's plausible distribution survives the instruction being executable.
    conditional_bits = 0.0
    for action in program.actions:
        if isinstance(action, Move) and isinstance(action.destination, RelativeOffset):
            moved_piece = _resolve_single(state, action.piece)
            if moved_piece is not None:
                bits, _, _ = conditional_leakage(program, context, plausible, moved_piece)
                conditional_bits += 0.0 if bits == float("inf") else bits

    if isinstance(guard_disclosures, GuardDisclosures):
        guard_goal = guard_disclosures.goal_leakage
        guard_named = guard_disclosures.named_captured_pieces
        guard_checks = guard_disclosures.check_disclosures
        guard_promotions = guard_disclosures.promotion_disclosures
        guard_squares = guard_disclosures.occupied_square_disclosures
        guard_length = guard_disclosures.line_length_revealed
    else:
        guard_goal = False
        guard_named = 0
        guard_checks = 0
        guard_promotions = 0
        guard_squares = 0
        guard_length = None

    absolute_squares = counts.absolute_source_squares + counts.absolute_destination_squares
    total = (
        counts.vector_leakage_bits
        + conditional_bits
        + (absolute_squares + counts.occupied_square_disclosures) * SQUARE_BITS
        + guard_named * CAPTURED_PIECE_BITS
        + (1.0 if guard_goal else 0.0)
        + (math.log2(max(2, guard_length)) if guard_length is not None else 0.0)
    )
    return ClueReport(
        absolute_source_squares=counts.absolute_source_squares,
        absolute_destination_squares=counts.absolute_destination_squares,
        occupied_square_disclosures=counts.occupied_square_disclosures,
        vector_leakage_bits=counts.vector_leakage_bits,
        conditional_leakage_bits=conditional_bits,
        anchor_uncertainty_bits=anchor_bits,
        weakest_anchor_uncertainty_bits=weakest_anchor,
        independent_anchors=independent_anchors,
        guard_goal_leakage=bool(guard_goal),
        guard_named_captured_pieces=int(guard_named),
        guard_check_disclosures=int(guard_checks),
        guard_promotion_disclosures=int(guard_promotions),
        guard_occupied_square_disclosures=int(guard_squares),
        guard_line_length_revealed=guard_length,
        distinct_next_positions_across_plausible=distinct_next_positions,
        total_clue_bits=total,
    )


def _resolve_single(state: _State, selector: Selector) -> StagePieceId | None:
    pieces = resolve_selector(state, selector)
    return pieces[0] if len(pieces) == 1 else None
