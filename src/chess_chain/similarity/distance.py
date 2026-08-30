"""Layered position distance between a stage endpoint and a next-start position.

The distance is layered rather than a single embedding:

1. Occupancy distance: ``popcount(endpoint_occupancy XOR start_occupancy)``.
2. Piece-placement distance over the 12 colour/type bitboards.
3. Material / inventory compatibility (on-board pieces, tray availability,
   normal-set limits, promotions and surplus pieces).
4. Minimum-cost physical piece assignment (see ``assignment.py``).
5. Relative displacement cost for matched pieces.
6. Whole-board transformations are handled by the retriever/candidate layer,
   which tries each symmetry and validates the reconstructed board.

Castling rights and the en-passant square are carried through every
comparison: two visually equal boards with different rights are *not* exact
matches. Endpoints that would require the player instructions to establish
non-visible rights are flagged (``castling_annotation_required`` /
``ep_annotation_required``) and may be rejected by product configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

from chess_chain.domain.pieces import PIECE_ORDER, PIECE_VALUES
from chess_chain.domain.position import CanonicalPosition
from chess_chain.similarity.assignment import (
    AssignmentResult,
    CostWeights,
    assign_pieces,
)

__all__ = [
    "EndpointComparison",
    "available_counts",
    "castling_annotation",
    "compare_endpoint",
]

#: Home squares involved in each standard castling right.
CASTLING_SQUARES: dict[str, tuple[chess.Square, chess.Square]] = {
    "K": (chess.E1, chess.H1),
    "Q": (chess.E1, chess.A1),
    "k": (chess.E8, chess.H8),
    "q": (chess.E8, chess.A8),
}

CASTLING_BITS: dict[str, int] = {
    "K": 1 << chess.H1,
    "Q": 1 << chess.A1,
    "k": 1 << chess.H8,
    "q": 1 << chess.A8,
}


@dataclass(frozen=True)
class EndpointComparison:
    """Full layered comparison of one accepted endpoint against one transformed
    next start."""

    endpoint_index: int
    endpoint_fingerprint: str
    endpoint_fen: str
    symmetry: str
    exact_position_match: bool
    d_occ: int
    d_placement: int
    piece_diff_remove: tuple[int, ...]  # per COUNT_NAMES: endpoint has extras
    piece_diff_add: tuple[int, ...]  # per COUNT_NAMES: start needs extras
    material_diff: tuple[int, ...]  # net per COUNT_NAMES: endpoint - start
    material_remove_w: int
    material_add_w: int
    material_remove_b: int
    material_add_b: int
    assignment: AssignmentResult
    inventory_valid: bool
    inventory_notes: tuple[str, ...]
    castling_annotation_required: bool
    ep_annotation_required: bool
    rights_note: str | None
    exact_journey: bool
    occupancy_journey: bool


def available_counts(
    start_counts: tuple[int, ...], endpoint_counts: tuple[int, ...]
) -> tuple[int, ...]:
    """Physical pieces available to the transition: everything on the endpoint
    board plus everything the stage started with that is now in the tray
    (captures and set-aside promoted pawns). Equivalent to the element-wise
    maximum of the two count vectors."""
    return tuple(max(s, e) for s, e in zip(start_counts, endpoint_counts, strict=True))


def compare_endpoint(
    endpoint: CanonicalPosition,
    start: CanonicalPosition,
    *,
    endpoint_index: int,
    source_start_counts: tuple[int, ...],
    symmetry: str,
    cost_weights: CostWeights | None = None,
) -> EndpointComparison:
    """Compare one accepted endpoint against the (transformed) next start.

    ``source_start_counts`` is the presented-start material of the *source*
    puzzle (the physical pieces that existed when the stage began). Endpoints
    with different occupancy or piece placement are compared independently and
    are never rejected for that reason.
    """
    endpoint_fp = endpoint.fingerprint
    start_fp = start.fingerprint
    endpoint_counts = _material_counts(endpoint.board)
    start_counts = _material_counts(start.board)

    d_occ = _popcount(endpoint_fp.occupied ^ start_fp.occupied)
    d_placement = sum(
        _popcount(eb ^ sb) for eb, sb in zip(endpoint_fp.bitboards, start_fp.bitboards, strict=True)
    )

    piece_diff_remove = tuple(
        max(0, e - s) for e, s in zip(endpoint_counts, start_counts, strict=True)
    )
    piece_diff_add = tuple(
        max(0, s - e) for e, s in zip(endpoint_counts, start_counts, strict=True)
    )
    material_diff = tuple(e - s for e, s in zip(endpoint_counts, start_counts, strict=True))
    material_remove_w = _material_score(piece_diff_remove, chess.WHITE)
    material_add_w = _material_score(piece_diff_add, chess.WHITE)
    material_remove_b = _material_score(piece_diff_remove, chess.BLACK)
    material_add_b = _material_score(piece_diff_add, chess.BLACK)

    available = available_counts(source_start_counts, endpoint_counts)
    inventory_notes = tuple(_inventory_notes(available))
    inventory_valid = all(s <= a for s, a in zip(start_counts, available, strict=True))

    assignment = assign_pieces(
        endpoint.board,
        start.board,
        source_start_counts,
        endpoint_counts,
        weights=cost_weights,
    )

    exact_position_match = endpoint_fp == start_fp
    castling_required, castling_note = castling_annotation(endpoint, start, assignment)
    ep_required = start_fp.ep_square is not None

    rights_parts = []
    if castling_note:
        rights_parts.append(castling_note)
    if ep_required:
        rights_parts.append("en-passant target square requires a non-visible setup")
    rights_note = "; ".join(rights_parts) or None

    exact_journey = (
        exact_position_match
        and assignment.moves_only()
        and assignment.replacements == 0
        and d_occ == 0
    )
    occupancy_journey = d_occ == 0

    return EndpointComparison(
        endpoint_index=endpoint_index,
        endpoint_fingerprint=endpoint_fp.to_key(),
        endpoint_fen=endpoint.fen,
        symmetry=symmetry,
        exact_position_match=exact_position_match,
        d_occ=d_occ,
        d_placement=d_placement,
        piece_diff_remove=piece_diff_remove,
        piece_diff_add=piece_diff_add,
        material_diff=material_diff,
        material_remove_w=material_remove_w,
        material_add_w=material_add_w,
        material_remove_b=material_remove_b,
        material_add_b=material_add_b,
        assignment=assignment,
        inventory_valid=inventory_valid,
        inventory_notes=inventory_notes,
        castling_annotation_required=castling_required,
        ep_annotation_required=ep_required,
        rights_note=rights_note,
        exact_journey=exact_journey,
        occupancy_journey=occupancy_journey,
    )


def castling_annotation(
    endpoint: CanonicalPosition,
    start: CanonicalPosition,
    assignment: AssignmentResult,
) -> tuple[bool, str | None]:
    """Determine whether the transition must establish a non-visible castling
    right.

    A right of the next start survives the whole physical process only when the
    endpoint itself still carries the right (the king/rook never moved during
    the stage) AND the king and rook are on their home squares and are not
    moved by the transition journey. Otherwise the player instructions must
    re-establish the right, which the mechanical (occupancy-only) checker
    cannot verify.
    """
    moved_sources = {
        pair.source_square
        for pair in assignment.matched
        if pair.source_square is not None and pair.source_square != pair.target_square
    }
    notes: list[str] = []
    for right, (king_square, rook_square) in CASTLING_SQUARES.items():
        if not start.fingerprint.castling & CASTLING_BITS[right]:
            continue
        endpoint_has_right = bool(endpoint.fingerprint.castling & CASTLING_BITS[right])
        king_piece = chess.Piece(chess.KING, chess.WHITE if right.isupper() else chess.BLACK)
        rook_piece = chess.Piece(chess.ROOK, chess.WHITE if right.isupper() else chess.BLACK)
        home_ok = (
            endpoint.board.piece_at(king_square) == king_piece
            and endpoint.board.piece_at(rook_square) == rook_piece
        )
        unmoved = king_square not in moved_sources and rook_square not in moved_sources
        if not (endpoint_has_right and home_ok and unmoved):
            notes.append(f"castling right {right} must be re-established")
    return (bool(notes), "; ".join(notes) or None)


def _inventory_notes(available: tuple[int, ...]) -> list[str]:
    from chess_chain.domain.pieces import STANDARD_SET_LIMITS

    notes: list[str] = []
    for index, (colour, piece_type) in enumerate(PIECE_ORDER):
        limit = STANDARD_SET_LIMITS[piece_type]
        if available[index] > limit:
            side = "white" if colour == chess.WHITE else "black"
            notes.append(
                f"{available[index]} {side} {chess.piece_name(piece_type)} pieces "
                f"exceed the normal set limit of {limit} (surplus required)"
            )
    return notes


def _material_counts(board: chess.Board) -> tuple[int, ...]:
    return tuple(len(board.pieces(piece_type, colour)) for colour, piece_type in PIECE_ORDER)


def _material_score(counts: tuple[int, ...], colour: chess.Color) -> int:
    total = 0
    for index, (piece_colour, piece_type) in enumerate(PIECE_ORDER):
        if piece_colour == colour:
            total += PIECE_VALUES[piece_type] * counts[index]
    return total


def _popcount(mask: int) -> int:
    return int(mask).bit_count()
