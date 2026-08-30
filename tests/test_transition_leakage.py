"""Tests for stage-5 transition leakage: vector geometry and anchor uncertainty."""

from __future__ import annotations

import chess

from chess_chain.analysis.policies import TRANSITION_READY_POLICY
from chess_chain.domain.position import CanonicalPosition
from chess_chain.transitions import ast as A
from chess_chain.transitions.config import TransitionConfig
from chess_chain.transitions.context import build_context_from_boards
from chess_chain.transitions.leakage import (
    anchor_uncertainty_bits,
    conditional_leakage,
    derive_plausible_lines,
    plausible_endpoint_set,
    relative_vector_leak_bits,
    vector_origin_count,
)
from chess_chain.transitions.verify import verify_program


def test_vector_six_six_four_origins_four_bits() -> None:
    # (dx, dy) = (6, 6): only 4 origin squares keep the offset on the board,
    # so the geometric estimate leaks exactly log2(64/4) = 4 bits.
    assert vector_origin_count(6, 6) == 4
    assert relative_vector_leak_bits(6, 6) == 4.0


def test_smaller_vectors_have_correct_origin_domains() -> None:
    assert vector_origin_count(0, 0) == 64  # any origin keeps it on board
    assert vector_origin_count(1, 0) == 56
    assert vector_origin_count(0, 1) == 56
    assert vector_origin_count(1, 1) == 49
    assert vector_origin_count(7, 7) == 1
    assert vector_origin_count(8, 0) == 0  # no on-board origin
    assert vector_origin_count(0, 8) == 0
    assert relative_vector_leak_bits(0, 0) == 0.0
    assert relative_vector_leak_bits(7, 7) == 6.0  # log2(64/1)
    assert relative_vector_leak_bits(8, 0) == float("inf")


def _queen_context():
    # An open board: the white queen (d4) stays put in the intended line but
    # has many legal destinations in plausible lines -> high uncertainty.
    return build_context_from_boards(
        puzzle_id="Q",
        presented_fen="4k3/8/8/8/8/3Q4/8/4K3 w - - 0 1",
        endpoint_fen="4k3/8/8/8/8/3Q4/8/5K2 b - - 1 1",
        solution_uci=("e1f1",),
        policy=TRANSITION_READY_POLICY,
    )


def test_unmoved_piece_with_high_uncertainty_is_a_strong_anchor() -> None:
    context = _queen_context()
    queen = _unique(context, chess.QUEEN, chess.WHITE)
    plausible = derive_plausible_lines(context, depth=1, node_cap=500)
    before = plausible_endpoint_set(plausible, queen)
    # The queen never moves in the intended line, yet its plausible endpoint
    # distribution is broad: the epistemic rule is about uncertainty before
    # solving, never about whether the piece moved.
    assert before.cardinality() >= 8
    assert anchor_uncertainty_bits(before) >= 2.0
    assert all(event.moving_piece_id != queen for event in context.events)


def _knight_context():
    # The white knight on a1 has exactly one legal move (a1b3, blocked on c2 by
    # its own pawn): it is the piece that moves, yet its plausible destination
    # set is a singleton.
    return build_context_from_boards(
        puzzle_id="N",
        presented_fen="4k3/8/8/8/8/8/2P5/N3K3 w - - 0 1",
        endpoint_fen="4k3/8/8/8/8/1N6/2P5/4K3 b - - 0 1",
        solution_uci=("a1b3",),
        policy=TRANSITION_READY_POLICY,
    )


def test_moved_piece_with_one_plausible_destination_has_near_zero_uncertainty() -> None:
    context = _knight_context()
    knight = _unique(context, chess.KNIGHT, chess.WHITE)
    # Accept an explicit plausible-endpoint distribution: in every near-best
    # line the knight is forced to b3 (its only legal move), so its uncertainty
    # is zero even though it is the piece that moves.
    from chess_chain.transitions.leakage import PlausibleLines

    plausible = PlausibleLines(
        depth=1,
        node_cap=1,
        lines=(("a1b3",),),
        squares=({knight: chess.B3},),
    )
    before = plausible_endpoint_set(plausible, knight)
    assert before.cardinality() == 1
    assert anchor_uncertainty_bits(before) < 0.01
    assert any(event.moving_piece_id == knight for event in context.events)


def test_anchor_requirement_can_reject_a_weak_anchor() -> None:
    context = _knight_context()
    knight = _unique(context, chess.KNIGHT, chess.WHITE)
    next_start = CanonicalPosition.from_board(chess.Board("4k3/8/8/8/8/1N6/2P5/4K3 b - - 0 1"))
    from chess_chain.transitions.leakage import PlausibleLines

    plausible = PlausibleLines(
        depth=1,
        node_cap=1,
        lines=(("a1b3",),),
        squares=({knight: chess.B3},),
    )
    selector = A.PieceBeganOnSquare(knight.start_square)
    program = A.TransitionProgram((A.Move(selector, A.RelativeOffset(selector, 0, 0)),))
    config = TransitionConfig(min_anchor_uncertainty_bits=0.5, plausible_depth=1)
    verification = verify_program(
        program,
        accepted=(context,),
        next_start=next_start,
        plausible=plausible,
        config=config,
    )
    assert any("anchor uncertainty" in issue for issue in verification.issues)


def test_conditional_leakage_is_h_before_minus_h_after() -> None:
    context = _queen_context()
    queen = _unique(context, chess.QUEEN, chess.WHITE)
    plausible = derive_plausible_lines(context, depth=1, node_cap=500)
    before = plausible_endpoint_set(plausible, queen)
    # A +1 file move keeps the queen on board for almost every plausible
    # square: conditional leakage is small; a 6+6 offset keeps it on board for
    # only the four corner-compatible squares, leaking more.
    selector = A.PieceBeganOnSquare(queen.start_square)
    small_program = A.TransitionProgram((A.Move(selector, A.RelativeOffset(selector, 1, 0)),))
    small, after_small, _ = conditional_leakage(small_program, context, plausible, queen)
    big_program = A.TransitionProgram((A.Move(selector, A.RelativeOffset(selector, 6, 6)),))
    big, after_big, _ = conditional_leakage(big_program, context, plausible, queen)
    assert small < big
    assert big > 0.0
    # ``after`` is a subset of ``before`` in every case.
    assert len(after_small.entries) <= len(before.entries)
    assert len(after_big.entries) <= len(before.entries)


def _unique(context: object, piece_type: int, colour: int):
    return next(
        piece
        for piece, square in context.piece_square.items()  # type: ignore[attr-defined]
        if square is not None and piece.piece_type == piece_type and piece.colour == colour
    )
