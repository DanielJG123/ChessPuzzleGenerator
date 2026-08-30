"""Tests for stage-boundary guards and their clue disclosures."""

from __future__ import annotations

import chess

from chess_chain.analysis.policies import (
    CHECKMATE_POLICY,
    EXPLICIT_PREDICATE,
    EXPLICIT_PREDICATE_QUEEN_CAPTURED,
    FIXED_PLIES,
    FIXED_PLIES_POLICY_4,
    TRANSITION_READY,
    get_policy,
)
from chess_chain.domain.position import CanonicalPosition
from chess_chain.transitions import ast as A
from chess_chain.transitions.context import build_context_from_boards
from chess_chain.transitions.guards import (
    StageBoundaryGuard,
    evaluate_guard,
    guard_disclosures,
    verify_guard,
)
from chess_chain.transitions.render import render_program


def _context(presented: str, endpoint: str, solution: tuple[str, ...], policy):
    return build_context_from_boards(
        puzzle_id="G",
        presented_fen=presented,
        endpoint_fen=endpoint,
        solution_uci=solution,
        policy=policy,
    )


def test_checkmate_guard_is_true_only_at_mate_and_leaks_the_goal() -> None:
    presented = "6k1/6pp/8/8/8/8/8/5R1K b - - 0 1"
    endpoint = "5R1k/6pp/8/8/8/8/8/7K b - - 1 2"
    context = _context(presented, endpoint, ("g8h8", "f1f8"), CHECKMATE_POLICY)
    guard = StageBoundaryGuard.from_policy(CHECKMATE_POLICY)
    evaluation = evaluate_guard(guard, context)
    assert evaluation.applicable is True
    assert evaluation.unique is True
    disclosures = guard_disclosures(guard)
    assert disclosures.goal_leakage is True  # the player learns mate is the goal
    # The guard is false on the presented start (a proper prefix).
    prefix = _context(presented, presented, (), CHECKMATE_POLICY)
    verification = verify_guard(guard, accepted=(context,), prefixes=(prefix,), wrong_states=())
    assert verification.passed is True
    assert all(not evaluation.ready() for evaluation in verification.prefix_evaluations)


def test_explicit_predicate_guard_discloses_named_captured_piece() -> None:
    presented = "3qk3/8/8/8/8/8/8/3QK3 w - - 0 1"
    endpoint = "3Qk3/8/8/8/8/8/8/4K3 b - - 0 1"
    policy = EXPLICIT_PREDICATE_QUEEN_CAPTURED
    context = _context(presented, endpoint, ("d1d8",), policy)
    guard = StageBoundaryGuard.from_policy(policy)
    assert guard.kind == EXPLICIT_PREDICATE
    evaluation = evaluate_guard(guard, context)
    assert evaluation.applicable is True
    disclosures = guard_disclosures(guard)
    # "absent:black_queen" names the captured piece.
    assert disclosures.named_captured_pieces == 1
    assert disclosures.goal_leakage is False
    # Not applicable before the capture.
    before = _context(presented, presented, (), policy)
    assert evaluate_guard(guard, before).applicable is False


def test_fixed_plies_guard_verifies_length_and_reveals_line_length() -> None:
    presented = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
    endpoint = "4k3/8/8/8/8/R7/8/4K3 w - - 4 3"
    solution = ("a1a2", "e8e7", "a2a3", "e7e8")
    policy = FIXED_PLIES_POLICY_4
    context = _context(presented, endpoint, solution, policy)
    guard = StageBoundaryGuard.from_policy(policy)
    assert guard.kind == FIXED_PLIES
    evaluation = evaluate_guard(guard, context)
    assert evaluation.applicable is True
    assert guard_disclosures(guard).line_length_revealed == 4
    # A shorter line is not applicable.
    short_endpoint = "8/4k3/8/8/8/8/R7/4K3 w - - 2 2"
    short = _context(presented, short_endpoint, solution[:2], policy)
    assert evaluate_guard(guard, short).applicable is False


def test_transition_ready_guard_uniquely_true_only_at_the_accepted_endpoint() -> None:
    presented = "4k3/8/8/8/8/8/8/4K2R w - - 0 1"
    endpoint = "8/4k3/8/8/8/8/7R/4K3 w - - 2 2"
    next_start = CanonicalPosition.from_board(chess.Board(presented))
    context = _context(presented, endpoint, ("h1h2", "e8e7"), get_policy("transition_ready"))
    rook = A.PieceBeganOnSquare(chess.H1)
    king = A.PieceBeganOnSquare(chess.E8)
    program = A.TransitionProgram(
        (
            A.Move(rook, A.RelativeOffset(rook, 0, -1)),
            A.Move(king, A.RelativeOffset(king, 0, 1)),
        )
    )
    guard = StageBoundaryGuard.from_policy(
        get_policy("transition_ready"),
        transition_program=program,
        expected_next=next_start,
    )
    assert guard.kind == TRANSITION_READY
    # Accepted endpoint: applicable and unique.
    assert evaluate_guard(guard, context).ready() is True
    # Presented start and every proper prefix: the guard must be false or
    # non-unique.
    prefixes = (
        _context(presented, presented, (), get_policy("transition_ready")),
        _context(
            presented,
            "4k3/8/8/8/8/8/7R/4K3 b - - 0 1",
            ("h1h2",),
            get_policy("transition_ready"),
        ),
    )
    verification = verify_guard(guard, accepted=(context,), prefixes=prefixes, wrong_states=())
    assert verification.passed is True
    for evaluation in verification.prefix_evaluations:
        assert evaluation.ready() is False


def test_castling_rights_are_rendered_explicitly_or_rejected_by_config() -> None:
    from chess_chain.transitions.config import TransitionConfig
    from chess_chain.transitions.verify import verify_program

    presented = "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1"
    context = _context(presented, presented, (), get_policy("transition_ready"))
    next_start = CanonicalPosition.from_board(chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"))
    mask = (1 << chess.H1) | (1 << chess.A1) | (1 << chess.H8) | (1 << chess.A8)
    rights_program = A.TransitionProgram((A.EstablishRights(castling_mask=mask),))
    rendered = render_program(rights_program)
    assert "castling rights are KQkq" in rendered

    allowed = verify_program(
        rights_program,
        accepted=(context,),
        next_start=next_start,
        config=TransitionConfig(allow_establish_rights=True),
    )
    assert allowed.passed is True
    forbidden = verify_program(
        rights_program,
        accepted=(context,),
        next_start=next_start,
        config=TransitionConfig(allow_establish_rights=False),
    )
    assert forbidden.passed is False
    assert any("EstablishRights" in issue for issue in forbidden.issues)


def test_ep_rights_are_rendered_explicitly() -> None:
    rights_program = A.TransitionProgram((A.EstablishRights(castling_mask=0, ep_square=chess.E3),))
    rendered = render_program(rights_program)
    assert "en-passant target exists on e3" in rendered
