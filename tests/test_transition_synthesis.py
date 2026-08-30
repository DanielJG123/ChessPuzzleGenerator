"""Tests for transition synthesis over multiple accepted endpoints."""

from __future__ import annotations

import chess

from chess_chain.analysis.policies import TRANSITION_READY_POLICY
from chess_chain.domain.position import CanonicalPosition
from chess_chain.transitions import ast as A
from chess_chain.transitions.config import TransitionConfig
from chess_chain.transitions.context import build_context_from_boards
from chess_chain.transitions.executor import STATUS_OK, execute_program
from chess_chain.transitions.synth import synthesize
from chess_chain.transitions.verify import verify_program

POL = TRANSITION_READY_POLICY


def _context(puzzle_id: str, presented: str, endpoint: str, solution: tuple[str, ...]):
    return build_context_from_boards(
        puzzle_id=puzzle_id,
        presented_fen=presented,
        endpoint_fen=endpoint,
        solution_uci=solution,
        policy=POL,
    )


def test_multiple_accepted_endpoints_mapping_to_one_exact_next_start() -> None:
    # Two accepted endpoints (black king ends on d8 vs f8) with different
    # occupied squares must both map to the exact same next start.
    presented = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
    ctx1 = _context("M1", presented, "3k4/8/8/8/8/8/R7/4K3 w - - 2 2", ("a1a2", "e8d8"))
    ctx2 = _context("M2", presented, "5k2/8/8/8/8/8/R7/4K3 w - - 2 2", ("a1a2", "e8f8"))
    assert ctx1.endpoint.fingerprint.occupied != ctx2.endpoint.fingerprint.occupied
    next_start = CanonicalPosition.from_board(chess.Board("4k3/8/8/8/8/8/8/1R2K3 w - - 0 1"))
    program = A.TransitionProgram(
        (
            A.Move(A.OccupantOfSquare(chess.A2), A.AbsoluteSquare(chess.B1)),
            A.Move(A.PieceBeganOnSquare(chess.E8), A.AbsoluteSquare(chess.E8)),
        )
    )
    verification = verify_program(
        program, accepted=(ctx1, ctx2), next_start=next_start, config=TransitionConfig()
    )
    assert verification.passed is True
    assert all(outcome.status == STATUS_OK for outcome in verification.accepted_outcomes)
    assert all(outcome.creates_next_start for outcome in verification.accepted_outcomes)


def test_conditional_branch_handles_present_or_captured_pieces() -> None:
    # The black knight is on the board in one accepted endpoint and captured in
    # the other; the conditional executes the empty branch in the first case and
    # re-adds the tray piece in the second, both reaching the exact same next
    # start. The white rook is removed in both.
    presented = "4k3/8/8/8/8/8/8/R1n3K1 w - - 0 1"
    ctx1 = _context("C1", presented, "8/4k3/8/8/8/8/R7/2n3K1 w - - 2 2", ("a1a2", "e8e7"))
    ctx2 = _context("C2", presented, "8/4k3/8/8/8/8/8/2R3K1 w - - 2 2", ("a1c1", "e8e7"))
    next_start = CanonicalPosition.from_board(chess.Board("8/4k3/8/8/8/8/8/2n3K1 w - - 0 1"))
    knight = A.PieceBeganOnSquare(chess.C1)
    rook = A.PieceBeganOnSquare(chess.A1)
    program = A.TransitionProgram(
        (
            A.Remove(rook),
            A.Conditional(
                (
                    A.Branch(A.OnBoard(knight), ()),
                    A.Branch(A.OffBoard(knight), (A.Add(knight, A.AbsoluteSquare(chess.C1)),)),
                )
            ),
        )
    )
    verification = verify_program(
        program, accepted=(ctx1, ctx2), next_start=next_start, config=TransitionConfig()
    )
    assert verification.passed is True
    assert all(outcome.creates_next_start for outcome in verification.accepted_outcomes)
    # Both branches resolve to the same target on their own context.
    assert execute_program(program, ctx1).fingerprint() == next_start.fingerprint
    assert execute_program(program, ctx2).fingerprint() == next_start.fingerprint


def test_multiple_endpoints_that_cannot_share_a_transition_are_rejected() -> None:
    # ctx1 ends white-to-move, ctx2 ends black-to-move; no physical program can
    # change the side to move, so the pair cannot share a transition.
    presented = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
    ctx1 = _context("S1", presented, "3k4/8/8/8/8/8/R7/4K3 w - - 2 2", ("a1a2", "e8d8"))
    ctx2 = _context(
        "S2",
        presented,
        "3k4/8/8/8/8/R7/8/4K3 b - - 3 3",
        ("a1a2", "e8d8", "a2a3"),
    )
    next_start = CanonicalPosition.from_board(chess.Board("4k3/8/8/8/8/8/8/1R2K3 w - - 0 1"))
    program = A.TransitionProgram(
        (
            A.Move(A.OccupantOfSquare(chess.A2), A.AbsoluteSquare(chess.B1)),
            A.Move(A.PieceBeganOnSquare(chess.E8), A.AbsoluteSquare(chess.E8)),
        )
    )
    verification = verify_program(
        program, accepted=(ctx1, ctx2), next_start=next_start, config=TransitionConfig()
    )
    assert verification.passed is False
    assert verification.accepted_outcomes[1].creates_next_start is False


def test_synthesize_finds_verified_programs_with_short_relative_instructions() -> None:
    presented = "4k3/8/8/8/8/8/8/R3K3 b - - 0 1"
    ctx = _context(
        "SIM-A",
        presented,
        "3k4/8/8/8/8/8/R7/4K3 b - - 1 2",
        ("e8d8", "a1a2"),
    )
    next_start = CanonicalPosition.from_board(chess.Board(presented))
    config = TransitionConfig(plausible_depth=1, plausible_node_cap=300)
    results = synthesize((ctx,), next_start, policy=POL, config=config)
    assert results, "expected at least one verified transition"
    assert results[0].verification.passed is True
    assert len(results[0].program.actions) <= config.default_max_actions
    # The relative-offset variant (no absolute destinations) outranks the
    # absolute fallback when both are found.
    relative = [
        candidate
        for candidate in results
        if candidate.verification.clue.absolute_destination_squares == 0
    ]
    absolute = [
        candidate
        for candidate in results
        if candidate.verification.clue.absolute_destination_squares > 0
    ]
    assert relative
    assert relative[0].rank_key() < (absolute[0].rank_key() if absolute else (float("inf"),))
