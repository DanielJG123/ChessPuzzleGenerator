"""Tests for stage-end policies and terminal-variation analysis."""

from __future__ import annotations

import chess

from chess_chain.analysis.policies import (
    CHECKMATE_POLICY,
    EXPLICIT_PREDICATE_QUEEN_CAPTURED,
    FIXED_PLIES_POLICY_4,
    STAGE_END_POLICIES,
    TRANSITION_READY_POLICY,
    evaluate_predicate,
)
from chess_chain.analysis.variations import (
    analyze_variations,
    enumerate_accepted_lines,
)
from tests.conftest import engine_fen


def test_checkmate_policy_verifies_mate_endpoints() -> None:
    # Presented start after the preceding g8h8: white delivers mate with Rf8#.
    fen = "7k/6pp/8/8/8/8/8/5R1K w - - 1 2"
    summary = analyze_variations(
        puzzle_id="MATE1",
        start_fen=fen,
        solution_uci=("f1f8",),
        end_policy=CHECKMATE_POLICY,
        acceptance="intended_only",
        max_plies=2,
        max_nodes=1000,
    )
    assert summary.line_count == 1
    assert summary.endpoints_verified is True
    assert summary.verification_note is None


def test_checkmate_policy_fails_for_non_mate_endpoint() -> None:
    fen = "7k/8/8/8/8/8/8/R6K w - - 0 1"
    summary = analyze_variations(
        puzzle_id="CHECK1",
        start_fen=fen,
        solution_uci=("a1a7",),
        end_policy=CHECKMATE_POLICY,
        acceptance="intended_only",
        max_plies=2,
        max_nodes=1000,
    )
    assert summary.endpoints_verified is False
    assert "not checkmate" in (summary.verification_note or "")


def test_transition_ready_policy_defers_proof() -> None:
    summary = analyze_variations(
        puzzle_id="ANY1",
        start_fen="4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        solution_uci=("e1e2",),
        end_policy=TRANSITION_READY_POLICY,
        acceptance="intended_only",
        max_plies=2,
        max_nodes=100,
    )
    assert summary.endpoints_verified is True
    assert "deferred" in (summary.verification_note or "")


def test_explicit_predicate_policy() -> None:
    # White queen captures the black queen: predicate "absent:black_queen".
    fen = "6k1/8/8/8/8/4q3/8/2Q3K1 w - - 0 1"
    summary = analyze_variations(
        puzzle_id="PRED1",
        start_fen=fen,
        solution_uci=("c1e3",),
        end_policy=EXPLICIT_PREDICATE_QUEEN_CAPTURED,
        acceptance="intended_only",
        max_plies=2,
        max_nodes=100,
    )
    assert summary.endpoints_verified is True
    assert summary.verification_note is None
    board = chess.Board(fen)
    board.push(chess.Move.from_uci("c1e3"))
    assert evaluate_predicate(board, EXPLICIT_PREDICATE_QUEEN_CAPTURED.predicate or "")


def test_fixed_plies_policy() -> None:
    summary = analyze_variations(
        puzzle_id="FIXED1",
        start_fen="4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        solution_uci=("e1e2", "e8d8", "e2e3", "d8d7"),
        end_policy=FIXED_PLIES_POLICY_4,
        acceptance="intended_only",
        max_plies=4,
        max_nodes=100,
    )
    assert summary.endpoints_verified is True
    assert summary.verification_note is None


def test_intermediate_lines_with_different_occupancy_are_not_rejected() -> None:
    # From a quiet position, different legal first moves end at different masks.
    fen = "7k/8/8/8/8/8/8/R5K1 w - - 0 1"
    summary = analyze_variations(
        puzzle_id="OCC1",
        start_fen=fen,
        solution_uci=("a1a2",),
        end_policy=TRANSITION_READY_POLICY,
        acceptance="all_legal",
        max_plies=1,
        max_nodes=1000,
    )
    assert summary.line_count >= 4  # rook + king moves
    assert summary.occupancy_invariant is False
    assert summary.exact_position_invariant is False
    # Descriptive facts only: the summary is returned, not rejected.
    assert isinstance(summary.endpoints_verified, bool)


def test_final_accepted_lines_identical_occupancy_different_identities() -> None:
    # Only the four promotion moves are accepted (scored); all leave e8 occupied.
    fen = "6k1/4P3/8/8/8/8/8/4K3 w - - 0 1"
    key = engine_fen(chess.Board(fen))
    move_scores = {
        key: {
            "e7e8q": 0,
            "e7e8r": 0,
            "e7e8b": 0,
            "e7e8n": 0,
        }
    }
    summary = analyze_variations(
        puzzle_id="ID1",
        start_fen=fen,
        solution_uci=("e7e8q",),
        end_policy=TRANSITION_READY_POLICY,
        acceptance="plausible",
        max_plies=1,
        max_nodes=1000,
        move_scores=move_scores,
        plausible_gap_cp=30,
    )
    assert summary.line_count == 4
    assert summary.occupancy_invariant is True
    assert summary.exact_position_invariant is False  # Q/R/B/N differ in identity
    assert len(summary.terminal_occupancies) == 1


def test_all_policies_are_registered_and_versioned() -> None:
    assert set(STAGE_END_POLICIES) == {
        "checkmate",
        "transition_ready",
        "explicit_black_queen_captured",
        "fixed_plies_4",
    }
    for policy in STAGE_END_POLICIES.values():
        assert policy.version >= 1
        assert policy.key().endswith(f"@v{policy.version}")


def test_enumerate_accepted_lines_bounded_and_truncated() -> None:
    board = chess.Board("7k/8/8/8/8/8/8/R5K1 w - - 0 1")
    lines, truncated = enumerate_accepted_lines(
        board,
        intended_uci=("a1a2",),
        acceptance="all_legal",
        max_plies=2,
        max_nodes=3,
    )
    assert truncated is True
    assert len(lines) < 4 * 4  # node budget cut the full tree
