"""Typed stage-boundary guards with a deterministic evaluator.

A ``StageBoundaryGuard`` is *separate* from the transition actions even when
both are rendered as one instruction. For every intermediate ``StageEndPolicy``
the same uniform rule applies:

- false or non-unique on the presented start, every proper intended prefix and
  the configured plausible wrong states;
- true and uniquely executable on every accepted endpoint.

Policy-specific disclosures are recorded as clue metadata: ``CHECKMATE`` leaks
that the goal is mate; ``EXPLICIT_PREDICATE`` leaks named captured pieces,
checks, promotions or occupied squares it mentions; ``FIXED_PLIES`` reveals the
stated line length. ``TRANSITION_READY`` leaks nothing beyond the transition
program's own disclosures (the guard *is* the program becoming executable).
"""

from __future__ import annotations

from dataclasses import dataclass

from chess_chain.analysis.policies import (
    CHECKMATE,
    EXPLICIT_PREDICATE,
    FIXED_PLIES,
    TRANSITION_READY,
    StageEndPolicy,
    evaluate_predicate,
)
from chess_chain.domain.position import CanonicalPosition
from chess_chain.transitions.ast import TransitionProgram
from chess_chain.transitions.context import EndpointContext
from chess_chain.transitions.executor import execute_program

__all__ = [
    "GuardDisclosures",
    "GuardEvaluation",
    "GuardVerification",
    "StageBoundaryGuard",
    "evaluate_guard",
    "guard_disclosures",
    "verify_guard",
]


@dataclass(frozen=True)
class StageBoundaryGuard:
    """The typed stopping rule for one intermediate stage."""

    kind: str
    policy_key: str
    predicate: str | None = None  # EXPLICIT_PREDICATE
    fixed_plies: int | None = None  # FIXED_PLIES
    transition_program: TransitionProgram | None = None  # TRANSITION_READY
    expected_next: CanonicalPosition | None = None  # TRANSITION_READY
    wording: str | None = None

    @classmethod
    def from_policy(
        cls,
        policy: StageEndPolicy,
        *,
        transition_program: TransitionProgram | None = None,
        expected_next: CanonicalPosition | None = None,
    ) -> StageBoundaryGuard:
        return cls(
            kind=policy.kind,
            policy_key=policy.key(),
            predicate=policy.predicate,
            fixed_plies=policy.fixed_plies,
            transition_program=transition_program,
            expected_next=expected_next,
            wording=policy.wording,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "policy_key": self.policy_key,
            "predicate": self.predicate,
            "fixed_plies": self.fixed_plies,
            "transition_program": (
                self.transition_program.to_json() if self.transition_program is not None else None
            ),
            "expected_next_fen": self.expected_next.fen if self.expected_next is not None else None,
            "wording": self.wording,
        }


@dataclass(frozen=True)
class GuardEvaluation:
    """The deterministic evaluation of one guard on one context."""

    applicable: bool
    unique: bool
    note: str | None = None

    def ready(self) -> bool:
        return self.applicable and self.unique


@dataclass(frozen=True)
class GuardDisclosures:
    """What the guard wording reveals to the player."""

    goal_leakage: bool = False
    named_captured_pieces: int = 0
    check_disclosures: int = 0
    promotion_disclosures: int = 0
    occupied_square_disclosures: int = 0
    line_length_revealed: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "goal_leakage": self.goal_leakage,
            "named_captured_pieces": self.named_captured_pieces,
            "check_disclosures": self.check_disclosures,
            "promotion_disclosures": self.promotion_disclosures,
            "occupied_square_disclosures": self.occupied_square_disclosures,
            "line_length_revealed": self.line_length_revealed,
        }


@dataclass(frozen=True)
class GuardVerification:
    """The outcome of verifying a guard against all relevant states."""

    passed: bool
    issues: tuple[str, ...]
    accepted_evaluations: tuple[GuardEvaluation, ...]
    prefix_evaluations: tuple[GuardEvaluation, ...]
    wrong_state_evaluations: tuple[GuardEvaluation, ...]
    disclosures: GuardDisclosures


def evaluate_guard(guard: StageBoundaryGuard, context: EndpointContext) -> GuardEvaluation:
    """Evaluate one guard on one context, deterministically."""
    if guard.kind == CHECKMATE:
        applicable = context.endpoint.board.is_checkmate()
        return GuardEvaluation(applicable=applicable, unique=True)
    if guard.kind == EXPLICIT_PREDICATE:
        assert guard.predicate is not None
        applicable = evaluate_predicate(context.endpoint.board, guard.predicate)
        return GuardEvaluation(applicable=applicable, unique=True)
    if guard.kind == FIXED_PLIES:
        assert guard.fixed_plies is not None
        applicable = len(context.events) == guard.fixed_plies
        return GuardEvaluation(applicable=applicable, unique=True)
    if guard.kind == TRANSITION_READY:
        assert guard.transition_program is not None
        assert guard.expected_next is not None
        result = execute_program(guard.transition_program, context)
        if not result.ok:
            return GuardEvaluation(
                applicable=False,
                unique=False,
                note=f"executor: {result.note}",
            )
        matches = result.fingerprint() == guard.expected_next.fingerprint
        return GuardEvaluation(applicable=matches, unique=True)
    raise ValueError(f"unknown guard kind {guard.kind!r}")


def guard_disclosures(guard: StageBoundaryGuard) -> GuardDisclosures:
    """Derive what the guard wording reveals to the player."""
    if guard.kind == CHECKMATE:
        return GuardDisclosures(goal_leakage=True)
    if guard.kind == EXPLICIT_PREDICATE:
        return _predicate_disclosures(guard.predicate or "")
    if guard.kind == FIXED_PLIES:
        return GuardDisclosures(line_length_revealed=guard.fixed_plies)
    if guard.kind == TRANSITION_READY:
        return GuardDisclosures()
    raise ValueError(f"unknown guard kind {guard.kind!r}")


def _predicate_disclosures(predicate: str) -> GuardDisclosures:
    kind, _, arg = predicate.partition(":")
    if kind == "checkmate":
        return GuardDisclosures(goal_leakage=True)
    if kind == "absent":
        return GuardDisclosures(named_captured_pieces=1)
    if kind == "present":
        symbol, _, square = arg.partition(":")
        return GuardDisclosures(
            named_captured_pieces=1 if symbol else 0,
            occupied_square_disclosures=1 if square else 0,
        )
    return GuardDisclosures()


def verify_guard(
    guard: StageBoundaryGuard,
    *,
    accepted: tuple[EndpointContext, ...],
    prefixes: tuple[EndpointContext, ...],
    wrong_states: tuple[EndpointContext, ...],
) -> GuardVerification:
    """Verify the uniform guard rule across every relevant state.

    - on every accepted endpoint: applicable AND unique;
    - on the presented start, every proper intended prefix and every configured
      plausible wrong state: NOT (applicable AND unique).
    """
    issues: list[str] = []
    accepted_evals = tuple(evaluate_guard(guard, context) for context in accepted)
    for index, evaluation in enumerate(accepted_evals):
        if not evaluation.ready():
            issues.append(
                f"accepted endpoint {index}: guard not applicable and unique "
                f"(applicable={evaluation.applicable}, unique={evaluation.unique})"
            )
    prefix_evals = tuple(evaluate_guard(guard, context) for context in prefixes)
    for index, evaluation in enumerate(prefix_evals):
        if evaluation.ready():
            issues.append(
                f"prefix {index}: guard unexpectedly applicable and unique on a non-accepted state"
            )
    wrong_evals = tuple(evaluate_guard(guard, context) for context in wrong_states)
    for index, evaluation in enumerate(wrong_evals):
        if evaluation.ready():
            issues.append(
                f"plausible wrong state {index}: guard unexpectedly applicable "
                "and unique on a non-accepted state"
            )
    return GuardVerification(
        passed=not issues,
        issues=tuple(issues),
        accepted_evaluations=accepted_evals,
        prefix_evaluations=prefix_evals,
        wrong_state_evaluations=wrong_evals,
        disclosures=guard_disclosures(guard),
    )

    raise ValueError(f"unknown guard kind {guard.kind!r}")
