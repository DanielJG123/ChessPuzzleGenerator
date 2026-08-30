"""Transition verification: correctness, non-accepted invalidity, and clues.

``verify_program`` checks, for a candidate transition program:

1. exact correctness: on every accepted endpoint it executes unambiguously and
   produces the exact next presented position (fingerprint equality, including
   castling rights and en-passant state);
2. invalidity elsewhere: applied to the presented start, every proper intended
   prefix and the configured plausible wrong states, it must *not* produce the
   correct next start (each state's status is recorded: executable, ambiguous,
   off-board, colliding, inventory-invalid, ...);
3. reconstructability: the number of distinct next positions produced across
   plausible endpoints; if all plausible endpoints collapse into the exact next
   start, the instructions may reveal it without solving and are rejected when
   ``reject_collapsing_reveal`` is set;
4. the clue report (absolute disclosures, geometric/conditional leakage, guard
   disclosures) and the configurable anchor-uncertainty thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

from chess_chain.domain.position import CanonicalPosition
from chess_chain.transitions.ast import (
    AbsoluteSquare,
    Add,
    Conditional,
    EstablishRights,
    Move,
    Remove,
    Replace,
    Swap,
    TransitionProgram,
)
from chess_chain.transitions.config import TransitionConfig
from chess_chain.transitions.context import (
    EndpointContext,
    build_context_from_boards,
    build_prefix_contexts,
)
from chess_chain.transitions.executor import STATUS_OK, execute_program
from chess_chain.transitions.guards import (
    GuardVerification,
    StageBoundaryGuard,
    guard_disclosures,
    verify_guard,
)
from chess_chain.transitions.leakage import (
    ClueReport,
    PlausibleLines,
    compute_clue_report,
    derive_plausible_lines,
)

__all__ = [
    "StateOutcome",
    "VerificationResult",
    "generate_plausible_wrong_contexts",
    "verify_program",
]


@dataclass(frozen=True)
class StateOutcome:
    """The outcome of applying the program to one non-accepted state."""

    label: str
    status: str
    note: str | None = None
    creates_next_start: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "status": self.status,
            "note": self.note,
            "creates_next_start": self.creates_next_start,
        }


@dataclass(frozen=True)
class VerificationResult:
    """The complete verification of one transition program."""

    passed: bool
    issues: tuple[str, ...]
    accepted_outcomes: tuple[StateOutcome, ...]
    non_accepted_outcomes: tuple[StateOutcome, ...]
    distinct_next_positions: int
    clue: ClueReport
    guard: GuardVerification | None

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "issues": list(self.issues),
            "accepted_outcomes": [outcome.to_dict() for outcome in self.accepted_outcomes],
            "non_accepted_outcomes": [outcome.to_dict() for outcome in self.non_accepted_outcomes],
            "distinct_next_positions": self.distinct_next_positions,
            "clue": self.clue.to_dict(),
            "guard": (
                {
                    "passed": self.guard.passed,
                    "issues": list(self.guard.issues),
                    "disclosures": self.guard.disclosures.to_dict(),
                }
                if self.guard is not None
                else None
            ),
        }


def verify_program(
    program: TransitionProgram,
    *,
    accepted: tuple[EndpointContext, ...],
    next_start: CanonicalPosition,
    prefixes: tuple[EndpointContext, ...] | None = None,
    wrong_states: tuple[EndpointContext, ...] | None = None,
    guard: StageBoundaryGuard | None = None,
    plausible: PlausibleLines | None = None,
    config: TransitionConfig | None = None,
) -> VerificationResult:
    """Verify a transition program (see module docstring)."""
    config = config or TransitionConfig()
    issues: list[str] = []

    if not config.allow_establish_rights and _contains_establish_rights(program):
        issues.append("program uses EstablishRights but the product configuration forbids it")
    if not config.allow_absolute_square_fallback and _contains_absolute_squares(program):
        issues.append("program uses absolute-square fallbacks but the product forbids them")

    accepted_outcomes: list[StateOutcome] = []
    for index, context in enumerate(accepted):
        result = execute_program(program, context)
        label = f"accepted:{index}"
        matches = result.fingerprint() == next_start.fingerprint
        accepted_outcomes.append(
            StateOutcome(
                label=label, status=result.status, note=result.note, creates_next_start=matches
            )
        )
        if not result.ok or not matches:
            issues.append(
                f"{label}: program does not create the exact next start "
                f"(status={result.status}, matches={matches})"
            )

    prefix_list = list(prefixes) if prefixes is not None else []
    wrong_list = list(wrong_states) if wrong_states is not None else []
    if not prefix_list and accepted:
        prefix_list = list(build_prefix_contexts(accepted[0], accepted[0].stage_end_policy))

    non_accepted: list[StateOutcome] = []
    distinct: set[str] = set()
    for index, context in enumerate(prefix_list):
        outcome = _run_non_accepted(program, context, next_start, f"prefix:{index}")
        non_accepted.append(outcome)
        if outcome.status == STATUS_OK:
            distinct.add(_outcome_fingerprint(program, context))
    for index, context in enumerate(wrong_list):
        outcome = _run_non_accepted(program, context, next_start, f"wrong:{index}")
        non_accepted.append(outcome)
        if outcome.status == STATUS_OK:
            distinct.add(_outcome_fingerprint(program, context))
    for outcome in non_accepted:
        if outcome.creates_next_start:
            issues.append(
                f"{outcome.label}: program creates the correct next start from a non-accepted state"
            )

    distinct_next = len(distinct)
    if (
        config.reject_collapsing_reveal
        and distinct
        and distinct == {next_start.fingerprint.to_key()}
    ):
        issues.append(
            "every executable non-accepted state collapses into the exact next start "
            "(the visible instructions reveal the next position before solving)"
        )

    if plausible is None:
        if accepted:
            plausible = derive_plausible_lines(
                accepted[0],
                depth=config.plausible_depth,
                node_cap=config.plausible_node_cap,
            )
        else:
            plausible = PlausibleLines(depth=0, node_cap=0, lines=(), squares=())
    clue = compute_clue_report(
        program,
        accepted[0] if accepted else _dummy_context(),
        plausible=plausible,
        guard_disclosures=guard_disclosures(guard) if guard is not None else None,
        distinct_next_positions=distinct_next,
    )
    if clue.independent_anchors < config.min_independent_anchors and _references_piece(program):
        issues.append(
            f"too few independent uncertain anchors: {clue.independent_anchors} < "
            f"{config.min_independent_anchors}"
        )
    if (
        _references_piece(program)
        and clue.weakest_anchor_uncertainty_bits is not None
        and clue.weakest_anchor_uncertainty_bits < config.min_anchor_uncertainty_bits
    ):
        issues.append(
            f"weakest anchor uncertainty {clue.weakest_anchor_uncertainty_bits:.3f} bits "
            f"below the configured minimum {config.min_anchor_uncertainty_bits}"
        )

    guard_verification: GuardVerification | None = None
    if guard is not None:
        guard_verification = verify_guard(
            guard,
            accepted=accepted,
            prefixes=tuple(prefix_list),
            wrong_states=tuple(wrong_list),
        )
        if not guard_verification.passed:
            issues.extend(guard_verification.issues)

    return VerificationResult(
        passed=not issues,
        issues=tuple(issues),
        accepted_outcomes=tuple(accepted_outcomes),
        non_accepted_outcomes=tuple(non_accepted),
        distinct_next_positions=distinct_next,
        clue=clue,
        guard=guard_verification,
    )


def _run_non_accepted(
    program: TransitionProgram,
    context: EndpointContext,
    next_start: CanonicalPosition,
    label: str,
) -> StateOutcome:
    result = execute_program(program, context)
    matches = result.fingerprint() == next_start.fingerprint
    return StateOutcome(
        label=label,
        status=result.status,
        note=result.note,
        creates_next_start=result.ok and matches,
    )


def _outcome_fingerprint(program: TransitionProgram, context: EndpointContext) -> str:
    result = execute_program(program, context)
    fingerprint = result.fingerprint()
    return fingerprint.to_key() if fingerprint is not None else f"error:{result.status}"


def _references_piece(program: TransitionProgram) -> bool:
    """True when the program references at least one piece selector (a pure
    whole-board transform program references none)."""

    def _action(action: object) -> bool:
        if isinstance(action, (Move, Add, Remove, Replace, Swap)):
            return True
        if isinstance(action, Conditional):
            return any(_action(sub) for branch in action.branches for sub in branch.actions)
        return False

    return any(_action(action) for action in program.actions)


def _contains_establish_rights(program: TransitionProgram) -> bool:
    return any(isinstance(action, EstablishRights) for action in program.actions)


def _contains_absolute_squares(program: TransitionProgram) -> bool:
    def _action(action: object) -> bool:
        if isinstance(action, Move):
            return isinstance(action.destination, AbsoluteSquare)
        if isinstance(action, Add):
            return isinstance(action.destination, AbsoluteSquare)
        if isinstance(action, Conditional):
            return any(_action(sub) for branch in action.branches for sub in branch.actions)
        return False

    return any(_action(action) for action in program.actions)


def _dummy_context() -> EndpointContext:
    from chess_chain.analysis.policies import TRANSITION_READY_POLICY

    return build_context_from_boards(
        puzzle_id="(none)",
        presented_fen="8/8/8/8/8/8/8/8 w - - 0 1",
        endpoint_fen="8/8/8/8/8/8/8/8 w - - 0 1",
        policy=TRANSITION_READY_POLICY,
    )


def generate_plausible_wrong_contexts(
    context: EndpointContext,
    *,
    depth: int = 2,
    node_cap: int = 2000,
    excluded_fens: frozenset[str] = frozenset(),
) -> tuple[EndpointContext, ...]:
    """Generate plausible wrong states: distinct positions reachable within
    ``depth`` plies that are *not* the presented start, not an intended prefix
    and not an accepted endpoint (``excluded_fens`` may list every accepted
    endpoint so that multi-endpoint transitions never treat a sibling accepted
    endpoint as a wrong state). Each is built as a full context (its own
    history) so history selectors resolve on it."""
    start_fen = context.presented_start.fen
    accepted_fens = {context.endpoint.fen} | set(excluded_fens)
    intended = tuple(event.uci for event in context.events)
    prefix_fens = {_fen_after(start_fen, intended[:length]) for length in range(len(intended) + 1)}

    board = context.presented_start.board
    seen: set[str] = set()
    contexts: list[EndpointContext] = []
    nodes = 0

    def visit(current: chess.Board, line: tuple[str, ...]) -> None:
        nonlocal nodes
        if nodes >= node_cap:
            return
        nodes += 1
        if line:
            fen = current.fen()
            if fen not in seen and fen not in prefix_fens and fen not in accepted_fens:
                seen.add(fen)
                contexts.append(
                    build_context_from_boards(
                        puzzle_id=context.puzzle_id,
                        presented_fen=start_fen,
                        endpoint_fen=fen,
                        solution_uci=line,
                        solver_colour=context.solver_colour,
                        policy=context.stage_end_policy,
                    )
                )
        if len(line) >= depth:
            return
        for move in current.legal_moves:
            if nodes >= node_cap:
                return
            current.push(move)
            visit(current, line + (move.uci(),))
            current.pop()

    visit(board.copy(), ())
    return tuple(contexts)


def _fen_after(fen: str, moves: tuple[str, ...]) -> str:
    board = chess.Board(fen)
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
    return board.fen()
