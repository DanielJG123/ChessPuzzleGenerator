"""Independent product validation (stage 7).

``validate_product`` reconstructs the entire product from the canonical
``sequence@1`` specification and re-verifies it: initial setup legality, every
accepted move, intermediate stage-ending guards, transition execution from
every accepted endpoint, exact next-start positions, explicit handling of
non-visible castling/en-passant rights, physical inventory feasibility, final
occupancy invariance, the final-only collision status against its declared
bounded scope, final checkmate, the sequence fingerprint, analysis/config
version staleness and provisional-rating internal consistency.

``--explain-failure`` (``explain_failure()``) identifies the exact stage, ply,
transition AST node or inventory item responsible for each failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess

from chess_chain.analysis.collision import CollisionStatus
from chess_chain.config import Settings
from chess_chain.product.common import (
    material_feasible_for_standard_set,
    replay_line,
)
from chess_chain.sequence.fingerprint import sequence_fingerprint
from chess_chain.sequence.rating import (
    SimpleAverageRatingEstimator,
    estimate_key,
)
from chess_chain.sequence.records import SequenceResult
from chess_chain.sequence.request import request_fingerprint

__all__ = ["ProductIssue", "ProductValidationReport", "validate_product"]


@dataclass(frozen=True)
class ProductIssue:
    """One validation failure with an actionable location."""

    stage: int  # 1-based stage index; 0 = product-level
    check: str
    location: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "check": self.check,
            "location": self.location,
            "message": self.message,
        }


@dataclass(frozen=True)
class ProductValidationReport:
    """The full independent validation result."""

    valid: bool
    checks: dict[str, bool]
    issues: tuple[ProductIssue, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "checks": self.checks,
            "issues": [issue.to_dict() for issue in self.issues],
            "warnings": list(self.warnings),
        }

    def explain_failure(self) -> str:
        """Human-readable failure explanation (stage/ply/AST/inventory)."""
        if self.valid:
            return "product validation passed"
        lines = ["product validation FAILED:"]
        for issue in self.issues:
            scope = f"stage {issue.stage}" if issue.stage else "product"
            lines.append(f"- [{issue.check}] {scope} @ {issue.location}: {issue.message}")
        return "\n".join(lines)


def validate_product(
    sequence: SequenceResult,
    settings: Settings,
    *,
    repo: Any = None,
) -> ProductValidationReport:
    """Independently re-verify ``sequence`` against every product constraint."""
    issues: list[ProductIssue] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}

    # 1. Initial setup legality.
    _check_initial_setup(sequence, issues, checks)
    # 2. Every accepted move.
    _check_accepted_moves(sequence, issues, checks)
    # 3. Intermediate guards + 7. transition execution + 8. exact next start.
    _check_transitions(sequence, settings, issues, checks, warnings)
    # 4/11. Intermediate variance allowed; final occupancy invariance required.
    _check_occupancy(sequence, issues, checks)
    # 6. Final-only collision status against its declared bounded scope/policy.
    _check_final_collision(sequence, settings, repo, issues, checks)
    # 9. Non-visible castling/en-passant rights handling.
    _check_rights(sequence, settings, issues, checks)
    # 10. Physical inventory feasibility.
    _check_inventory(sequence, issues, checks)
    # 10b. Final checkmate.
    _check_final_checkmate(sequence, issues, checks)
    # 12. Sequence fingerprint.
    _check_fingerprint(sequence, issues, checks)
    # 13. No stale analysis/config versions.
    _check_versions(sequence, settings, issues, checks)
    # 14. Provisional rating internal consistency.
    _check_rating(sequence, issues, checks)

    return ProductValidationReport(
        valid=not issues,
        checks=checks,
        issues=tuple(issues),
        warnings=tuple(warnings),
    )


def _check_initial_setup(
    sequence: SequenceResult, issues: list[ProductIssue], checks: dict[str, bool]
) -> None:
    failed = False
    for index, stage in enumerate(sequence.stages, start=1):
        try:
            board = chess.Board(str(stage.row["start_fen"]))
        except ValueError as exc:
            failed = True
            issues.append(ProductIssue(index, "initial_setup_legal", "start FEN", str(exc)))
            continue
        if not board.is_valid():
            failed = True
            issues.append(
                ProductIssue(
                    index,
                    "initial_setup_legal",
                    "start FEN",
                    "position is not a legal chess position",
                )
            )
    checks["initial_setup_legal"] = not failed


def _check_accepted_moves(
    sequence: SequenceResult, issues: list[ProductIssue], checks: dict[str, bool]
) -> None:
    failed = False
    for index, stage in enumerate(sequence.stages, start=1):
        for line in stage.accepted_lines:
            try:
                replay_line(str(stage.row["start_fen"]), line)
            except ValueError as exc:
                failed = True
                issues.append(
                    ProductIssue(index, "accepted_moves_legal", "accepted line", str(exc))
                )
    checks["accepted_moves_legal"] = not failed


def _check_transitions(
    sequence: SequenceResult,
    settings: Settings,
    issues: list[ProductIssue],
    checks: dict[str, bool],
    warnings: list[str],
) -> None:
    from chess_chain.transitions.context import build_context_from_boards
    from chess_chain.transitions.executor import execute_program

    transition_ok = True
    next_start_ok = True
    guard_ok = True
    for index, stage in enumerate(sequence.stages, start=1):
        if stage.edge is None or stage.edge.program is None:
            continue
        if stage.index == 0:
            continue  # the first stage has no incoming transition
        source_stage = sequence.stages[stage.index - 1]
        next_start = chess.Board(str(stage.row["start_fen"]))
        for line_index, line in enumerate(source_stage.accepted_lines):
            try:
                context = build_context_from_boards(
                    puzzle_id=source_stage.puzzle_id,
                    presented_fen=str(source_stage.row["start_fen"]),
                    endpoint_fen=_endpoint_fen(source_stage, line),
                    solution_uci=line,
                    solver_colour=(
                        chess.WHITE
                        if str(source_stage.row["solver_colour"]) == "white"
                        else chess.BLACK
                    ),
                    policy=source_stage.policy,
                )
            except ValueError as exc:
                transition_ok = False
                issues.append(
                    ProductIssue(
                        index,
                        "transition_execution",
                        f"source accepted line {line_index}",
                        str(exc),
                    )
                )
                continue
            result = execute_program(stage.edge.program, context)
            location = f"transition AST on source accepted line {line_index} ({line})"
            if not result.ok:
                transition_ok = False
                issues.append(
                    ProductIssue(
                        index,
                        "transition_execution",
                        location,
                        f"executor: {result.note}",
                    )
                )
                continue
            after = result.board
            matches = (
                after is not None and after.fen().split()[0:4] == next_start.fen().split()[0:4]
            )
            if not matches:
                next_start_ok = False
                issues.append(
                    ProductIssue(
                        index,
                        "next_start_exact",
                        location,
                        "executed transition does not produce the exact next start",
                    )
                )

        # Guard re-check: the boundary must be ready on every accepted endpoint.
        # (The final stage's guard is the separate checkmate verification.)
        if not stage.is_final:
            guard = _reverify_guard(sequence, settings, stage, index)
            if guard is not None and not guard.passed:
                guard_ok = False
                for issue in guard.issues:
                    issues.append(ProductIssue(index, "intermediate_guards", "guard", issue))
    checks["transition_execution"] = transition_ok
    checks["next_start_exact"] = next_start_ok
    checks["intermediate_guards"] = guard_ok


def _endpoint_fen(stage: Any, line: tuple[str, ...]) -> str:
    endpoint_fens: tuple[str, ...] = stage.endpoint_fens()
    target = _fen_after(stage, line)
    for fen in endpoint_fens:
        if _fen_equivalent(fen, target):
            return fen
    return target


def _fen_after(stage: Any, line: tuple[str, ...]) -> str:
    _, board = replay_line(str(stage.row["start_fen"]), line)
    return board.fen()


def _fen_equivalent(fen_a: str, fen_b: str) -> bool:
    return fen_a.split()[0:4] == fen_b.split()[0:4]


def _reverify_guard(
    sequence: SequenceResult, settings: Settings, stage: Any, index: int
) -> Any | None:
    from chess_chain.analysis.policies import get_policy
    from chess_chain.transitions.context import build_context_from_boards
    from chess_chain.transitions.guards import StageBoundaryGuard
    from chess_chain.transitions.leakage import derive_plausible_lines
    from chess_chain.transitions.verify import (
        generate_plausible_wrong_contexts,
        verify_program,
    )

    if stage.edge is None or stage.edge.program is None or stage.index == 0:
        return None
    source_stage = sequence.stages[stage.index - 1]
    policy = get_policy(stage.end_policy_key.split("@", 1)[0])
    contexts = tuple(
        build_context_from_boards(
            puzzle_id=source_stage.puzzle_id,
            presented_fen=str(source_stage.row["start_fen"]),
            endpoint_fen=_endpoint_fen(source_stage, line),
            solution_uci=line,
            solver_colour=(
                chess.WHITE if str(source_stage.row["solver_colour"]) == "white" else chess.BLACK
            ),
            policy=policy,
        )
        for line in source_stage.accepted_lines
    )
    next_start = _next_start_canonical(sequence, stage)
    if next_start is None:
        return None
    config = settings.transition_config
    prefixes = _prefixes_for(contexts)
    wrong_states = generate_plausible_wrong_contexts(
        contexts[0],
        depth=config.plausible_depth,
        node_cap=config.plausible_node_cap,
        excluded_fens=frozenset(context.endpoint.fen for context in contexts),
    )
    plausible = derive_plausible_lines(
        contexts[0], depth=config.plausible_depth, node_cap=config.plausible_node_cap
    )
    guard = StageBoundaryGuard.from_policy(
        policy,
        transition_program=stage.edge.program if policy.kind == "TRANSITION_READY" else None,
        expected_next=next_start if policy.kind == "TRANSITION_READY" else None,
    )
    verification = verify_program(
        stage.edge.program,
        accepted=contexts,
        next_start=next_start,
        prefixes=prefixes,
        wrong_states=wrong_states,
        guard=guard,
        plausible=plausible,
        config=config,
    )
    return verification.guard


def _prefixes_for(contexts: tuple[Any, ...]) -> tuple[Any, ...]:
    from chess_chain.transitions.context import build_prefix_contexts

    return build_prefix_contexts(contexts[0], contexts[0].stage_end_policy)


def _next_start_canonical(sequence: SequenceResult, stage: Any) -> Any:
    """The transition's expected next start: the stage's own presented start."""
    from chess_chain.domain.position import CanonicalPosition

    if stage.index == 0:
        return None
    return CanonicalPosition.from_board(chess.Board(str(stage.row["start_fen"])))


def _check_occupancy(
    sequence: SequenceResult, issues: list[ProductIssue], checks: dict[str, bool]
) -> None:
    intermediate_variance_allowed = True
    final_invariant = True
    final_stage = sequence.stages[-1]
    if not _occupancy_invariant(str(final_stage.row["start_fen"]), final_stage.accepted_lines):
        final_invariant = False
        issues.append(
            ProductIssue(
                len(sequence.stages),
                "final_occupancy_invariance",
                "accepted lines",
                "final stage accepted lines do not share one occupancy",
            )
        )
    checks["intermediate_occupancy_variance_allowed"] = intermediate_variance_allowed
    checks["final_occupancy_invariance"] = final_invariant


def _check_final_collision(
    sequence: SequenceResult,
    settings: Settings,
    repo: Any,
    issues: list[ProductIssue],
    checks: dict[str, bool],
) -> None:
    final_stage = sequence.stages[-1]
    selection = final_stage.final_selection
    if selection is None:
        checks["final_collision_status"] = False
        issues.append(
            ProductIssue(
                len(sequence.stages),
                "final_collision_status",
                "final stage",
                "no final-collision analysis is recorded",
            )
        )
        return
    stored_status = selection.collision.status
    expected = CollisionStatus(sequence.request.final_collision_status)
    if stored_status == expected:
        checks["final_collision_status"] = True
        return
    inconclusive_ok = (
        stored_status == CollisionStatus.INCONCLUSIVE and not sequence.request.collision_strict
    )
    if inconclusive_ok:
        checks["final_collision_status"] = True
        return
    checks["final_collision_status"] = False
    issues.append(
        ProductIssue(
            len(sequence.stages),
            "final_collision_status",
            "final stage",
            f"stored collision status {stored_status.value} does not meet the declared "
            f"scope/policy (expected {expected.value})",
        )
    )


def _check_rights(
    sequence: SequenceResult,
    settings: Settings,
    issues: list[ProductIssue],
    checks: dict[str, bool],
) -> None:
    from chess_chain.transitions import ast as A

    failed = False
    for index, stage in enumerate(sequence.stages, start=1):
        if stage.edge is None or stage.edge.program is None or stage.index == 0:
            continue
        source_stage = sequence.stages[stage.index - 1]
        next_start_fen = _next_start_fen_str(sequence, stage)
        if next_start_fen is None:
            continue
        next_board = chess.Board(next_start_fen)
        endpoint_board = chess.Board(_endpoint_fen(source_stage, source_stage.accepted_lines[0]))
        needs_rights = _rights_missing(endpoint_board, next_board)
        establishes = any(
            isinstance(action, A.EstablishRights) for action in stage.edge.program.actions
        )
        if needs_rights and not establishes:
            failed = True
            issues.append(
                ProductIssue(
                    index,
                    "rights_handling",
                    "next start FEN",
                    "next start has castling/en-passant rights not established by any "
                    "transition action",
                )
            )
        if establishes and not settings.transition_config.allow_establish_rights:
            failed = True
            issues.append(
                ProductIssue(
                    index,
                    "rights_handling",
                    "transition AST",
                    "transition establishes non-visible rights but the configuration forbids it",
                )
            )
    checks["rights_handling"] = not failed


def _rights_missing(endpoint_board: chess.Board, next_board: chess.Board) -> bool:
    endpoint_rights = endpoint_board.fen().split()[2]
    next_rights = next_board.fen().split()[2]
    if not set(next_rights).issubset(set(endpoint_rights)):
        return True
    return endpoint_board.ep_square is None and next_board.ep_square is not None


def _next_start_fen_str(sequence: SequenceResult, stage: Any) -> str | None:
    if stage.is_final or stage.index + 1 >= len(sequence.stages):
        return None
    return str(sequence.stages[stage.index + 1].row["start_fen"])


def _check_inventory(
    sequence: SequenceResult, issues: list[ProductIssue], checks: dict[str, bool]
) -> None:
    failed = False
    for index, stage in enumerate(sequence.stages, start=1):
        start = chess.Board(str(stage.row["start_fen"]))
        boards = [start]
        for line in stage.accepted_lines:
            _, endpoint = replay_line(str(stage.row["start_fen"]), line)
            boards.append(endpoint)
        next_start_fen = _next_start_fen_str(sequence, stage)
        if next_start_fen is not None:
            boards.append(chess.Board(next_start_fen))
        for board in boards:
            if not material_feasible_for_standard_set(board):
                failed = True
                issues.append(
                    ProductIssue(
                        index,
                        "inventory_feasible",
                        _inventory_item(board),
                        "position material exceeds the configured standard physical set",
                    )
                )
                break
    checks["inventory_feasible"] = not failed


def _inventory_item(board: chess.Board) -> str:
    from chess_chain.product.common import STANDARD_SET_LIMITS, material_counts

    for (colour, piece_type), count in material_counts(board).items():
        limit = STANDARD_SET_LIMITS.get((chess.Color(colour), chess.PieceType(piece_type)), 0)
        if count > limit:
            name = chess.piece_name(piece_type)
            side = "white" if colour == chess.WHITE else "black"
            return f"inventory: {count} {side} {name}(s) (limit {limit})"
    return "inventory"


def _check_final_checkmate(
    sequence: SequenceResult, issues: list[ProductIssue], checks: dict[str, bool]
) -> None:
    final_stage = sequence.stages[-1]
    _, final_board = replay_line(
        str(final_stage.row["start_fen"]),
        tuple(str(move) for move in final_stage.row["solution_uci"]),
    )
    mate = final_board.is_checkmate()
    checks["final_checkmate"] = mate
    if not mate:
        issues.append(
            ProductIssue(
                len(sequence.stages),
                "final_checkmate",
                "last stored solution",
                "last stored solution does not end in python-chess checkmate",
            )
        )


def _check_fingerprint(
    sequence: SequenceResult, issues: list[ProductIssue], checks: dict[str, bool]
) -> None:
    request_key = request_fingerprint(sequence.request)
    stage_summaries = tuple(
        {
            "puzzle_id": stage.puzzle_id,
            "symmetry": stage.edge.symmetry if stage.edge is not None else None,
            "accepted_lines": stage.accepted_lines,
            "transition_ast": (
                stage.edge.program.to_json()
                if stage.edge is not None and stage.edge.program is not None
                else None
            ),
        }
        for stage in sequence.stages
    )
    recomputed = sequence_fingerprint(request_key, stage_summaries)
    matches = recomputed == sequence.fingerprint
    checks["sequence_fingerprint"] = matches
    if not matches:
        issues.append(ProductIssue(0, "sequence_fingerprint", "product", "fingerprint mismatch"))


def _check_versions(
    sequence: SequenceResult,
    settings: Settings,
    issues: list[ProductIssue],
    checks: dict[str, bool],
) -> None:
    current = {
        "request": int(sequence.request.version),
        "transition": int(settings.transition_config.version),
        "collision_analyser": int(settings.collision_analyser_version),
        "similarity": int(settings.similarity_version),
        "transform": int(settings.transform_version),
    }
    stored = sequence.versions
    stale = [
        f"{name}: stored {stored.get(name)} != current {value}"
        for name, value in current.items()
        if stored.get(name) != value
    ]
    checks["no_stale_versions"] = not stale
    if stale:
        issues.append(ProductIssue(0, "no_stale_versions", "product", "; ".join(stale)))


def _check_rating(
    sequence: SequenceResult, issues: list[ProductIssue], checks: dict[str, bool]
) -> None:
    estimator = SimpleAverageRatingEstimator()
    key = estimate_key("mean_source_puzzle_rating", estimator.estimator_version)
    stored = sequence.difficulty_estimates.get(key)
    if stored is None:
        checks["rating_consistency"] = True
        return
    expected = estimator.estimate(sequence.stage_ratings())
    consistent = (
        int(stored.get("estimator_version", -1)) == estimator.estimator_version
        and tuple(int(r) for r in stored.get("stage_ratings", [])) == tuple(expected.stage_ratings)
        and int(stored.get("value", -1)) == expected.value
    )
    checks["rating_consistency"] = consistent
    if not consistent:
        issues.append(
            ProductIssue(
                0,
                "rating_consistency",
                "difficulty_estimates",
                "stored provisional estimate is inconsistent with the stored "
                "source ratings and estimator version",
            )
        )


def _occupancy_invariant(start_fen: str, accepted_lines: tuple[tuple[str, ...], ...]) -> bool:
    board = chess.Board(start_fen)
    occupancies: set[int] = set()
    for line in accepted_lines:
        current = board.copy()
        for uci in line:
            move = chess.Move.from_uci(uci)
            if move not in current.legal_moves:
                return False
            current.push(move)
        occupancies.add(int(current.occupied))
    return len(occupancies) <= 1
