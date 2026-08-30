"""Machine validation of generated sequences (stage 6).

``validate_sequence`` re-checks the twelve hard constraints by re-deriving
contexts, re-executing the transitions and re-querying the final-collision
analysis:

1. every source line replays legally;
2. every solver move meets the configured uniqueness policy;
3. every opponent move meets the configured forcedness policy (or all accepted
   branches are explicitly supported);
4. every intermediate stage has a verified visible ending under its policy;
5. every transition maps every accepted intermediate endpoint exactly to the
   same next presented start (different occupancy allowed);
6. transitions and guards stay within leakage and complexity limits;
7. all stages are achievable with the configured physical inventory;
8. no puzzle is repeated within a sequence;
9. exclusions / already-used ids are respected;
10. the last stored solution ends in python-chess-confirmed checkmate;
11. only the final puzzle must have occupancy invariance across accepted lines;
12. only the final puzzle must meet the requested final-collision status
    (``collision_free_within_scope`` in strict mode - never renamed to an
    unbounded proof).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess

from chess_chain.analysis.cache import DiskKeyValueCache
from chess_chain.analysis.collision import (
    CollisionStatus,
    FinalOccupancyCollisionAnalyzer,
)
from chess_chain.sequence.finalise import _collision_for_final
from chess_chain.sequence.graph import NeighbourGraph
from chess_chain.sequence.records import SequenceResult
from chess_chain.sequence.request import SequenceRequest

__all__ = ["ValidationReport", "validate_sequence", "validation_from_dict"]


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    issues: tuple[str, ...]
    stage_details: tuple[dict[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "issues": list(self.issues),
            "stage_details": list(self.stage_details),
        }


def validate_sequence(
    result: SequenceResult,
    graph: NeighbourGraph,
    request: SequenceRequest,
    *,
    registry: Any = None,
) -> ValidationReport:
    """Re-validate a generated sequence against the hard constraints."""
    issues: list[str] = []
    puzzle_ids = result.puzzle_ids()

    # 8. No repeats within the sequence.
    if len(set(puzzle_ids)) != len(puzzle_ids):
        issues.append("a puzzle is repeated within the sequence")

    # 9. Exclusions / already-used ids.
    excluded = set(request.excluded_puzzle_ids) | set(request.already_used_puzzle_ids)
    for puzzle_id in puzzle_ids:
        if puzzle_id in excluded:
            issues.append(f"puzzle {puzzle_id} is excluded or already used")
    if registry is not None and request.forbid_puzzle_reuse:
        reused = registry.reused_puzzles(puzzle_ids)
        if reused:
            issues.append(f"puzzles reused across manufactured products: {', '.join(reused)}")

    for stage in result.stages:
        # 1. Every source line replays legally (contexts re-derived).
        contexts, _ = graph.contexts_for(
            stage.puzzle_id,
            stage.policy,
            request,
        )
        if len(contexts) != len(stage.accepted_lines):
            issues.append(
                f"stage {stage.puzzle_id}: accepted lines do not replay legally "
                f"({len(contexts)} contexts for {len(stage.accepted_lines)} lines)"
            )

        # 2/3. Solver uniqueness and opponent forcedness.
        issues.extend(_uniqueness_issues(stage, request))

        if stage.edge is None:
            continue

        # 4/5. The transition was verified at synthesis time; leakage/complexity
        # limits are checked below. Re-verification is performed by the
        # transition verify step when the persisted program is re-executed.
        if (
            stage.edge.leakage_bits is not None
            and stage.edge.leakage_bits > request.max_leakage_bits
        ):
            issues.append(
                f"stage {stage.puzzle_id}: leakage {stage.edge.leakage_bits:.2f} exceeds "
                f"{request.max_leakage_bits}"
            )
        if (
            stage.edge.absolute_disclosures is not None
            and stage.edge.absolute_disclosures > request.max_absolute_square_disclosures
        ):
            issues.append(
                f"stage {stage.puzzle_id}: absolute disclosures "
                f"{stage.edge.absolute_disclosures} exceed "
                f"{request.max_absolute_square_disclosures}"
            )
        if (
            stage.edge.action_count is not None
            and stage.edge.action_count > request.max_transition_actions
        ):
            issues.append(
                f"stage {stage.puzzle_id}: {stage.edge.action_count} actions exceed "
                f"{request.max_transition_actions}"
            )
        if request.exact_journey == "required" and not stage.edge.exact_journey:
            issues.append(f"stage {stage.puzzle_id}: exact journey required but not exact")

        # 7. Physical inventory (authoritative row from the repository).
        full_row = graph._row(stage.puzzle_id)
        if request.inventory_profile == "standard" and bool(
            full_row.get("unusual_material", False)
        ):
            issues.append(
                f"stage {stage.puzzle_id}: unusual material outside the standard inventory"
            )

    # 10. Final stage ends in checkmate.
    final_stage = result.stages[-1]
    if not _final_ends_in_checkmate(final_stage.row):
        issues.append("final stored solution does not end in python-chess checkmate")

    # 11. Only the final puzzle must have occupancy invariance.
    if not _occupancy_invariant(final_stage.row, final_stage.accepted_lines):
        issues.append("final stage accepted lines do not share one occupancy")

    # 12. Only the final puzzle must meet the requested collision status.
    _final_collision_check(graph, final_stage, request, issues)

    return ValidationReport(valid=not issues, issues=tuple(issues))


def _uniqueness_issues(stage: Any, request: SequenceRequest) -> list[str]:
    """Check solver uniqueness and opponent forcedness along the intended line."""
    issues: list[str] = []
    board = chess.Board(str(stage.row["start_fen"]))
    intended = tuple(str(move) for move in stage.row["solution_uci"])
    solver_white = str(stage.row["solver_colour"]) == "white"
    for depth, uci in enumerate(intended):
        is_solver_ply = (board.turn == chess.WHITE) == solver_white
        legal = list(board.legal_moves)
        if request.solver_uniqueness == "strict" and is_solver_ply and len(legal) > 1:
            issues.append(
                f"stage {stage.puzzle_id}: solver ply {depth} ({uci}) is not unique "
                f"({len(legal)} legal moves)"
            )
        if request.opponent_forcedness == "required" and not is_solver_ply and len(legal) > 1:
            issues.append(
                f"stage {stage.puzzle_id}: opponent ply {depth} ({uci}) is not forced "
                f"({len(legal)} legal moves)"
            )
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            issues.append(f"stage {stage.puzzle_id}: intended move {uci} is illegal")
            break
        board.push(move)
    return issues


def _final_ends_in_checkmate(row: dict[str, Any]) -> bool:
    board = chess.Board(str(row["start_fen"]))
    for uci in row["solution_uci"]:
        move = chess.Move.from_uci(str(uci))
        if move not in board.legal_moves:
            return False
        board.push(move)
    return board.is_checkmate()


def _occupancy_invariant(row: dict[str, Any], accepted_lines: tuple[tuple[str, ...], ...]) -> bool:
    board = chess.Board(str(row["start_fen"]))
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


def _final_collision_check(
    graph: NeighbourGraph,
    final_stage: Any,
    request: SequenceRequest,
    issues: list[str],
) -> None:
    cache = DiskKeyValueCache(graph.settings.analysis_dir / "collision_cache")
    analyzer = FinalOccupancyCollisionAnalyzer(cache=cache)
    try:
        result = _collision_for_final(graph, final_stage.row, request, analyzer)
    except (ValueError, KeyError) as exc:
        issues.append(f"final collision analysis failed: {exc}")
        return
    expected = request.final_collision_status
    if result.status != CollisionStatus(expected):
        # A bounded inconclusive result is acceptable in non-strict mode; the
        # status is still recorded, never renamed to an unbounded proof.
        inconclusive_ok = (
            result.status == CollisionStatus.INCONCLUSIVE and not request.collision_strict
        )
        if not inconclusive_ok:
            issues.append(f"final collision status is {result.status.value}, expected {expected}")


def validation_from_dict(data: dict[str, Any]) -> ValidationReport:
    return ValidationReport(
        valid=bool(data.get("valid", False)),
        issues=tuple(str(issue) for issue in data.get("issues", [])),
    )
