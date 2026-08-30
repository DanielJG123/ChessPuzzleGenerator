"""Lazy final-candidate selection (stage 6).

Collision analysis never runs for every puzzle and never for intermediate
stages: finalists are ranked first using cheap/deep cached features, then each
finalist is checked in rank order against the persistent final-collision cache
with the complete scope/configuration key; a miss runs
``FinalOccupancyCollisionAnalyzer`` for that candidate only. Collisions are
rejected, and under strict policy inconclusive results are rejected too. A
configurable number of validated finalist alternatives is retained so the
earlier path search is not committed to a single brittle ending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess

from chess_chain.analysis.cache import DiskKeyValueCache
from chess_chain.analysis.collision import (
    CollisionRequest,
    CollisionResult,
    CollisionStatus,
    FinalOccupancyCollisionAnalyzer,
)
from chess_chain.analysis.policies import get_policy
from chess_chain.domain.position import PositionFingerprint
from chess_chain.sequence.graph import EdgeAttempt, NeighbourGraph
from chess_chain.sequence.request import SequenceRequest
from chess_chain.sequence.search import PathState, SearchStage

__all__ = ["FinalSelection", "FinalistRank", "select_final"]


@dataclass(frozen=True)
class FinalistRank:
    """One finalist's cheap-feature rank (before any collision analysis)."""

    puzzle_id: str
    rank: int
    edge: EdgeAttempt

    def to_dict(self) -> dict[str, object]:
        return {
            "puzzle_id": self.puzzle_id,
            "rank": self.rank,
            "rating": self.edge.next_rating,
            "themes": list(self.edge.next_themes),
            "actions": self.edge.action_count,
            "leakage_bits": self.edge.leakage_bits,
            "absolute_disclosures": self.edge.absolute_disclosures,
        }


@dataclass(frozen=True)
class FinalSelection:
    """The validated final stage plus its bounded collision report."""

    stage: SearchStage
    edge: EdgeAttempt
    collision: CollisionResult
    rank: FinalistRank
    alternatives: tuple[SearchStage, ...]
    final_ends_in_checkmate: bool
    final_occupancy_invariant: bool
    final_accepted_lines: tuple[tuple[str, ...], ...]
    path: Any = None  # the PathState that led to this finalist

    def to_dict(self) -> dict[str, object]:
        return {
            "puzzle_id": self.stage.puzzle_id,
            "rank": self.rank.to_dict(),
            "collision_status": self.collision.status.value,
            "collision_cached": self.collision.cached,
            "collision_report": self.collision.to_dict(),
            "final_ends_in_checkmate": self.final_ends_in_checkmate,
            "final_occupancy_invariant": self.final_occupancy_invariant,
            "alternatives": [
                {
                    "puzzle_id": stage.puzzle_id,
                    "edge": stage.edge.to_dict() if stage.edge else None,
                }
                for stage in self.alternatives
            ],
        }


def select_final(
    graph: NeighbourGraph,
    frontier: list[PathState],
    request: SequenceRequest,
    *,
    analyzer: FinalOccupancyCollisionAnalyzer | None = None,
    run_log: list[str] | None = None,
) -> FinalSelection | None:
    """Rank finalists by cached features, then lazily validate collisions."""
    if analyzer is None:
        cache = DiskKeyValueCache(graph.settings.analysis_dir / "collision_cache")
        analyzer = FinalOccupancyCollisionAnalyzer(cache=cache)

    finalists: list[tuple[PathState, EdgeAttempt]] = []
    seen: set[str] = set()
    for path in sorted(frontier, key=lambda p: p.score):
        edges = graph.edges(
            path.last().puzzle_id,
            request=request,
            used_ids=path.puzzle_ids(),
            for_final=True,
        )
        for edge in edges:
            if edge.next_puzzle_id in seen:
                continue
            seen.add(edge.next_puzzle_id)
            finalists.append((path, edge))

    # Rank by cheap/deep cached features (rating, quality, transition cost,
    # journey, theme variety) WITHOUT running collision analysis yet.
    ranked = sorted(finalists, key=lambda item: _finalist_key(item[1], item[0], request))
    ranked = ranked[: request.max_finalists_checked]
    ranks = {
        edge.next_puzzle_id: FinalistRank(puzzle_id=edge.next_puzzle_id, rank=index, edge=edge)
        for index, (_, edge) in enumerate(ranked)
    }

    validated: list[tuple[int, PathState, EdgeAttempt, CollisionResult]] = []
    for index, (path, edge) in enumerate(ranked):
        row = graph._row(edge.next_puzzle_id)
        result = _collision_for_final(graph, row, request, analyzer)
        if run_log is not None:
            run_log.append(f"{edge.next_puzzle_id}:{index}:{result.status.value}")
        if result.status == CollisionStatus.COLLISION_FOUND:
            continue
        if result.status == CollisionStatus.INCONCLUSIVE and request.collision_strict:
            continue
        stage = _final_stage(graph, edge, row, request)
        if stage is None:
            continue
        accepted_lines, _ = _final_accepted_lines(graph, row, request)
        # Constraint 11: only the final puzzle must have occupancy invariance
        # across accepted lines - a varying-occupancy final is rejected.
        if not _occupancy_invariant(row, accepted_lines):
            if run_log is not None:
                run_log.append(f"{edge.next_puzzle_id}:{index}:occupancy_variant")
            continue
        validated.append((index, path, edge, result))
        if len(validated) >= request.validated_finalist_alternatives:
            break

    if not validated:
        return None

    validated.sort(key=lambda item: item[0])
    chosen_edge = validated[0][2]
    chosen_collision = validated[0][3]
    alternatives: list[SearchStage] = []
    for _, _, edge, _ in validated[1:]:
        stage = _final_stage(graph, edge, graph._row(edge.next_puzzle_id), request)
        if stage is not None:
            alternatives.append(stage)
    stage = _final_stage(graph, chosen_edge, graph._row(chosen_edge.next_puzzle_id), request)
    if stage is None:
        return None
    accepted_lines, truncated = _final_accepted_lines(graph, stage.row, request)
    return FinalSelection(
        stage=stage,
        edge=chosen_edge,
        collision=chosen_collision,
        rank=ranks[chosen_edge.next_puzzle_id],
        alternatives=tuple(alternatives),
        final_ends_in_checkmate=_final_ends_in_checkmate(stage.row),
        final_occupancy_invariant=_occupancy_invariant(stage.row, accepted_lines),
        final_accepted_lines=accepted_lines,
        path=validated[0][1],
    )


def _finalist_key(
    edge: EdgeAttempt, path: PathState, request: SequenceRequest
) -> tuple[float, ...]:
    """Cheap-feature rank: requested difficulty first, then transition cost,
    journey quality, popularity and theme variety (no collision analysis)."""
    target = request.target_rating
    rating_dist = abs((edge.next_rating or 0) - target) / 500.0 if target is not None else 0.0
    transition_cost = (edge.action_count or 0) * 2.0 + (edge.leakage_bits or 0.0) * 0.05
    transition_cost += (edge.absolute_disclosures or 0) * 3.0
    journey = -4.0 if edge.exact_journey else (edge.d_occ_best or 0.0) * 0.5
    popularity = -(edge.next_popularity or 0) * 0.01
    path_themes = {str(t) for stage in path.stages for t in stage.row.get("themes", [])}
    new_themes = len(set(edge.next_themes) - path_themes)
    return (rating_dist, transition_cost, journey, popularity, -new_themes)


def _final_accepted_lines(
    graph: NeighbourGraph, row: dict[str, Any], request: SequenceRequest
) -> tuple[tuple[tuple[str, ...], ...], bool]:
    from chess_chain.sequence.graph import accepted_lines_for_puzzle

    return accepted_lines_for_puzzle(row, get_policy("checkmate"), request)


def _collision_for_final(
    graph: NeighbourGraph,
    row: dict[str, Any],
    request: SequenceRequest,
    analyzer: FinalOccupancyCollisionAnalyzer,
) -> CollisionResult:
    """Run (or reuse from cache) the final-only occupancy-collision analysis
    for one final candidate, using the complete scope/configuration key. The
    authoritative full row is always fetched from the repository."""
    full_row = graph._row(str(row["puzzle_id"]))
    board = chess.Board(str(full_row["start_fen"]))
    accepted_lines, _ = _final_accepted_lines(graph, full_row, request)
    terminals = [_terminal(board, line) for line in accepted_lines]
    target = int(terminals[0].occupied) if terminals else int(full_row["final_occupancy"])
    collision_request = CollisionRequest(
        puzzle_id=str(full_row["puzzle_id"]),
        source_version=int(full_row["transform_version"]),
        presented_fingerprint=PositionFingerprint.from_board(board).to_key(),
        target_occupancy=target,
        accepted_line_policy=request.acceptance,
        stage_end_policy=get_policy("checkmate").key(),
        scope=request.collision_scope,
        max_depth=request.collision_max_depth,
        max_nodes=request.collision_max_nodes,
        pruning_config="transposition",
        analyser_version=int(graph.settings.collision_analyser_version),
    )
    return analyzer.analyze(
        request=collision_request,
        board=board,
        accepted_lines=accepted_lines,
    )


def _terminal(board: chess.Board, line: tuple[str, ...]) -> chess.Board:
    current = board.copy()
    for uci in line:
        move = chess.Move.from_uci(uci)
        if move not in current.legal_moves:
            raise ValueError(f"illegal final line move {uci!r}")
        current.push(move)
    return current


def _final_stage(
    graph: NeighbourGraph,
    edge: EdgeAttempt,
    row: dict[str, Any],
    request: SequenceRequest,
) -> SearchStage | None:
    policy = get_policy("checkmate")
    from chess_chain.sequence.search import SearchStage

    lines, truncated = _final_accepted_lines(graph, row, request)
    if not lines:
        return None
    return SearchStage(
        puzzle_id=edge.next_puzzle_id,
        row=row,
        policy=policy,
        accepted_lines=lines,
        truncated=truncated,
        edge=edge,
    )


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
    occupancies = {int(_terminal(board, line).occupied) for line in accepted_lines}
    return len(occupancies) <= 1
