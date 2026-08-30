"""Top-level sequence generation orchestration (stage 6).

``generate_sequence`` resolves a start puzzle, runs the bounded beam search
with backtracking over on-demand neighbour edges, lazily selects and validates
the final stage (collision analysis only for finalists, in rank order), builds
the canonical ``sequence@1`` document, computes the stable fingerprint,
validates every hard constraint and optionally registers the product.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from chess_chain.config import Settings
from chess_chain.sequence.finalise import FinalSelection, select_final
from chess_chain.sequence.fingerprint import sequence_fingerprint
from chess_chain.sequence.graph import NeighbourGraph
from chess_chain.sequence.records import (
    SequenceResult,
    SequenceStageRecord,
)
from chess_chain.sequence.registry import SequenceRegistry
from chess_chain.sequence.request import SequenceRequest, request_fingerprint
from chess_chain.sequence.search import BeamSearchSearcher, PathState, SearchStage
from chess_chain.sequence.validate import validate_sequence

__all__ = ["SequenceGenerationError", "generate_sequence"]


class SequenceGenerationError(RuntimeError):
    """Raised when no sequence satisfies the request constraints."""

    def __init__(self, message: str, report: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.report = report


def generate_sequence(
    request: SequenceRequest,
    settings: Settings,
    repo: Any,
    *,
    index: Any = None,
    registry: SequenceRegistry | None = None,
    register: bool = True,
    searcher: Any = None,
    collision_analyzer: Any = None,
    collision_run_log: list[str] | None = None,
    edge_cache: Any = None,
) -> SequenceResult:
    """Generate one validated sequence for ``request`` (deterministic)."""
    transition_config = dataclasses.replace(
        settings.transition_config,
        default_max_actions=request.max_transition_actions,
    )
    graph = NeighbourGraph(
        settings,
        repo,
        transition_config=transition_config,
        index=index,
        edge_cache=edge_cache,
    )

    start_puzzle_id = _resolve_start(request, repo)
    searcher = searcher or BeamSearchSearcher()

    def _finalise(frontier: list[Any]) -> Any:
        return select_final(
            graph,
            frontier,
            request,
            analyzer=collision_analyzer,
            run_log=collision_run_log,
        )

    if hasattr(searcher, "finaliser"):
        searcher.finaliser = _finalise
    frontier, selection, search_report = searcher.search(graph, request, start_puzzle_id)
    if selection is None:
        raise SequenceGenerationError(
            f"no sequence found (search report: {json.dumps(search_report.to_dict())})",
            report=search_report,
        )

    result = _build_result(request, graph, selection, search_report)
    validation = validate_sequence(result, graph, request, registry=registry)
    result = SequenceResult(
        request=result.request,
        fingerprint=result.fingerprint,
        stages=result.stages,
        validation=validation,
        search_report=result.search_report,
        product_id=result.product_id,
        versions=_collect_versions(request, settings),
    )

    if registry is not None:
        if register and not registry.is_registered(request.product_id):
            if request.forbid_puzzle_reuse:
                reused = registry.reused_puzzles(result.puzzle_ids())
                if reused:
                    raise SequenceGenerationError(
                        f"puzzle reuse across manufactured products: {', '.join(reused)}"
                    )
            registry.register(
                product_id=request.product_id,
                fingerprint=result.fingerprint,
                puzzle_ids=result.puzzle_ids(),
                request_summary=request.to_dict(),
            )
        elif registry.is_registered(request.product_id):
            existing = registry.entries[request.product_id]
            if str(existing.get("fingerprint")) != result.fingerprint:
                raise SequenceGenerationError(
                    f"product {request.product_id} is already registered with a "
                    "different fingerprint"
                )
    return result


def _resolve_start(request: SequenceRequest, repo: Any) -> str:
    if request.start_puzzle_id is not None:
        return request.start_puzzle_id
    # Deterministic start pick from the rating band (seeded, bounded).
    import random

    rng = random.Random(request.random_seed)
    sql = "SELECT puzzle_id FROM puzzles"
    conditions: list[str] = []
    params: list[object] = []
    if request.rating_min is not None:
        conditions.append("rating >= ?")
        params.append(request.rating_min)
    if request.rating_max is not None:
        conditions.append("rating <= ?")
        params.append(request.rating_max)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY puzzle_id LIMIT 200"
    rows = repo.query(sql, params)
    candidates = [str(row[0]) for row in rows.iter_rows()]
    if not candidates:
        raise SequenceGenerationError("no start puzzle matches the request constraints")
    return rng.choice(candidates)


def _build_result(
    request: SequenceRequest,
    graph: NeighbourGraph,
    selection: FinalSelection,
    search_report: Any,
) -> SequenceResult:
    path: PathState = selection.path
    records: list[SequenceStageRecord] = []
    for index, stage in enumerate(path.stages):
        records.append(
            SequenceStageRecord(
                index=index,
                puzzle_id=stage.puzzle_id,
                is_final=False,
                end_policy_key=stage.policy.key(),
                accepted_lines=stage.accepted_lines,
                truncated=stage.truncated,
                row=stage.row,
                edge=stage.edge,
            )
        )
    final_stage: SearchStage = selection.stage
    records.append(
        SequenceStageRecord(
            index=len(records),
            puzzle_id=final_stage.puzzle_id,
            is_final=True,
            end_policy_key=final_stage.policy.key(),
            accepted_lines=final_stage.accepted_lines,
            truncated=final_stage.truncated,
            row=final_stage.row,
            edge=selection.edge,
            final_selection=selection,
        )
    )

    request_key = request_fingerprint(request)
    stage_summaries = tuple(
        {
            "puzzle_id": record.puzzle_id,
            "symmetry": record.edge.symmetry if record.edge is not None else None,
            "accepted_lines": record.accepted_lines,
            "transition_ast": (
                record.edge.program.to_json()
                if record.edge is not None and record.edge.program is not None
                else None
            ),
        }
        for record in records
    )
    fingerprint = sequence_fingerprint(request_key, stage_summaries)
    return SequenceResult(
        request=request,
        fingerprint=fingerprint,
        stages=tuple(records),
        validation=None,
        search_report=search_report,
        product_id=request.product_id,
    )


def _collect_versions(request: SequenceRequest, settings: Settings) -> dict[str, int]:
    """Analysis/config versions at generation time for stale-version checks."""
    return {
        "request": int(request.version),
        "transition": int(settings.transition_config.version),
        "collision_analyser": int(settings.collision_analyser_version),
        "similarity": int(settings.similarity_version),
        "transform": int(settings.transform_version),
    }
