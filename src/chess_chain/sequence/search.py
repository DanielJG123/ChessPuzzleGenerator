"""Bounded beam search with backtracking for sequence construction (stage 6).

The search strategy sits behind a ``SequenceSearcher`` protocol so an
OR-Tools/CP-SAT or other solver can be added later without changing the graph
or finalist modules. The default implementation is a deterministic beam search
(width configurable) that keeps, per depth, a frontier of scored partial paths
plus per-path iterators over ranked neighbour edges. A dead end - or a failed
final selection - triggers bounded backtracking which simply consumes the
next-ranked neighbours of the previous frontier, so regeneration from the same
seed and configuration is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from chess_chain.analysis.policies import StageEndPolicy
from chess_chain.sequence.graph import EdgeAttempt, NeighbourGraph
from chess_chain.sequence.request import SequenceRequest

__all__ = [
    "PathState",
    "SearchReport",
    "SearchStage",
    "SequenceSearcher",
    "BeamSearchSearcher",
    "score_path",
]


@dataclass(frozen=True)
class SearchStage:
    """One node on a sequence path."""

    puzzle_id: str
    row: dict[str, Any]
    policy: StageEndPolicy
    accepted_lines: tuple[tuple[str, ...], ...]
    truncated: bool
    edge: EdgeAttempt | None = None  # the verified transition INTO this stage


@dataclass
class SearchReport:
    """Concise accounting for one search run."""

    expanded_nodes: int = 0
    edges_attempted: int = 0
    edges_rejected: list[tuple[str, str, str]] = field(default_factory=list)
    backtrack_steps: int = 0
    dead_end: bool = False
    no_path: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "expanded_nodes": self.expanded_nodes,
            "edges_attempted": self.edges_attempted,
            "edges_rejected": [
                {"from": src, "to": dst, "reason": reason}
                for src, dst, reason in self.edges_rejected
            ],
            "backtrack_steps": self.backtrack_steps,
            "dead_end": self.dead_end,
            "no_path": self.no_path,
        }


class PathState:
    """A partial sequence with a stateful iterator over its neighbour edges."""

    def __init__(
        self,
        stages: tuple[SearchStage, ...],
        *,
        pending: list[EdgeAttempt],
        score: float,
        target_rating: int | None,
    ) -> None:
        self.stages = stages
        self.pending = pending
        self.cursor = 0
        self.score = score
        self._target_rating = target_rating

    def puzzle_ids(self) -> frozenset[str]:
        return frozenset(stage.puzzle_id for stage in self.stages)

    def last(self) -> SearchStage:
        return self.stages[-1]

    def consume_edges(
        self,
        count: int,
        *,
        graph: NeighbourGraph,
        request: SequenceRequest,
        report: SearchReport,
    ) -> list[EdgeAttempt]:
        """Pull the next ``count`` ranked neighbour edges (bounded by the
        cached per-node list). Backtracking advances this cursor later."""
        if self.cursor == 0:
            self.pending = graph.edges(
                self.last().puzzle_id,
                request=request,
                used_ids=self.puzzle_ids(),
            )
            report.expanded_nodes += 1
        start = self.cursor
        end = min(start + count, len(self.pending))
        self.cursor = end
        return self.pending[start:end]

    def extend(self, edge: EdgeAttempt, stage: SearchStage) -> PathState:
        return PathState(
            self.stages + (stage,),
            pending=[],
            score=score_path(self.stages + (stage,), self._target_rating),
            target_rating=self._target_rating,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "puzzle_ids": [stage.puzzle_id for stage in self.stages],
            "score": round(self.score, 4),
        }


class SequenceSearcher(Protocol):
    """Interface for sequence search strategies (beam search today; other
    solvers can implement the same protocol later)."""

    def search(
        self,
        graph: NeighbourGraph,
        request: SequenceRequest,
        start_puzzle_id: str,
    ) -> tuple[list[PathState], Any, SearchReport]:
        """Return ``(frontier, final_selection, report)``; ``final_selection``
        is None when no sequence satisfies the constraints within the budget."""
        ...


def score_path(stages: tuple[SearchStage, ...], target_rating: int | None) -> float:
    """Transparent weighted score; lower is better.

    Preferences (in order): reliable recognisability (always satisfied for
    verified transitions), requested difficulty (rating proximity), exact
    continuous journey, occupancy continuity, minimal actions, low leakage,
    low absolute disclosure, varied themes, minimal setup burden.
    """
    score = 0.0
    themes_seen: set[str] = set()
    for stage in stages:
        themes = tuple(str(t) for t in stage.row.get("themes", []))
        themes_seen.update(themes)
        if stage.edge is not None:
            score += _edge_cost(stage.edge, target_rating)
    score -= len(themes_seen) * 0.5
    for stage in stages:
        score += int(stage.row.get("presented_piece_count", 32)) * 0.05
    return score


def _edge_cost(edge: EdgeAttempt, target_rating: int | None) -> float:
    cost = 0.0
    if target_rating is not None and edge.next_rating is not None:
        cost += abs(edge.next_rating - target_rating) / 500.0
    if edge.exact_journey:
        cost -= 4.0
    cost += (edge.d_occ_best or 0.0) * 0.5
    cost += (edge.action_count or 0) * 2.0
    cost += (edge.leakage_bits or 0.0) * 0.05
    cost += (edge.absolute_disclosures or 0) * 3.0
    return cost


class BeamSearchSearcher:
    """Deterministic beam search with configurable width and bounded
    backtracking over ranked per-node neighbour edges.

    Ties are broken with a hash of ``(seed, path)`` so the same seed reproduces
    the same output while different seeds can pick different valid sequences
    when choices exist. Final-stage selection is driven by an injected
    ``finaliser`` callback (lazy collision validation); a failed final
    selection backtracks into the intermediate frontier just like a dead end.
    """

    def __init__(
        self,
        width: int | None = None,
        per_node: int | None = None,
        finaliser: Any = None,
    ) -> None:
        self.width = width
        self.per_node = per_node
        self.finaliser = finaliser

    def search(
        self,
        graph: NeighbourGraph,
        request: SequenceRequest,
        start_puzzle_id: str,
    ) -> tuple[list[PathState], Any, SearchReport]:
        width = self.width or request.beam_width
        report = SearchReport()
        seed = request.random_seed

        start_row = graph._row(start_puzzle_id)
        start_policy = graph._source_policy(request, for_final=False)
        start_lines, start_truncated = graph.accepted_lines(start_puzzle_id, start_policy, request)
        start_stage = SearchStage(
            puzzle_id=start_puzzle_id,
            row=start_row,
            policy=start_policy,
            accepted_lines=start_lines,
            truncated=start_truncated,
        )
        start_path = PathState(
            (start_stage,),
            pending=[],
            score=0.0,
            target_rating=request.target_rating,
        )

        # Depth-first expansion with bounded backtracking: each expanded path
        # retains its top-``width`` ranked children (the beam); a dead end - or
        # a failed final selection - pops the next sibling at the deepest
        # unexhausted level. The backtrack budget bounds the total work.
        stack: list[tuple[PathState, list[PathState], int]] = []
        current = start_path
        depth = 1
        budget = request.backtrack_budget
        selection: Any = None

        while True:
            if depth < request.num_stages - 1:
                per_node = self.per_node or request.per_node_candidates
                edges = current.consume_edges(per_node, graph=graph, request=request, report=report)
                report.edges_attempted += len(edges)
                children: list[PathState] = []
                for edge in edges:
                    stage = self._stage_for(graph, edge, request)
                    if stage is None:
                        continue
                    children.append(current.extend(edge, stage))
                children.sort(key=lambda path: (path.score, _tie_break(path, seed)))
                retained = children[:width]
                stack.append((current, retained, 0))
                if retained:
                    current = retained[0]
                    depth = len(current.stages)
                    continue
            else:
                if self.finaliser is not None:
                    selection = self.finaliser([current])
                if selection is not None:
                    return [current], selection, report

            # Dead end (intermediate) or failed final selection: backtrack.
            report.dead_end = True
            if budget <= 0:
                report.no_path = True
                return [], None, report
            budget -= 1
            report.backtrack_steps += 1
            moved = False
            while stack:
                parent, siblings, index = stack[-1]
                if index + 1 < len(siblings):
                    stack[-1] = (parent, siblings, index + 1)
                    current = siblings[index + 1]
                    depth = len(current.stages)
                    moved = True
                    break
                stack.pop()
            if not moved:
                report.no_path = True
                return [], None, report

    def _stage_for(
        self,
        graph: NeighbourGraph,
        edge: EdgeAttempt,
        request: SequenceRequest,
    ) -> SearchStage | None:
        row = graph._row(edge.next_puzzle_id)
        policy = graph._source_policy(request, for_final=False)
        lines, truncated = graph.accepted_lines(edge.next_puzzle_id, policy, request)
        return SearchStage(
            puzzle_id=edge.next_puzzle_id,
            row=row,
            policy=policy,
            accepted_lines=lines,
            truncated=truncated,
            edge=edge,
        )


def _tie_break(path: PathState, seed: int) -> str:
    import hashlib

    payload = f"{seed}:{path.to_dict()['puzzle_ids']}".encode()
    return hashlib.sha256(payload).hexdigest()
