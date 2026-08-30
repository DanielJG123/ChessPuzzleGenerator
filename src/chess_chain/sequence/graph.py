"""On-demand neighbour generation for sequence search (stage 6).

Puzzles are nodes and synthesised transitions are directed edges. No global
graph over the corpus is materialised: neighbours are retrieved per node by
querying the similarity index for candidate next starts in the requested rating
band, running exact piece assignment, attempting transition synthesis, and
caching successful and failed edge attempts by the complete version/config
fingerprint. Every returned edge carries decomposed costs and rejection
reasons.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess

from chess_chain.analysis.cache import DiskKeyValueCache
from chess_chain.analysis.policies import (
    CHECKMATE,
    EXPLICIT_PREDICATE,
    FIXED_PLIES,
    TRANSITION_READY,
    StageEndPolicy,
    evaluate_predicate,
    get_policy,
)
from chess_chain.analysis.variations import enumerate_accepted_lines
from chess_chain.domain.pieces import MaterialSignature
from chess_chain.domain.position import CanonicalPosition
from chess_chain.sequence.request import SequenceRequest
from chess_chain.similarity.backend import get_backend
from chess_chain.similarity.candidate import PairCandidate
from chess_chain.similarity.retrieval import RetrievalStats, retrieve
from chess_chain.transitions import ast as A
from chess_chain.transitions.config import TransitionConfig
from chess_chain.transitions.context import build_context_from_boards
from chess_chain.transitions.synth import synthesize


@dataclass(frozen=True)
class EdgeAttempt:
    """One source -> next edge: a verified transition (or the rejection)."""

    source_puzzle_id: str
    next_puzzle_id: str
    end_policy_key: str
    ok: bool
    reason: str | None = None

    # Verified transition (when ok).
    program: A.TransitionProgram | None = None
    verification: Any | None = None
    rendered: str = ""
    symmetry: str = "identity"
    action_count: int | None = None
    leakage_bits: float | None = None
    absolute_disclosures: int | None = None
    d_occ_best: float | None = None
    cost_best: float | None = None
    exact_journey: bool = False

    # Persisted verification summary (self-contained records for rendering).
    clue: dict[str, object] | None = None
    guard: dict[str, object] | None = None
    verification_issues: tuple[str, ...] = ()

    # Candidate quality facts.
    next_rating: int | None = None
    next_themes: tuple[str, ...] = ()
    next_piece_count: int | None = None
    next_popularity: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_puzzle_id": self.source_puzzle_id,
            "next_puzzle_id": self.next_puzzle_id,
            "end_policy_key": self.end_policy_key,
            "ok": self.ok,
            "reason": self.reason,
            "program": self.program.to_json() if self.program is not None else None,
            "rendered": self.rendered,
            "symmetry": self.symmetry,
            "action_count": self.action_count,
            "leakage_bits": self.leakage_bits,
            "absolute_disclosures": self.absolute_disclosures,
            "d_occ_best": self.d_occ_best,
            "cost_best": self.cost_best,
            "exact_journey": self.exact_journey,
            "clue": self.clue,
            "guard": self.guard,
            "verification_issues": list(self.verification_issues),
            "next_rating": self.next_rating,
            "next_themes": list(self.next_themes),
            "next_piece_count": self.next_piece_count,
            "next_popularity": self.next_popularity,
        }


class EdgeCache:
    """Persistent cache of edge attempts keyed by the complete configuration."""

    def __init__(self, directory: Path | None = None) -> None:
        self._store = DiskKeyValueCache(directory) if directory is not None else None
        self._memory: dict[str, dict[str, object]] = {}

    def _key(
        self,
        *,
        source: str,
        next_puzzle_id: str,
        end_policy_key: str,
        acceptance: str,
        max_plies: int,
        max_nodes: int,
        max_accepted_lines: int,
        transition_config: TransitionConfig,
    ) -> str:
        return DiskKeyValueCache.key_from(
            {
                "edge_attempt_version": 1,
                "source": source,
                "next": next_puzzle_id,
                "end_policy": end_policy_key,
                "acceptance": acceptance,
                "max_plies": max_plies,
                "max_nodes": max_nodes,
                "max_accepted_lines": max_accepted_lines,
                "transition_config": transition_config.as_dict(),
            }
        )

    def get(self, key: str) -> dict[str, object] | None:
        if key in self._memory:
            return self._memory[key]
        if self._store is not None:
            stored = self._store.get(key)
            if stored is not None:
                self._memory[key] = stored
                return stored
        return None

    def put(self, key: str, value: dict[str, object]) -> None:
        self._memory[key] = value
        if self._store is not None:
            self._store.put(key, value)


def _endpoint_satisfies(end_policy: StageEndPolicy, board: chess.Board, line_length: int) -> bool:
    kind = end_policy.kind
    if kind == CHECKMATE:
        return board.is_checkmate()
    if kind == EXPLICIT_PREDICATE:
        assert end_policy.predicate is not None
        return evaluate_predicate(board, end_policy.predicate)
    if kind == FIXED_PLIES:
        assert end_policy.fixed_plies is not None
        return line_length == end_policy.fixed_plies
    if kind == TRANSITION_READY:
        return True  # the transition itself is the guard; verified later
    raise ValueError(f"unknown stage-end kind {kind!r}")


def accepted_lines_for_puzzle(
    row: dict[str, Any],
    end_policy: StageEndPolicy,
    request: SequenceRequest,
) -> tuple[tuple[tuple[str, ...], ...], bool]:
    """Enumerate the accepted lines for a puzzle under a stage-end policy.

    The accepted set is the bounded enumeration (``acceptance``,
    ``max_plies``, ``max_nodes``) whose endpoints satisfy the policy, capped at
    ``max_accepted_lines``. Truncation is recorded so search reports can say
    the accepted set was bounded rather than silently complete.
    """
    board = chess.Board(str(row["start_fen"]))
    intended = tuple(str(move) for move in row["solution_uci"])
    lines, truncated = enumerate_accepted_lines(
        board,
        intended_uci=intended,
        acceptance=request.acceptance,
        max_plies=request.max_plies,
        max_nodes=request.max_nodes,
    )
    accepted: list[tuple[str, ...]] = []
    for line in lines:
        current = board.copy()
        for uci in line:
            move = chess.Move.from_uci(uci)
            if move not in current.legal_moves:
                continue
            current.push(move)
        if _endpoint_satisfies(end_policy, current, len(line)):
            accepted.append(line)
    accepted.sort()
    if len(accepted) > request.max_accepted_lines:
        accepted = accepted[: request.max_accepted_lines]
        truncated = True
    return tuple(accepted), truncated


def _fen_after(fen: str, moves: tuple[str, ...]) -> str:
    board = chess.Board(fen)
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
    return board.fen()


class NeighbourGraph:
    """On-demand node -> neighbour-edge retrieval with a persistent attempt cache.

    ``index`` and ``backend`` may be injected (tests) or built from settings on
    first use. All puzzle rows are fetched through the repository; no global
    graph is ever materialised.
    """

    def __init__(
        self,
        settings: Any,
        repo: Any,
        *,
        transition_config: TransitionConfig,
        index: Any = None,
        backend: Any = None,
        edge_cache: EdgeCache | None = None,
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.transition_config = transition_config
        self._index = index
        self._backend = backend
        directory = getattr(settings, "sequence_dir", None)
        self.edge_cache = edge_cache or EdgeCache(directory / "edge_cache" if directory else None)
        self._context_cache: dict[
            tuple[str, str, str, int, int], tuple[tuple[tuple[str, ...], ...], bool]
        ] = {}
        self._row_cache: dict[str, dict[str, Any]] = {}
        self._attempts_cache: dict[tuple[str, str], EdgeAttempt] = {}
        self.retrieval_stats: list[RetrievalStats] = []

    def _row(self, puzzle_id: str) -> dict[str, Any]:
        if puzzle_id not in self._row_cache:
            row = self.repo.fetch_puzzle(puzzle_id)
            if row is None:
                raise KeyError(f"puzzle {puzzle_id!r} not found in the active dataset")
            self._row_cache[puzzle_id] = row
        return self._row_cache[puzzle_id]

    def accepted_lines(
        self, puzzle_id: str, policy: StageEndPolicy, request: SequenceRequest
    ) -> tuple[tuple[tuple[str, ...], ...], bool]:
        key = (puzzle_id, policy.key(), request.acceptance, request.max_plies, request.max_nodes)
        if key not in self._context_cache:
            row = self._row(puzzle_id)
            self._context_cache[key] = accepted_lines_for_puzzle(row, policy, request)
        return self._context_cache[key]

    def contexts_for(
        self, puzzle_id: str, policy: StageEndPolicy, request: SequenceRequest
    ) -> tuple[tuple[Any, ...], bool]:
        lines, truncated = self.accepted_lines(puzzle_id, policy, request)
        row = self._row(puzzle_id)
        contexts = tuple(
            build_context_from_boards(
                puzzle_id=puzzle_id,
                presented_fen=str(row["start_fen"]),
                endpoint_fen=_fen_after(str(row["start_fen"]), line),
                solution_uci=line,
                solver_colour=(
                    chess.WHITE if str(row["solver_colour"]) == "white" else chess.BLACK
                ),
                policy=policy,
            )
            for line in lines
        )
        return contexts, truncated

    def _edge_cache_key(
        self, source: str, next_puzzle_id: str, policy: StageEndPolicy, request: SequenceRequest
    ) -> str:
        return self.edge_cache._key(
            source=source,
            next_puzzle_id=next_puzzle_id,
            end_policy_key=policy.key(),
            acceptance=request.acceptance,
            max_plies=request.max_plies,
            max_nodes=request.max_nodes,
            max_accepted_lines=request.max_accepted_lines,
            transition_config=self.transition_config,
        )

    def _attempt_edge(
        self,
        source: str,
        candidate: PairCandidate,
        policy: StageEndPolicy,
        request: SequenceRequest,
    ) -> EdgeAttempt:
        pair_key = (source, candidate.next_puzzle_id)
        if pair_key in self._attempts_cache:
            return self._attempts_cache[pair_key]

        cache_key = self._edge_cache_key(source, candidate.next_puzzle_id, policy, request)
        stored = self.edge_cache.get(cache_key)
        if stored is not None:
            attempt = self._attempt_from_dict(stored)
            self._attempts_cache[pair_key] = attempt
            return attempt

        contexts, _ = self.contexts_for(source, policy, request)
        next_row = self._row(candidate.next_puzzle_id)
        next_start = CanonicalPosition.from_board(chess.Board(str(next_row["start_fen"])))
        results = synthesize(
            contexts,
            next_start,
            policy=policy,
            config=self.transition_config,
        )
        if not results:
            reason = "no verified transition within the synthesis budget"
            if candidate.reasons_for_rejection:
                reason = " | ".join(candidate.reasons_for_rejection)
            attempt = EdgeAttempt(
                source_puzzle_id=source,
                next_puzzle_id=candidate.next_puzzle_id,
                end_policy_key=policy.key(),
                ok=False,
                reason=reason,
                next_rating=int(next_row["rating"]),
                next_themes=tuple(str(t) for t in next_row["themes"]),
                next_piece_count=int(next_row["presented_piece_count"]),
                next_popularity=int(next_row["popularity"]),
                d_occ_best=candidate.d_occ_best,
                cost_best=candidate.cost_best,
            )
        else:
            best = results[0]
            verification = best.verification
            guard_dict = _guard_to_dict(verification.guard) if verification.guard else None
            attempt = EdgeAttempt(
                source_puzzle_id=source,
                next_puzzle_id=candidate.next_puzzle_id,
                end_policy_key=policy.key(),
                ok=True,
                program=best.program,
                verification=verification,
                rendered=best.rendered,
                symmetry=_program_symmetry(best.program),
                action_count=len(best.program.actions),
                leakage_bits=verification.clue.total_clue_bits,
                absolute_disclosures=(
                    verification.clue.absolute_source_squares
                    + verification.clue.absolute_destination_squares
                ),
                d_occ_best=candidate.d_occ_best,
                cost_best=candidate.cost_best,
                exact_journey=best.program.actions == (),
                clue=verification.clue.to_dict(),
                guard=guard_dict,
                verification_issues=tuple(verification.issues),
                next_rating=int(next_row["rating"]),
                next_themes=tuple(str(t) for t in next_row["themes"]),
                next_piece_count=int(next_row["presented_piece_count"]),
                next_popularity=int(next_row["popularity"]),
            )
        self.edge_cache.put(cache_key, attempt.to_dict())
        self._attempts_cache[pair_key] = attempt
        return attempt

    def _attempt_from_dict(self, data: dict[str, Any]) -> EdgeAttempt:
        program_data = data.get("program")
        return EdgeAttempt(
            source_puzzle_id=str(data["source_puzzle_id"]),
            next_puzzle_id=str(data["next_puzzle_id"]),
            end_policy_key=str(data["end_policy_key"]),
            ok=bool(data["ok"]),
            reason=str(data["reason"]) if data.get("reason") else None,
            program=(
                A.TransitionProgram.from_json(program_data) if program_data is not None else None
            ),
            rendered=str(data.get("rendered", "")),
            symmetry=str(data.get("symmetry", "identity")),
            action_count=(
                int(data["action_count"]) if data.get("action_count") is not None else None
            ),
            leakage_bits=(
                float(data["leakage_bits"]) if data.get("leakage_bits") is not None else None
            ),
            absolute_disclosures=(
                int(data["absolute_disclosures"])
                if data.get("absolute_disclosures") is not None
                else None
            ),
            d_occ_best=float(data["d_occ_best"]) if data.get("d_occ_best") is not None else None,
            cost_best=float(data["cost_best"]) if data.get("cost_best") is not None else None,
            exact_journey=bool(data.get("exact_journey", False)),
            clue=dict(data["clue"]) if data.get("clue") is not None else None,
            guard=dict(data["guard"]) if data.get("guard") is not None else None,
            verification_issues=tuple(str(i) for i in data.get("verification_issues", [])),
            next_rating=(int(data["next_rating"]) if data.get("next_rating") is not None else None),
            next_themes=tuple(str(t) for t in data.get("next_themes", [])),
            next_piece_count=(
                int(data["next_piece_count"]) if data.get("next_piece_count") is not None else None
            ),
            next_popularity=(
                int(data["next_popularity"]) if data.get("next_popularity") is not None else None
            ),
        )

    # -- public neighbour API -------------------------------------------------------------

    def edges(
        self,
        source_puzzle_id: str,
        *,
        request: SequenceRequest,
        used_ids: frozenset[str] = frozenset(),
        for_final: bool = False,
    ) -> list[EdgeAttempt]:
        """Return ranked, verified neighbour edges for ``source_puzzle_id``.

        Intermediate edges use the preferred allowed intermediate policy;
        final edges additionally require a checkmate endpoint. Candidate next
        starts come from the similarity index in the requested rating band and
        are ranked by the decomposable transition-complexity score; the exact
        piece assignment and transition synthesis run per candidate, bounded by
        ``per_node_candidates``.
        """
        policy = self._source_policy(request, for_final=False)
        contexts, _ = self.contexts_for(source_puzzle_id, policy, request)
        if not contexts:
            return []
        row = self._row(source_puzzle_id)
        source_start_counts = MaterialSignature.from_board(
            chess.Board(str(row["start_fen"]))
        ).counts
        endpoints = tuple(context.endpoint for context in contexts)

        index = self._ensure_index()
        backend = self._backend or get_backend(name="exact")
        candidates, stats = retrieve(
            source_puzzle_id=source_puzzle_id,
            source_start_counts=source_start_counts,
            endpoints=endpoints,
            index=index,
            backend=backend,
            end_policy_key=policy.key(),
            endpoint_source=f"sequence:{policy.key()}:{request.acceptance}",
            rating_min=request.rating_min,
            rating_max=request.rating_max,
            max_rating_deviation=request.max_rating_deviation,
            min_popularity=request.min_popularity,
            top_k=request.per_node_candidates * 8,
            exclude_unusual_material=request.inventory_profile == "standard",
        )
        self.retrieval_stats.append(stats)

        edges: list[EdgeAttempt] = []
        seen: set[str] = set()
        current_rating = int(row["rating"])
        for candidate in candidates:
            next_id = candidate.next_puzzle_id
            if next_id in used_ids or next_id in seen:
                continue
            if not self._passes_quality_gates(next_id, request, current_rating, for_final):
                continue
            seen.add(next_id)
            attempt = self._attempt_edge(source_puzzle_id, candidate, policy, request)
            if not attempt.ok:
                continue
            if not self._passes_transition_budget(attempt, request):
                continue
            edges.append(attempt)
            if len(edges) >= request.per_node_candidates:
                break
        return edges

    # -- helpers ----------------------------------------------------------------------------

    def _source_policy(self, request: SequenceRequest, for_final: bool) -> StageEndPolicy:
        if for_final:
            return get_policy("checkmate")
        preferred = request.preferred_intermediate_policies
        allowed = request.allowed_intermediate_policies
        name = preferred[0] if preferred else allowed[0]
        return get_policy(name)

    def _passes_quality_gates(
        self,
        next_id: str,
        request: SequenceRequest,
        current_rating: int,
        for_final: bool,
    ) -> bool:
        if next_id in request.excluded_puzzle_ids or next_id in request.already_used_puzzle_ids:
            return False
        row = self._row(next_id)
        rating = int(row["rating"])
        if request.rating_spread and abs(rating - current_rating) > request.rating_spread:
            return False
        if int(row["rating_deviation"]) > request.max_rating_deviation:
            return False
        if int(row["popularity"]) < request.min_popularity:
            return False
        if int(row["nb_plays"]) < request.min_nb_plays:
            return False
        if bool(row.get("unusual_material", False)) and request.inventory_profile == "standard":
            return False
        themes = tuple(str(t) for t in row["themes"])
        if request.allowed_themes and not any(theme in themes for theme in request.allowed_themes):
            return False
        if any(theme in themes for theme in request.forbidden_themes):
            return False
        return not (for_final and not self._ends_in_checkmate(row))

    def _passes_transition_budget(self, attempt: EdgeAttempt, request: SequenceRequest) -> bool:
        if (
            attempt.action_count is not None
            and attempt.action_count > request.max_transition_actions
        ):
            return False
        if attempt.leakage_bits is not None and attempt.leakage_bits > request.max_leakage_bits:
            return False
        if (
            attempt.absolute_disclosures is not None
            and attempt.absolute_disclosures > request.max_absolute_square_disclosures
        ):
            return False
        return not (request.exact_journey == "required" and not attempt.exact_journey)

    def _ends_in_checkmate(self, row: dict[str, Any]) -> bool:
        board = chess.Board(str(row["start_fen"]))
        for uci in row["solution_uci"]:
            move = chess.Move.from_uci(str(uci))
            if move not in board.legal_moves:
                return False
            board.push(move)
        return board.is_checkmate()

    def _ensure_index(self) -> Any:
        if self._index is None:
            from chess_chain.similarity.retrieval import load_or_build_index

            self._index = load_or_build_index(
                self.settings,
                self.repo,
                version=int(self.settings.similarity_version),
            )
        return self._index


def _program_symmetry(program: A.TransitionProgram) -> str:
    if len(program.actions) == 1 and isinstance(program.actions[0], A.Transform):
        return program.actions[0].symmetry
    return "identity"


def _guard_to_dict(guard: Any) -> dict[str, object]:
    """Compact serialisation of a ``GuardVerification`` for the canonical record."""
    return {
        "passed": bool(guard.passed),
        "issues": list(guard.issues),
        "accepted_ready": sum(1 for evaluation in guard.accepted_evaluations if evaluation.ready()),
        "accepted_total": len(guard.accepted_evaluations),
        "prefix_ready": sum(1 for evaluation in guard.prefix_evaluations if evaluation.ready()),
        "wrong_state_ready": sum(
            1 for evaluation in guard.wrong_state_evaluations if evaluation.ready()
        ),
        "disclosures": guard.disclosures.to_dict(),
    }


__all__ = ["EdgeAttempt", "EdgeCache", "NeighbourGraph", "accepted_lines_for_puzzle"]
