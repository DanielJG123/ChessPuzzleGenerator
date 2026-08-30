"""Lazy final-only occupancy-collision analysis.

The mechanical validator opens the box once, on the final puzzle of a complete
sequence. This analyser is therefore only invoked for puzzles being considered
as the *final* stage. It searches for legal *non-accepted* lines that reach the
target occupancy mask within a declared scope, including earlier prefixes (the
mechanism could open as soon as the mask appears).

A bounded search never claims global impossibility: the positive result is
``collision_free_within_scope``, not "globally safe".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import chess

from chess_chain.analysis.cache import DiskKeyValueCache
from chess_chain.domain.position import PositionFingerprint

__all__ = [
    "CollisionRequest",
    "CollisionResult",
    "CollisionStatus",
    "FinalOccupancyCollisionAnalyzer",
]

SCOPES = ("exact_plies", "up_to_plies", "plausible_horizon")


class CollisionStatus(StrEnum):
    COLLISION_FOUND = "collision_found"
    COLLISION_FREE_WITHIN_SCOPE = "collision_free_within_scope"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class CollisionRequest:
    """Everything that participates in the collision cache identity."""

    puzzle_id: str
    source_version: int
    presented_fingerprint: str
    target_occupancy: int
    accepted_line_policy: str
    stage_end_policy: str
    scope: str
    max_depth: int
    max_nodes: int
    pruning_config: str
    analyser_version: int

    def __post_init__(self) -> None:
        if self.scope not in SCOPES:
            raise ValueError(f"unknown scope {self.scope!r}; expected one of {SCOPES}")
        if not (0 <= self.target_occupancy < 1 << 64):
            raise ValueError("target occupancy out of 64-bit range")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def cache_key(self, cache: DiskKeyValueCache) -> str:
        return cache.key_from(self.to_dict())


@dataclass(frozen=True)
class CollisionResult:
    request: CollisionRequest
    status: CollisionStatus
    witness_line: tuple[str, ...] | None
    witness_depth: int | None
    nodes_searched: int
    verified_acceptance_occupancy: bool
    note: str | None
    cached: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            **self.request.to_dict(),
            "status": self.status.value,
            "witness_line": list(self.witness_line) if self.witness_line else None,
            "witness_depth": self.witness_depth,
            "nodes_searched": self.nodes_searched,
            "verified_acceptance_occupancy": self.verified_acceptance_occupancy,
            "note": self.note,
            "cached": self.cached,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CollisionResult:
        request = CollisionRequest(
            puzzle_id=str(data["puzzle_id"]),
            source_version=int(data["source_version"]),
            presented_fingerprint=str(data["presented_fingerprint"]),
            target_occupancy=int(data["target_occupancy"]),
            accepted_line_policy=str(data["accepted_line_policy"]),
            stage_end_policy=str(data["stage_end_policy"]),
            scope=str(data["scope"]),
            max_depth=int(data["max_depth"]),
            max_nodes=int(data["max_nodes"]),
            pruning_config=str(data["pruning_config"]),
            analyser_version=int(data["analyser_version"]),
        )
        witness = data.get("witness_line")
        return cls(
            request=request,
            status=CollisionStatus(str(data["status"])),
            witness_line=(tuple(str(item) for item in witness) if witness is not None else None),
            witness_depth=(
                int(data["witness_depth"]) if data["witness_depth"] is not None else None
            ),
            nodes_searched=int(data["nodes_searched"]),
            verified_acceptance_occupancy=bool(data["verified_acceptance_occupancy"]),
            note=str(data["note"]) if data["note"] else None,
            cached=bool(data.get("cached", False)),
        )


class FinalOccupancyCollisionAnalyzer:
    """Bounded search for non-accepted lines reaching the target occupancy."""

    def __init__(self, cache: DiskKeyValueCache | None = None) -> None:
        self._cache = cache

    def analyze(
        self,
        *,
        request: CollisionRequest,
        board: chess.Board,
        accepted_lines: tuple[tuple[str, ...], ...],
        move_filter: Callable[[chess.Board, chess.Move], bool] | None = None,
    ) -> CollisionResult:
        """Run (or reuse) the collision analysis for ``request``.

        ``accepted_lines`` are the accepted final lines; every one must end at
        ``request.target_occupancy``. ``move_filter`` restricts the search (used
        by the plausible_horizon scope); None searches all legal moves.
        """
        if self._cache is not None:
            key = request.cache_key(self._cache)
            if self._cache.has(key):
                stored = CollisionResult.from_dict(self._cache.get(key) or {})
                return CollisionResult(
                    request=stored.request,
                    status=stored.status,
                    witness_line=stored.witness_line,
                    witness_depth=stored.witness_depth,
                    nodes_searched=stored.nodes_searched,
                    verified_acceptance_occupancy=stored.verified_acceptance_occupancy,
                    note=stored.note,
                    cached=True,
                )

        verified = self._verify_accepted_occupancy(board, accepted_lines, request)
        if not verified:
            result = CollisionResult(
                request=request,
                status=CollisionStatus.INCONCLUSIVE,
                witness_line=None,
                witness_depth=None,
                nodes_searched=0,
                verified_acceptance_occupancy=False,
                note="accepted final lines do not all end at the target occupancy mask",
            )
        else:
            result = self._search(
                request=request,
                board=board,
                accepted_lines=accepted_lines,
                move_filter=move_filter,
            )

        if self._cache is not None:
            self._cache.put(request.cache_key(self._cache), result.to_dict())
        return result

    def _verify_accepted_occupancy(
        self,
        board: chess.Board,
        accepted_lines: tuple[tuple[str, ...], ...],
        request: CollisionRequest,
    ) -> bool:
        for line in accepted_lines:
            current = board.copy()
            for uci in line:
                move = chess.Move.from_uci(uci)
                if move not in current.legal_moves:
                    return False
                current.push(move)
            if int(current.occupied) != request.target_occupancy:
                return False
        return True

    def _search(
        self,
        *,
        request: CollisionRequest,
        board: chess.Board,
        accepted_lines: tuple[tuple[str, ...], ...],
        move_filter: Callable[[chess.Board, chess.Move], bool] | None,
    ) -> CollisionResult:
        visited: dict[PositionFingerprint, int] = {}
        nodes = 0
        witness: tuple[str, ...] | None = None
        witness_depth: int | None = None
        hit_limit = False
        accepted_set = set(accepted_lines)

        def dfs(current: chess.Board, line: tuple[str, ...], depth: int) -> None:
            nonlocal nodes, witness, witness_depth, hit_limit

            # 1. Collision check runs before any pruning: a non-accepted line
            #    reaching the target mask at ANY depth (including prefixes and
            #    transpositions of the accepted final position) is a collision.
            is_accepted_final = line in accepted_set
            if int(current.occupied) == request.target_occupancy and not is_accepted_final:
                witness = line
                witness_depth = depth
                return

            # 2. The accepted final node is the intended endpoint: the box opens
            #    there, so continuations beyond it are not explored.
            if is_accepted_final:
                return

            if depth >= request.max_depth:
                return

            # 3. Transposition cache with safe pruning (only after the collision
            #    check, so transposed target positions are still reported).
            fingerprint = PositionFingerprint.from_board(current)
            previous = visited.get(fingerprint)
            if previous is not None and previous <= depth:
                return
            visited[fingerprint] = depth

            nodes += 1
            if nodes > request.max_nodes:
                hit_limit = True
                return
            for move in current.legal_moves:
                if move_filter is not None and not move_filter(current, move):
                    continue
                current.push(move)
                dfs(current, line + (move.uci(),), depth + 1)
                current.pop()
                if witness is not None or hit_limit:
                    return

        dfs(board.copy(), (), 0)

        if witness is not None:
            status = CollisionStatus.COLLISION_FOUND
        elif hit_limit:
            status = CollisionStatus.INCONCLUSIVE
        else:
            status = CollisionStatus.COLLISION_FREE_WITHIN_SCOPE
        return CollisionResult(
            request=request,
            status=status,
            witness_line=witness,
            witness_depth=witness_depth,
            nodes_searched=nodes,
            verified_acceptance_occupancy=True,
            note=None,
        )
