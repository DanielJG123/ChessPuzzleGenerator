"""Bounded accepted-line enumeration and terminal-variation analysis.

Intermediate candidates are not rejected merely because accepted endpoints
have different occupancy: later transition synthesis may normalise all accepted
endpoints into one exact next start. Exact-position and occupancy invariance
are recorded as descriptive facts only.
"""

from __future__ import annotations

from dataclasses import dataclass
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
)

__all__ = [
    "TerminalVariationSummary",
    "analyze_variations",
    "estimate_scope_along_reference",
]


@dataclass(frozen=True)
class TerminalVariationSummary:
    puzzle_id: str
    end_policy_key: str
    end_policy_kind: str
    acceptance: str
    line_count: int
    terminal_fens: tuple[str, ...]
    terminal_occupancies: tuple[int, ...]
    exact_position_invariant: bool
    occupancy_invariant: bool
    endpoints_verified: bool
    verification_note: str | None
    truncated: bool
    max_plies: int
    max_nodes: int
    cache_key: str

    def to_dict(self) -> dict[str, object]:
        return {
            "puzzle_id": self.puzzle_id,
            "end_policy_key": self.end_policy_key,
            "end_policy_kind": self.end_policy_kind,
            "acceptance": self.acceptance,
            "line_count": self.line_count,
            "terminal_fens": list(self.terminal_fens),
            "terminal_occupancies": list(self.terminal_occupancies),
            "exact_position_invariant": self.exact_position_invariant,
            "occupancy_invariant": self.occupancy_invariant,
            "endpoints_verified": self.endpoints_verified,
            "verification_note": self.verification_note,
            "truncated": self.truncated,
            "max_plies": self.max_plies,
            "max_nodes": self.max_nodes,
            "cache_key": self.cache_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TerminalVariationSummary:
        return cls(
            puzzle_id=str(data["puzzle_id"]),
            end_policy_key=str(data["end_policy_key"]),
            end_policy_kind=str(data["end_policy_kind"]),
            acceptance=str(data["acceptance"]),
            line_count=int(data["line_count"]),
            terminal_fens=tuple(str(fen) for fen in data["terminal_fens"]),
            terminal_occupancies=tuple(int(mask) for mask in data["terminal_occupancies"]),
            exact_position_invariant=bool(data["exact_position_invariant"]),
            occupancy_invariant=bool(data["occupancy_invariant"]),
            endpoints_verified=bool(data["endpoints_verified"]),
            verification_note=(
                str(data["verification_note"]) if data["verification_note"] else None
            ),
            truncated=bool(data["truncated"]),
            max_plies=int(data["max_plies"]),
            max_nodes=int(data["max_nodes"]),
            cache_key=str(data["cache_key"]),
        )


def chess_engine_fen(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])


def estimate_scope_along_reference(
    board: chess.Board, intended_uci: tuple[str, ...], max_plies: int
) -> int:
    """Upper-bound line count by multiplying legal-move counts along the line."""
    estimate = 1
    current = board.copy()
    for depth in range(max_plies):
        legal = len(list(current.legal_moves))
        estimate *= max(legal, 1)
        if estimate >= 1_000_000_000:
            return 1_000_000_000
        if depth < len(intended_uci):
            move = chess.Move.from_uci(intended_uci[depth])
            if move not in current.legal_moves:
                break
            current.push(move)
        else:
            break
    return estimate


def _accepted_moves(
    board: chess.Board,
    *,
    depth: int,
    intended_uci: tuple[str, ...],
    acceptance: str,
    move_scores: dict[str, dict[str, int]] | None,
    plausible_gap_cp: int | None,
) -> list[str]:
    legal = [move.uci() for move in board.legal_moves]
    if acceptance == "intended_only":
        if depth < len(intended_uci) and intended_uci[depth] in legal:
            return [intended_uci[depth]]
        return []
    if acceptance == "all_legal":
        return legal
    if acceptance == "plausible":
        if move_scores is None or plausible_gap_cp is None:
            return []
        scores = move_scores.get(chess_engine_fen(board))
        if not scores:
            return []
        best = max(scores.values())
        return [
            move
            for move in legal
            if scores.get(move) is not None and best - scores[move] <= plausible_gap_cp
        ]
    raise ValueError(f"unknown acceptance policy {acceptance!r}")


def enumerate_accepted_lines(
    board: chess.Board,
    *,
    intended_uci: tuple[str, ...],
    acceptance: str,
    max_plies: int,
    max_nodes: int,
    move_scores: dict[str, dict[str, int]] | None = None,
    plausible_gap_cp: int | None = None,
) -> tuple[tuple[tuple[str, ...], ...], bool]:
    """Enumerate accepted lines; returns ``(lines, truncated)``."""
    lines: list[tuple[str, ...]] = []
    nodes = 0
    truncated = False

    def dfs(current: chess.Board, line: tuple[str, ...], depth: int) -> None:
        nonlocal nodes, truncated
        if depth == max_plies:
            lines.append(line)
            return
        moves = _accepted_moves(
            current,
            depth=depth,
            intended_uci=intended_uci,
            acceptance=acceptance,
            move_scores=move_scores,
            plausible_gap_cp=plausible_gap_cp,
        )
        if not moves:
            lines.append(line)
            return
        for uci in moves:
            nodes += 1
            if nodes > max_nodes:
                truncated = True
                return
            current.push(chess.Move.from_uci(uci))
            dfs(current, line + (uci,), depth + 1)
            current.pop()

    dfs(board.copy(), (), 0)
    return tuple(lines), truncated


def analyze_variations(
    *,
    puzzle_id: str,
    start_fen: str,
    solution_uci: tuple[str, ...] | list[str],
    end_policy: StageEndPolicy,
    acceptance: str,
    max_plies: int,
    max_nodes: int,
    cache: DiskKeyValueCache | None = None,
    move_scores: dict[str, dict[str, int]] | None = None,
    plausible_gap_cp: int | None = None,
) -> TerminalVariationSummary:
    """Enumerate accepted lines and verify the end policy over their endpoints."""
    intended = tuple(solution_uci)
    cache_key = (
        cache.key_from(
            {
                "puzzle_id": puzzle_id,
                "start_fen": start_fen,
                "end_policy": end_policy.key(),
                "acceptance": acceptance,
                "max_plies": max_plies,
                "max_nodes": max_nodes,
                "plausible_gap_cp": plausible_gap_cp,
            }
        )
        if cache is not None
        else ""
    )

    board = chess.Board(start_fen)
    lines, truncated = enumerate_accepted_lines(
        board,
        intended_uci=intended,
        acceptance=acceptance,
        max_plies=max_plies,
        max_nodes=max_nodes,
        move_scores=move_scores,
        plausible_gap_cp=plausible_gap_cp,
    )

    terminals: list[chess.Board] = []
    for line in lines:
        current = board.copy()
        for uci in line:
            current.push(chess.Move.from_uci(uci))
        terminals.append(current)

    terminal_fens = tuple(sorted({terminal.fen() for terminal in terminals}))
    terminal_occupancies = tuple(sorted({int(terminal.occupied) for terminal in terminals}))
    exact_position_invariant = len(terminal_fens) == 1
    occupancy_invariant = len(terminal_occupancies) == 1

    verified, note = _verify_endpoints(
        end_policy=end_policy,
        lines=lines,
        terminals=terminals,
    )

    return TerminalVariationSummary(
        puzzle_id=puzzle_id,
        end_policy_key=end_policy.key(),
        end_policy_kind=end_policy.kind,
        acceptance=acceptance,
        line_count=len(lines),
        terminal_fens=terminal_fens,
        terminal_occupancies=terminal_occupancies,
        exact_position_invariant=exact_position_invariant,
        occupancy_invariant=occupancy_invariant,
        endpoints_verified=verified,
        verification_note=note,
        truncated=truncated,
        max_plies=max_plies,
        max_nodes=max_nodes,
        cache_key=cache_key,
    )


def _verify_endpoints(
    *,
    end_policy: StageEndPolicy,
    lines: tuple[tuple[str, ...], ...],
    terminals: list[chess.Board],
) -> tuple[bool, str | None]:
    kind = end_policy.kind
    if kind == CHECKMATE:
        failures = sum(1 for terminal in terminals if not terminal.is_checkmate())
        if failures == 0:
            return True, None
        return False, f"{failures} of {len(terminals)} accepted endpoints are not checkmate"
    if kind == TRANSITION_READY:
        return True, "transition precondition proof deferred to stage 5"
    if kind == EXPLICIT_PREDICATE:
        assert end_policy.predicate is not None
        failures = sum(
            1 for terminal in terminals if not evaluate_predicate(terminal, end_policy.predicate)
        )
        if failures == 0:
            return True, None
        return False, (
            f"{failures} of {len(terminals)} accepted endpoints fail predicate "
            f"{end_policy.predicate!r}"
        )
    if kind == FIXED_PLIES:
        assert end_policy.fixed_plies is not None
        failures = sum(1 for line in lines if len(line) != end_policy.fixed_plies)
        if failures == 0:
            return True, None
        return False, (
            f"{failures} of {len(lines)} accepted lines have length != {end_policy.fixed_plies}"
        )
    raise ValueError(f"unknown stage-end kind {kind!r}")
