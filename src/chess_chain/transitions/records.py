"""Transition persistence: the AST plus machine-verifiable before/after FENs.

The persisted ``transition.json`` stores the program AST (never the rendered
text as the source of truth), the stage-boundary guard, every accepted
endpoint's presented-start FEN and solution line, the exact next-start FEN,
the verification result and the clue report. ``verify transition.json`` reloads
the AST and re-executes it, so a renderer change can never alter semantics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chess_chain.analysis.policies import get_policy
from chess_chain.domain.position import CanonicalPosition
from chess_chain.transitions import ast as A
from chess_chain.transitions.config import TransitionConfig
from chess_chain.transitions.context import EndpointContext, build_context_from_boards
from chess_chain.transitions.guards import StageBoundaryGuard
from chess_chain.transitions.render import render_program
from chess_chain.transitions.verify import VerificationResult, verify_program

SCHEMA = "transition@1"

__all__ = [
    "SCHEMA",
    "TransitionArtifact",
    "build_transition_document",
    "load_transition_document",
    "write_transition_document",
]


@dataclass(frozen=True)
class TransitionArtifact:
    """The reloadable form of one verified transition."""

    schema: str
    source_puzzle_id: str
    next_puzzle_id: str
    policy_key: str
    presented_fens: tuple[str, ...]
    endpoint_fens: tuple[str, ...]
    solutions: tuple[tuple[str, ...], ...]
    after_fen: str
    program: A.TransitionProgram
    guard: StageBoundaryGuard
    rendered: str
    verification: VerificationResult | None
    config: TransitionConfig

    def reconstruct_contexts(self) -> tuple[EndpointContext, ...]:
        name = self.policy_key.split("@", 1)[0]
        policy = get_policy(name)
        return tuple(
            build_context_from_boards(
                puzzle_id=self.source_puzzle_id,
                presented_fen=presented_fen,
                endpoint_fen=endpoint_fen,
                solution_uci=solution,
                policy=policy,
            )
            for presented_fen, endpoint_fen, solution in zip(
                self.presented_fens, self.endpoint_fens, self.solutions, strict=True
            )
        )

    def next_start(self) -> CanonicalPosition:
        import chess

        return CanonicalPosition.from_board(chess.Board(self.after_fen))

    def re_verify(self) -> VerificationResult:
        contexts = self.reconstruct_contexts()
        return verify_program(
            self.program,
            accepted=contexts,
            next_start=self.next_start(),
            guard=self.guard,
            config=self.config,
        )


def build_transition_document(
    *,
    source_puzzle_id: str,
    next_puzzle_id: str,
    contexts: tuple[EndpointContext, ...],
    next_start: CanonicalPosition,
    program: A.TransitionProgram,
    guard: StageBoundaryGuard,
    verification: VerificationResult,
    config: TransitionConfig,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "source_puzzle_id": source_puzzle_id,
        "next_puzzle_id": next_puzzle_id,
        "policy_key": guard.policy_key,
        "presented_fens": [context.presented_start.fen for context in contexts],
        "endpoint_fens": [context.endpoint.fen for context in contexts],
        "solutions": [list(event.uci for event in context.events) for context in contexts],
        "after_fen": next_start.fen,
        "program": program.to_json(),
        "guard": guard.to_json(),
        "rendered": render_program(program),
        "verification": verification.to_dict(),
        "config": config.as_dict(),
    }


def write_transition_document(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".part")
    temp_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def load_transition_document(path: Path) -> TransitionArtifact | None:
    """Reload a persisted transition; returns None for unreadable files."""
    import chess

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        program = A.TransitionProgram.from_json(data["program"])
        guard_data = data.get("guard") or {}
        transition_program = program if guard_data.get("kind") == "TRANSITION_READY" else None
        expected_next = (
            CanonicalPosition.from_board(chess.Board(str(guard_data["expected_next_fen"])))
            if guard_data.get("expected_next_fen")
            else None
        )
        guard = StageBoundaryGuard(
            kind=str(guard_data.get("kind", "")),
            policy_key=str(data["policy_key"]),
            predicate=str(guard_data["predicate"]) if guard_data.get("predicate") else None,
            fixed_plies=(
                int(guard_data["fixed_plies"])
                if guard_data.get("fixed_plies") is not None
                else None
            ),
            transition_program=transition_program,
            expected_next=expected_next,
            wording=str(guard_data["wording"]) if guard_data.get("wording") else None,
        )
        config_data = data.get("config") or {}
        fields = TransitionConfig.__dataclass_fields__
        config = TransitionConfig(
            **{key: value for key, value in config_data.items() if key in fields}
        )
        return TransitionArtifact(
            schema=str(data.get("schema", SCHEMA)),
            source_puzzle_id=str(data["source_puzzle_id"]),
            next_puzzle_id=str(data["next_puzzle_id"]),
            policy_key=str(data["policy_key"]),
            presented_fens=tuple(str(item) for item in data["presented_fens"]),
            endpoint_fens=tuple(str(item) for item in data["endpoint_fens"]),
            solutions=tuple(
                tuple(str(move) for move in solution) for solution in data["solutions"]
            ),
            after_fen=str(data["after_fen"]),
            program=program,
            guard=guard,
            rendered=str(data.get("rendered", "")),
            verification=None,
            config=config,
        )
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None
