"""Canonical ``sequence@1`` documents (stage 6).

A generated sequence is persisted as one JSON document with:

- the versioned request and the stable sequence fingerprint;
- per-stage records: puzzle id, end policy, accepted lines, the verified
  transition edge, guard disclosures, and the final collision report;
- the machine validation report;
- a concise search report (chosen nodes/edges and rejected alternatives);
- a provisional player-facing representation (first setup, side to move when
  required, all visible transitions - never hidden codes);
- private manufacturing/QA data (intended lines, accepted branches, endpoint
  FENs, intermediate guards, the single final mechanical occupancy target and
  its bounded collision report).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chess_chain.sequence.finalise import FinalSelection
from chess_chain.sequence.graph import EdgeAttempt
from chess_chain.sequence.request import SequenceRequest, request_from_dict
from chess_chain.sequence.search import SearchReport

SCHEMA = "sequence@1"

__all__ = ["SCHEMA", "SequenceStageRecord", "SequenceResult", "load_sequence"]


@dataclass(frozen=True)
class SequenceStageRecord:
    """One stage of a manufactured sequence."""

    index: int
    puzzle_id: str
    is_final: bool
    end_policy_key: str
    accepted_lines: tuple[tuple[str, ...], ...]
    truncated: bool
    row: dict[str, Any]
    edge: EdgeAttempt | None = None  # verified transition INTO this stage
    final_selection: FinalSelection | None = None  # final stage only

    def endpoint_fens(self) -> tuple[str, ...]:
        import chess

        board = chess.Board(str(self.row["start_fen"]))
        fens: list[str] = []
        for line in self.accepted_lines:
            current = board.copy()
            for uci in line:
                current.push(chess.Move.from_uci(uci))
            fens.append(current.fen())
        return tuple(sorted(set(fens)))

    @property
    def policy(self) -> Any:
        from chess_chain.analysis.policies import get_policy

        return get_policy(self.end_policy_key.split("@", 1)[0])

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "index": self.index,
            "puzzle_id": self.puzzle_id,
            "is_final": self.is_final,
            "end_policy_key": self.end_policy_key,
            "accepted_lines": [list(line) for line in self.accepted_lines],
            "accepted_endpoint_fens": list(self.endpoint_fens()),
            "truncated": self.truncated,
            "start_fen": str(self.row["start_fen"]),
            "solution_uci": [str(move) for move in self.row["solution_uci"]],
            "solver_colour": str(self.row["solver_colour"]),
            "rating": int(self.row["rating"]),
            "themes": [str(theme) for theme in self.row["themes"]],
            "edge": self.edge.to_dict() if self.edge is not None else None,
        }
        if self.final_selection is not None:
            data["final"] = self.final_selection.to_dict()
        return data


@dataclass(frozen=True)
class SequenceResult:
    """The full generated sequence."""

    request: SequenceRequest
    fingerprint: str
    stages: tuple[SequenceStageRecord, ...]
    validation: Any
    search_report: SearchReport
    product_id: str
    #: Forward-compatible difficulty-estimate store keyed by
    #: ``method@estimator_version`` (stage 7 provisional placeholder).
    difficulty_estimates: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Analysis/config versions at generation time (stale-version detection).
    versions: dict[str, int] = field(default_factory=dict)

    def puzzle_ids(self) -> tuple[str, ...]:
        return tuple(stage.puzzle_id for stage in self.stages)

    def stage_ratings(self) -> tuple[int | None, ...]:
        return tuple(int(stage.row["rating"]) for stage in self.stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "product_id": self.product_id,
            "fingerprint": self.fingerprint,
            "request": self.request.to_dict(),
            "stages": [stage.to_dict() for stage in self.stages],
            "validation": self.validation.to_dict(),
            "search_report": self.search_report.to_dict(),
            "difficulty_estimates": self.difficulty_estimates,
            "versions": self.versions,
            "player_facing": _player_facing(self),
            "manufacturing_qa": _manufacturing_qa(self),
        }


def _player_facing(result: SequenceResult) -> dict[str, object]:
    first = result.stages[0]
    stages_visible: list[dict[str, object]] = []
    for stage in result.stages:
        visible: dict[str, object] = {
            "puzzle_id": stage.puzzle_id,
            "side_to_move": str(stage.row["solver_colour"]),
        }
        if stage.edge is not None:
            if stage.edge.exact_journey:
                visible["transition"] = (
                    "No changes needed - the position already matches the next puzzle."
                )
            else:
                visible["transition"] = stage.edge.rendered
        stages_visible.append(visible)
    return {
        "first_setup_fen": str(first.row["start_fen"]),
        "stages": stages_visible,
        "note": "All transitions are visible to the player before solving; "
        "there are no hidden codes or staged reveals.",
    }


def _manufacturing_qa(result: SequenceResult) -> dict[str, object]:
    final_record = result.stages[-1]
    final = final_record.final_selection
    return {
        "stages": [
            {
                "puzzle_id": stage.puzzle_id,
                "is_final": stage.is_final,
                "intended_line": [str(move) for move in stage.row["solution_uci"]],
                "accepted_branches": [list(line) for line in stage.accepted_lines],
                "endpoint_fens": list(stage.endpoint_fens()),
                "end_policy": stage.end_policy_key,
                "guard_disclosures": _guard_disclosures(stage),
                "transition_ast": (
                    stage.edge.program.to_json() if stage.edge and stage.edge.program else None
                ),
            }
            for stage in result.stages
        ],
        "final_mechanical_target": {
            "occupancy_mask": (
                f"{int(final.collision.request.target_occupancy):#x}" if final is not None else None
            ),
            "collision_report": final.collision.to_dict() if final is not None else None,
        },
    }


def _guard_disclosures(stage: SequenceStageRecord) -> dict[str, object]:
    if stage.edge is None or stage.edge.verification is None:
        return {}
    clue = stage.edge.verification.clue
    return {
        "goal_leakage": clue.guard_goal_leakage,
        "named_captured_pieces": clue.guard_named_captured_pieces,
        "line_length_revealed": clue.guard_line_length_revealed,
        "occupied_square_disclosures": clue.guard_occupied_square_disclosures,
    }


def load_sequence(path: Path) -> SequenceResult | None:
    """Reload a persisted sequence document (None when unreadable)."""
    from chess_chain.sequence.validate import validation_from_dict

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        request = request_from_dict(dict(data["request"]))
        stages: list[SequenceStageRecord] = []
        for stage_data in data["stages"]:
            edge_data = stage_data.get("edge")
            edge = _edge_from_dict(edge_data) if edge_data else None
            accepted_lines = tuple(
                tuple(str(m) for m in line) for line in stage_data["accepted_lines"]
            )
            record = SequenceStageRecord(
                index=int(stage_data["index"]),
                puzzle_id=str(stage_data["puzzle_id"]),
                is_final=bool(stage_data["is_final"]),
                end_policy_key=str(stage_data["end_policy_key"]),
                accepted_lines=accepted_lines,
                truncated=bool(stage_data.get("truncated", False)),
                row={
                    "puzzle_id": str(stage_data["puzzle_id"]),
                    "start_fen": str(stage_data["start_fen"]),
                    "solution_uci": list(stage_data["solution_uci"]),
                    "solver_colour": str(stage_data["solver_colour"]),
                    "rating": int(stage_data["rating"]),
                    "themes": list(stage_data["themes"]),
                },
                edge=edge,
            )
            final_data = stage_data.get("final")
            if final_data is not None:
                record = _restore_final_selection(record, edge, dict(final_data))
            stages.append(record)
        return SequenceResult(
            request=request,
            fingerprint=str(data["fingerprint"]),
            stages=tuple(stages),
            validation=validation_from_dict(dict(data["validation"])),
            search_report=_search_report_from_dict(dict(data.get("search_report", {}))),
            product_id=str(data["product_id"]),
            difficulty_estimates={
                str(key): dict(value) for key, value in data.get("difficulty_estimates", {}).items()
            },
            versions={str(key): int(value) for key, value in data.get("versions", {}).items()},
        )
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


def write_sequence(sequence: SequenceResult, path: Path) -> None:
    """Persist ``sequence`` as a canonical ``sequence@1`` document."""
    path.write_text(json.dumps(sequence.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _restore_final_selection(
    stage: SequenceStageRecord, edge: EdgeAttempt | None, data: dict[str, Any]
) -> SequenceStageRecord:
    """Rehydrate the final-stage ``FinalSelection`` from the persisted record."""
    import dataclasses
    from typing import cast

    from chess_chain.analysis.collision import CollisionResult
    from chess_chain.sequence.finalise import FinalistRank, FinalSelection

    collision = CollisionResult.from_dict(dict(data["collision_report"]))
    rank_data = dict(data["rank"])
    rank = FinalistRank(
        puzzle_id=str(rank_data["puzzle_id"]),
        rank=int(rank_data["rank"]),
        edge=edge
        if edge is not None
        else EdgeAttempt(source_puzzle_id="", next_puzzle_id="", end_policy_key="", ok=False),
    )
    selection = FinalSelection(
        stage=cast(Any, stage),
        edge=edge if edge is not None else rank.edge,
        collision=collision,
        rank=rank,
        alternatives=(),
        final_ends_in_checkmate=bool(data.get("final_ends_in_checkmate", False)),
        final_occupancy_invariant=bool(data.get("final_occupancy_invariant", False)),
        final_accepted_lines=stage.accepted_lines,
        path=None,
    )
    return dataclasses.replace(stage, final_selection=selection)


def _edge_from_dict(data: dict[str, Any]) -> EdgeAttempt:
    from chess_chain.transitions import ast as A

    program_data = data.get("program")
    return EdgeAttempt(
        source_puzzle_id=str(data["source_puzzle_id"]),
        next_puzzle_id=str(data["next_puzzle_id"]),
        end_policy_key=str(data["end_policy_key"]),
        ok=bool(data["ok"]),
        reason=str(data["reason"]) if data.get("reason") else None,
        program=(A.TransitionProgram.from_json(program_data) if program_data is not None else None),
        rendered=str(data.get("rendered", "")),
        symmetry=str(data.get("symmetry", "identity")),
        action_count=(int(data["action_count"]) if data.get("action_count") is not None else None),
        leakage_bits=(
            float(data["leakage_bits"]) if data.get("leakage_bits") is not None else None
        ),
        absolute_disclosures=(
            int(data["absolute_disclosures"])
            if data.get("absolute_disclosures") is not None
            else None
        ),
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


def _search_report_from_dict(data: dict[str, Any]) -> SearchReport:
    return SearchReport(
        expanded_nodes=int(data.get("expanded_nodes", 0)),
        edges_attempted=int(data.get("edges_attempted", 0)),
        edges_rejected=[
            (str(item["from"]), str(item["to"]), str(item["reason"]))
            for item in data.get("edges_rejected", [])
        ],
        backtrack_steps=int(data.get("backtrack_steps", 0)),
        dead_end=bool(data.get("dead_end", False)),
        no_path=bool(data.get("no_path", False)),
    )
