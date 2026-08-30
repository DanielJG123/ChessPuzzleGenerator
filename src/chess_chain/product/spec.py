"""Manufacturing specification builder (stage 7).

The canonical product specification is a JSON document independent of any
rendering format. It contains everything needed to manufacture or inspect the
physical product, including the single final mechanical occupancy target, the
provisional difficulty estimate and the deterministic fingerprints - but never
a fabricated uncertainty interval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import chess

from chess_chain.analysis.policies import get_policy
from chess_chain.product.render import DATA_LICENCE
from chess_chain.sequence.rating import DifficultyEstimate
from chess_chain.sequence.records import SequenceResult

if TYPE_CHECKING:
    from chess_chain.config import Settings

__all__ = ["build_manufacturing_spec"]


def build_manufacturing_spec(
    sequence: SequenceResult,
    settings: Settings,
    estimate: DifficultyEstimate,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the ``manufacturing_spec.json`` document."""
    stages: list[dict[str, Any]] = []
    for stage in sequence.stages:
        policy = get_policy(stage.end_policy_key.split("@", 1)[0])
        stage_spec: dict[str, Any] = {
            "index": stage.index,
            "puzzle_id": stage.puzzle_id,
            "is_final": stage.is_final,
            "source_rating": int(stage.row["rating"]),
            "side_to_move": str(stage.row["solver_colour"]),
            "start_fen": str(stage.row["start_fen"]),
            "stage_end_policy": {
                "key": policy.key(),
                "kind": policy.kind,
                "wording": policy.wording,
            },
            "intended_line": [str(move) for move in stage.row["solution_uci"]],
            "accepted_lines": [list(line) for line in stage.accepted_lines],
            "accepted_endpoint_fens": list(stage.endpoint_fens()),
            "accepted_occupancy_invariant": _occupancy_invariant(
                str(stage.row["start_fen"]), stage.accepted_lines
            ),
        }
        if stage.edge is not None:
            stage_spec["transition"] = {
                "rendered": stage.edge.rendered,
                "ast": (stage.edge.program.to_json() if stage.edge.program is not None else None),
                "action_count": stage.edge.action_count,
                "exact_journey": stage.edge.exact_journey,
                "symmetry": stage.edge.symmetry,
                "clue_report": stage.edge.clue,
                "guard": stage.edge.guard,
                "next_start_fen": _next_start_fen(sequence, stage),
                "next_puzzle_id": stage.edge.next_puzzle_id,
            }
        if stage.final_selection is not None:
            stage_spec["final"] = {
                "checkmate": stage.final_selection.final_ends_in_checkmate,
                "occupancy_invariant": stage.final_selection.final_occupancy_invariant,
                "mechanical_occupancy_mask": (
                    f"{int(stage.final_selection.collision.request.target_occupancy):#x}"
                ),
                "collision": {
                    "status": stage.final_selection.collision.status.value,
                    "scope": stage.final_selection.collision.request.scope,
                    "max_depth": stage.final_selection.collision.request.max_depth,
                    "max_nodes": stage.final_selection.collision.request.max_nodes,
                    "witness_line": stage.final_selection.collision.witness_line,
                    "witness_depth": stage.final_selection.collision.witness_depth,
                    "cached": stage.final_selection.collision.cached,
                    "analyser_version": stage.final_selection.collision.request.analyser_version,
                },
            }
        stages.append(stage_spec)

    return {
        "schema": "manufacturing_spec@1",
        "product_id": sequence.product_id,
        "generator_version": {
            "sequence_schema": "sequence@1",
            "request": sequence.request.version,
            "versions": sequence.versions,
        },
        "generated_at": generated_at,
        "random_seed": sequence.request.random_seed,
        "sequence_fingerprint": sequence.fingerprint,
        "source_data": {
            "licence": DATA_LICENCE,
            "puzzle_ids": list(sequence.puzzle_ids()),
            "start_puzzle_id": sequence.request.start_puzzle_id,
        },
        "first_position": {
            "fen": str(sequence.stages[0].row["start_fen"]),
            "side_to_move": str(sequence.stages[0].row["solver_colour"]),
        },
        "stages": stages,
        "difficulty": {
            "provisional_average": estimate.to_dict(),
            "label": estimate.label,
            "uncertainty_interval": None,
            "note": (
                "No uncertainty interval is fabricated. This is a provisional "
                "source-rating average for manual product review only; it "
                "ignores the both-sides protocol, visible transition clues, "
                "stage-end information and the multiplicative difficulty of "
                "completing several stages."
            ),
        },
        "final_stage": _final_summary(sequence),
        "reproduce": {
            "command": (f"chess-chain sequence reproduce --product-id {sequence.product_id}"),
            "request_fingerprint": sequence.request.to_dict(),
        },
    }


def _occupancy_invariant(start_fen: str, accepted_lines: tuple[tuple[str, ...], ...]) -> bool:
    board = chess.Board(start_fen)
    occupancies: set[int] = set()
    for line in accepted_lines:
        current = board.copy()
        for uci in line:
            current.push(chess.Move.from_uci(uci))
        occupancies.add(int(current.occupied))
    return len(occupancies) <= 1


def _next_start_fen(sequence: SequenceResult, stage: Any) -> str | None:
    if stage.is_final or stage.index + 1 >= len(sequence.stages):
        return None
    return str(sequence.stages[stage.index + 1].row["start_fen"])


def _final_summary(sequence: SequenceResult) -> dict[str, Any]:
    final_stage = sequence.stages[-1]
    selection = final_stage.final_selection
    if selection is None:
        return {"checkmate": False, "collision_status": None}
    return {
        "puzzle_id": final_stage.puzzle_id,
        "checkmate": selection.final_ends_in_checkmate,
        "mechanical_occupancy_mask": (f"{int(selection.collision.request.target_occupancy):#x}"),
        "collision_status": selection.collision.status.value,
        "collision_scope": selection.collision.request.scope,
    }
