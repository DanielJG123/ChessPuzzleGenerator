"""Record builders for variations and collision parquet persistence."""

from __future__ import annotations

from typing import Any

from chess_chain.analysis.collision import (
    CollisionRequest,
    CollisionResult,
    CollisionStatus,
)
from chess_chain.analysis.variations import TerminalVariationSummary

__all__ = ["collision_from_record", "collision_to_record", "variation_to_record"]


def variation_to_record(summary: TerminalVariationSummary) -> dict[str, object]:
    return {
        "puzzle_id": summary.puzzle_id,
        "end_policy_key": summary.end_policy_key,
        "end_policy_kind": summary.end_policy_kind,
        "acceptance": summary.acceptance,
        "line_count": summary.line_count,
        "terminal_fens": list(summary.terminal_fens),
        "terminal_occupancies": list(summary.terminal_occupancies),
        "exact_position_invariant": summary.exact_position_invariant,
        "occupancy_invariant": summary.occupancy_invariant,
        "endpoints_verified": summary.endpoints_verified,
        "verification_note": summary.verification_note,
        "truncated": summary.truncated,
        "max_plies": summary.max_plies,
        "max_nodes": summary.max_nodes,
        "cache_key": summary.cache_key,
    }


def collision_to_record(result: CollisionResult, *, cache_key: str) -> dict[str, object]:
    request = result.request
    return {
        "puzzle_id": request.puzzle_id,
        "source_version": request.source_version,
        "presented_fingerprint": request.presented_fingerprint,
        "target_occupancy": request.target_occupancy,
        "accepted_line_policy": request.accepted_line_policy,
        "stage_end_policy": request.stage_end_policy,
        "scope": request.scope,
        "max_depth": request.max_depth,
        "max_nodes": request.max_nodes,
        "pruning_config": request.pruning_config,
        "analyser_version": request.analyser_version,
        "cache_key": cache_key,
        "status": result.status.value,
        "witness_line": list(result.witness_line) if result.witness_line else None,
        "witness_depth": result.witness_depth,
        "nodes_searched": result.nodes_searched,
        "verified_acceptance_occupancy": result.verified_acceptance_occupancy,
        "note": result.note,
        "cached": result.cached,
    }


def collision_from_record(record: dict[str, Any]) -> CollisionResult:
    request = CollisionRequest(
        puzzle_id=str(record["puzzle_id"]),
        source_version=int(record["source_version"]),
        presented_fingerprint=str(record["presented_fingerprint"]),
        target_occupancy=int(record["target_occupancy"]),
        accepted_line_policy=str(record["accepted_line_policy"]),
        stage_end_policy=str(record["stage_end_policy"]),
        scope=str(record["scope"]),
        max_depth=int(record["max_depth"]),
        max_nodes=int(record["max_nodes"]),
        pruning_config=str(record["pruning_config"]),
        analyser_version=int(record["analyser_version"]),
    )
    witness = record.get("witness_line")
    return CollisionResult(
        request=request,
        status=CollisionStatus(str(record["status"])),
        witness_line=(tuple(str(item) for item in witness) if witness is not None else None),
        witness_depth=(
            int(record["witness_depth"]) if record["witness_depth"] is not None else None
        ),
        nodes_searched=int(record["nodes_searched"]),
        verified_acceptance_occupancy=bool(record["verified_acceptance_occupancy"]),
        note=str(record["note"]) if record["note"] else None,
        cached=bool(record.get("cached", False)),
    )
