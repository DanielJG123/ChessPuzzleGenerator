"""Record builders for similarity persistence (candidate pairs and index)."""

from __future__ import annotations

import json
from typing import Any

from chess_chain.similarity.candidate import PairCandidate

__all__ = ["candidate_to_record"]


def candidate_to_record(candidate: PairCandidate) -> dict[str, Any]:
    """Flatten a ``PairCandidate`` for the ``CANDIDATE_SCHEMA`` columns.

    Scalar and aggregate fields stay typed/queryable; the detailed per-endpoint
    journeys, score components and material diffs are stored as JSON strings
    for the explain command and machine-readable reports.
    """
    data = candidate.to_dict()
    return {
        "source_puzzle_id": data["source_puzzle_id"],
        "next_puzzle_id": data["next_puzzle_id"],
        "source_stage_end_policy": data["source_stage_end_policy"],
        "endpoint_source": data["endpoint_source"],
        "applied_symmetry": data["applied_symmetry"],
        "accepted_endpoint_fingerprints": data["accepted_endpoint_fingerprints"],
        "d_occ_best": data["aggregate"]["d_occ"]["best"],
        "d_occ_mean": data["aggregate"]["d_occ"]["mean"],
        "d_occ_worst": data["aggregate"]["d_occ"]["worst"],
        "d_placement_best": data["aggregate"]["d_placement"]["best"],
        "d_placement_mean": data["aggregate"]["d_placement"]["mean"],
        "d_placement_worst": data["aggregate"]["d_placement"]["worst"],
        "cost_best": data["aggregate"]["cost"]["best"],
        "cost_mean": data["aggregate"]["cost"]["mean"],
        "cost_worst": data["aggregate"]["cost"]["worst"],
        "total_moves": data["counts"]["moves"],
        "total_swaps": data["counts"]["swaps"],
        "total_replacements": data["counts"]["replacements"],
        "total_additions": data["counts"]["additions"],
        "total_removals": data["counts"]["removals"],
        "inventory_valid": data["inventory_valid"],
        "rights_annotations_required": data["rights_annotations_required"],
        "transition_complexity_score": data["transition_complexity_score"],
        "reasons_for_rejection": data["reasons_for_rejection"],
        "per_endpoint_json": json.dumps(
            data["per_endpoint"], sort_keys=True, separators=(",", ":")
        ),
        "score_components_json": json.dumps(
            data["score_components"], sort_keys=True, separators=(",", ":")
        ),
        "material_diff_json": json.dumps(
            data["start_end_material_diff"], sort_keys=True, separators=(",", ":")
        ),
    }
