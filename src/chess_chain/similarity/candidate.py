"""Structured similarity candidate pairs.

A ``PairCandidate`` links a source puzzle (with its stage-end policy and
accepted endpoints) to a next puzzle, under one applied symmetry. Per-endpoint
journeys carry the layered distances and the physical piece-assignment plans;
the candidate aggregates at least the best, mean and worst per-endpoint cost
and exposes a decomposable transition-complexity score with explicit
components and rejection reasons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess

from chess_chain.domain.pieces import COUNT_NAMES
from chess_chain.domain.position import CanonicalPosition
from chess_chain.similarity.distance import EndpointComparison, compare_endpoint
from chess_chain.similarity.scoring import (
    SimilarityScoreWeights,
    compute_candidate_score,
)
from chess_chain.similarity.symmetry import (
    SYMMETRIES,
    apply_symmetry,
)

__all__ = ["PairCandidate", "build_candidate"]


@dataclass(frozen=True)
class PairCandidate:
    """One ranked source->next puzzle pair under a chosen symmetry."""

    source_puzzle_id: str
    next_puzzle_id: str
    source_stage_end_policy: str
    endpoint_source: str  # "variation_analysis:<policy>" or "final_position:<policy>"
    accepted_endpoint_ids: tuple[str, ...]
    accepted_endpoint_fingerprints: tuple[str, ...]
    applied_symmetry: str
    per_endpoint: tuple[EndpointComparison, ...]

    d_occ_best: float
    d_occ_mean: float
    d_occ_worst: float
    d_placement_best: float
    d_placement_mean: float
    d_placement_worst: float
    cost_best: float
    cost_mean: float
    cost_worst: float

    total_moves: int
    total_swaps: int
    total_replacements: int
    total_additions: int
    total_removals: int

    start_end_material_diff: dict[str, int]
    inventory_valid: bool
    inventory_notes: tuple[str, ...]
    rights_annotations_required: bool

    transition_complexity_score: float
    score_components: dict[str, float]
    reasons_for_rejection: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for reports and persistence."""
        return {
            "source_puzzle_id": self.source_puzzle_id,
            "next_puzzle_id": self.next_puzzle_id,
            "source_stage_end_policy": self.source_stage_end_policy,
            "endpoint_source": self.endpoint_source,
            "accepted_endpoint_ids": list(self.accepted_endpoint_ids),
            "accepted_endpoint_fingerprints": list(self.accepted_endpoint_fingerprints),
            "applied_symmetry": self.applied_symmetry,
            "per_endpoint": [_endpoint_to_dict(comparison) for comparison in self.per_endpoint],
            "aggregate": {
                "d_occ": {
                    "best": self.d_occ_best,
                    "mean": self.d_occ_mean,
                    "worst": self.d_occ_worst,
                },
                "d_placement": {
                    "best": self.d_placement_best,
                    "mean": self.d_placement_mean,
                    "worst": self.d_placement_worst,
                },
                "cost": {
                    "best": self.cost_best,
                    "mean": self.cost_mean,
                    "worst": self.cost_worst,
                },
            },
            "counts": {
                "moves": self.total_moves,
                "swaps": self.total_swaps,
                "replacements": self.total_replacements,
                "additions": self.total_additions,
                "removals": self.total_removals,
            },
            "start_end_material_diff": self.start_end_material_diff,
            "inventory_valid": self.inventory_valid,
            "inventory_notes": list(self.inventory_notes),
            "rights_annotations_required": self.rights_annotations_required,
            "transition_complexity_score": self.transition_complexity_score,
            "score_components": self.score_components,
            "reasons_for_rejection": list(self.reasons_for_rejection),
        }


def _endpoint_to_dict(comparison: EndpointComparison) -> dict[str, Any]:
    assignment = comparison.assignment
    return {
        "endpoint_index": comparison.endpoint_index,
        "endpoint_fingerprint": comparison.endpoint_fingerprint,
        "endpoint_fen": comparison.endpoint_fen,
        "exact_position_match": comparison.exact_position_match,
        "d_occ": comparison.d_occ,
        "d_placement": comparison.d_placement,
        "piece_diff_remove": list(comparison.piece_diff_remove),
        "piece_diff_add": list(comparison.piece_diff_add),
        "material_diff": list(comparison.material_diff),
        "material_remove_w": comparison.material_remove_w,
        "material_add_w": comparison.material_add_w,
        "material_remove_b": comparison.material_remove_b,
        "material_add_b": comparison.material_add_b,
        "assignment": {
            "feasible": assignment.feasible,
            "total_cost": assignment.total_cost,
            "moves": assignment.moves,
            "swaps": assignment.swaps,
            "replacements": assignment.replacements,
            "additions": assignment.additions,
            "removals": assignment.removals,
            "note": assignment.note,
            "matched": [
                {
                    "source": pair.source_key(),
                    "target_square": chess.square_name(pair.target_square),
                    "cost": pair.cost,
                    "from_tray": pair.from_tray,
                }
                for pair in assignment.matched
            ],
            "displacement_vectors": [list(v) for v in assignment.displacement_vectors],
        },
        "inventory_valid": comparison.inventory_valid,
        "inventory_notes": list(comparison.inventory_notes),
        "castling_annotation_required": comparison.castling_annotation_required,
        "ep_annotation_required": comparison.ep_annotation_required,
        "rights_note": comparison.rights_note,
        "exact_journey": comparison.exact_journey,
        "occupancy_journey": comparison.occupancy_journey,
    }


def build_candidate(
    *,
    source_puzzle_id: str,
    next_puzzle_id: str,
    source_stage_end_policy: str,
    endpoint_source: str,
    endpoints: tuple[CanonicalPosition, ...],
    source_start_counts: tuple[int, ...],
    next_start: CanonicalPosition,
    symmetries: tuple[str, ...] = SYMMETRIES,
    cost_weights: Any | None = None,
    score_weights: SimilarityScoreWeights | None = None,
    reject_non_visible_rights: bool = False,
) -> PairCandidate:
    """Score every accepted endpoint of ``source`` against ``next_start`` under
    each symmetry and choose the single symmetry minimising the total
    per-endpoint assignment cost (one program must handle every endpoint).

    ``next_start`` is the *untransformed* presented start of the next puzzle;
    each symmetry is reconstructed and validated before scoring.
    """
    from chess_chain.similarity.assignment import CostWeights

    weights = cost_weights or CostWeights()
    endpoint_ids = tuple(f"{source_puzzle_id}:e{i}" for i in range(len(endpoints)))
    endpoint_fingerprints = tuple(endpoint.fingerprint.to_key() for endpoint in endpoints)

    best: tuple[float, str, tuple[EndpointComparison, ...]] | None = None
    for symmetry in symmetries:
        transformed = apply_symmetry(next_start, symmetry)
        if transformed is None:
            continue
        comparisons = tuple(
            compare_endpoint(
                endpoint,
                transformed,
                endpoint_index=index,
                source_start_counts=source_start_counts,
                symmetry=symmetry,
                cost_weights=weights,
            )
            for index, endpoint in enumerate(endpoints)
        )
        total = sum(comparison.assignment.total_cost for comparison in comparisons)
        candidate = (total, symmetry, comparisons)
        if best is None or (total, SYMMETRIES.index(symmetry)) < (
            best[0],
            SYMMETRIES.index(best[1]),
        ):
            best = candidate

    if best is None:
        return PairCandidate(
            source_puzzle_id=source_puzzle_id,
            next_puzzle_id=next_puzzle_id,
            source_stage_end_policy=source_stage_end_policy,
            endpoint_source=endpoint_source,
            accepted_endpoint_ids=endpoint_ids,
            accepted_endpoint_fingerprints=endpoint_fingerprints,
            applied_symmetry=SYMMETRIES[0],
            per_endpoint=(),
            d_occ_best=float("inf"),
            d_occ_mean=float("inf"),
            d_occ_worst=float("inf"),
            d_placement_best=float("inf"),
            d_placement_mean=float("inf"),
            d_placement_worst=float("inf"),
            cost_best=float("inf"),
            cost_mean=float("inf"),
            cost_worst=float("inf"),
            total_moves=0,
            total_swaps=0,
            total_replacements=0,
            total_additions=0,
            total_removals=0,
            start_end_material_diff={},
            inventory_valid=False,
            inventory_notes=(),
            rights_annotations_required=False,
            transition_complexity_score=float("inf"),
            score_components={},
            reasons_for_rejection=(
                "no symmetry produced a legal reconstructed standard-chess position",
            ),
        )

    _, applied_symmetry, comparisons = best
    score, components = compute_candidate_score(
        comparisons,
        applied_symmetry=applied_symmetry,
        weights=score_weights,
    )

    reasons: list[str] = []
    inventory_notes: set[str] = set()
    rights_annotations_required = False
    for comparison in comparisons:
        inventory_notes.update(comparison.inventory_notes)
        if not comparison.inventory_valid:
            reasons.append(
                f"endpoint {comparison.endpoint_index}: next start material is not "
                "reachable from the available physical inventory"
            )
        if not comparison.assignment.feasible:
            reasons.append(
                f"endpoint {comparison.endpoint_index}: piece assignment infeasible "
                f"({comparison.assignment.note or 'unknown'})"
            )
        if comparison.castling_annotation_required or comparison.ep_annotation_required:
            rights_annotations_required = True
    if reject_non_visible_rights and rights_annotations_required:
        reasons.append(
            "product configuration rejects transitions that must establish "
            "non-visible castling/en-passant rights"
        )

    material_diff: dict[str, int] = {}
    for index, name in enumerate(COUNT_NAMES):
        material_diff[name] = sum(comparison.material_diff[index] for comparison in comparisons)

    return PairCandidate(
        source_puzzle_id=source_puzzle_id,
        next_puzzle_id=next_puzzle_id,
        source_stage_end_policy=source_stage_end_policy,
        endpoint_source=endpoint_source,
        accepted_endpoint_ids=endpoint_ids,
        accepted_endpoint_fingerprints=endpoint_fingerprints,
        applied_symmetry=applied_symmetry,
        per_endpoint=comparisons,
        d_occ_best=min(float(c.d_occ) for c in comparisons),
        d_occ_mean=sum(float(c.d_occ) for c in comparisons) / len(comparisons),
        d_occ_worst=max(float(c.d_occ) for c in comparisons),
        d_placement_best=min(float(c.d_placement) for c in comparisons),
        d_placement_mean=sum(float(c.d_placement) for c in comparisons) / len(comparisons),
        d_placement_worst=max(float(c.d_placement) for c in comparisons),
        cost_best=min(float(c.assignment.total_cost) for c in comparisons),
        cost_mean=sum(float(c.assignment.total_cost) for c in comparisons) / len(comparisons),
        cost_worst=max(float(c.assignment.total_cost) for c in comparisons),
        total_moves=sum(c.assignment.moves for c in comparisons),
        total_swaps=sum(c.assignment.swaps for c in comparisons),
        total_replacements=sum(c.assignment.replacements for c in comparisons),
        total_additions=sum(c.assignment.additions for c in comparisons),
        total_removals=sum(c.assignment.removals for c in comparisons),
        start_end_material_diff=material_diff,
        inventory_valid=all(c.inventory_valid for c in comparisons),
        inventory_notes=tuple(sorted(inventory_notes)),
        rights_annotations_required=rights_annotations_required,
        transition_complexity_score=score,
        score_components=components,
        reasons_for_rejection=tuple(reasons),
    )
