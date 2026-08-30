"""Decomposable, configurable candidate scoring.

The transition-complexity score is a weighted sum of explainable components:

- best / mean / worst occupancy Hamming distance over the accepted endpoints;
- best / mean / worst per-endpoint physical-assignment cost;
- total swaps and replacements (lifts/crossings beyond plain moves);
- a penalty for applying a non-identity symmetry;
- a penalty when any endpoint requires establishing non-visible rights
  (castling / en-passant) that the occupancy-only checker cannot verify;
- a penalty when the physical inventory cannot reach the next start;
- a per-branch penalty when several accepted endpoints must be normalised by
  one program with verified branches.

Lower is better. Every component and its weight is exposed by
``compute_candidate_score`` so the number is never unexplained.
"""

from __future__ import annotations

from dataclasses import dataclass

from chess_chain.similarity.distance import EndpointComparison
from chess_chain.similarity.symmetry import SYMMETRY_IDENTITY

__all__ = [
    "DEFAULT_SCORE_WEIGHTS",
    "SimilarityScoreWeights",
    "compute_candidate_score",
]


@dataclass(frozen=True)
class SimilarityScoreWeights:
    """Weights for the decomposable transition-complexity score."""

    occupancy_best: float = 3.0
    occupancy_mean: float = 1.5
    occupancy_worst: float = 0.5
    cost_best: float = 1.0
    cost_mean: float = 0.5
    cost_worst: float = 0.25
    swap: float = 1.0
    replace: float = 4.0
    branch: float = 2.0
    symmetry_non_identity: float = 3.0
    rights_annotation: float = 5.0
    inventory_invalid: float = 20.0

    def as_dict(self) -> dict[str, float]:
        return {
            "occupancy_best": self.occupancy_best,
            "occupancy_mean": self.occupancy_mean,
            "occupancy_worst": self.occupancy_worst,
            "cost_best": self.cost_best,
            "cost_mean": self.cost_mean,
            "cost_worst": self.cost_worst,
            "swap": self.swap,
            "replace": self.replace,
            "branch": self.branch,
            "symmetry_non_identity": self.symmetry_non_identity,
            "rights_annotation": self.rights_annotation,
            "inventory_invalid": self.inventory_invalid,
        }


DEFAULT_SCORE_WEIGHTS = SimilarityScoreWeights()


def compute_candidate_score(
    comparisons: tuple[EndpointComparison, ...],
    *,
    applied_symmetry: str,
    weights: SimilarityScoreWeights | None = None,
) -> tuple[float, dict[str, float]]:
    """Return ``(score, components)``; every component is a named scalar."""
    w = weights or DEFAULT_SCORE_WEIGHTS
    if not comparisons:
        return float("inf"), {}

    d_occ_values = [float(c.d_occ) for c in comparisons]
    cost_values = [float(c.assignment.total_cost) for c in comparisons]
    rights = any(c.castling_annotation_required or c.ep_annotation_required for c in comparisons)
    inventory_invalid = any(not c.inventory_valid for c in comparisons)

    components = {
        "occupancy_best": w.occupancy_best * min(d_occ_values),
        "occupancy_mean": w.occupancy_mean * (sum(d_occ_values) / len(d_occ_values)),
        "occupancy_worst": w.occupancy_worst * max(d_occ_values),
        "cost_best": w.cost_best * min(cost_values),
        "cost_mean": w.cost_mean * (sum(cost_values) / len(cost_values)),
        "cost_worst": w.cost_worst * max(cost_values),
        "swaps": w.swap * sum(c.assignment.swaps for c in comparisons),
        "replacements": w.replace * sum(c.assignment.replacements for c in comparisons),
        "branches": w.branch * max(0, len(comparisons) - 1),
        "symmetry_non_identity": (
            w.symmetry_non_identity if applied_symmetry != SYMMETRY_IDENTITY else 0.0
        ),
        "rights_annotation": w.rights_annotation if rights else 0.0,
        "inventory_invalid": w.inventory_invalid if inventory_invalid else 0.0,
    }
    return sum(components.values()), components
