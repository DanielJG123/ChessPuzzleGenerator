"""Product-suitability summaries assembled from the analysis tiers."""

from __future__ import annotations

from typing import Any

from chess_chain.analysis.collision import CollisionResult
from chess_chain.analysis.variations import TerminalVariationSummary
from chess_chain.domain.analysis import ForcednessCategory, PlyAnalysis

__all__ = ["summarize_puzzle"]


def summarize_puzzle(
    *,
    puzzle_id: str,
    outcome: str,
    cheap_plies: list[PlyAnalysis] | None = None,
    deep_plies: list[PlyAnalysis] | None = None,
    variation: TerminalVariationSummary | None = None,
    collision: CollisionResult | None = None,
) -> dict[str, Any]:
    """Return the product-suitability summary for one puzzle."""
    reference_plies = deep_plies if deep_plies is not None else cheap_plies
    if reference_plies is None:
        reference_plies = []

    solver_plies = [ply for ply in reference_plies if ply.is_solver_ply]
    opponent_plies = [ply for ply in reference_plies if not ply.is_solver_ply]

    solver_unique = sum(1 for ply in solver_plies if ply.only_legal_move)
    solver_uniqueness = solver_unique / len(solver_plies) if solver_plies else None

    opponent_forcedness: dict[str, int] = {}
    for ply in opponent_plies:
        if ply.forcedness is not None:
            opponent_forcedness[ply.forcedness.value] = (
                opponent_forcedness.get(ply.forcedness.value, 0) + 1
            )

    plausible_counts = [
        ply.plausible_count for ply in opponent_plies if ply.plausible_count is not None
    ]
    max_plausible = max(plausible_counts) if plausible_counts else None
    mean_plausible = sum(plausible_counts) / len(plausible_counts) if plausible_counts else None

    line_wide_ambiguity = any(
        ply.forcedness == ForcednessCategory.AMBIGUOUS for ply in reference_plies
    )

    possible_policies = _possible_policies(outcome, variation)

    summary: dict[str, Any] = {
        "puzzle_id": puzzle_id,
        "outcome": outcome,
        "ply_count": len(reference_plies),
        "solver_move_count": len(solver_plies),
        "solver_move_uniqueness": solver_uniqueness,
        "opponent_move_count": len(opponent_plies),
        "opponent_forcedness": opponent_forcedness,
        "max_plausible_opponent_choices": max_plausible,
        "mean_plausible_opponent_choices": mean_plausible,
        "line_wide_ambiguity": line_wide_ambiguity,
        "possible_stage_end_policies": possible_policies,
    }
    if variation is not None:
        summary["terminal_exact_position_variation"] = len(variation.terminal_fens)
        summary["terminal_occupancy_variation"] = len(variation.terminal_occupancies)
        summary["terminal_variation_verified"] = variation.endpoints_verified
        summary["terminal_variation_note"] = variation.verification_note
    if collision is not None:
        summary["final_collision_status"] = collision.status.value
        summary["final_collision_witness"] = (
            list(collision.witness_line) if collision.witness_line else None
        )
        summary["final_collision_cached"] = collision.cached
    return summary


def _possible_policies(outcome: str, variation: TerminalVariationSummary | None) -> list[str]:
    policies: list[str] = []
    if outcome == "checkmate":
        policies.append("checkmate")
    policies.append("transition_ready")
    if variation is not None:
        if variation.end_policy_kind == "CHECKMATE" and variation.endpoints_verified:
            policies.append("checkmate")
        elif (
            variation.end_policy_kind == "EXPLICIT_PREDICATE"
            or variation.end_policy_kind == "FIXED_PLIES"
        ):
            policies.append(variation.end_policy_key)
        elif variation.end_policy_kind == "TRANSITION_READY":
            policies.append("transition_ready")
    return sorted(set(policies))
