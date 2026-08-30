"""Stable pyarrow schemas for similarity outputs.

The candidate schema stores the queryable aggregate/scalar fields as typed
columns and keeps the detailed per-endpoint journeys and score components as
JSON strings (consumed by the explain command and machine-readable reports).
"""

from __future__ import annotations

import pyarrow as pa

from chess_chain.domain.pieces import COUNT_NAMES

INDEX_SCHEMA = pa.schema(
    [
        pa.field("puzzle_id", pa.string()),
        pa.field("source", pa.string()),
        pa.field("transform_version", pa.uint16()),
        pa.field("rating", pa.int32()),
        pa.field("rating_deviation", pa.int32()),
        pa.field("popularity", pa.int32()),
        pa.field("nb_plays", pa.int32()),
        pa.field("start_fen", pa.string()),
        pa.field("presented_piece_count", pa.uint8()),
        pa.field("start_occupancy", pa.uint64()),
        pa.field("start_castling", pa.uint64()),
        pa.field("start_ep_square", pa.uint16()),
        *[pa.field(f"start_{name}", pa.uint8()) for name in COUNT_NAMES],
        pa.field("duplicate_queens", pa.bool_()),
        pa.field("unusual_material", pa.bool_()),
    ]
)

CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("source_puzzle_id", pa.string()),
        pa.field("next_puzzle_id", pa.string()),
        pa.field("source_stage_end_policy", pa.string()),
        pa.field("endpoint_source", pa.string()),
        pa.field("applied_symmetry", pa.string()),
        pa.field("accepted_endpoint_fingerprints", pa.list_(pa.string())),
        pa.field("d_occ_best", pa.float64()),
        pa.field("d_occ_mean", pa.float64()),
        pa.field("d_occ_worst", pa.float64()),
        pa.field("d_placement_best", pa.float64()),
        pa.field("d_placement_mean", pa.float64()),
        pa.field("d_placement_worst", pa.float64()),
        pa.field("cost_best", pa.float64()),
        pa.field("cost_mean", pa.float64()),
        pa.field("cost_worst", pa.float64()),
        pa.field("total_moves", pa.uint32()),
        pa.field("total_swaps", pa.uint32()),
        pa.field("total_replacements", pa.uint32()),
        pa.field("total_additions", pa.uint32()),
        pa.field("total_removals", pa.uint32()),
        pa.field("inventory_valid", pa.bool_()),
        pa.field("rights_annotations_required", pa.bool_()),
        pa.field("transition_complexity_score", pa.float64()),
        pa.field("reasons_for_rejection", pa.list_(pa.string())),
        pa.field("per_endpoint_json", pa.string()),
        pa.field("score_components_json", pa.string()),
        pa.field("material_diff_json", pa.string()),
    ]
)
