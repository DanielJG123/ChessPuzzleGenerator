"""Stable pyarrow schemas for stage-3 analysis outputs."""

from __future__ import annotations

import pyarrow as pa

CHEAP_PLY_SCHEMA = pa.schema(
    [
        pa.field("puzzle_id", pa.string()),
        pa.field("source", pa.string()),
        pa.field("transform_version", pa.uint16()),
        pa.field("ply_index", pa.uint16()),
        pa.field("side", pa.uint8()),
        pa.field("is_solver_ply", pa.bool_()),
        pa.field("uci", pa.string()),
        pa.field("san", pa.string()),
        pa.field("legal_move_count", pa.uint16()),
        pa.field("evasion_count", pa.uint16()),
        pa.field("in_check", pa.bool_()),
        pa.field("only_legal_move", pa.bool_()),
        pa.field("is_capture", pa.bool_()),
        pa.field("is_recapture", pa.bool_()),
        pa.field("is_check", pa.bool_()),
        pa.field("is_forcing_check", pa.bool_()),
        pa.field("is_mate", pa.bool_()),
        pa.field("is_castling", pa.bool_()),
        pa.field("is_en_passant", pa.bool_()),
        pa.field("is_promotion", pa.bool_()),
        pa.field("promotion_piece", pa.string()),
        pa.field("occupancy_before", pa.uint64()),
        pa.field("occupancy_after", pa.uint64()),
        pa.field("squares_removed", pa.list_(pa.uint8())),
        pa.field("squares_added", pa.list_(pa.uint8())),
        pa.field("transposing_move_count", pa.uint16()),
        pa.field("forcedness", pa.string()),
    ]
)

DEEP_PLY_SCHEMA = pa.schema(
    [
        *CHEAP_PLY_SCHEMA,
        pa.field("engine_rank", pa.uint16()),
        pa.field("engine_score_cp", pa.int32()),
        pa.field("engine_score_mate", pa.int16()),
        pa.field("second_best_gap_cp", pa.int32()),
        pa.field("plausible_count", pa.uint16()),
        pa.field("plausible_moves", pa.list_(pa.string())),
        pa.field("engine_name", pa.string()),
        pa.field("engine_version", pa.string()),
        pa.field("analysis_budget", pa.string()),
        pa.field("engine_fingerprint", pa.string()),
        pa.field("forcedness_version", pa.uint16()),
    ]
)

VARIATIONS_SCHEMA = pa.schema(
    [
        pa.field("puzzle_id", pa.string()),
        pa.field("end_policy_key", pa.string()),
        pa.field("end_policy_kind", pa.string()),
        pa.field("acceptance", pa.string()),
        pa.field("line_count", pa.uint32()),
        pa.field("terminal_fens", pa.list_(pa.string())),
        pa.field("terminal_occupancies", pa.list_(pa.uint64())),
        pa.field("exact_position_invariant", pa.bool_()),
        pa.field("occupancy_invariant", pa.bool_()),
        pa.field("endpoints_verified", pa.bool_()),
        pa.field("verification_note", pa.string()),
        pa.field("truncated", pa.bool_()),
        pa.field("max_plies", pa.uint16()),
        pa.field("max_nodes", pa.uint32()),
        pa.field("cache_key", pa.string()),
    ]
)

COLLISION_SCHEMA = pa.schema(
    [
        pa.field("puzzle_id", pa.string()),
        pa.field("source_version", pa.uint16()),
        pa.field("presented_fingerprint", pa.string()),
        pa.field("target_occupancy", pa.uint64()),
        pa.field("accepted_line_policy", pa.string()),
        pa.field("stage_end_policy", pa.string()),
        pa.field("scope", pa.string()),
        pa.field("max_depth", pa.uint16()),
        pa.field("max_nodes", pa.uint32()),
        pa.field("pruning_config", pa.string()),
        pa.field("analyser_version", pa.uint16()),
        pa.field("cache_key", pa.string()),
        pa.field("status", pa.string()),
        pa.field("witness_line", pa.list_(pa.string())),
        pa.field("witness_depth", pa.uint16()),
        pa.field("nodes_searched", pa.uint32()),
        pa.field("verified_acceptance_occupancy", pa.bool_()),
        pa.field("note", pa.string()),
        pa.field("cached", pa.bool_()),
    ]
)
