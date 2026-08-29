"""Stable pyarrow schemas for processed puzzle data."""

from __future__ import annotations

import pyarrow as pa

from chess_chain.domain.pieces import COUNT_NAMES

#: Column names for the 12 colour/piece-type material counts and bitboards.
#: The `start_bb_*` columns are 64-bit bitboards stored as unsigned integers.

PUZZLE_SCHEMA = pa.schema(
    [
        pa.field("puzzle_id", pa.string()),
        pa.field("source", pa.string()),
        pa.field("transform_version", pa.uint16()),
        pa.field("rating", pa.int32()),
        pa.field("rating_deviation", pa.int32()),
        pa.field("popularity", pa.int32()),
        pa.field("nb_plays", pa.int32()),
        pa.field("themes", pa.list_(pa.string())),
        pa.field("opening_tags", pa.list_(pa.string())),
        pa.field("game_url", pa.string()),
        pa.field("daily_date", pa.string()),
        pa.field("start_fen", pa.string()),
        pa.field("start_fingerprint", pa.string()),
        *[pa.field(f"start_bb_{name}", pa.uint64()) for name in COUNT_NAMES],
        pa.field("start_occupancy", pa.uint64()),
        pa.field("start_castling", pa.uint64()),
        pa.field("start_ep_square", pa.uint16()),
        *[pa.field(f"start_{name}", pa.uint8()) for name in COUNT_NAMES],
        pa.field("start_material_w", pa.int16()),
        pa.field("start_material_b", pa.int16()),
        pa.field("presented_piece_count", pa.uint8()),
        pa.field("solution_uci", pa.list_(pa.string())),
        pa.field("solution_san", pa.list_(pa.string())),
        pa.field("solution_plies", pa.uint8()),
        pa.field("final_fen", pa.string()),
        pa.field("final_fingerprint", pa.string()),
        pa.field("final_occupancy", pa.uint64()),
        *[pa.field(f"final_{name}", pa.uint8()) for name in COUNT_NAMES],
        pa.field("final_material_w", pa.int16()),
        pa.field("final_material_b", pa.int16()),
        pa.field("outcome", pa.string()),
        pa.field("is_check", pa.bool_()),
        pa.field("is_checkmate", pa.bool_()),
        pa.field("is_stalemate", pa.bool_()),
        pa.field("solver_colour", pa.string()),
        pa.field("moved_piece_ids", pa.list_(pa.string())),
        pa.field("unmoved_piece_ids", pa.list_(pa.string())),
        pa.field("captured_piece_ids", pa.list_(pa.string())),
        pa.field("last_moved_piece_id", pa.string()),
        pa.field("final_checking_piece_ids", pa.list_(pa.string())),
        pa.field("promotion_events", pa.list_(pa.string())),
        pa.field("has_promotion", pa.bool_()),
        pa.field("duplicate_queens", pa.bool_()),
        pa.field("unusual_material", pa.bool_()),
    ]
)

QUARANTINE_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string()),
        pa.field("line_number", pa.int64()),
        pa.field("puzzle_id", pa.string()),
        pa.field("reason", pa.string()),
        pa.field("raw_fields", pa.list_(pa.string())),
    ]
)
