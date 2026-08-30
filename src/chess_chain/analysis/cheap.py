"""Cheap tier: pure python-chess structural analysis for all ingested puzzles."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import chess
import pyarrow as pa
import pyarrow.parquet as pq

from chess_chain.analysis.ply import analyze_ply_structural
from chess_chain.analysis.schema import CHEAP_PLY_SCHEMA
from chess_chain.domain.analysis import PlyAnalysis

__all__ = [
    "analyze_cheap_puzzle",
    "ply_to_cheap_record",
    "stream_cheap_analysis",
]


def _solver_colour(value: str) -> chess.Color:
    return chess.WHITE if value == "white" else chess.BLACK


def analyze_cheap_puzzle(
    *,
    puzzle_id: str,
    start_fen: str,
    solution_uci: tuple[str, ...] | list[str],
    solver_colour: str,
) -> list[PlyAnalysis]:
    """Replay ``solution_uci`` from ``start_fen`` and analyse every ply."""
    board = chess.Board(start_fen)
    colour = _solver_colour(solver_colour)
    plies: list[PlyAnalysis] = []
    previous_was_capture = False
    for ply_index, uci in enumerate(solution_uci):
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"{puzzle_id}: illegal stored move {uci!r} at ply {ply_index}")
        analysis = analyze_ply_structural(
            board,
            move,
            ply_index=ply_index,
            is_solver_ply=board.turn == colour,
            previous_was_capture=previous_was_capture,
        )
        previous_was_capture = analysis.is_capture
        plies.append(analysis)
    return plies


def ply_to_cheap_record(
    *,
    puzzle_id: str,
    source: str,
    transform_version: int,
    ply: PlyAnalysis,
) -> dict[str, object]:
    return {
        "puzzle_id": puzzle_id,
        "source": source,
        "transform_version": transform_version,
        "ply_index": ply.ply_index,
        "side": int(ply.side),
        "is_solver_ply": ply.is_solver_ply,
        "uci": ply.uci,
        "san": ply.san,
        "legal_move_count": ply.legal_move_count,
        "evasion_count": ply.evasion_count,
        "in_check": ply.in_check,
        "only_legal_move": ply.only_legal_move,
        "is_capture": ply.is_capture,
        "is_recapture": ply.is_recapture,
        "is_check": ply.is_check,
        "is_forcing_check": ply.is_forcing_check,
        "is_mate": ply.is_mate,
        "is_castling": ply.is_castling,
        "is_en_passant": ply.is_en_passant,
        "is_promotion": ply.is_promotion,
        "promotion_piece": (
            chess.piece_symbol(ply.promotion_piece) if ply.promotion_piece else None
        ),
        "occupancy_before": ply.occupancy_before,
        "occupancy_after": ply.occupancy_after,
        "squares_removed": list(ply.squares_removed),
        "squares_added": list(ply.squares_added),
        "transposing_move_count": ply.transposing_move_count,
        "forcedness": ply.forcedness.value if ply.forcedness is not None else None,
    }


def stream_cheap_analysis(
    *,
    rows: Iterable[tuple[Any, ...]],
    output_path: object,
) -> int:
    """Write cheap ply records for ``rows`` to Parquet; return ply count.

    Each row is ``(puzzle_id, source, transform_version, start_fen,
    solution_uci, solver_colour)``. ``output_path`` is a path-like object.
    """
    writer = pq.ParquetWriter(str(output_path), CHEAP_PLY_SCHEMA, compression="zstd")
    total = 0
    try:
        buffer: list[dict[str, object]] = []
        for row in rows:
            puzzle_id, source, transform_version, start_fen, solution_uci, solver_colour = (
                row[0],
                row[1],
                int(row[2]),
                row[3],
                row[4],
                row[5],
            )
            plies = analyze_cheap_puzzle(
                puzzle_id=str(puzzle_id),
                start_fen=str(start_fen),
                solution_uci=list(solution_uci),
                solver_colour=str(solver_colour),
            )
            for ply in plies:
                buffer.append(
                    ply_to_cheap_record(
                        puzzle_id=str(puzzle_id),
                        source=str(source),
                        transform_version=transform_version,
                        ply=ply,
                    )
                )
                total += 1
                if len(buffer) >= 2000:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=CHEAP_PLY_SCHEMA))
                    buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=CHEAP_PLY_SCHEMA))
    finally:
        writer.close()
    return total
