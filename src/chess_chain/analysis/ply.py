"""Shared structural per-ply analysis (pure python-chess)."""

from __future__ import annotations

from typing import Any

import chess

from chess_chain.domain.analysis import ForcednessCategory, PlyAnalysis
from chess_chain.domain.position import PositionFingerprint

__all__ = ["analyze_ply_structural", "ply_from_record"]


def analyze_ply_structural(
    board: chess.Board,
    move: chess.Move,
    *,
    ply_index: int,
    is_solver_ply: bool,
    previous_was_capture: bool,
) -> PlyAnalysis:
    """Analyse ``move`` on ``board`` and advance ``board`` past it.

    The move must be legal on ``board``. Occupancy deltas are computed exactly:
    castling moves two pieces (four squares), en passant removes a pawn from a
    non-destination square, ordinary captures remove two squares.
    """
    legal_count = len(list(board.legal_moves))
    side = board.turn
    in_check_before = board.is_check()
    evasion_count = legal_count if in_check_before else None
    only_legal_move = legal_count == 1

    squares_before = set(chess.scan_reversed(int(board.occupied)))
    occupancy_before = int(board.occupied)
    san = board.san(move)
    is_capture = board.is_capture(move)
    is_castling = board.is_castling(move)
    is_en_passant = board.is_en_passant(move)
    is_promotion = move.promotion is not None

    board.push(move)
    occupancy_after = int(board.occupied)
    squares_after = set(chess.scan_reversed(occupancy_after))
    is_check_after = board.is_check()
    is_mate_after = board.is_checkmate()
    is_forcing_check = is_check_after and len(list(board.legal_moves)) == 1
    is_recapture = is_capture and previous_was_capture
    after_fingerprint = PositionFingerprint.from_board(board)
    board.pop()

    # In standard chess no two distinct moves from one position produce the same
    # resulting position, so this is always 0; recorded explicitly rather than
    # assumed. (Different *move orders* can transpose later, which is a search
    # fact, not a same-ply fact.)
    transposing_move_count = 0
    for alternative in board.legal_moves:
        if alternative == move:
            continue
        board.push(alternative)
        if PositionFingerprint.from_board(board) == after_fingerprint:
            transposing_move_count += 1
        board.pop()

    board.push(move)

    return PlyAnalysis(
        ply_index=ply_index,
        side=side,
        is_solver_ply=is_solver_ply,
        uci=move.uci(),
        san=san,
        legal_move_count=legal_count,
        evasion_count=evasion_count,
        in_check=in_check_before,
        only_legal_move=only_legal_move,
        is_capture=is_capture,
        is_recapture=is_recapture,
        is_check=is_check_after,
        is_forcing_check=is_forcing_check,
        is_mate=is_mate_after,
        is_castling=is_castling,
        is_en_passant=is_en_passant,
        is_promotion=is_promotion,
        promotion_piece=move.promotion,
        occupancy_before=occupancy_before,
        occupancy_after=occupancy_after,
        squares_removed=tuple(sorted(squares_before - squares_after)),
        squares_added=tuple(sorted(squares_after - squares_before)),
        transposing_move_count=transposing_move_count,
        engine_rank=None,
        engine_score_cp=None,
        engine_score_mate=None,
        second_best_gap_cp=None,
        plausible_count=None,
        plausible_moves=(),
        forcedness=(ForcednessCategory.LITERAL_FORCED if only_legal_move else None),
    )


def ply_from_record(record: dict[str, Any]) -> PlyAnalysis:
    """Reconstruct a PlyAnalysis from a parquet/JSON record."""
    promotion = record.get("promotion_piece")
    return PlyAnalysis(
        ply_index=int(record["ply_index"]),
        side=chess.WHITE if int(record["side"]) == 0 else chess.BLACK,
        is_solver_ply=bool(record["is_solver_ply"]),
        uci=str(record["uci"]),
        san=str(record["san"]),
        legal_move_count=int(record["legal_move_count"]),
        evasion_count=(
            int(record["evasion_count"]) if record.get("evasion_count") is not None else None
        ),
        in_check=bool(record["in_check"]),
        only_legal_move=bool(record["only_legal_move"]),
        is_capture=bool(record["is_capture"]),
        is_recapture=bool(record["is_recapture"]),
        is_check=bool(record["is_check"]),
        is_forcing_check=bool(record["is_forcing_check"]),
        is_mate=bool(record["is_mate"]),
        is_castling=bool(record["is_castling"]),
        is_en_passant=bool(record["is_en_passant"]),
        is_promotion=bool(record["is_promotion"]),
        promotion_piece=(chess.PIECE_SYMBOLS.index(promotion) if promotion else None),
        occupancy_before=int(record["occupancy_before"]),
        occupancy_after=int(record["occupancy_after"]),
        squares_removed=tuple(int(square) for square in record["squares_removed"]),
        squares_added=tuple(int(square) for square in record["squares_added"]),
        transposing_move_count=int(record["transposing_move_count"]),
        engine_rank=(int(record["engine_rank"]) if record.get("engine_rank") is not None else None),
        engine_score_cp=(
            int(record["engine_score_cp"]) if record.get("engine_score_cp") is not None else None
        ),
        engine_score_mate=(
            int(record["engine_score_mate"])
            if record.get("engine_score_mate") is not None
            else None
        ),
        second_best_gap_cp=(
            int(record["second_best_gap_cp"])
            if record.get("second_best_gap_cp") is not None
            else None
        ),
        plausible_count=(
            int(record["plausible_count"]) if record.get("plausible_count") is not None else None
        ),
        plausible_moves=tuple(str(move) for move in (record.get("plausible_moves") or [])),
        forcedness=(
            ForcednessCategory(str(record["forcedness"])) if record.get("forcedness") else None
        ),
    )
