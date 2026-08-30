"""Deep tier: Stockfish MultiPV analysis for shortlisted puzzles.

Analysis is cached by the canonical engine FEN plus a complete
engine/configuration fingerprint. Every result records the engine name/version
and the budget used.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import chess

from chess_chain.analysis.cache import DiskKeyValueCache
from chess_chain.analysis.engine import (
    EngineAnalysisResult,
    EngineLike,
    engine_fen,
    engine_fingerprint_from_identity,
)
from chess_chain.analysis.forcedness import ForcednessConfig, classify_forcedness
from chess_chain.analysis.ply import analyze_ply_structural
from chess_chain.domain.analysis import PlyAnalysis

__all__ = ["analyze_deep_puzzle", "estimate_engine_calls", "ply_to_deep_record"]


def _solver_colour(value: str) -> chess.Color:
    return chess.WHITE if value == "white" else chess.BLACK


def estimate_engine_calls(
    *,
    start_fen: str,
    solution_uci: tuple[str, ...] | list[str],
    cache: DiskKeyValueCache,
    engine_fingerprint: str,
) -> tuple[int, int]:
    """Count total positions and how many are already cached."""
    board = chess.Board(start_fen)
    total = 0
    cached = 0
    for uci in solution_uci:
        move = chess.Move.from_uci(uci)
        key = cache.key_from({"fen": engine_fen(board), "engine": engine_fingerprint})
        if cache.has(key):
            cached += 1
        total += 1
        board.push(move)
    return total, cached


def analyze_deep_puzzle(
    *,
    puzzle_id: str,
    start_fen: str,
    solution_uci: tuple[str, ...] | list[str],
    solver_colour: str,
    engine: EngineLike,
    cache: DiskKeyValueCache,
    forcedness: ForcednessConfig,
    engine_fingerprint: str | None = None,
) -> tuple[list[PlyAnalysis], dict[str, object]]:
    """Analyse every ply with MultiPV engine scores; returns (plies, engine meta)."""
    board = chess.Board(start_fen)
    colour = _solver_colour(solver_colour)
    engine_name, engine_version = engine.engine_identity()
    budget = str(getattr(engine, "budget", "{}"))
    if engine_fingerprint is None:
        engine_fingerprint = engine_fingerprint_from_identity(
            budget=budget, name=engine_name, version=engine_version
        )

    plies: list[PlyAnalysis] = []
    previous_was_capture = False
    for ply_index, uci in enumerate(solution_uci):
        move = chess.Move.from_uci(uci)
        fen_key = engine_fen(board)
        cache_key = cache.key_from({"fen": fen_key, "engine": engine_fingerprint})
        cached_value = cache.get(cache_key)
        if cached_value is not None:
            result = EngineAnalysisResult.from_dict(cached_value)
        else:
            result = engine.analyze(board, mate_value_cp=forcedness.mate_value_cp)
            cache.put(cache_key, result.to_dict())

        infos = result.moves
        best = infos[0] if infos else None
        intended = next((info for info in infos if info.uci == uci), None)
        second = infos[1] if len(infos) > 1 else None

        intended_gap: int | None = None
        if (
            best is not None
            and intended is not None
            and best.score_cp is not None
            and intended.score_cp is not None
        ):
            intended_gap = best.score_cp - intended.score_cp
        second_gap: int | None = None
        if (
            best is not None
            and second is not None
            and best.score_cp is not None
            and second.score_cp is not None
        ):
            second_gap = best.score_cp - second.score_cp
        plausible = [
            info
            for info in infos
            if best is not None
            and best.score_cp is not None
            and info.score_cp is not None
            and best.score_cp - info.score_cp <= forcedness.plausible_gap_cp
        ]

        analysis = analyze_ply_structural(
            board,
            move,
            ply_index=ply_index,
            is_solver_ply=board.turn == colour,
            previous_was_capture=previous_was_capture,
        )
        previous_was_capture = analysis.is_capture
        forcedness_category = classify_forcedness(
            legal_move_count=analysis.legal_move_count,
            intended_rank=intended.rank if intended is not None else None,
            intended_gap_cp=intended_gap,
            second_gap_cp=second_gap,
            engine_analysed=bool(infos),
            config=forcedness,
        )
        deep_ply = replace(
            analysis,
            engine_rank=intended.rank if intended is not None else None,
            engine_score_cp=intended.score_cp if intended is not None else None,
            engine_score_mate=intended.score_mate if intended is not None else None,
            second_best_gap_cp=second_gap,
            plausible_count=len(plausible),
            plausible_moves=tuple(info.uci for info in plausible),
            forcedness=forcedness_category,
        )
        plies.append(deep_ply)

    meta: dict[str, object] = {
        "engine_name": engine_name,
        "engine_version": engine_version,
        "engine_fingerprint": engine_fingerprint,
        "budget": budget,
        "forcedness_version": forcedness.version,
    }
    return plies, meta


def ply_to_deep_record(
    *,
    puzzle_id: str,
    source: str,
    transform_version: int,
    ply: PlyAnalysis,
    engine_meta: dict[str, Any],
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
        "engine_rank": ply.engine_rank,
        "engine_score_cp": ply.engine_score_cp,
        "engine_score_mate": ply.engine_score_mate,
        "second_best_gap_cp": ply.second_best_gap_cp,
        "plausible_count": ply.plausible_count,
        "plausible_moves": list(ply.plausible_moves),
        "engine_name": str(engine_meta["engine_name"]),
        "engine_version": str(engine_meta["engine_version"]),
        "analysis_budget": str(engine_meta["budget"]),
        "engine_fingerprint": str(engine_meta["engine_fingerprint"]),
        "forcedness_version": int(engine_meta["forcedness_version"]),
    }
