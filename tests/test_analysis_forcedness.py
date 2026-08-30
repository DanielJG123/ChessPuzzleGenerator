"""Tests for forcedness categories, literal/unique/ambiguous replies and deep analysis."""

from __future__ import annotations

import chess

from chess_chain.analysis.cheap import analyze_cheap_puzzle
from chess_chain.analysis.forcedness import (
    ForcednessConfig,
    classify_forcedness,
)
from chess_chain.domain.analysis import ForcednessCategory
from tests.conftest import FakeEngineAdapter, engine_fen

CONFIG = ForcednessConfig(
    version=1,
    unique_best_gap_cp=50,
    practically_forced_gap_cp=200,
    mate_value_cp=10_000,
    plausible_gap_cp=30,
)


def _forced_position_fen() -> str:
    # Black to move, in check from the rook on g8 (defended by the queen on g1);
    # exactly one legal move (h8h7).
    return "6Rk/8/8/8/8/8/8/6QK b - - 1 2"


def test_literal_forced_king_response() -> None:
    plies = analyze_cheap_puzzle(
        puzzle_id="FORCE1",
        start_fen=_forced_position_fen(),
        solution_uci=("h8h7",),
        solver_colour="black",
    )
    assert len(plies) == 1
    ply = plies[0]
    assert ply.legal_move_count == 1
    assert ply.only_legal_move is True
    assert ply.in_check is True
    assert ply.evasion_count == 1
    assert ply.is_solver_ply is True
    assert ply.forcedness == ForcednessCategory.LITERAL_FORCED


def test_classify_forcedness_unit_cases() -> None:
    assert (
        classify_forcedness(
            legal_move_count=1,
            intended_rank=0,
            intended_gap_cp=0,
            second_gap_cp=None,
            engine_analysed=True,
            config=CONFIG,
        )
        == ForcednessCategory.LITERAL_FORCED
    )
    # Engine absent -> None (deferred to the deep tier).
    assert (
        classify_forcedness(
            legal_move_count=5,
            intended_rank=None,
            intended_gap_cp=None,
            second_gap_cp=None,
            engine_analysed=False,
            config=CONFIG,
        )
        is None
    )
    # Intended not among the scored candidates -> invalid source line.
    assert (
        classify_forcedness(
            legal_move_count=3,
            intended_rank=None,
            intended_gap_cp=None,
            second_gap_cp=10,
            engine_analysed=True,
            config=CONFIG,
        )
        == ForcednessCategory.INVALID_SOURCE_LINE
    )
    # Unique best: clear gap to second.
    assert (
        classify_forcedness(
            legal_move_count=3,
            intended_rank=0,
            intended_gap_cp=0,
            second_gap_cp=80,
            engine_analysed=True,
            config=CONFIG,
        )
        == ForcednessCategory.UNIQUE_BEST
    )
    # Practically forced: every alternative decisively worse.
    assert (
        classify_forcedness(
            legal_move_count=3,
            intended_rank=0,
            intended_gap_cp=0,
            second_gap_cp=250,
            engine_analysed=True,
            config=CONFIG,
        )
        == ForcednessCategory.PRACTICALLY_FORCED
    )
    # Ambiguous: close second.
    assert (
        classify_forcedness(
            legal_move_count=3,
            intended_rank=0,
            intended_gap_cp=0,
            second_gap_cp=20,
            engine_analysed=True,
            config=CONFIG,
        )
        == ForcednessCategory.AMBIGUOUS
    )
    # Intended worse than best but not decisively -> ambiguous.
    assert (
        classify_forcedness(
            legal_move_count=3,
            intended_rank=2,
            intended_gap_cp=120,
            second_gap_cp=10,
            engine_analysed=True,
            config=CONFIG,
        )
        == ForcednessCategory.AMBIGUOUS
    )
    # Intended decisively worse -> invalid source line.
    assert (
        classify_forcedness(
            legal_move_count=3,
            intended_rank=2,
            intended_gap_cp=400,
            second_gap_cp=10,
            engine_analysed=True,
            config=CONFIG,
        )
        == ForcednessCategory.INVALID_SOURCE_LINE
    )


def _three_move_position_fen() -> str:
    # Black to move (not in check); exactly three legal king moves: h7, g8, g7.
    return "7k/8/8/8/8/8/8/R5K1 b - - 0 1"


def _run_deep(start_fen: str, results: dict[str, list[tuple[str, int]]], tmp_path):
    from chess_chain.analysis.cache import DiskKeyValueCache
    from chess_chain.analysis.deep import analyze_deep_puzzle

    engine = FakeEngineAdapter(results)
    cache = DiskKeyValueCache(tmp_path / "engine_cache")
    plies, meta = analyze_deep_puzzle(
        puzzle_id="DEEP1",
        start_fen=start_fen,
        solution_uci=("h8h7",),
        solver_colour="black",
        engine=engine,
        cache=cache,
        forcedness=CONFIG,
    )
    return plies, engine


def test_unique_best_defence_with_fake_engine(tmp_path) -> None:
    fen = engine_fen(chess.Board(_three_move_position_fen()))
    plies, engine = _run_deep(
        _three_move_position_fen(),
        {fen: [("h8h7", 0), ("h8g8", -80), ("h8g7", -300)]},
        tmp_path,
    )
    assert engine.calls == [fen]
    ply = plies[0]
    assert ply.engine_rank == 0
    assert ply.engine_score_cp == 0
    assert ply.second_best_gap_cp == 80
    assert ply.forcedness == ForcednessCategory.UNIQUE_BEST


def test_genuinely_ambiguous_reply_with_fake_engine(tmp_path) -> None:
    fen = engine_fen(chess.Board(_three_move_position_fen()))
    plies, _ = _run_deep(
        _three_move_position_fen(),
        {fen: [("h8h7", 0), ("h8g8", -10), ("h8g7", -20)]},
        tmp_path,
    )
    assert plies[0].forcedness == ForcednessCategory.AMBIGUOUS
    assert plies[0].plausible_count == 3


def test_practically_forced_reply_with_fake_engine(tmp_path) -> None:
    fen = engine_fen(chess.Board(_three_move_position_fen()))
    plies, _ = _run_deep(
        _three_move_position_fen(),
        {fen: [("h8h7", 0), ("h8g8", -250), ("h8g7", -500)]},
        tmp_path,
    )
    assert plies[0].forcedness == ForcednessCategory.PRACTICALLY_FORCED


def test_invalid_source_line_when_intended_not_scored(tmp_path) -> None:
    fen = engine_fen(chess.Board(_three_move_position_fen()))
    plies, _ = _run_deep(
        _three_move_position_fen(),
        # Intended h8h7 is deliberately missing from the engine output.
        {fen: [("h8g8", 0), ("h8g7", -50)]},
        tmp_path,
    )
    assert plies[0].forcedness == ForcednessCategory.INVALID_SOURCE_LINE


def test_deep_analysis_is_deterministic_and_cached(tmp_path) -> None:
    from chess_chain.analysis.cache import DiskKeyValueCache
    from chess_chain.analysis.deep import analyze_deep_puzzle

    fen = " ".join(chess.Board(_three_move_position_fen()).fen().split()[:4])
    engine = FakeEngineAdapter({fen: [("h8h7", 0), ("h8g8", -80), ("h8g7", -300)]})
    cache = DiskKeyValueCache(tmp_path / "engine_cache")
    first, meta1 = analyze_deep_puzzle(
        puzzle_id="DEEP1",
        start_fen=_three_move_position_fen(),
        solution_uci=("h8h7",),
        solver_colour="black",
        engine=engine,
        cache=cache,
        forcedness=CONFIG,
    )
    calls_after_first = list(engine.calls)
    second, meta2 = analyze_deep_puzzle(
        puzzle_id="DEEP1",
        start_fen=_three_move_position_fen(),
        solution_uci=("h8h7",),
        solver_colour="black",
        engine=engine,
        cache=cache,
        forcedness=CONFIG,
    )
    # Second run reused the cache: no new engine calls, identical results.
    assert engine.calls == calls_after_first
    assert meta1 == meta2
    assert second[0].engine_score_cp == first[0].engine_score_cp
    assert meta1["engine_name"] == "FakeStockfish"


def test_engine_absence_is_clear_and_cheap_still_works(tmp_path) -> None:
    from chess_chain.analysis.engine import EngineAdapter, EngineConfig, EngineUnavailableError

    engine = EngineAdapter(EngineConfig(path=""))
    try:
        engine.engine_identity()
    except EngineUnavailableError as exc:
        assert "engine" in str(exc).lower()
    else:
        raise AssertionError("expected EngineUnavailableError for an empty engine path")

    # Cheap analysis never touches an engine.
    plies = analyze_cheap_puzzle(
        puzzle_id="FORCE1",
        start_fen=_forced_position_fen(),
        solution_uci=("h8h7",),
        solver_colour="black",
    )
    assert plies[0].legal_move_count == 1
