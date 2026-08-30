"""Tests for the puzzle replay / derivation logic."""

from __future__ import annotations

import chess
import pytest

from chess_chain.chess.replay import (
    castling_rook_squares,
    classify_terminal,
    derive_core,
)
from chess_chain.domain.errors import IngestRejectError
from chess_chain.domain.pieces import StagePieceId
from chess_chain.domain.source import source_from_fields
from tests.conftest import (
    BAD_ROWS,
    CAPTURE_ROWS,
    CASTLING_ROWS,
    CHECK_ROWS,
    MATE_ROWS,
    PROMOTION_ROWS,
    SAMPLE_ROWS,
)


def _parse(row: str):
    return source_from_fields(row.rstrip("\n").split(","), line_number=1)


def test_preceding_move_is_applied_before_presenting() -> None:
    # 00sHx raw FEN has the black king on e8; Moves[0] = e8d7.
    core = derive_core(_parse(SAMPLE_ROWS[0]))
    presented = core.presented_start.board
    assert presented.piece_at(chess.D7) == chess.Piece(chess.KING, chess.BLACK)
    assert presented.piece_at(chess.E8) is None
    assert core.solution_uci == ("a2e6", "d7d8", "f7f8")
    assert core.solver_colour == chess.WHITE


def test_official_sample_rows_replay_without_quarantine() -> None:
    for row in SAMPLE_ROWS:
        core = derive_core(_parse(row))
        expected_solution = tuple(m for m in _parse(row).moves[1:])
        assert core.solution_uci == expected_solution
        assert core.solution_san
        assert len(core.solution_san) == len(core.solution_uci)


def test_mate_fixture() -> None:
    core = derive_core(_parse(MATE_ROWS[0]))
    assert core.outcome == "checkmate"
    assert core.is_checkmate
    assert core.is_check
    assert core.final_checking_ids == {core.last_moved_id}


def test_check_fixture() -> None:
    core = derive_core(_parse(CHECK_ROWS[0]))
    assert core.outcome == "check"
    assert core.is_check
    assert not core.is_checkmate
    assert core.final_checking_ids == {core.last_moved_id}


def test_capture_fixture_tracks_piece_identities() -> None:
    core = derive_core(_parse(CAPTURE_ROWS[0]))
    assert len(core.captured_ids) == 1
    captured = core.captured_ids[0]
    assert captured.key() == "bpg7"
    assert captured not in core.moved_ids
    assert captured not in core.unmoved_ids
    # The capturing rook moved and remains on the board.
    assert core.last_moved_id.key() == "wrg1"
    assert core.last_moved_id in core.moved_ids


def test_promotion_fixture() -> None:
    core = derive_core(_parse(PROMOTION_ROWS[0]))
    assert core.has_promotion
    assert core.promotion_events == ("e7e8q",)
    # The promoting pawn keeps its stage identity (start square e7).
    event = core.events[0]
    assert event.promotion == chess.QUEEN
    assert event.moving_piece_id.key() == "wpe7"
    assert event.moving_piece_type == chess.PAWN


def test_stage_piece_identity_moves_through_solution() -> None:
    core = derive_core(_parse(SAMPLE_ROWS[0]))
    # a2e6: the white bishop that began on a2 moves to e6.
    first_event = core.events[0]
    assert first_event.moving_piece_id.key() == "wba2"
    assert first_event.origin == chess.A2
    assert first_event.destination == chess.E6
    # d7d8: the black king that began the presented start on d7 is the mover.
    second_event = core.events[1]
    assert second_event.moving_piece_id.key() == "bkd7"
    assert second_event.origin == chess.D7
    assert second_event.destination == chess.D8


def test_unmoved_pieces_are_on_final_board() -> None:
    core = derive_core(_parse(SAMPLE_ROWS[0]))
    final_ids = {
        *core.moved_ids,
        *core.unmoved_ids,
        *(identity for identity in core.captured_ids),
    }
    presented_count = len(core.presented_start.board.piece_map())
    assert len(final_ids) == presented_count


def test_fingerprint_and_occupancy() -> None:
    core = derive_core(_parse(SAMPLE_ROWS[0]))
    start_fp = core.presented_start.fingerprint
    assert start_fp.occupied == int(core.presented_start.board.occupied)
    assert start_fp.bitboards[0] == int(core.presented_start.board.pieces(chess.PAWN, chess.WHITE))
    # Occupancy is the union of the 12 colour/piece bitboards.
    assert start_fp.occupied == sum(start_fp.bitboards)


def test_castling_fixture_tracks_rook_identity() -> None:
    # Real Lichess export row 090As: the final ply e8g8 is black kingside
    # castling; the castled rook lands on f8 and delivers the check. Before the
    # fix the rook identity was left on h8 and derive_core crashed with
    # ``KeyError: (False, 61)`` when computing final_checking_ids.
    core = derive_core(_parse(CASTLING_ROWS[0]))
    last = core.events[-1]
    assert last.uci == "e8g8"
    assert last.san == "O-O+"  # castling that delivers check
    assert last.moving_piece_id.key() == "bke8"
    rook = StagePieceId(chess.BLACK, chess.ROOK, chess.H8)
    assert core.final_position.board.piece_at(chess.F8) == chess.Piece(chess.ROOK, chess.BLACK)
    assert rook.key() == "brh8"
    # The rook genuinely moved: tracked at f8, recorded as moved, and it is the
    # piece that gives the final check.
    assert rook in core.moved_ids
    assert rook not in core.unmoved_ids
    assert core.is_check
    assert core.final_checking_ids == {rook}


def test_queenside_castling_fixture() -> None:
    # White queenside castling in the solution: king e1 -> c1, rook a1 -> d1.
    core = derive_core(_parse(CASTLING_ROWS[1]))
    event = core.events[0]
    assert event.uci == "e1c1"
    assert event.san == "O-O-O+"  # the castled rook gives check on d1
    assert event.moving_piece_id.key() == "wke1"
    rook = StagePieceId(chess.WHITE, chess.ROOK, chess.A1)
    assert core.final_position.board.piece_at(chess.D1) == chess.Piece(chess.ROOK, chess.WHITE)
    assert rook in core.moved_ids
    assert core.is_check
    assert core.final_checking_ids == {rook}


def test_castling_rook_squares_helper() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1")
    kingside = chess.Move.from_uci("e1g1")
    # Only queenside rights exist, so a kingside king move is not castling.
    no_rights = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    assert castling_rook_squares(board, kingside) is None
    queenside = chess.Move.from_uci("e1c1")
    assert castling_rook_squares(board, queenside) == (chess.A1, chess.D1)
    assert castling_rook_squares(no_rights, queenside) is None
    # The king-takes-rook UCI form is not emitted by generate_castling_moves in
    # standard mode (Lichess exports e1c1/e1g1), so it is not castling here.
    assert castling_rook_squares(board, chess.Move.from_uci("e1a1")) is None
    black_board = chess.Board("4k2r/8/8/8/8/8/8/R3K3 b k - 0 1")
    assert castling_rook_squares(black_board, chess.Move.from_uci("e8g8")) == (
        chess.H8,
        chess.F8,
    )
    assert castling_rook_squares(black_board, chess.Move.from_uci("e8c8")) is None


def test_bad_rows_are_rejected_with_reason() -> None:
    for row, reason in BAD_ROWS:
        with pytest.raises(IngestRejectError) as excinfo:
            derive_core(_parse(row))
        assert reason in str(excinfo.value)


def test_classify_terminal_stalemate() -> None:
    board = chess.Board("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")
    outcome, is_check, is_mate, is_stalemate = classify_terminal(board)
    assert outcome == "stalemate"
    assert is_stalemate
    assert not is_check
    assert not is_mate


def test_canonical_equality_ignores_move_counters() -> None:
    a = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    b = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 10 42")
    from chess_chain.domain.position import CanonicalPosition

    assert CanonicalPosition.from_board(a) == CanonicalPosition.from_board(b)
    assert hash(CanonicalPosition.from_board(a)) == hash(CanonicalPosition.from_board(b))
    assert CanonicalPosition.from_board(a).fen != b.fen
