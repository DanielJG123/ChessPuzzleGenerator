"""Tests for the puzzle replay / derivation logic."""

from __future__ import annotations

import chess
import pytest

from chess_chain.chess.replay import classify_terminal, derive_core
from chess_chain.domain.errors import IngestRejectError
from chess_chain.domain.source import source_from_fields
from tests.conftest import (
    BAD_ROWS,
    CAPTURE_ROWS,
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
