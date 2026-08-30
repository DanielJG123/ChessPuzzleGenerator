"""Tests for the deterministic transition executor and verification."""

from __future__ import annotations

import chess

from chess_chain.analysis.policies import TRANSITION_READY_POLICY
from chess_chain.domain.position import CanonicalPosition
from chess_chain.transitions import ast as A
from chess_chain.transitions.config import TransitionConfig
from chess_chain.transitions.context import build_context_from_boards
from chess_chain.transitions.executor import (
    STATUS_AMBIGUOUS,
    STATUS_COLLISION,
    STATUS_DUPLICATE_USE,
    STATUS_MISSING,
    STATUS_OFF_BOARD,
    STATUS_OK,
    execute_program,
)
from chess_chain.transitions.render import render_program
from chess_chain.transitions.verify import verify_program

POL = TRANSITION_READY_POLICY


def _context(presented: str, endpoint: str, solution: tuple[str, ...] = ()):
    return build_context_from_boards(
        puzzle_id="T",
        presented_fen=presented,
        endpoint_fen=endpoint,
        solution_uci=solution,
        policy=POL,
    )


def test_transition_valid_only_at_the_intended_endpoint() -> None:
    presented = "4k3/8/8/8/8/8/8/4K2R w - - 0 1"
    endpoint = "8/4k3/8/8/8/8/7R/4K3 w - - 2 2"
    next_start = CanonicalPosition.from_board(chess.Board(presented))
    context = _context(presented, endpoint, ("h1h2", "e8e7"))
    rook = A.PieceBeganOnSquare(chess.H1)
    king = A.PieceBeganOnSquare(chess.E8)
    program = A.TransitionProgram(
        (
            A.Move(rook, A.RelativeOffset(rook, 0, -1)),
            A.Move(king, A.RelativeOffset(king, 0, 1)),
        )
    )
    verification = verify_program(
        program, accepted=(context,), next_start=next_start, config=TransitionConfig()
    )
    assert verification.passed is True
    assert all(outcome.status == STATUS_OK for outcome in verification.accepted_outcomes)
    assert all(not outcome.creates_next_start for outcome in verification.non_accepted_outcomes)


def test_transition_accidentally_valid_from_a_prefix_is_rejected() -> None:
    presented = "4k3/8/8/8/8/8/8/4K2R w - - 0 1"
    endpoint = "8/4k3/8/8/8/8/7R/4K3 w - - 2 2"
    next_start = CanonicalPosition.from_board(chess.Board(presented))
    context = _context(presented, endpoint, ("h1h2", "e8e7"))
    # Generic absolute moves: at the presented start both pieces already sit on
    # their targets, so the program silently produces the next start there.
    program = A.TransitionProgram(
        (
            A.Move(A.PieceBeganOnSquare(chess.H1), A.AbsoluteSquare(chess.H1)),
            A.Move(A.PieceBeganOnSquare(chess.E8), A.AbsoluteSquare(chess.E8)),
        )
    )
    verification = verify_program(
        program, accepted=(context,), next_start=next_start, config=TransitionConfig()
    )
    assert verification.passed is False
    assert any(
        "creates the correct next start from a non-accepted state" in issue
        for issue in verification.issues
    )


def test_absolute_square_fallback_receives_a_leakage_penalty() -> None:
    presented = "4k3/8/8/8/8/8/8/4K2R w - - 0 1"
    endpoint = "8/4k3/8/8/8/8/7R/4K3 w - - 2 2"
    context = _context(presented, endpoint, ("h1h2", "e8e7"))
    next_start = CanonicalPosition.from_board(chess.Board(presented))
    rook = A.PieceBeganOnSquare(chess.H1)
    king = A.PieceBeganOnSquare(chess.E8)
    relative_program = A.TransitionProgram(
        (
            A.Move(rook, A.RelativeOffset(rook, 0, -1)),
            A.Move(king, A.RelativeOffset(king, 0, 1)),
        )
    )
    absolute_program = A.TransitionProgram(
        (
            A.Move(rook, A.AbsoluteSquare(chess.H1)),
            A.Move(king, A.AbsoluteSquare(chess.E8)),
        )
    )
    relative = verify_program(
        relative_program, accepted=(context,), next_start=next_start, config=TransitionConfig()
    )
    absolute = verify_program(
        absolute_program, accepted=(context,), next_start=next_start, config=TransitionConfig()
    )
    # The absolute variant names two more squares outright (2 * 6 bits).
    assert (
        absolute.clue.absolute_source_squares + absolute.clue.absolute_destination_squares
        > relative.clue.absolute_source_squares + relative.clue.absolute_destination_squares
    )
    assert absolute.clue.total_clue_bits > relative.clue.total_clue_bits


def test_ambiguous_nearest_selector_is_rejected() -> None:
    # Two white knights equidistant from the white king: the nearest selector
    # must refuse to choose silently.
    context = _context(
        "4k3/8/8/8/8/2N3N1/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/2N3N1/8/4K3 w - - 0 1",
    )
    selector = A.NearestTo(A.UniquePiece(chess.WHITE, chess.KING), colour=chess.WHITE)
    program = A.TransitionProgram((A.Move(selector, A.RelativeOffset(selector, 1, 0)),))
    result = execute_program(program, context)
    assert result.status == STATUS_AMBIGUOUS
    assert "ambiguous" in (result.note or "")


def test_missing_selector_and_off_board_and_duplicate_use() -> None:
    context = _context(
        "4k3/8/8/8/8/8/8/4K2R w - - 0 1",
        "4k3/8/8/8/8/8/8/4K2R w - - 0 1",
    )
    missing_selector = A.OccupantOfSquare(chess.H2)
    missing = A.TransitionProgram(
        (A.Move(missing_selector, A.RelativeOffset(missing_selector, 0, 1)),)
    )
    assert execute_program(missing, context).status == STATUS_MISSING

    off_selector = A.OccupantOfSquare(chess.H1)
    off_board = A.TransitionProgram((A.Move(off_selector, A.RelativeOffset(off_selector, 0, -1)),))
    assert execute_program(off_board, context).status == STATUS_OFF_BOARD

    dup_selector = A.PieceBeganOnSquare(chess.H1)
    duplicate = A.TransitionProgram(
        (
            A.Move(dup_selector, A.RelativeOffset(dup_selector, -1, 0)),
            A.Move(dup_selector, A.RelativeOffset(dup_selector, -1, 0)),
        )
    )
    assert execute_program(duplicate, context).status == STATUS_DUPLICATE_USE


def test_captured_piece_replacement() -> None:
    presented = "4r1k1/8/8/8/8/8/8/4R1K1 w - - 0 1"
    endpoint = "4R1k1/8/8/8/8/8/8/6K1 b - - 0 1"
    context = _context(presented, endpoint, ("e1e8",))
    # The white rook captured the black rook on e8; replace it with the
    # captured piece.
    program = A.TransitionProgram(
        (
            A.Replace(
                A.OccupantOfSquare(chess.E8),
                A.CapturedPieceBeganOnSquare(chess.E8),
            ),
        )
    )
    result = execute_program(program, context)
    assert result.status == STATUS_OK
    assert result.board is not None
    assert result.board.piece_at(chess.E8) == chess.Piece(chess.ROOK, chess.BLACK)
    assert result.board.piece_at(chess.E8) != chess.Piece(chess.ROOK, chess.WHITE)


def test_collision_is_rejected() -> None:
    context = _context(
        "4k3/8/8/8/8/8/8/RN2K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/RN2K3 w - - 0 1",
    )
    program = A.TransitionProgram(
        (A.Move(A.OccupantOfSquare(chess.A1), A.SquareOf(A.OccupantOfSquare(chess.B1))),)
    )
    assert execute_program(program, context).status == STATUS_COLLISION


def test_swap_exchanges_two_pieces() -> None:
    context = _context(
        "4k3/8/8/8/8/8/8/RN2K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/RN2K3 w - - 0 1",
    )
    program = A.TransitionProgram(
        (
            A.Swap(
                A.OccupantOfSquare(chess.A1),
                A.OccupantOfSquare(chess.B1),
            ),
        )
    )
    result = execute_program(program, context)
    assert result.status == STATUS_OK
    assert result.board is not None
    assert result.board.piece_at(chess.A1) == chess.Piece(chess.KNIGHT, chess.WHITE)
    assert result.board.piece_at(chess.B1) == chess.Piece(chess.ROOK, chess.WHITE)


def test_renderer_executor_round_trip_preserves_semantics() -> None:
    presented = "4k3/8/8/8/8/8/8/4K2R w - - 0 1"
    endpoint = "8/4k3/8/8/8/8/7R/4K3 w - - 2 2"
    context = _context(presented, endpoint, ("h1h2", "e8e7"))
    rook = A.PieceBeganOnSquare(chess.H1)
    king = A.PieceBeganOnSquare(chess.E8)
    program = A.TransitionProgram(
        (
            A.Move(rook, A.RelativeOffset(rook, 0, -1)),
            A.Move(king, A.RelativeOffset(king, 0, 1)),
        )
    )
    rendered = render_program(program)
    assert "towards rank 1" in rendered
    assert "towards rank 8" in rendered
    # AST serialisation round-trips to an identical program.
    reparsed = A.TransitionProgram.from_json(program.to_json())
    assert render_program(reparsed) == rendered
    assert execute_program(reparsed, context).status == STATUS_OK
    assert (
        execute_program(reparsed, context).fingerprint()
        == execute_program(program, context).fingerprint()
    )


def test_castling_solution_tracks_rook_in_context() -> None:
    # White queenside castling: the endpoint maps both castled pieces to their
    # real squares (regression: the rook identity used to stay on a1).
    presented = "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1"
    endpoint = "4k3/8/8/8/8/8/8/2KR4 b - - 0 1"
    ctx = _context(presented, endpoint, ("e1c1",))
    assert ctx.events[0].san == "O-O-O"
    assert ctx.identity_on(chess.C1).key() == "wke1"
    rook = ctx.identity_on(chess.D1)
    assert rook is not None
    assert rook.key() == "wra1"
    assert ctx.piece_square[rook] == chess.D1


def test_castling_core_builds_consistent_context() -> None:
    # Real Lichess row 090As ends with black kingside castling where the rook
    # delivers the final check; the derived context must agree with the core.
    from chess_chain.chess.replay import derive_core
    from chess_chain.domain.source import source_from_fields
    from chess_chain.transitions.context import build_context_from_core
    from tests.conftest import CASTLING_ROWS

    source = source_from_fields(CASTLING_ROWS[0].rstrip("\n").split(","), line_number=1)
    core = derive_core(source)
    ctx = build_context_from_core(core, POL)
    rook = ctx.identity_on(chess.F8)
    assert rook is not None
    assert rook.key() == "brh8"
    assert rook in ctx.final_checking
    assert ctx.identity_on(chess.G8) is not None  # the castled king
