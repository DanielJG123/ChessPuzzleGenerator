"""Replay of Lichess puzzle lines with python-chess.

Source convention (see docs/architecture.md):

    raw_board       = Board(FEN)
    preceding_move  = Moves[0]
    presented_start = raw_board after preceding_move
    solution_moves  = Moves[1:]
    final_board     = presented_start after all solution_moves
"""

from __future__ import annotations

import chess

from chess_chain.domain.errors import IngestRejectError
from chess_chain.domain.moves import MoveEvent
from chess_chain.domain.pieces import MaterialSignature, StagePieceId
from chess_chain.domain.position import CanonicalPosition
from chess_chain.domain.puzzle import PuzzleCore
from chess_chain.domain.source import SourcePuzzle

__all__ = ["PieceTracker", "classify_terminal", "derive_core"]


def classify_terminal(board: chess.Board) -> tuple[str, bool, bool, bool]:
    """Return ``(outcome, is_check, is_checkmate, is_stalemate)`` for ``board``."""
    if board.is_checkmate():
        return "checkmate", True, True, False
    if board.is_stalemate():
        return "stalemate", False, False, True
    if board.is_check():
        return "check", True, False, False
    return "neither", False, False, False


class PieceTracker:
    """Tracks stage-local piece identities through a solution.

    Identities are assigned from the presented start position and followed
    square by square. Captured pieces are removed; promoted pawns keep their
    original identity (their current piece type is recorded in the MoveEvent).
    """

    def __init__(self, board: chess.Board) -> None:
        self._by_square: dict[tuple[chess.Color, chess.Square], StagePieceId] = {}
        self._squares: dict[StagePieceId, chess.Square] = {}
        for square, piece in board.piece_map().items():
            identity = StagePieceId(piece.color, piece.piece_type, square)
            self._by_square[(piece.color, square)] = identity
            self._squares[identity] = square

    def identity_at(self, square: chess.Square, colour: chess.Color) -> StagePieceId:
        return self._by_square[(colour, square)]

    def move(self, identity: StagePieceId, destination: chess.Square) -> None:
        origin = self._squares[identity]
        del self._by_square[(identity.colour, origin)]
        self._squares[identity] = destination
        self._by_square[(identity.colour, destination)] = identity

    def capture(self, identity: StagePieceId) -> None:
        square = self._squares.pop(identity)
        del self._by_square[(identity.colour, square)]

    def on_board(self) -> set[StagePieceId]:
        return set(self._squares)


def derive_core(source: SourcePuzzle) -> PuzzleCore:
    """Replay ``source`` into a PuzzleCore.

    Raises :class:`IngestRejectError` for malformed FENs, illegal preceding
    moves, illegal solution moves and impossible lines so the streaming
    pipeline can quarantine the row without aborting the run.
    """
    try:
        raw_board = chess.Board(source.fen)
    except ValueError as exc:
        raise IngestRejectError(f"malformed FEN: {exc}") from exc

    if len(source.moves) < 2:
        raise IngestRejectError(
            "no solution moves (moves list must contain a preceding move "
            "and at least one solution move)"
        )

    try:
        preceding = chess.Move.from_uci(source.moves[0])
    except ValueError as exc:
        raise IngestRejectError(f"malformed preceding move {source.moves[0]!r}") from exc
    if preceding not in raw_board.legal_moves:
        raise IngestRejectError(f"illegal preceding move {source.moves[0]!r}")

    raw_board.push(preceding)
    presented_start = CanonicalPosition.from_board(raw_board)
    solver_colour = raw_board.turn

    tracker = PieceTracker(raw_board)
    events: list[MoveEvent] = []
    moved_ids: set[StagePieceId] = set()
    captured_order: list[StagePieceId] = []
    promotion_events: list[str] = []
    solution_uci: list[str] = []
    solution_san: list[str] = []
    last_moved_id: StagePieceId | None = None

    for ply_index, uci in enumerate(source.moves[1:]):
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as exc:
            raise IngestRejectError(f"malformed solution move {uci!r}") from exc
        if move not in raw_board.legal_moves:
            raise IngestRejectError(f"illegal solution move {uci!r} at ply {ply_index}")

        origin_piece = raw_board.piece_at(move.from_square)
        assert origin_piece is not None
        moving_side = raw_board.turn
        moving_id = tracker.identity_at(move.from_square, moving_side)
        moving_type = origin_piece.piece_type

        captured_id: StagePieceId | None = None
        captured_type: chess.PieceType | None = None
        capture_square: chess.Square | None = None
        if raw_board.is_capture(move):
            if raw_board.is_en_passant(move):
                capture_square = chess.square(
                    chess.square_file(move.to_square),
                    chess.square_rank(move.from_square),
                )
            else:
                capture_square = move.to_square
            captured_piece = raw_board.piece_at(capture_square)
            assert captured_piece is not None
            captured_type = captured_piece.piece_type
            captured_id = tracker.identity_at(capture_square, not moving_side)

        san = raw_board.san(move)
        raw_board.push(move)
        is_check = raw_board.is_check()
        is_mate = raw_board.is_checkmate()

        if captured_id is not None:
            tracker.capture(captured_id)
            captured_order.append(captured_id)
        tracker.move(moving_id, move.to_square)
        moved_ids.add(moving_id)
        last_moved_id = moving_id

        if move.promotion is not None:
            promotion_events.append(uci)

        events.append(
            MoveEvent(
                ply_index=ply_index,
                side=moving_side,
                uci=uci,
                san=san,
                moving_piece_id=moving_id,
                moving_piece_type=moving_type,
                origin=move.from_square,
                destination=move.to_square,
                captured_piece_id=captured_id,
                captured_piece_type=captured_type,
                capture_square=capture_square,
                promotion=move.promotion,
                is_check=is_check,
                is_mate=is_mate,
            )
        )
        solution_uci.append(uci)
        solution_san.append(san)

    final_position = CanonicalPosition.from_board(raw_board)
    outcome, is_check, is_checkmate, is_stalemate = classify_terminal(raw_board)

    checkers = raw_board.checkers()
    final_checking_ids = frozenset(
        tracker.identity_at(square, not raw_board.turn) for square in checkers
    )

    start_material = MaterialSignature.from_board(presented_start.board)
    final_material = MaterialSignature.from_board(raw_board)
    moved_ids_frozen = frozenset(moved_ids)
    captured_ids_tuple = tuple(captured_order)
    final_ids = tracker.on_board()
    unmoved_ids = frozenset(final_ids - moved_ids_frozen)

    if last_moved_id is None:
        raise IngestRejectError("no solution moves")

    return PuzzleCore(
        source=source,
        presented_start=presented_start,
        solution_uci=tuple(solution_uci),
        solution_san=tuple(solution_san),
        final_position=final_position,
        solver_colour=solver_colour,
        outcome=outcome,
        is_check=is_check,
        is_checkmate=is_checkmate,
        is_stalemate=is_stalemate,
        events=tuple(events),
        moved_ids=moved_ids_frozen,
        unmoved_ids=unmoved_ids,
        captured_ids=captured_ids_tuple,
        last_moved_id=last_moved_id,
        final_checking_ids=final_checking_ids,
        promotion_events=tuple(promotion_events),
        has_promotion=bool(promotion_events),
        duplicate_queens=start_material.duplicate_queens() or final_material.duplicate_queens(),
        unusual_material=start_material.exceeds_standard_set()
        or final_material.exceeds_standard_set(),
    )
