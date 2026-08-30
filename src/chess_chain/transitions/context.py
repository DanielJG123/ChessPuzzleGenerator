"""EndpointContext: everything a transition program may refer to.

The context captures, for one accepted endpoint of a stage:

- the stage presented-start position;
- the exact solved endpoint;
- tracked stage-local piece identities (physical pieces);
- the complete intended move history;
- captured pieces in capture order;
- the last-moved piece;
- final checking and mating pieces where applicable;
- occupied/vacant squares (derivable from the endpoint board);
- castling rights and en-passant state (in the endpoint fingerprint);
- the selected ``StageEndPolicy``.

Builders exist for PuzzleCore-derived contexts, stored puzzle rows, explicit
boards (synthetic tests) and intermediate prefixes / plausible wrong states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess

from chess_chain.analysis.policies import StageEndPolicy
from chess_chain.chess.replay import castling_rook_squares
from chess_chain.domain.moves import MoveEvent
from chess_chain.domain.pieces import StagePieceId
from chess_chain.domain.position import CanonicalPosition
from chess_chain.domain.puzzle import PuzzleCore

__all__ = [
    "EndpointContext",
    "build_context_from_core",
    "build_context_from_row",
    "build_context_from_boards",
    "build_prefix_contexts",
    "replay_solution",
]


@dataclass(frozen=True)
class EndpointContext:
    """Everything a transition program may refer to (see module docstring)."""

    puzzle_id: str
    presented_start: CanonicalPosition
    endpoint: CanonicalPosition
    identities: dict[chess.Square, StagePieceId]  # endpoint square -> physical piece
    piece_square: dict[StagePieceId, chess.Square | None]  # None = off board (tray)
    start_square: dict[StagePieceId, chess.Square]
    events: tuple[MoveEvent, ...]
    captured_order: tuple[StagePieceId, ...]
    last_moved: StagePieceId | None
    final_checking: frozenset[StagePieceId]
    mating_pieces: frozenset[StagePieceId]
    stage_end_policy: StageEndPolicy
    solver_colour: chess.Color

    def endpoint_board(self) -> chess.Board:
        return self.endpoint.board

    def identity_on(self, square: chess.Square) -> StagePieceId | None:
        return self.identities.get(square)

    def tray_pieces(self) -> frozenset[StagePieceId]:
        return frozenset(piece for piece, square in self.piece_square.items() if square is None)

    def on_board(self) -> frozenset[StagePieceId]:
        return frozenset(piece for piece, square in self.piece_square.items() if square is not None)


def replay_solution(
    start_board: chess.Board,
    solution_uci: tuple[str, ...],
) -> tuple[chess.Board, tuple[MoveEvent, ...], dict[chess.Square, StagePieceId]]:
    """Replay the intended solution and return ``(board, events, identities)``.

    Physical identities are stage-local: ``(colour, type, start_square)``
    assigned from the presented start and tracked square by square.
    """
    board = start_board.copy()
    tracker = _IdentityTracker(board)
    events: list[MoveEvent] = []
    for ply_index, uci in enumerate(solution_uci):
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal solution move {uci!r} at ply {ply_index}")
        side = board.turn
        moving_id = tracker.identity_at(move.from_square)
        captured_id: StagePieceId | None = None
        captured_type: int | None = None
        capture_square: chess.Square | None = None
        if board.is_capture(move):
            if board.is_en_passant(move):
                capture_square = chess.square(
                    chess.square_file(move.to_square),
                    chess.square_rank(move.from_square),
                )
            else:
                capture_square = move.to_square
            captured_piece = board.piece_at(capture_square)
            assert captured_piece is not None
            captured_type = captured_piece.piece_type
            captured_id = tracker.identity_at(capture_square)
        san = board.san(move)
        castling = castling_rook_squares(board, move)
        board.push(move)
        is_check = board.is_check()
        is_mate = board.is_checkmate()
        if captured_id is not None:
            tracker.capture(captured_id)
        tracker.move(moving_id, move.to_square)
        if castling is not None:
            rook_from, rook_to = castling
            tracker.move(tracker.identity_at(rook_from), rook_to)
        events.append(
            MoveEvent(
                ply_index=ply_index,
                side=side,
                uci=uci,
                san=san,
                moving_piece_id=moving_id,
                moving_piece_type=tracker.piece_type(moving_id),
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
    return board, tuple(events), dict(tracker.by_square)


class _IdentityTracker:
    """Square <-> stage-local identity bookkeeping during a replay."""

    def __init__(self, board: chess.Board) -> None:
        self.by_square: dict[chess.Square, StagePieceId] = {}
        self.square_of: dict[StagePieceId, chess.Square] = {}
        self._type_of: dict[StagePieceId, chess.PieceType] = {}
        for square, piece in board.piece_map().items():
            identity = StagePieceId(piece.color, piece.piece_type, square)
            self.by_square[square] = identity
            self.square_of[identity] = square
            self._type_of[identity] = piece.piece_type

    def identity_at(self, square: chess.Square) -> StagePieceId:
        try:
            return self.by_square[square]
        except KeyError as exc:
            raise ValueError(f"no piece on {chess.square_name(square)}") from exc

    def capture(self, identity: StagePieceId) -> None:
        square = self.square_of.pop(identity)
        self.by_square.pop(square, None)
        self._type_of.pop(identity, None)

    def move(self, identity: StagePieceId, destination: chess.Square) -> None:
        origin = self.square_of.pop(identity)
        self.by_square.pop(origin, None)
        self.square_of[identity] = destination
        self.by_square[destination] = identity

    def piece_type(self, identity: StagePieceId) -> chess.PieceType:
        return self._type_of[identity]


def _assemble_context(
    *,
    puzzle_id: str,
    presented_start: CanonicalPosition,
    endpoint: CanonicalPosition,
    events: tuple[MoveEvent, ...],
    captured_order: tuple[StagePieceId, ...],
    last_moved: StagePieceId | None,
    identities: dict[chess.Square, StagePieceId],
    piece_square: dict[StagePieceId, chess.Square | None],
    start_square: dict[StagePieceId, chess.Square],
    policy: StageEndPolicy,
    solver_colour: chess.Color,
) -> EndpointContext:
    final_checking = frozenset(
        piece
        for square in endpoint.board.checkers()
        if (piece := identities.get(square)) is not None
    )
    mating_pieces = final_checking if endpoint.board.is_checkmate() else frozenset()
    return EndpointContext(
        puzzle_id=puzzle_id,
        presented_start=presented_start,
        endpoint=endpoint,
        identities=identities,
        piece_square=piece_square,
        start_square=start_square,
        events=events,
        captured_order=captured_order,
        last_moved=last_moved,
        final_checking=final_checking,
        mating_pieces=mating_pieces,
        stage_end_policy=policy,
        solver_colour=solver_colour,
    )


def build_context_from_core(core: PuzzleCore, policy: StageEndPolicy) -> EndpointContext:
    """Build the full-fidelity context from a stage-2 ``PuzzleCore``."""
    return build_context_from_boards(
        puzzle_id=core.source.puzzle_id,
        presented_fen=core.presented_start.fen,
        endpoint_fen=core.final_position.fen,
        solution_uci=core.solution_uci,
        solver_colour=core.solver_colour,
        policy=policy,
    )


def piece_squares_after(
    start_board: chess.Board,
    solution_uci: tuple[str, ...],
    pieces: frozenset[StagePieceId] | None = None,
) -> dict[StagePieceId, chess.Square | None]:
    """Track physical pieces through a line and return their final squares
    (``None`` = captured). Used by the plausible-line generator."""
    board = start_board.copy()
    tracker = _IdentityTracker(board)
    for uci in solution_uci:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal line move {uci!r}")
        if board.is_capture(move):
            if board.is_en_passant(move):
                capture_square = chess.square(
                    chess.square_file(move.to_square),
                    chess.square_rank(move.from_square),
                )
            else:
                capture_square = move.to_square
            tracker.capture(tracker.identity_at(capture_square))
        moving = tracker.identity_at(move.from_square)
        castling = castling_rook_squares(board, move)
        tracker.move(moving, move.to_square)
        if castling is not None:
            rook_from, rook_to = castling
            tracker.move(tracker.identity_at(rook_from), rook_to)
        board.push(move)
    if pieces is None:
        pieces = frozenset(tracker.square_of)
    return {piece: tracker.square_of.get(piece) for piece in pieces}


def build_context_from_row(row: dict[str, Any], policy: StageEndPolicy) -> EndpointContext:
    """Build a context from a stored puzzle row (presented start, solution,
    solver colour) by replaying the intended line."""
    return build_context_from_boards(
        puzzle_id=str(row["puzzle_id"]),
        presented_fen=str(row["start_fen"]),
        endpoint_fen=str(row["final_fen"]),
        solution_uci=tuple(str(move) for move in row["solution_uci"]),
        solver_colour=chess.WHITE if str(row["solver_colour"]) == "white" else chess.BLACK,
        policy=policy,
    )


def build_context_from_boards(
    *,
    puzzle_id: str,
    presented_fen: str,
    endpoint_fen: str,
    solution_uci: tuple[str, ...] = (),
    solver_colour: chess.Color = chess.WHITE,
    policy: StageEndPolicy,
) -> EndpointContext:
    """Build a context from explicit boards.

    When ``solution_uci`` is empty the endpoint is taken as-is: physical
    identities fall back to ``(colour, type, current_square)`` and history
    selectors (last-moved, captured pieces, ...) resolve to errors.
    """
    start_board = chess.Board(presented_fen)
    endpoint_board = chess.Board(endpoint_fen)
    presented_start = CanonicalPosition.from_board(start_board)
    endpoint = CanonicalPosition.from_board(endpoint_board)

    if not solution_uci:
        identities: dict[chess.Square, StagePieceId] = {}
        empty_piece_square: dict[StagePieceId, chess.Square | None] = {}
        empty_start_square: dict[StagePieceId, chess.Square] = {}
        for square, piece in endpoint_board.piece_map().items():
            identity = StagePieceId(piece.color, piece.piece_type, square)
            identities[square] = identity
            empty_piece_square[identity] = square
            empty_start_square[identity] = square
        return _assemble_context(
            puzzle_id=puzzle_id,
            presented_start=presented_start,
            endpoint=endpoint,
            events=(),
            captured_order=(),
            last_moved=None,
            identities=identities,
            piece_square=empty_piece_square,
            start_square=empty_start_square,
            policy=policy,
            solver_colour=solver_colour,
        )

    board, events, identities = replay_solution(start_board, solution_uci)
    if CanonicalPosition.from_board(board).fingerprint != endpoint.fingerprint:
        raise ValueError("endpoint_fen does not match the replayed solution")
    captured_order = tuple(
        event.captured_piece_id for event in events if event.captured_piece_id is not None
    )
    last_moved = events[-1].moving_piece_id if events else None
    piece_square: dict[StagePieceId, chess.Square | None] = {}
    start_square: dict[StagePieceId, chess.Square] = {}
    for event in events:
        identity = event.moving_piece_id
        if identity not in start_square:
            start_square[identity] = event.origin
        if event.captured_piece_id is not None:
            piece_square[event.captured_piece_id] = None
    # Pieces that never moved keep their start square; captured pieces are not
    # on the board; every endpoint square maps to a physical piece.
    for square, identity in identities.items():
        piece_square[identity] = square
        start_square.setdefault(identity, square)
    # The StagePieceId carries the true start square for every physical piece,
    # including captured ones that never appear in ``identities``.
    for tray_piece in piece_square:
        start_square.setdefault(tray_piece, tray_piece.start_square)
    return _assemble_context(
        puzzle_id=puzzle_id,
        presented_start=presented_start,
        endpoint=endpoint,
        events=events,
        captured_order=captured_order,
        last_moved=last_moved,
        identities=identities,
        piece_square=piece_square,
        start_square=start_square,
        policy=policy,
        solver_colour=solver_colour,
    )


def build_prefix_contexts(
    context: EndpointContext, policy: StageEndPolicy
) -> tuple[EndpointContext, ...]:
    """Contexts for the presented start and every proper intended prefix.

    These are the states the transition (and the stage-boundary guard) must be
    invalid or non-unique on.
    """
    solution = tuple(event.uci for event in context.events)
    contexts: list[EndpointContext] = []
    for length in range(0, len(solution)):
        if length == 0:
            contexts.append(
                build_context_from_boards(
                    puzzle_id=context.puzzle_id,
                    presented_fen=context.presented_start.fen,
                    endpoint_fen=context.presented_start.fen,
                    solver_colour=context.solver_colour,
                    policy=policy,
                )
            )
        else:
            contexts.append(
                build_context_from_boards(
                    puzzle_id=context.puzzle_id,
                    presented_fen=context.presented_start.fen,
                    endpoint_fen=_fen_after(context.presented_start.fen, solution[:length]),
                    solution_uci=solution[:length],
                    solver_colour=context.solver_colour,
                    policy=policy,
                )
            )
    return tuple(contexts)


def _fen_after(fen: str, moves: tuple[str, ...]) -> str:
    board = chess.Board(fen)
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
    return board.fen()
