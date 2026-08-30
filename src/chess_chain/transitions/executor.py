"""Deterministic transition executor.

The executor is the only semantic authority for the AST. It detects ambiguity,
missing selectors, off-board destinations, collisions, duplicate use of a
physical piece and inventory violations, and never silently chooses among ties.

Physical semantics:

- pieces are stage-local ``StagePieceId`` objects; each may be referenced by at
  most one action (``duplicate_piece_use``);
- a ``Move`` lands on an empty square (or its own square); landing on another
  piece is a ``collision``;
- ``Remove`` sends a piece to the tray; ``Add`` / ``Replace`` place a *tray*
  piece back on the board;
- castling rights are tracked like standard chess: moving a king or a rook off
  its home square voids the corresponding right. A transition that needs the
  next start's rights must state them explicitly with ``EstablishRights``;
- a whole-board ``Transform`` is reconstructed and validated with python-chess.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import chess

from chess_chain.domain.pieces import StagePieceId
from chess_chain.domain.position import CanonicalPosition, PositionFingerprint
from chess_chain.similarity.symmetry import apply_symmetry
from chess_chain.transitions import ast as A
from chess_chain.transitions.context import EndpointContext

STATUS_OK = "ok"
STATUS_AMBIGUOUS = "ambiguous_selector"
STATUS_MISSING = "missing_selector"
STATUS_OFF_BOARD = "off_board"
STATUS_COLLISION = "collision"
STATUS_DUPLICATE_USE = "duplicate_piece_use"
STATUS_INVENTORY = "inventory_violation"
STATUS_INVALID_TRANSFORM = "invalid_transform"
STATUS_RIGHTS_CONFLICT = "rights_conflict"
STATUS_NO_BRANCH = "no_branch_applies"
STATUS_INVALID_DESTINATION = "invalid_destination"
STATUS_INVALID_PROGRAM = "invalid_program"

__all__ = [
    "STATUS_AMBIGUOUS",
    "STATUS_COLLISION",
    "STATUS_DUPLICATE_USE",
    "STATUS_INVALID_DESTINATION",
    "STATUS_INVALID_PROGRAM",
    "STATUS_INVALID_TRANSFORM",
    "STATUS_INVENTORY",
    "STATUS_MISSING",
    "STATUS_NO_BRANCH",
    "STATUS_OFF_BOARD",
    "STATUS_OK",
    "STATUS_RIGHTS_CONFLICT",
    "ProgramResult",
    "TransitionError",
    "execute_program",
    "evaluate_condition",
    "resolve_destination",
    "resolve_selector",
]


class TransitionError(Exception):
    """An execution failure with a machine-readable ``status``."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ProgramResult:
    """The outcome of executing a transition program on one context."""

    status: str
    board: chess.Board | None
    note: str | None = None
    used: frozenset[StagePieceId] | None = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def fingerprint(self) -> PositionFingerprint | None:
        return PositionFingerprint.from_board(self.board) if self.board is not None else None


class _State:
    """Mutable execution state: working board, piece positions and usage."""

    def __init__(self, context: EndpointContext) -> None:
        self.context = context
        self.board = context.endpoint.board.copy()
        self.piece_square: dict[StagePieceId, chess.Square | None] = dict(context.piece_square)
        self.used: set[StagePieceId] = set()

    def owner_at(self, square: chess.Square) -> StagePieceId | None:
        for piece, piece_square in self.piece_square.items():
            if piece_square == square:
                return piece
        return None

    def on_board(self, piece: StagePieceId) -> chess.Square | None:
        return self.piece_square.get(piece)

    def mark_used(self, piece: StagePieceId) -> None:
        if piece in self.used:
            raise TransitionError(
                STATUS_DUPLICATE_USE,
                f"physical piece {piece.key()} is referenced by more than one action",
            )
        self.used.add(piece)

    def check_unused(self, piece: StagePieceId, role: str) -> None:
        if piece in self.used:
            raise TransitionError(
                STATUS_DUPLICATE_USE,
                f"{role} {piece.key()} is already used by an earlier action",
            )

    def set_square(self, piece: StagePieceId, square: chess.Square | None) -> None:
        self.piece_square[piece] = square

    def void_rights_for(self, piece: StagePieceId, origin: chess.Square) -> None:
        """Void castling rights in standard-chess fashion when a king or rook
        leaves its home square."""
        piece_at_origin = self.board.piece_at(origin)
        if piece_at_origin is None:
            return
        if piece_at_origin.piece_type == chess.KING:
            if origin == chess.E1:
                self.board.castling_rights &= ~((1 << chess.H1) | (1 << chess.A1))
            elif origin == chess.E8:
                self.board.castling_rights &= ~((1 << chess.H8) | (1 << chess.A8))
        elif piece_at_origin.piece_type == chess.ROOK:
            if origin == chess.H1:
                self.board.castling_rights &= ~(1 << chess.H1)
            elif origin == chess.A1:
                self.board.castling_rights &= ~(1 << chess.A1)
            elif origin == chess.H8:
                self.board.castling_rights &= ~(1 << chess.H8)
            elif origin == chess.A8:
                self.board.castling_rights &= ~(1 << chess.A8)


def resolve_selector(state: _State, selector: A.Selector) -> tuple[StagePieceId, ...]:
    """Resolve a selector to its candidate pieces (may be empty)."""
    context = state.context
    if isinstance(selector, A.PieceBeganOnSquare):
        return tuple(
            piece for piece, square in context.start_square.items() if square == selector.square
        )
    if isinstance(selector, A.OccupantOfSquare):
        owner = state.owner_at(selector.square)
        return (owner,) if owner is not None else ()
    if isinstance(selector, A.UniquePiece):
        return tuple(
            piece
            for piece, square in state.piece_square.items()
            if square is not None
            and _piece_type_of(state, piece) == selector.piece_type
            and _colour_of(state, piece) == selector.colour
        )
    if isinstance(selector, A.LastMovedPiece):
        return (context.last_moved,) if context.last_moved is not None else ()
    if isinstance(selector, A.PieceMadePly):
        if 0 <= selector.ply_index < len(context.events):
            return (context.events[selector.ply_index].moving_piece_id,)
        return ()
    if isinstance(selector, A.FirstCapturedPiece):
        return (context.captured_order[0],) if context.captured_order else ()
    if isinstance(selector, A.LastCapturedPiece):
        return (context.captured_order[-1],) if context.captured_order else ()
    if isinstance(selector, A.CapturedPieceBeganOnSquare):
        return tuple(
            piece
            for piece, square in state.piece_square.items()
            if square is None and piece.start_square == selector.square
        )
    if isinstance(selector, A.CheckingPiece):
        return _checking_pieces(state)
    if isinstance(selector, A.MatingPiece):
        return _checking_pieces(state) if state.board.is_checkmate() else ()
    if isinstance(selector, A.CheckedKing):
        if not state.board.is_check():
            return ()
        king_square = state.board.king(state.board.turn)
        if king_square is None:
            return ()
        owner = state.owner_at(king_square)
        return (owner,) if owner is not None else ()
    if isinstance(selector, A.NearestTo):
        return _nearest_or_farthest(state, selector)
    if isinstance(selector, A.InDirection):
        return _in_direction(state, selector)
    raise TransitionError(STATUS_INVALID_PROGRAM, f"unsupported selector {selector!r}")


def resolve_unique_selector(state: _State, selector: A.Selector, role: str) -> StagePieceId:
    """Resolve to exactly one piece; never silently chooses among ties."""
    candidates = resolve_selector(state, selector)
    if not candidates:
        raise TransitionError(STATUS_MISSING, f"{role}: selector resolves to no piece")
    if len(candidates) > 1:
        names = ", ".join(piece.key() for piece in candidates)
        raise TransitionError(STATUS_AMBIGUOUS, f"{role}: selector is ambiguous ({names})")
    return candidates[0]


def _checking_pieces(state: _State) -> tuple[StagePieceId, ...]:
    pieces: list[StagePieceId] = []
    for square in state.board.checkers():
        owner = state.owner_at(square)
        if owner is not None:
            pieces.append(owner)
    return tuple(pieces)


def _nearest_or_farthest(state: _State, selector: A.NearestTo) -> tuple[StagePieceId, ...]:
    anchor = resolve_unique_selector(state, selector.anchor, "nearest anchor")
    anchor_square = state.on_board(anchor)
    if anchor_square is None:
        raise TransitionError(STATUS_OFF_BOARD, "nearest anchor is off the board")
    candidates = [
        piece
        for piece, square in state.piece_square.items()
        if square is not None and piece != anchor
    ]
    if selector.colour is not None:
        candidates = [piece for piece in candidates if _colour_of(state, piece) == selector.colour]
    if not candidates:
        return ()
    ranked = sorted(
        candidates,
        key=lambda piece: _manhattan(state.on_board(piece), anchor_square),  # type: ignore[arg-type]
    )
    if selector.farthest:
        ranked = list(reversed(ranked))
    best = _manhattan(state.on_board(ranked[0]), anchor_square)  # type: ignore[arg-type]
    return tuple(
        piece
        for piece in ranked
        if _manhattan(state.on_board(piece), anchor_square) == best  # type: ignore[arg-type]
    )


def _in_direction(state: _State, selector: A.InDirection) -> tuple[StagePieceId, ...]:
    anchor = resolve_unique_selector(state, selector.anchor, "direction anchor")
    anchor_square = state.on_board(anchor)
    if anchor_square is None:
        raise TransitionError(STATUS_OFF_BOARD, "direction anchor is off the board")
    delta = _DIRECTION_DELTAS[selector.direction]
    file = chess.square_file(anchor_square) + delta[0]
    rank = chess.square_rank(anchor_square) + delta[1]
    while 0 <= file < 8 and 0 <= rank < 8:
        square = chess.square(file, rank)
        owner = state.owner_at(square)
        if owner is not None:
            return (owner,)
        file += delta[0]
        rank += delta[1]
    return ()


_DIRECTION_DELTAS: dict[str, tuple[int, int]] = {
    "n": (0, 1),
    "s": (0, -1),
    "e": (1, 0),
    "w": (-1, 0),
    "ne": (1, 1),
    "nw": (-1, 1),
    "se": (1, -1),
    "sw": (-1, -1),
}


def _piece_type_of(state: _State, piece: StagePieceId) -> chess.PieceType:
    square = state.on_board(piece)
    if square is None:
        raise TransitionError(STATUS_OFF_BOARD, f"{piece.key()} is off the board")
    piece_at = state.board.piece_at(square)
    assert piece_at is not None
    return piece_at.piece_type


def _colour_of(state: _State, piece: StagePieceId) -> chess.Color:
    square = state.on_board(piece)
    if square is None:
        raise TransitionError(STATUS_OFF_BOARD, f"{piece.key()} is off the board")
    piece_at = state.board.piece_at(square)
    assert piece_at is not None
    return piece_at.color


def _manhattan(a: chess.Square, b: chess.Square) -> int:
    return abs(chess.square_file(a) - chess.square_file(b)) + abs(
        chess.square_rank(a) - chess.square_rank(b)
    )


def resolve_destination(state: _State, destination: A.Destination) -> chess.Square:
    """Resolve a destination expression to a square (off-board raises)."""
    if isinstance(destination, A.RelativeOffset):
        piece = resolve_unique_selector(state, destination.piece, "relative destination")
        origin = state.on_board(piece)
        if origin is None:
            raise TransitionError(STATUS_OFF_BOARD, "relative offset source is off the board")
        return _offset_square(origin, destination.dx, destination.dy)
    if isinstance(destination, A.SquareOf):
        piece = resolve_unique_selector(state, destination.piece, "square-of destination")
        square = state.on_board(piece)
        if square is None:
            raise TransitionError(STATUS_OFF_BOARD, "square-of piece is off the board")
        return square
    if isinstance(destination, A.VacatedBy):
        piece = resolve_unique_selector(state, destination.piece, "vacated-by destination")
        origin = _last_origin_of(state.context, piece)
        if origin is None:
            raise TransitionError(
                STATUS_INVALID_DESTINATION, f"{piece.key()} never moved (no vacated square)"
            )
        return origin
    if isinstance(destination, A.PlyOrigin):
        events = state.context.events
        if not (0 <= destination.ply_index < len(events)):
            raise TransitionError(STATUS_MISSING, "ply index out of range")
        return events[destination.ply_index].origin
    if isinstance(destination, A.RelativeToAnchor):
        anchor = resolve_unique_selector(state, destination.anchor, "anchor destination")
        anchor_square = state.on_board(anchor)
        if anchor_square is None:
            raise TransitionError(STATUS_OFF_BOARD, "anchor is off the board")
        return _offset_square(anchor_square, destination.dx, destination.dy)
    if isinstance(destination, A.AbsoluteSquare):
        if not (0 <= destination.square < 64):
            raise TransitionError(STATUS_INVALID_DESTINATION, "absolute square out of range")
        return chess.Square(destination.square)
    raise TransitionError(STATUS_INVALID_PROGRAM, f"unsupported destination {destination!r}")


def _offset_square(origin: chess.Square, dx: int, dy: int) -> chess.Square:
    file = chess.square_file(origin) + dx
    rank = chess.square_rank(origin) + dy
    if not (0 <= file < 8) or not (0 <= rank < 8):
        raise TransitionError(
            STATUS_OFF_BOARD,
            f"offset ({dx:+d},{dy:+d}) from {chess.square_name(origin)} leaves the board",
        )
    return chess.square(file, rank)


def _last_origin_of(context: EndpointContext, piece: StagePieceId) -> chess.Square | None:
    for event in reversed(context.events):
        if event.moving_piece_id == piece:
            return event.origin
    return None


def evaluate_condition(state: _State, condition: A.Condition) -> bool:
    """Deterministically evaluate a branch condition on the current state."""
    if isinstance(condition, A.SquareOccupied):
        return state.board.piece_at(condition.square) is not None
    if isinstance(condition, A.SquareEmpty):
        return state.board.piece_at(condition.square) is None
    if isinstance(condition, A.PieceAt):
        piece = state.board.piece_at(condition.square)
        return (
            piece is not None
            and piece.piece_type == condition.piece_type
            and int(piece.color) == condition.colour
        )
    if isinstance(condition, A.SelectorResolves):
        try:
            return len(resolve_selector(state, condition.selector)) == 1
        except TransitionError:
            return False
    if isinstance(condition, A.OnBoard):
        pieces = resolve_selector(state, condition.selector)
        return len(pieces) == 1 and state.on_board(pieces[0]) is not None
    if isinstance(condition, A.OffBoard):
        pieces = resolve_selector(state, condition.selector)
        return len(pieces) == 1 and state.on_board(pieces[0]) is None
    if isinstance(condition, A.LastPlyWas):
        events = state.context.events
        return bool(events) and events[-1].uci == condition.uci
    raise TransitionError(STATUS_INVALID_PROGRAM, f"unsupported condition {condition!r}")


def execute_program(
    program: A.TransitionProgram,
    context: EndpointContext,
) -> ProgramResult:
    """Execute ``program`` on ``context`` and return the resulting board.

    The resulting board carries the *natural* castling rights after the
    physical rearrangements (rights voided by king/rook moves) plus any
    explicitly established rights via ``EstablishRights``.
    """
    state = _State(context)
    try:
        for action in program.actions:
            _execute_action(state, action)
    except TransitionError as exc:
        return ProgramResult(status=exc.status, board=None, note=str(exc))
    return ProgramResult(status=STATUS_OK, board=state.board, used=frozenset(state.used))


def _execute_action(state: _State, action: A.Action) -> None:
    if isinstance(action, A.Move):
        _action_move(state, action)
    elif isinstance(action, A.Swap):
        _action_swap(state, action)
    elif isinstance(action, A.Remove):
        _action_remove(state, action)
    elif isinstance(action, A.Replace):
        _action_replace(state, action)
    elif isinstance(action, A.Add):
        _action_add(state, action)
    elif isinstance(action, A.Transform):
        _action_transform(state, action)
    elif isinstance(action, A.EstablishRights):
        _action_establish_rights(state, action)
    elif isinstance(action, A.Conditional):
        _action_conditional(state, action)
    else:
        raise TransitionError(STATUS_INVALID_PROGRAM, f"unsupported action {action!r}")


def _action_move(state: _State, action: A.Move) -> None:
    piece = resolve_unique_selector(state, action.piece, "move piece")
    state.check_unused(piece, "move")
    origin = state.on_board(piece)
    if origin is None:
        raise TransitionError(STATUS_OFF_BOARD, f"move: {piece.key()} is off the board")
    destination = resolve_destination(state, action.destination)
    occupant = state.owner_at(destination)
    if occupant is not None and occupant != piece:
        raise TransitionError(
            STATUS_COLLISION,
            f"move {piece.key()} to {chess.square_name(destination)} "
            f"collides with {occupant.key()}",
        )
    state.void_rights_for(piece, origin)
    if origin == destination:
        state.mark_used(piece)
        return
    moving = state.board.piece_at(origin)
    assert moving is not None
    state.board.remove_piece_at(origin)
    state.board.set_piece_at(destination, moving)
    state.set_square(piece, destination)
    state.mark_used(piece)


def _action_swap(state: _State, action: A.Swap) -> None:
    piece_a = resolve_unique_selector(state, action.a, "swap piece A")
    piece_b = resolve_unique_selector(state, action.b, "swap piece B")
    if piece_a == piece_b:
        raise TransitionError(STATUS_COLLISION, "swap requires two distinct pieces")
    state.check_unused(piece_a, "swap")
    state.check_unused(piece_b, "swap")
    square_a = state.on_board(piece_a)
    square_b = state.on_board(piece_b)
    if square_a is None or square_b is None:
        raise TransitionError(STATUS_OFF_BOARD, "swap pieces must both be on the board")
    state.void_rights_for(piece_a, square_a)
    state.void_rights_for(piece_b, square_b)
    piece_on_a = state.board.piece_at(square_a)
    piece_on_b = state.board.piece_at(square_b)
    assert piece_on_a is not None and piece_on_b is not None
    state.board.remove_piece_at(square_a)
    state.board.remove_piece_at(square_b)
    state.board.set_piece_at(square_b, piece_on_a)
    state.board.set_piece_at(square_a, piece_on_b)
    state.set_square(piece_a, square_b)
    state.set_square(piece_b, square_a)
    state.mark_used(piece_a)
    state.mark_used(piece_b)


def _action_remove(state: _State, action: A.Remove) -> None:
    piece = resolve_unique_selector(state, action.piece, "remove piece")
    state.check_unused(piece, "remove")
    origin = state.on_board(piece)
    if origin is None:
        raise TransitionError(STATUS_OFF_BOARD, f"remove: {piece.key()} is already off the board")
    state.void_rights_for(piece, origin)
    state.board.remove_piece_at(origin)
    state.set_square(piece, None)
    state.mark_used(piece)


def _action_replace(state: _State, action: A.Replace) -> None:
    occupant = resolve_unique_selector(state, action.occupant, "replace occupant")
    replacement = resolve_unique_selector(state, action.replacement, "replace source")
    state.check_unused(occupant, "replace")
    state.check_unused(replacement, "replace")
    origin = state.on_board(occupant)
    if origin is None:
        raise TransitionError(STATUS_OFF_BOARD, f"replace: {occupant.key()} is off the board")
    if state.on_board(replacement) is not None:
        raise TransitionError(
            STATUS_INVENTORY,
            f"replace source {replacement.key()} is on the board (must be a tray piece)",
        )
    state.void_rights_for(occupant, origin)
    state.board.remove_piece_at(origin)
    state.board.set_piece_at(origin, chess.Piece(replacement.piece_type, replacement.colour))
    state.set_square(occupant, None)
    state.set_square(replacement, origin)
    state.mark_used(occupant)
    state.mark_used(replacement)


def _action_add(state: _State, action: A.Add) -> None:
    source = resolve_unique_selector(state, action.source, "add source")
    state.check_unused(source, "add")
    if state.on_board(source) is not None:
        raise TransitionError(
            STATUS_INVENTORY, f"add source {source.key()} is already on the board"
        )
    destination = resolve_destination(state, action.destination)
    if state.owner_at(destination) is not None:
        raise TransitionError(
            STATUS_COLLISION,
            f"add {source.key()} to {chess.square_name(destination)} collides with an occupant",
        )
    state.board.set_piece_at(destination, chess.Piece(source.piece_type, source.colour))
    state.set_square(source, destination)
    state.mark_used(source)


def _action_transform(state: _State, action: A.Transform) -> None:
    transformed = apply_symmetry(CanonicalPosition.from_board(state.board), action.symmetry)
    if transformed is None:
        raise TransitionError(
            STATUS_INVALID_TRANSFORM,
            f"symmetry {action.symmetry!r} does not preserve a legal chess position",
        )
    mapping = _square_mapping(action.symmetry)
    new_squares: dict[StagePieceId, chess.Square | None] = {}
    for piece, square in state.piece_square.items():
        new_squares[piece] = mapping(square) if square is not None else None
    state.board = transformed.board.copy()
    state.piece_square = new_squares


def _square_mapping(
    symmetry: str,
) -> Callable[[chess.Square | None], chess.Square | None]:
    from chess_chain.similarity.symmetry import (
        SYMMETRY_IDENTITY,
    )
    from chess_chain.similarity.symmetry import (
        _square_mapper as _mapper,
    )

    if symmetry == SYMMETRY_IDENTITY:
        return lambda square: square
    mapper = _mapper(symmetry)
    return lambda square: mapper(square) if square is not None else None


def _action_establish_rights(state: _State, action: A.EstablishRights) -> None:
    allowed = (1 << chess.H1) | (1 << chess.A1) | (1 << chess.H8) | (1 << chess.A8)
    if action.castling_mask & ~allowed:
        raise TransitionError(
            STATUS_RIGHTS_CONFLICT, "castling mask names squares outside the four home rooks"
        )
    _validate_rights_consistency(state.board, action.castling_mask, action.ep_square)
    state.board.castling_rights = action.castling_mask
    state.board.ep_square = action.ep_square


def _validate_rights_consistency(
    board: chess.Board, castling_mask: int, ep_square: chess.Square | None
) -> None:
    """Stated rights must agree with static piece placement: the king and rook
    must sit on their home squares for every stated right."""
    required = {
        1 << chess.H1: (chess.E1, chess.H1, chess.WHITE),
        1 << chess.A1: (chess.E1, chess.A1, chess.WHITE),
        1 << chess.H8: (chess.E8, chess.H8, chess.BLACK),
        1 << chess.A8: (chess.E8, chess.A8, chess.BLACK),
    }
    for bit, (king_square, rook_square, colour) in required.items():
        if not castling_mask & bit:
            continue
        if board.piece_at(king_square) != chess.Piece(chess.KING, colour):
            side = "white" if colour == chess.WHITE else "black"
            raise TransitionError(
                STATUS_RIGHTS_CONFLICT,
                f"stated castling right requires the {side} "
                f"king on {chess.square_name(king_square)}",
            )
        if board.piece_at(rook_square) != chess.Piece(chess.ROOK, colour):
            side = "white" if colour == chess.WHITE else "black"
            raise TransitionError(
                STATUS_RIGHTS_CONFLICT,
                f"stated castling right requires the {side} "
                f"rook on {chess.square_name(rook_square)}",
            )
    if ep_square is not None:
        if not (0 <= ep_square < 64):
            raise TransitionError(STATUS_RIGHTS_CONFLICT, "en-passant square out of range")
        if chess.square_rank(ep_square) not in (2, 5):
            raise TransitionError(
                STATUS_RIGHTS_CONFLICT, "en-passant square must be on rank 3 or rank 6"
            )


def _action_conditional(state: _State, action: A.Conditional) -> None:
    for branch in action.branches:
        if evaluate_condition(state, branch.condition):
            for sub_action in branch.actions:
                _execute_action(state, sub_action)
            return
    raise TransitionError(STATUS_NO_BRANCH, "no conditional branch condition matched")
