"""Deterministic natural-language rendering of transition ASTs.

The renderer is display-only: it never changes semantics (the AST is the
semantic authority and is stored beside the rendered text). Orientation is
expressed explicitly with file and rank names - "two files towards h", "three
ranks towards rank 8" - rather than player-relative "left" or "up", because the
player-sheet convention is board-global.
"""

from __future__ import annotations

import chess

from chess_chain.transitions import ast as A

__all__ = ["render_program", "render_guard"]

_DIRECTION_WORDS: dict[str, str] = {
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "north-east",
    "nw": "north-west",
    "se": "south-east",
    "sw": "south-west",
}


def _square_name(square: int) -> str:
    return chess.square_name(chess.Square(square))


def _ordinal(index: int) -> str:
    return (
        f"{index + 1}{'st' if index == 0 else 'nd' if index == 1 else 'rd' if index == 2 else 'th'}"
    )


def _colour_word(colour: int) -> str:
    return "white" if colour == chess.WHITE else "black"


def render_selector(selector: A.Selector) -> str:
    if isinstance(selector, A.PieceBeganOnSquare):
        return f"the piece that began the stage on {_square_name(selector.square)}"
    if isinstance(selector, A.OccupantOfSquare):
        return f"the piece on {_square_name(selector.square)}"
    if isinstance(selector, A.UniquePiece):
        return f"the {_colour_word(selector.colour)} {chess.piece_name(selector.piece_type)}"
    if isinstance(selector, A.LastMovedPiece):
        return "the last-moved piece"
    if isinstance(selector, A.PieceMadePly):
        return f"the piece that made the {_ordinal(selector.ply_index)} half-move"
    if isinstance(selector, A.FirstCapturedPiece):
        return "the first piece captured"
    if isinstance(selector, A.LastCapturedPiece):
        return "the last piece captured"
    if isinstance(selector, A.CapturedPieceBeganOnSquare):
        return f"the captured piece that began on {_square_name(selector.square)}"
    if isinstance(selector, A.CheckingPiece):
        return "the piece giving check"
    if isinstance(selector, A.MatingPiece):
        return "the piece delivering mate"
    if isinstance(selector, A.CheckedKing):
        return "the checked king"
    if isinstance(selector, A.NearestTo):
        word = "farthest" if selector.farthest else "nearest"
        colour = f" {_colour_word(selector.colour)}" if selector.colour is not None else ""
        return f"the{colour} piece {word} to {render_selector(selector.anchor)}"
    if isinstance(selector, A.InDirection):
        return (
            f"the piece immediately {_DIRECTION_WORDS[selector.direction]} of "
            f"{render_selector(selector.anchor)}"
        )
    return f"<{selector.kind}>"


def render_destination(destination: A.Destination) -> str:
    if isinstance(destination, A.RelativeOffset):
        return _render_offset(destination.dx, destination.dy)
    if isinstance(destination, A.SquareOf):
        return f"the square of {render_selector(destination.piece)}"
    if isinstance(destination, A.VacatedBy):
        return f"the square vacated by {render_selector(destination.piece)}"
    if isinstance(destination, A.PlyOrigin):
        return f"the origin square of the {_ordinal(destination.ply_index)} half-move"
    if isinstance(destination, A.RelativeToAnchor):
        return (
            f"{_render_offset(destination.dx, destination.dy)} from "
            f"{render_selector(destination.anchor)}"
        )
    if isinstance(destination, A.AbsoluteSquare):
        return f"square {_square_name(destination.square)}"
    return f"<{destination.kind}>"


def _render_offset(dx: int, dy: int) -> str:
    parts: list[str] = []
    if dx != 0:
        direction = "h" if dx > 0 else "a"
        parts.append(f"{abs(dx)} file{'s' if abs(dx) != 1 else ''} towards {direction}")
    if dy != 0:
        rank = "8" if dy > 0 else "1"
        parts.append(f"{abs(dy)} rank{'s' if abs(dy) != 1 else ''} towards rank {rank}")
    if not parts:
        return "its current square"
    return " and ".join(parts)


def render_condition(condition: A.Condition) -> str:
    if isinstance(condition, A.SquareOccupied):
        return f"{_square_name(condition.square)} is occupied"
    if isinstance(condition, A.SquareEmpty):
        return f"{_square_name(condition.square)} is empty"
    if isinstance(condition, A.PieceAt):
        return (
            f"the {_colour_word(condition.colour)} {chess.piece_name(condition.piece_type)} "
            f"stands on {_square_name(condition.square)}"
        )
    if isinstance(condition, A.SelectorResolves):
        return f"{render_selector(condition.selector)} is present"
    if isinstance(condition, A.OnBoard):
        return f"{render_selector(condition.selector)} is on the board"
    if isinstance(condition, A.OffBoard):
        return f"{render_selector(condition.selector)} is off the board"
    if isinstance(condition, A.LastPlyWas):
        return f"the last half-move was {condition.uci}"
    return f"<{condition.kind}>"


def render_action(action: A.Action) -> str:
    if isinstance(action, A.Move):
        return f"Move {render_selector(action.piece)} to {render_destination(action.destination)}."
    if isinstance(action, A.Swap):
        return f"Swap {render_selector(action.a)} with {render_selector(action.b)}."
    if isinstance(action, A.Remove):
        return f"Remove {render_selector(action.piece)} from the board."
    if isinstance(action, A.Replace):
        return (
            f"Replace {render_selector(action.occupant)} with "
            f"{render_selector(action.replacement)}."
        )
    if isinstance(action, A.Add):
        return (
            f"Place {render_selector(action.source)} on {render_destination(action.destination)}."
        )
    if isinstance(action, A.Transform):
        return f"Mirror the whole board: {action.symmetry.replace('_', ' ')}."
    if isinstance(action, A.EstablishRights):
        return _render_rights(action)
    if isinstance(action, A.Conditional):
        branches = []
        for branch in action.branches:
            if branch.actions:
                body = " ".join(render_action(sub) for sub in branch.actions)
            else:
                body = "do nothing"
            branches.append(f"if {render_condition(branch.condition)}, then {body}")
        text = " otherwise ".join(branches)
        return text if text.endswith(".") else text + "."
    return f"<{action.kind}>"


def _render_rights(action: A.EstablishRights) -> str:
    letters: list[str] = []
    if action.castling_mask & (1 << chess.H1):
        letters.append("K")
    if action.castling_mask & (1 << chess.A1):
        letters.append("Q")
    if action.castling_mask & (1 << chess.H8):
        letters.append("k")
    if action.castling_mask & (1 << chess.A8):
        letters.append("q")
    parts = []
    if letters:
        parts.append(f"castling rights are {''.join(letters)}")
    if action.ep_square is not None:
        parts.append(f"an en-passant target exists on {_square_name(action.ep_square)}")
    if not parts:
        return "Note: no castling rights or en-passant target."
    return "Note: " + "; ".join(parts) + "."


def render_program(program: A.TransitionProgram) -> str:
    return " ".join(render_action(action) for action in program.actions)


def render_guard(guard: object) -> str:
    from chess_chain.transitions.guards import StageBoundaryGuard

    if not isinstance(guard, StageBoundaryGuard):
        return ""
    if guard.kind == "TRANSITION_READY" and guard.transition_program is not None:
        return render_program(guard.transition_program)
    return guard.wording or ""
