"""Typed transition AST (stage 5).

The AST is the *only* semantic authority: the executor executes the AST, the
renderer only formats it, and the persisted artifact stores the AST plus
machine-verifiable before/after FENs. Every node is JSON-serialisable
(``to_json`` / ``from_json``) so transitions can be verified offline without
re-deriving them.

Selectors resolve to *physical* pieces (stage-local ``StagePieceId``
identities). The executor never silently picks among ties: any selector that
resolves to zero pieces is ``missing_selector`` and any selector resolving to
more than one piece is ``ambiguous_selector``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

DIRECTIONS = ("n", "s", "e", "w", "ne", "nw", "se", "sw")

__all__ = [
    "AbsoluteSquare",
    "Add",
    "Branch",
    "CapturedPieceBeganOnSquare",
    "CheckedKing",
    "CheckingPiece",
    "Condition",
    "Conditional",
    "Destination",
    "DIRECTIONS",
    "EstablishRights",
    "FirstCapturedPiece",
    "InDirection",
    "LastCapturedPiece",
    "LastMovedPiece",
    "MatingPiece",
    "Move",
    "NearestTo",
    "OccupantOfSquare",
    "PieceBeganOnSquare",
    "PieceMadePly",
    "PlyOrigin",
    "RelativeOffset",
    "RelativeToAnchor",
    "Remove",
    "Replace",
    "Selector",
    "SquareOf",
    "SquareEmpty",
    "SquareOccupied",
    "Swap",
    "Transform",
    "TransitionProgram",
    "UniquePiece",
    "VacatedBy",
    "from_condition",
    "from_destination",
    "from_selector",
]


class _Node(Protocol):
    @property
    def kind(self) -> str:
        raise NotImplementedError

    def to_json(self) -> dict[str, object]:
        raise NotImplementedError

    @classmethod
    def from_json(cls, data: dict[str, object]) -> object:
        raise NotImplementedError


class Selector(_Node):
    """A typed selector for one or more physical pieces."""

    @property
    def kind(self) -> str:
        raise NotImplementedError

    def to_json(self) -> dict[str, object]:
        raise NotImplementedError

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Selector:
        raise NotImplementedError


class Destination(_Node):
    """A typed destination square expression."""

    @property
    def kind(self) -> str:
        raise NotImplementedError

    def to_json(self) -> dict[str, object]:
        raise NotImplementedError

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Destination:
        raise NotImplementedError


class Condition(_Node):
    """A deterministic branch condition over an EndpointContext."""

    @property
    def kind(self) -> str:
        raise NotImplementedError

    def to_json(self) -> dict[str, object]:
        raise NotImplementedError

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Condition:
        raise NotImplementedError


class Action(_Node):
    """A typed physical transition action."""

    @property
    def kind(self) -> str:
        raise NotImplementedError

    def to_json(self) -> dict[str, object]:
        raise NotImplementedError

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Action:
        raise NotImplementedError


@dataclass(frozen=True)
class PieceBeganOnSquare(Selector):
    """The physical piece that began the stage on ``square``."""

    square: int

    @property
    def kind(self) -> str:
        return "piece_began_on_square"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind, "square": self.square}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Selector:
        return cls(square=int(data["square"]))


@dataclass(frozen=True)
class OccupantOfSquare(Selector):
    """The current occupant of ``square`` (a missing piece is an error)."""

    square: int

    @property
    def kind(self) -> str:
        return "occupant_of_square"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind, "square": self.square}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Selector:
        return cls(square=int(data["square"]))


@dataclass(frozen=True)
class UniquePiece(Selector):
    """The unique piece of the given colour and type (ambiguous if two match)."""

    colour: int  # chess.Color: 0 white, 1 black
    piece_type: int

    @property
    def kind(self) -> str:
        return "unique_piece"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind, "colour": self.colour, "piece_type": self.piece_type}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Selector:
        return cls(colour=int(data["colour"]), piece_type=int(data["piece_type"]))


@dataclass(frozen=True)
class LastMovedPiece(Selector):
    """The piece that made the last intended half-move."""

    @property
    def kind(self) -> str:
        return "last_moved_piece"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Selector:
        return cls()


@dataclass(frozen=True)
class PieceMadePly(Selector):
    """The piece that made the named (0-based) intended half-move."""

    ply_index: int

    @property
    def kind(self) -> str:
        return "piece_made_ply"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind, "ply_index": self.ply_index}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Selector:
        return cls(ply_index=int(data["ply_index"]))


@dataclass(frozen=True)
class FirstCapturedPiece(Selector):
    """The first piece captured during the stage."""

    @property
    def kind(self) -> str:
        return "first_captured_piece"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Selector:
        return cls()


@dataclass(frozen=True)
class LastCapturedPiece(Selector):
    """The last piece captured during the stage."""

    @property
    def kind(self) -> str:
        return "last_captured_piece"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Selector:
        return cls()


@dataclass(frozen=True)
class CapturedPieceBeganOnSquare(Selector):
    """A captured piece that began the stage on ``square``."""

    square: int

    @property
    def kind(self) -> str:
        return "captured_piece_began_on_square"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind, "square": self.square}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Selector:
        return cls(square=int(data["square"]))


@dataclass(frozen=True)
class CheckingPiece(Selector):
    """A piece currently giving check."""

    @property
    def kind(self) -> str:
        return "checking_piece"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Selector:
        return cls()


@dataclass(frozen=True)
class MatingPiece(Selector):
    """A piece delivering checkmate (empty outside a mate)."""

    @property
    def kind(self) -> str:
        return "mating_piece"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Selector:
        return cls()


@dataclass(frozen=True)
class CheckedKing(Selector):
    """The king currently in check."""

    @property
    def kind(self) -> str:
        return "checked_king"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Selector:
        return cls()


@dataclass(frozen=True)
class NearestTo(Selector):
    """The piece nearest (or farthest) to the anchor's current square.

    Ties are handled deterministically by *rejection*: an equidistant pair
    makes the selector ambiguous and the executor refuses to choose silently.
    """

    anchor: Selector
    colour: int | None = None
    farthest: bool = False

    @property
    def kind(self) -> str:
        return "nearest_to"

    def to_json(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "anchor": self.anchor.to_json(),
            "colour": self.colour,
            "farthest": self.farthest,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Selector:
        return cls(
            anchor=from_selector(data["anchor"]),
            colour=(int(data["colour"]) if data.get("colour") is not None else None),
            farthest=bool(data.get("farthest", False)),
        )


@dataclass(frozen=True)
class InDirection(Selector):
    """The first piece immediately in ``direction`` from the anchor's square."""

    anchor: Selector
    direction: str  # one of DIRECTIONS

    @property
    def kind(self) -> str:
        return "in_direction"

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"unknown direction {self.direction!r}; expected one of {DIRECTIONS}")

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind, "anchor": self.anchor.to_json(), "direction": self.direction}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Selector:
        return cls(
            anchor=from_selector(data["anchor"]),
            direction=str(data["direction"]),
        )


@dataclass(frozen=True)
class RelativeOffset(Destination):
    """``(dx, dy)`` from the selected piece's current square (no wrap).

    ``dx`` counts files towards h, ``dy`` counts ranks towards rank 8. The
    origin count for the offset alone on an 8x8 board is
    ``(8 - abs(dx)) * (8 - abs(dy))`` (see leakage.py).
    """

    piece: Selector
    dx: int
    dy: int

    @property
    def kind(self) -> str:
        return "relative_offset"

    def to_json(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "piece": self.piece.to_json(),
            "dx": self.dx,
            "dy": self.dy,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Destination:
        return cls(
            piece=from_selector(data["piece"]),
            dx=int(data["dx"]),
            dy=int(data["dy"]),
        )


@dataclass(frozen=True)
class SquareOf(Destination):
    """The square currently occupied by another selected piece."""

    piece: Selector

    @property
    def kind(self) -> str:
        return "square_of"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind, "piece": self.piece.to_json()}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Destination:
        return cls(piece=from_selector(data["piece"]))


@dataclass(frozen=True)
class VacatedBy(Destination):
    """The square the selected piece vacated on its most recent move (an
    unmoved piece never vacated a square: resolution error)."""

    piece: Selector

    @property
    def kind(self) -> str:
        return "vacated_by"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind, "piece": self.piece.to_json()}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Destination:
        return cls(piece=from_selector(data["piece"]))


@dataclass(frozen=True)
class PlyOrigin(Destination):
    """The origin square of the named (0-based) intended half-move."""

    ply_index: int

    @property
    def kind(self) -> str:
        return "ply_origin"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind, "ply_index": self.ply_index}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Destination:
        return cls(ply_index=int(data["ply_index"]))


@dataclass(frozen=True)
class RelativeToAnchor(Destination):
    """``(dx, dy)`` from a dynamically selected anchor's current square."""

    anchor: Selector
    dx: int
    dy: int

    @property
    def kind(self) -> str:
        return "relative_to_anchor"

    def to_json(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "anchor": self.anchor.to_json(),
            "dx": self.dx,
            "dy": self.dy,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Destination:
        return cls(
            anchor=from_selector(data["anchor"]),
            dx=int(data["dx"]),
            dy=int(data["dy"]),
        )


@dataclass(frozen=True)
class AbsoluteSquare(Destination):
    """An absolute square name: permitted but penalised as a fallback."""

    square: int

    @property
    def kind(self) -> str:
        return "absolute_square"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind, "square": self.square}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Destination:
        return cls(square=int(data["square"]))


@dataclass(frozen=True)
class SquareOccupied(Condition):
    """True when ``square`` is occupied on the current board."""

    square: int

    @property
    def kind(self) -> str:
        return "square_occupied"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind, "square": self.square}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Condition:
        return cls(square=int(data["square"]))


@dataclass(frozen=True)
class SquareEmpty(Condition):
    """True when ``square`` is empty on the current board."""

    square: int

    @property
    def kind(self) -> str:
        return "square_empty"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind, "square": self.square}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Condition:
        return cls(square=int(data["square"]))


@dataclass(frozen=True)
class PieceAt(Condition):
    """True when the piece on ``square`` has the given colour and type."""

    square: int
    colour: int
    piece_type: int

    @property
    def kind(self) -> str:
        return "piece_at"

    def to_json(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "square": self.square,
            "colour": self.colour,
            "piece_type": self.piece_type,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Condition:
        return cls(
            square=int(data["square"]),
            colour=int(data["colour"]),
            piece_type=int(data["piece_type"]),
        )


@dataclass(frozen=True)
class SelectorResolves(Condition):
    """True when the selector resolves to exactly one piece."""

    selector: Selector

    @property
    def kind(self) -> str:
        return "selector_resolves"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind, "selector": self.selector.to_json()}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Condition:
        return cls(selector=from_selector(data["selector"]))


@dataclass(frozen=True)
class OnBoard(Condition):
    """True when the selected piece is currently on the board."""

    selector: Selector

    @property
    def kind(self) -> str:
        return "on_board"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind, "selector": self.selector.to_json()}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Condition:
        return cls(selector=from_selector(data["selector"]))


@dataclass(frozen=True)
class OffBoard(Condition):
    """True when the selected piece is currently off the board (tray)."""

    selector: Selector

    @property
    def kind(self) -> str:
        return "off_board"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind, "selector": self.selector.to_json()}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Condition:
        return cls(selector=from_selector(data["selector"]))


@dataclass(frozen=True)
class LastPlyWas(Condition):
    """True when the last intended half-move is exactly ``uci``."""

    uci: str

    @property
    def kind(self) -> str:
        return "last_ply_was"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind, "uci": self.uci}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Condition:
        return cls(uci=str(data["uci"]))


@dataclass(frozen=True)
class Move(Action):
    """Move a physical piece to a destination square (collision-free)."""

    piece: Selector
    destination: Destination

    @property
    def kind(self) -> str:
        return "move"

    def to_json(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "piece": self.piece.to_json(),
            "destination": self.destination.to_json(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Action:
        return cls(
            piece=from_selector(data["piece"]),
            destination=from_destination(data["destination"]),
        )


@dataclass(frozen=True)
class Swap(Action):
    """Exchange the squares of two distinct on-board pieces."""

    a: Selector
    b: Selector

    @property
    def kind(self) -> str:
        return "swap"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind, "a": self.a.to_json(), "b": self.b.to_json()}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Action:
        return cls(
            a=from_selector(data["a"]),
            b=from_selector(data["b"]),
        )


@dataclass(frozen=True)
class Remove(Action):
    """Take a piece off the board (it joins the tray)."""

    piece: Selector

    @property
    def kind(self) -> str:
        return "remove"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind, "piece": self.piece.to_json()}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Action:
        return cls(piece=from_selector(data["piece"]))


@dataclass(frozen=True)
class Replace(Action):
    """Replace the on-board occupant with a captured/off-board piece."""

    occupant: Selector
    replacement: Selector

    @property
    def kind(self) -> str:
        return "replace"

    def to_json(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "occupant": self.occupant.to_json(),
            "replacement": self.replacement.to_json(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Action:
        return cls(
            occupant=from_selector(data["occupant"]),
            replacement=from_selector(data["replacement"]),
        )


@dataclass(frozen=True)
class Add(Action):
    """Place an available off-board (captured) piece on a destination."""

    source: Selector
    destination: Destination

    @property
    def kind(self) -> str:
        return "add"

    def to_json(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "source": self.source.to_json(),
            "destination": self.destination.to_json(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Action:
        return cls(
            source=from_selector(data["source"]),
            destination=from_destination(data["destination"]),
        )


@dataclass(frozen=True)
class Transform(Action):
    """Apply an approved whole-board transformation (reconstructed and
    validated with python-chess)."""

    symmetry: str

    @property
    def kind(self) -> str:
        return "transform"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind, "symmetry": self.symmetry}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Action:
        return cls(symmetry=str(data["symmetry"]))


@dataclass(frozen=True)
class EstablishRights(Action):
    """Explicitly establish next-stage castling rights / en-passant history.

    These are instruction *metadata*: the mechanical checker only sees occupied
    squares, so rights cannot be inferred from static piece placement. The
    action is allowed only when product configuration permits it.
    """

    castling_mask: int = 0
    ep_square: int | None = None

    @property
    def kind(self) -> str:
        return "establish_rights"

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind, "castling_mask": self.castling_mask, "ep_square": self.ep_square}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Action:
        ep = data.get("ep_square")
        return cls(
            castling_mask=int(data["castling_mask"]),
            ep_square=int(ep) if ep is not None else None,
        )


@dataclass(frozen=True)
class Branch:
    """One conditional branch: a deterministic condition plus its actions."""

    condition: Condition
    actions: tuple[Action, ...]


@dataclass(frozen=True)
class Conditional(Action):
    """A conditional branch set.

    Allowed only when every branch is fully verifiable and necessary for some
    accepted endpoint variant. The executor applies the first branch whose
    condition is true; verification requires that every accepted endpoint takes
    exactly one branch and every branch produces the same next start.
    """

    branches: tuple[Branch, ...] = field(default_factory=tuple)

    @property
    def kind(self) -> str:
        return "conditional"

    def to_json(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "branches": [
                {
                    "condition": branch.condition.to_json(),
                    "actions": [action.to_json() for action in branch.actions],
                }
                for branch in self.branches
            ],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Action:
        raw_branches = data.get("branches", [])
        assert isinstance(raw_branches, list)
        branches = tuple(
            Branch(
                condition=from_condition(item["condition"]),
                actions=tuple(from_action(action) for action in item["actions"]),
            )
            for item in raw_branches
        )
        return cls(branches=branches)


@dataclass(frozen=True)
class TransitionProgram:
    """A sequence of transition actions. The AST is the semantic authority."""

    actions: tuple[Action, ...]

    def to_json(self) -> dict[str, object]:
        return {"actions": [action.to_json() for action in self.actions]}

    @classmethod
    def from_json(cls, data: dict[str, object]) -> TransitionProgram:
        raw_actions = data.get("actions", [])
        assert isinstance(raw_actions, list)
        return cls(actions=tuple(from_action(action) for action in raw_actions))


_SELECTOR_KINDS: dict[str, type] = {
    "piece_began_on_square": PieceBeganOnSquare,
    "occupant_of_square": OccupantOfSquare,
    "unique_piece": UniquePiece,
    "last_moved_piece": LastMovedPiece,
    "piece_made_ply": PieceMadePly,
    "first_captured_piece": FirstCapturedPiece,
    "last_captured_piece": LastCapturedPiece,
    "captured_piece_began_on_square": CapturedPieceBeganOnSquare,
    "checking_piece": CheckingPiece,
    "mating_piece": MatingPiece,
    "checked_king": CheckedKing,
    "nearest_to": NearestTo,
    "in_direction": InDirection,
}

_DESTINATION_KINDS: dict[str, type] = {
    "relative_offset": RelativeOffset,
    "square_of": SquareOf,
    "vacated_by": VacatedBy,
    "ply_origin": PlyOrigin,
    "relative_to_anchor": RelativeToAnchor,
    "absolute_square": AbsoluteSquare,
}

_CONDITION_KINDS: dict[str, type] = {
    "square_occupied": SquareOccupied,
    "square_empty": SquareEmpty,
    "piece_at": PieceAt,
    "selector_resolves": SelectorResolves,
    "on_board": OnBoard,
    "off_board": OffBoard,
    "last_ply_was": LastPlyWas,
}

_ACTION_KINDS: dict[str, type] = {
    "move": Move,
    "swap": Swap,
    "remove": Remove,
    "replace": Replace,
    "add": Add,
    "transform": Transform,
    "establish_rights": EstablishRights,
    "conditional": Conditional,
}


def _parse(data: object, registry: dict[str, type], name: str) -> object:
    if not isinstance(data, dict) or "kind" not in data:
        raise ValueError(f"malformed {name} node: {data!r}")
    node_type = registry.get(str(data["kind"]))
    if node_type is None:
        raise ValueError(f"unknown {name} kind {data['kind']!r}")
    return node_type.from_json(data)  # type: ignore[attr-defined]


def from_selector(data: object) -> Selector:
    return _parse(data, _SELECTOR_KINDS, "selector")  # type: ignore[return-value]


def from_destination(data: object) -> Destination:
    return _parse(data, _DESTINATION_KINDS, "destination")  # type: ignore[return-value]


def from_condition(data: object) -> Condition:
    return _parse(data, _CONDITION_KINDS, "condition")  # type: ignore[return-value]


def from_action(data: object) -> Action:
    return _parse(data, _ACTION_KINDS, "action")  # type: ignore[return-value]
