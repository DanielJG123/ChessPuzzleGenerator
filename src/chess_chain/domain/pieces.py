"""Stage-local piece identities and material signatures."""

from __future__ import annotations

from dataclasses import dataclass

import chess

# Fixed ordering used for bitboards, material counts and fingerprints:
# white P,N,B,R,Q,K followed by black P,N,B,R,Q,K.
PIECE_ORDER: tuple[tuple[chess.Color, chess.PieceType], ...] = (
    (chess.WHITE, chess.PAWN),
    (chess.WHITE, chess.KNIGHT),
    (chess.WHITE, chess.BISHOP),
    (chess.WHITE, chess.ROOK),
    (chess.WHITE, chess.QUEEN),
    (chess.WHITE, chess.KING),
    (chess.BLACK, chess.PAWN),
    (chess.BLACK, chess.KNIGHT),
    (chess.BLACK, chess.BISHOP),
    (chess.BLACK, chess.ROOK),
    (chess.BLACK, chess.QUEEN),
    (chess.BLACK, chess.KING),
)

# Short column names matching PIECE_ORDER (one per colour/piece-type pair).
COUNT_NAMES: tuple[str, ...] = (
    "wp",
    "wn",
    "wb",
    "wr",
    "wq",
    "wk",
    "bp",
    "bn",
    "bb",
    "br",
    "bq",
    "bk",
)

# What a single normal chess set provides per colour.
STANDARD_SET_LIMITS: dict[chess.PieceType, int] = {
    chess.PAWN: 8,
    chess.KNIGHT: 2,
    chess.BISHOP: 2,
    chess.ROOK: 2,
    chess.QUEEN: 1,
    chess.KING: 1,
}

STANDARD_SET_PIECE_LIMIT = 16  # maximum pieces per colour in one standard set

PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


@dataclass(frozen=True, order=True)
class StagePieceId:
    """A stage-local physical identity for one piece.

    The identity is derived from the piece's colour, piece type and *start
    square* in the presented start position, and is tracked through the
    intended solution. This allows language such as "the piece that began this
    puzzle on a8". No historical game identity is assumed to be recoverable
    from the source data.
    """

    colour: chess.Color
    piece_type: chess.PieceType
    start_square: chess.Square

    def key(self) -> str:
        """Return a stable serialised key, e.g. ``"wra8"`` for a white rook on a8."""
        return (
            f"{'w' if self.colour == chess.WHITE else 'b'}"
            f"{chess.piece_symbol(self.piece_type)}"
            f"{chess.square_name(self.start_square)}"
        )

    @classmethod
    def from_key(cls, key: str) -> StagePieceId:
        if len(key) != 4:
            raise ValueError(f"invalid stage piece id {key!r}")
        colour = chess.WHITE if key[0] == "w" else chess.BLACK
        piece_type = chess.Piece.from_symbol(key[1]).piece_type
        start_square = chess.parse_square(key[2:])
        return cls(colour=colour, piece_type=piece_type, start_square=start_square)


@dataclass(frozen=True)
class MaterialSignature:
    """Piece counts per colour/piece-type in PIECE_ORDER."""

    counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.counts) != len(PIECE_ORDER):
            raise ValueError(f"expected {len(PIECE_ORDER)} counts, got {len(self.counts)}")

    @classmethod
    def from_board(cls, board: chess.Board) -> MaterialSignature:
        return cls(
            tuple(len(board.pieces(piece_type, colour)) for colour, piece_type in PIECE_ORDER)
        )

    @classmethod
    def from_counts(cls, **counts: int) -> MaterialSignature:
        return cls(tuple(counts.get(name, 0) for name in COUNT_NAMES))

    def piece_count(self) -> int:
        return sum(self.counts)

    def counts_by_colour(self, colour: chess.Color) -> dict[chess.PieceType, int]:
        result: dict[chess.PieceType, int] = {}
        for i, (piece_colour, piece_type) in enumerate(PIECE_ORDER):
            if piece_colour == colour:
                result[piece_type] = self.counts[i]
        return result

    def material_score(self, colour: chess.Color) -> int:
        total = 0
        for i, (piece_colour, piece_type) in enumerate(PIECE_ORDER):
            if piece_colour == colour:
                total += PIECE_VALUES[piece_type] * self.counts[i]
        return total

    def duplicate_queens(self) -> bool:
        for colour in (chess.WHITE, chess.BLACK):
            if self.counts_by_colour(colour).get(chess.QUEEN, 0) > 1:
                return True
        return False

    def exceeds_standard_set(self) -> bool:
        for colour in (chess.WHITE, chess.BLACK):
            by_colour = self.counts_by_colour(colour)
            if sum(by_colour.values()) > STANDARD_SET_PIECE_LIMIT:
                return True
            for piece_type, limit in STANDARD_SET_LIMITS.items():
                if by_colour.get(piece_type, 0) > limit:
                    return True
        return False
