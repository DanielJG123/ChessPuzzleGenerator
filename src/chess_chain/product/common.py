"""Shared helpers for product rendering and validation (stage 7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess

__all__ = [
    "PlyRecord",
    "replay_line",
    "per_ply_forcedness",
    "piece_list",
    "material_counts",
    "material_feasible_for_standard_set",
]

#: A standard physical set can supply at most these counts of each piece type.
STANDARD_SET_LIMITS: dict[tuple[chess.Color, chess.PieceType], int] = {
    (chess.WHITE, chess.PAWN): 8,
    (chess.BLACK, chess.PAWN): 8,
    (chess.WHITE, chess.ROOK): 2,
    (chess.BLACK, chess.ROOK): 2,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.BLACK, chess.BISHOP): 2,
    (chess.WHITE, chess.KNIGHT): 2,
    (chess.BLACK, chess.KNIGHT): 2,
    (chess.WHITE, chess.QUEEN): 1,
    (chess.BLACK, chess.QUEEN): 1,
    (chess.WHITE, chess.KING): 1,
    (chess.BLACK, chess.KING): 1,
}


@dataclass(frozen=True)
class PlyRecord:
    """One replayed ply: move number, side, SAN and UCI, and the board after."""

    move_number: int
    side: str
    san: str
    uci: str
    fen: str

    def to_dict(self) -> dict[str, str]:
        return {
            "move_number": str(self.move_number),
            "side": self.side,
            "san": self.san,
            "uci": self.uci,
            "fen": self.fen,
        }


def replay_line(start_fen: str, line: tuple[str, ...]) -> tuple[list[PlyRecord], chess.Board]:
    """Replay ``line`` from ``start_fen``; raises ``ValueError`` on an illegal
    move. Returns ``(plies, final_board)``."""
    board = chess.Board(start_fen)
    plies: list[PlyRecord] = []
    for half, uci in enumerate(line):
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {uci} at half-move {half} from {start_fen}")
        san = board.san(move)
        board.push(move)
        plies.append(
            PlyRecord(
                move_number=(half + 2) // 2,
                side="white" if board.turn == chess.BLACK else "black",
                san=san,
                uci=uci,
                fen=board.fen(),
            )
        )
    return plies, board


def per_ply_forcedness(row: dict[str, Any]) -> list[dict[str, object]]:
    """Per-ply legal-move counts along the intended line, for QA reporting.

    Solver plies report how many legal moves the side to move has (1 means
    strictly forced); opponent plies report the same count, which the
    configured opponent-forcedness policy is checked against.
    """
    board = chess.Board(str(row["start_fen"]))
    intended = tuple(str(move) for move in row["solution_uci"])
    solver_white = str(row["solver_colour"]) == "white"
    records: list[dict[str, object]] = []
    for depth, uci in enumerate(intended):
        legal = list(board.legal_moves)
        is_solver_ply = (board.turn == chess.WHITE) == solver_white
        records.append(
            {
                "ply": depth,
                "side": "solver" if is_solver_ply else "opponent",
                "uci": uci,
                "legal_moves": len(legal),
                "forced": len(legal) == 1,
            }
        )
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            break
        board.push(move)
    return records


def piece_list(fen: str) -> str:
    """Human-readable piece list, e.g. ``White: Ke1, Ra1; Black: Ke8``."""
    board = chess.Board(fen)
    parts: list[str] = []
    for colour, label in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        pieces = [
            f"{chess.piece_symbol(piece.piece_type).upper()}{chess.square_name(square)}"
            for square, piece in board.piece_map().items()
            if piece.color == colour
        ]
        pieces.sort()
        parts.append(f"{label}: {', '.join(pieces) if pieces else '(none)'}")
    return "; ".join(parts)


def material_counts(board: chess.Board) -> dict[tuple[int, int], int]:
    """Piece-type counts keyed by ``(color, piece_type)``."""
    counts: dict[tuple[int, int], int] = {}
    for _square, piece in board.piece_map().items():
        key = (int(piece.color), int(piece.piece_type))
        counts[key] = counts.get(key, 0) + 1
    return counts


def material_feasible_for_standard_set(board: chess.Board) -> bool:
    """True when the position is reachable with one standard physical set."""
    for (colour, piece_type), count in material_counts(board).items():
        if count > STANDARD_SET_LIMITS.get((chess.Color(colour), chess.PieceType(piece_type)), 0):
            return False
    return True
