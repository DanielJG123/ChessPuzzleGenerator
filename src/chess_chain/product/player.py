"""Player-safe product instructions (stage 7)."""

from __future__ import annotations

import chess

from chess_chain.analysis.policies import get_policy
from chess_chain.config import Settings
from chess_chain.product.common import piece_list
from chess_chain.product.render import BOARD_ORIENTATION
from chess_chain.sequence.records import SequenceResult

__all__ = ["build_player_instructions"]


def build_player_instructions(sequence: SequenceResult, settings: Settings) -> str:
    """The only player-safe artifact: no solution moves, no hidden codes and no
    references to private review/QA material."""
    lines: list[str] = []
    lines.append(f"# {sequence.product_id} - player instructions")
    lines.append("")
    lines.append(
        f"This product is sequence fingerprint `{sequence.fingerprint[:16]}...`. "
        "All instructions below are visible before you begin; there are no "
        "hidden codes and no concealed stage assumptions."
    )
    lines.append("")
    lines.append("## Board orientation")
    lines.append("")
    lines.append(BOARD_ORIENTATION)
    lines.append("")
    lines.append("## Initial setup")
    lines.append("")
    first = sequence.stages[0]
    lines.append(f"FEN: `{first.row['start_fen']}`")
    lines.append("")
    lines.append(f"Pieces: {piece_list(str(first.row['start_fen']))}")
    lines.append("")
    lines.append(f"Side to move: **{first.row['solver_colour']}**")
    lines.append("")

    rights_statement = _rights_statement(sequence)
    lines.append("## Castling and en-passant rights")
    lines.append("")
    lines.append(rights_statement)
    lines.append("")

    lines.append("## Global rule")
    lines.append("")
    lines.append(settings.transition_config.global_rule_wording)
    lines.append("")

    lines.append("## Stage boundaries")
    lines.append("")
    lines.append(
        "The sequence ends at the final stage. Each intermediate stage ends "
        "when its boundary condition first becomes true:"
    )
    for stage in sequence.stages:
        policy = get_policy(stage.end_policy_key.split("@", 1)[0])
        wording = policy.wording or policy.name
        lines.append(f"- Stage {stage.index + 1} ({stage.puzzle_id}): ends {wording}.")
    lines.append("")

    lines.append("## Transitions")
    lines.append("")
    lines.append(
        "When a stage's boundary condition becomes true, apply exactly one "
        "instruction to the board:"
    )
    for stage in sequence.stages:
        if stage.edge is None:
            continue
        lines.append(f"### After stage {stage.index} (towards stage {stage.index + 2})")
        lines.append("")
        if stage.edge.exact_journey:
            lines.append("No changes needed - the position already matches the next puzzle.")
        else:
            lines.append(stage.edge.rendered)
        lines.append("")

    lines.append("## Solutions")
    lines.append("")
    lines.append(
        "Solution moves are intentionally not included. Choose the strongest "
        "move for the side to move at every turn, as stated in the global rule."
    )
    lines.append("")
    return "\n".join(lines)


def _rights_statement(sequence: SequenceResult) -> str:
    board = chess.Board(str(sequence.stages[0].row["start_fen"]))
    castling_field = board.fen().split()[2]
    rights = [right for right in ("K", "Q", "k", "q") if right in castling_field]
    rights_text = ", ".join(rights) if rights else "none (no castling rights)"
    ep = board.ep_square
    ep_text = chess.square_name(ep) if ep is not None else "none (no en-passant square)"
    rights_line = (
        f"Initial castling rights: {rights_text}. Initial en-passant square: "
        f"{ep_text}. The product policy filters out transitions that must "
        "establish non-visible castling/en-passant rights, so every right is "
        "either present in the initial position or explicitly instructed."
    )
    if not rights and ep is None:
        rights_line = (
            "No castling or en-passant rights are in play in the initial "
            "position. The product policy filters out transitions that must "
            "establish non-visible castling/en-passant rights."
        )
    return rights_line
