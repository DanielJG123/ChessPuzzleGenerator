"""Private visual review report (stage 7).

``sequence_review.html`` is a self-contained developer/manual-review document
that may reveal every solution. Boards are rendered with python-chess SVG and
embedded together with CSS into one file that opens locally without a server.
It is strictly private: ``player_instructions.md`` never links to or names it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import chess

from chess_chain.analysis.policies import get_policy
from chess_chain.product.common import (
    material_feasible_for_standard_set,
    per_ply_forcedness,
    replay_line,
)
from chess_chain.product.render import BOARD_ORIENTATION
from chess_chain.sequence.rating import DifficultyEstimate
from chess_chain.sequence.records import SequenceResult

if TYPE_CHECKING:
    from chess_chain.config import Settings

__all__ = ["render_review_html", "render_review_notes"]

TEMPLATE_NAME = "review_template.html.j2"


def render_review_html(
    sequence: SequenceResult,
    settings: Settings,
    estimate: DifficultyEstimate,
) -> str:
    """Render the self-contained review HTML for ``sequence``."""
    from jinja2 import Environment, PackageLoader, select_autoescape

    environment = Environment(
        loader=PackageLoader("chess_chain.product", "templates"),
        autoescape=select_autoescape(["html"]),
    )
    template = environment.get_template(TEMPLATE_NAME)
    model = _review_model(sequence, settings, estimate)
    return template.render(**model)


def render_review_notes(sequence: SequenceResult) -> str:
    """A human review-aid template; never read back by any command."""
    lines: list[str] = []
    lines.append(f"# Review notes - {sequence.product_id}")
    lines.append(f"sequence_fingerprint: {sequence.fingerprint}")
    lines.append("")
    lines.append("status: open  # open | accepted | rejected")
    lines.append("reviewer: ''")
    lines.append("notes: |")
    lines.append("    ")
    lines.append("")
    lines.append("rejected_stages: []  # e.g. [1, 3]")
    lines.append("rejected_transitions: []  # e.g. ['P2 -> P3']")
    lines.append("suggested_request_changes: {}")
    lines.append("")
    lines.append("# Copy-ready exclusion example (uses the actual SequenceRequest field):")
    lines.append("#   excluded_puzzle_ids: [<puzzle-id-1>, <puzzle-id-2>]")
    lines.append("")
    return "\n".join(lines)


def _review_model(
    sequence: SequenceResult, settings: Settings, estimate: DifficultyEstimate
) -> dict[str, Any]:
    stages = []
    for stage in sequence.stages:
        stages.append(_stage_model(sequence, stage))
    return {
        "product_id": sequence.product_id,
        "sequence_fingerprint": sequence.fingerprint,
        "request_fingerprint": _request_fingerprint(sequence),
        "seed": sequence.request.random_seed,
        "reproduce_command": (f"chess-chain sequence reproduce --product-id {sequence.product_id}"),
        "board_orientation": BOARD_ORIENTATION,
        "stages": stages,
        "estimate": estimate.to_dict(),
    }


def _request_fingerprint(sequence: SequenceResult) -> str:
    from chess_chain.sequence.request import request_fingerprint

    return request_fingerprint(sequence.request)


def _stage_model(sequence: SequenceResult, stage: Any) -> dict[str, Any]:
    start_board = chess.Board(str(stage.row["start_fen"]))
    intended = tuple(str(move) for move in stage.row["solution_uci"])
    plies = _ply_svgs(start_board, intended)

    accepted_alternatives: list[dict[str, object]] = []
    endpoint_svgs: list[str] = []
    for line in stage.accepted_lines:
        _, final_board = replay_line(str(stage.row["start_fen"]), line)
        endpoint_svgs.append(_svg(final_board))
        if line != intended:
            accepted_alternatives.append(
                {
                    "line": list(line),
                    "endpoint_fen": final_board.fen(),
                    "endpoint_svg": _svg(final_board),
                }
            )

    model: dict[str, Any] = {
        "index": stage.index,
        "puzzle_id": stage.puzzle_id,
        "is_final": stage.is_final,
        "source_rating": int(stage.row["rating"]),
        "side_to_move": str(stage.row["solver_colour"]),
        "start_fen": str(stage.row["start_fen"]),
        "start_svg": _svg(start_board),
        "plies": plies,
        "accepted_alternatives": accepted_alternatives,
        "endpoint_svgs": endpoint_svgs,
        "end_policy_key": stage.end_policy_key,
        "end_policy_wording": _policy_wording(stage),
        "guard": stage.edge.guard if stage.edge is not None else None,
        "forcedness": per_ply_forcedness(stage.row),
        "clue": stage.edge.clue if stage.edge is not None else None,
        "inventory_warning": _inventory_warning(stage, sequence),
    }

    if stage.edge is not None:
        model["transition_rendered"] = stage.edge.rendered
        model["transition_ast_summary"] = _ast_summary(stage.edge)
        model["transition_exact_journey"] = stage.edge.exact_journey
        model["action_count"] = stage.edge.action_count
        model["symmetry"] = stage.edge.symmetry
        model["verification_issues"] = list(stage.edge.verification_issues)
        if not stage.is_final and stage.index + 1 < len(sequence.stages):
            next_stage = sequence.stages[stage.index + 1]
            next_board = chess.Board(str(next_stage.row["start_fen"]))
            after_board = _after_transition_board(stage, next_board)
            model["next_start_fen"] = str(next_stage.row["start_fen"])
            model["next_start_svg"] = _svg(next_board)
            model["after_transition_fen"] = after_board.fen()
            model["after_transition_svg"] = _svg(after_board)
            model["exact_match"] = _position_match(after_board, next_board)

    if stage.final_selection is not None:
        selection = stage.final_selection
        model["final_checkmate"] = selection.final_ends_in_checkmate
        model["final_occupancy_mask"] = f"{int(selection.collision.request.target_occupancy):#x}"
        model["final_collision_status"] = selection.collision.status.value
        model["final_collision_scope"] = selection.collision.request.scope
        model["final_collision_max_depth"] = selection.collision.request.max_depth
        model["final_collision_witness"] = selection.collision.witness_line
        model["final_collision_cached"] = selection.collision.cached
    return model


def _policy_wording(stage: Any) -> str:
    policy = get_policy(stage.end_policy_key.split("@", 1)[0])
    return policy.wording or policy.name


def _ply_svgs(start_board: chess.Board, line: tuple[str, ...]) -> list[dict[str, str]]:
    plies: list[dict[str, str]] = []
    board = start_board.copy()
    for half, uci in enumerate(line):
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            break
        san = board.san(move)
        board.push(move)
        plies.append(
            {
                "move_number": str((half + 2) // 2),
                "side": "white" if board.turn == chess.BLACK else "black",
                "san": san,
                "uci": uci,
                "fen": board.fen(),
                "svg": _svg(board),
            }
        )
    return plies


def _svg(board: chess.Board) -> str:
    import chess.svg

    return chess.svg.board(board, size=320)


def _position_match(after_board: chess.Board, next_board: chess.Board) -> bool:
    return after_board.fen().split()[0:4] == next_board.fen().split()[0:4]


def _after_transition_board(stage: Any, fallback: chess.Board) -> chess.Board:
    if stage.edge is None or stage.edge.program is None:
        return fallback.copy()
    from chess_chain.transitions.context import build_context_from_boards
    from chess_chain.transitions.executor import execute_program

    policy = get_policy(stage.end_policy_key.split("@", 1)[0])
    context = build_context_from_boards(
        puzzle_id=stage.puzzle_id,
        presented_fen=str(stage.row["start_fen"]),
        endpoint_fen=stage.endpoint_fens()[0],
        solution_uci=stage.accepted_lines[0],
        policy=policy,
    )
    result = execute_program(stage.edge.program, context)
    if not result.ok or result.board is None:
        return fallback.copy()
    return result.board


def _ast_summary(edge: Any) -> str:
    if edge.program is None:
        return "(exact journey: empty program)"
    return _actions_text(edge.program.actions)


def _actions_text(actions: tuple[Any, ...]) -> str:
    from chess_chain.transitions import ast as A

    return "; ".join(_action_text(action, A) for action in actions) if actions else "(none)"


def _action_text(action: Any, ast_module: Any) -> str:
    name = type(action).__name__
    if hasattr(action, "symmetry"):
        return f"{name}(symmetry={action.symmetry})"
    if hasattr(action, "condition"):
        return f"{name}(condition={_conditions_text(action.condition, ast_module)})"
    if hasattr(action, "piece"):
        return f"{name}(piece={_piece_text(action.piece, ast_module)})"
    return name


def _piece_text(selector: Any, ast_module: Any) -> str:
    if isinstance(selector, ast_module.PieceBeganOnSquare):
        import chess as _chess

        return f"PieceBeganOnSquare({_chess.square_name(selector.square)})"
    return type(selector).__name__


def _conditions_text(condition: Any, ast_module: Any) -> str:
    if isinstance(condition, ast_module.OnBoard):
        return f"OnBoard({_piece_text(condition.selector, ast_module)})"
    if isinstance(condition, ast_module.OffBoard):
        return f"OffBoard({_piece_text(condition.selector, ast_module)})"
    return type(condition).__name__


def _inventory_warning(stage: Any, sequence: SequenceResult) -> str | None:
    start = chess.Board(str(stage.row["start_fen"]))
    if not material_feasible_for_standard_set(start):
        return "start position exceeds a standard physical set"
    for line in stage.accepted_lines:
        _, endpoint = replay_line(str(stage.row["start_fen"]), line)
        if not material_feasible_for_standard_set(endpoint):
            return f"accepted line {list(line)} endpoint exceeds a standard physical set"
    return None
