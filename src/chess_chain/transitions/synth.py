"""Bounded transition synthesis.

Strategy (ordered by preference):

1. whole-board transforms first: a single ``Transform`` action (with an
   ``EstablishRights`` annotation when the product allows it) is the cheapest
   valid program;
2. short relative operations derived from the similarity piece-assignment
   plans (identity-anchored selectors such as "the piece that began on a1",
   relative offsets where the offset is identical across accepted endpoints,
   absolute squares only as a penalised fallback);
3. replacements / additions / removals / swaps only when needed.

Search is bounded: candidate plans come from the deterministic assignment of
the first accepted endpoint, unified over every accepted endpoint by physical
piece identity, with a hard ``max_actions`` cap. No LLM is involved; every
candidate is executed and verified (see ``verify.py``) before it is ranked.
"""

from __future__ import annotations

import chess

from chess_chain.analysis.policies import StageEndPolicy
from chess_chain.domain.pieces import PIECE_ORDER, StagePieceId
from chess_chain.domain.position import CanonicalPosition
from chess_chain.similarity.assignment import assign_pieces
from chess_chain.similarity.symmetry import SYMMETRIES
from chess_chain.transitions import ast as A
from chess_chain.transitions.config import TransitionConfig
from chess_chain.transitions.context import EndpointContext
from chess_chain.transitions.guards import StageBoundaryGuard
from chess_chain.transitions.leakage import PlausibleLines, derive_plausible_lines
from chess_chain.transitions.render import render_program
from chess_chain.transitions.verify import (
    VerificationResult,
    generate_plausible_wrong_contexts,
    verify_program,
)

__all__ = ["SynthesisCandidate", "synthesize"]


class SynthesisCandidate:
    """One synthesised transition, already executed and verified."""

    def __init__(
        self,
        *,
        program: A.TransitionProgram,
        guard: StageBoundaryGuard,
        verification: VerificationResult,
        next_start: CanonicalPosition,
    ) -> None:
        self.program = program
        self.guard = guard
        self.verification = verification
        self.next_start = next_start
        self.rendered = render_program(program)

    def rank_key(self) -> tuple[object, ...]:
        clue = self.verification.clue
        creates_next = sum(
            1 for outcome in self.verification.non_accepted_outcomes if outcome.creates_next_start
        )
        return (
            not self.verification.passed,
            creates_next,
            # More distinct outcomes across plausible endpoints means the next
            # start is *harder* to reconstruct without solving -> better.
            -clue.distinct_next_positions_across_plausible,
            round(clue.vector_leakage_bits + clue.conditional_leakage_bits, 6),
            clue.absolute_source_squares + clue.absolute_destination_squares,
            len(self.program.actions),
            round(clue.total_clue_bits, 6),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "program": self.program.to_json(),
            "guard": self.guard.to_json(),
            "rendered": self.rendered,
            "rank_key": [str(item) for item in self.rank_key()],
            "verification": self.verification.to_dict(),
        }


def synthesize(
    contexts: tuple[EndpointContext, ...],
    next_start: CanonicalPosition,
    *,
    policy: StageEndPolicy,
    config: TransitionConfig | None = None,
    plausible: PlausibleLines | None = None,
) -> list[SynthesisCandidate]:
    """Search for verified transition programs mapping every accepted endpoint
    to ``next_start``, ranked transparently (see ``rank_key``)."""
    config = config or TransitionConfig()
    if not contexts:
        return []

    candidates: list[SynthesisCandidate] = []

    # 1. Whole-board transforms.
    for symmetry in SYMMETRIES:
        program = A.TransitionProgram((A.Transform(symmetry),))
        candidate = _try_candidate(
            program,
            contexts=contexts,
            next_start=next_start,
            policy=policy,
            config=config,
            plausible=plausible,
        )
        if candidate is not None:
            candidates.append(candidate)

    # 2. Assignment-derived plans (relative, then absolute destinations).
    for use_absolute in (False, True):
        plan_program = _build_plan_program(
            contexts,
            next_start,
            use_absolute=use_absolute,
            max_actions=config.default_max_actions,
        )
        if plan_program is None:
            continue
        candidate = _try_candidate(
            plan_program,
            contexts=contexts,
            next_start=next_start,
            policy=policy,
            config=config,
            plausible=plausible,
        )
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(key=lambda candidate: candidate.rank_key())
    return [candidate for candidate in candidates if candidate.verification.passed]


def _try_candidate(
    program: A.TransitionProgram,
    *,
    contexts: tuple[EndpointContext, ...],
    next_start: CanonicalPosition,
    policy: StageEndPolicy,
    config: TransitionConfig,
    plausible: PlausibleLines | None,
) -> SynthesisCandidate | None:
    prefixes = _prefixes_for(contexts) if contexts else None
    wrong_states = (
        _wrong_for(
            contexts[0],
            config,
            frozenset(context.endpoint.fen for context in contexts),
        )
        if contexts
        else None
    )
    if plausible is None and contexts:
        plausible = derive_plausible_lines(
            contexts[0],
            depth=config.plausible_depth,
            node_cap=config.plausible_node_cap,
        )
    guard = StageBoundaryGuard.from_policy(
        policy,
        transition_program=program if policy.kind == "TRANSITION_READY" else None,
        expected_next=next_start if policy.kind == "TRANSITION_READY" else None,
    )
    verification = verify_program(
        program,
        accepted=contexts,
        next_start=next_start,
        prefixes=prefixes,
        wrong_states=wrong_states,
        guard=guard,
        plausible=plausible,
        config=config,
    )
    return SynthesisCandidate(
        program=program,
        guard=guard,
        verification=verification,
        next_start=next_start,
    )


def _prefixes_for(contexts: tuple[EndpointContext, ...]) -> tuple[EndpointContext, ...]:
    from chess_chain.transitions.context import build_prefix_contexts

    return build_prefix_contexts(contexts[0], contexts[0].stage_end_policy)


def _wrong_for(
    context: EndpointContext,
    config: TransitionConfig,
    accepted_fens: frozenset[str],
) -> tuple[EndpointContext, ...]:
    return generate_plausible_wrong_contexts(
        context,
        depth=config.plausible_depth,
        node_cap=config.plausible_node_cap,
        excluded_fens=accepted_fens,
    )


def _build_plan_program(
    contexts: tuple[EndpointContext, ...],
    next_start: CanonicalPosition,
    *,
    use_absolute: bool,
    max_actions: int,
) -> A.TransitionProgram | None:
    """Build a generic action plan from the first endpoint's piece assignment,
    unified across every accepted endpoint by physical piece identity.

    Returns None when no consistent plan fits within ``max_actions``.
    """
    from chess_chain.transitions.executor import execute_program

    first = contexts[0]

    # Exact journey: every accepted endpoint already equals the next start.
    if all(context.endpoint.fingerprint == next_start.fingerprint for context in contexts):
        return A.TransitionProgram(())

    first_counts = _material_counts(first.endpoint.board)
    start_counts = _material_counts(first.presented_start.board)
    assignment = assign_pieces(
        first.endpoint.board,
        next_start.board,
        start_counts,
        first_counts,
    )
    if not assignment.feasible:
        return None

    target_for: dict[StagePieceId, chess.Square] = {}
    for pair in assignment.matched:
        if pair.source_square is not None:
            piece = first.identity_on(pair.source_square)
            if piece is not None:
                target_for[piece] = pair.target_square
        else:
            tray_piece = _pick_tray_piece(first, pair.colour, pair.piece_type, contexts)
            if tray_piece is None:
                return None
            target_for[tray_piece] = pair.target_square

    removed = frozenset(
        piece
        for pair in assignment.matched
        if pair.source_square is not None
        for piece in (first.identity_on(pair.source_square),)
        if piece is not None
    )
    all_on_board = {piece for square, piece in first.identities.items()}
    removed = frozenset(all_on_board - set(target_for))

    actions: list[A.Action] = []
    used_in_swap: set[StagePieceId] = set()

    # Emit Swap for 2-cycles first.
    targets = {piece: square for piece, square in target_for.items()}
    for piece in list(targets):
        if piece in used_in_swap:
            continue
        swap_dest = targets[piece]
        partner = _piece_at(targets, swap_dest)
        if (
            partner is not None
            and partner != piece
            and targets.get(partner) == _square_of(first, piece)
        ):
            sel_a = _selector_for(piece)
            sel_b = _selector_for(partner)
            actions.append(A.Swap(sel_a, sel_b))
            used_in_swap.add(piece)
            used_in_swap.add(partner)

    for piece in sorted(targets, key=lambda p: p.key()):
        if piece in used_in_swap:
            continue
        target = targets[piece]
        selector = _selector_for(piece)
        squares = [context.piece_square.get(piece) for context in contexts]
        on_board_squares = [square for square in squares if square is not None]
        if on_board_squares and all(square == target for square in on_board_squares):
            continue  # the piece already sits on its next-start square everywhere
        if use_absolute:
            destination: A.Destination = A.AbsoluteSquare(target)
        else:
            relative = _relative_offsets(contexts, piece, target)
            if relative is None:
                destination = A.AbsoluteSquare(target)
            else:
                destination = A.RelativeOffset(selector, relative[0], relative[1])
        present = [context.piece_square.get(piece) is not None for context in contexts]
        if all(present):
            actions.append(A.Move(selector, destination))
        elif not any(present):
            actions.append(A.Add(selector, destination))
        else:
            actions.append(
                A.Conditional(
                    (
                        A.Branch(A.OnBoard(selector), (A.Move(selector, destination),)),
                        A.Branch(A.OffBoard(selector), (A.Add(selector, destination),)),
                    )
                )
            )

    for piece in sorted(removed, key=lambda p: p.key()):
        selector = _selector_for(piece)
        present = [context.piece_square.get(piece) is not None for context in contexts]
        if all(present):
            actions.append(A.Remove(selector))
        elif any(present):
            actions.append(
                A.Conditional(
                    (
                        A.Branch(A.OnBoard(selector), (A.Remove(selector),)),
                        A.Branch(A.OffBoard(selector), ()),
                    )
                )
            )

    if not actions:
        return None
    if len(actions) > max_actions:
        return None

    program = A.TransitionProgram(tuple(actions))
    # Rights: the physical actions only void rights; if the next start carries
    # rights/ep the program must state them explicitly (when permitted). The
    # plan is only valid when the resulting *placement* matches; rights are
    # then established via EstablishRights metadata.
    result = execute_program(program, first)
    if not result.ok or result.board is None:
        return None
    from chess_chain.domain.position import PositionFingerprint

    fingerprint = PositionFingerprint.from_board(result.board)
    if (
        fingerprint.bitboards != next_start.fingerprint.bitboards
        or fingerprint.turn != next_start.fingerprint.turn
    ):
        return None
    if fingerprint.occupied != next_start.fingerprint.occupied:
        return None
    if (
        fingerprint.castling != next_start.fingerprint.castling
        or fingerprint.ep_square != next_start.fingerprint.ep_square
    ):
        rights_action = A.EstablishRights(
            castling_mask=next_start.fingerprint.castling,
            ep_square=next_start.fingerprint.ep_square,
        )
        program = A.TransitionProgram(tuple(actions) + (rights_action,))
    return program


def _selector_for(piece: StagePieceId) -> A.Selector:
    """Identity-anchored selector: works across endpoints because the physical
    piece (start square) is the same object everywhere."""
    return A.PieceBeganOnSquare(piece.start_square)


def _relative_offsets(
    contexts: tuple[EndpointContext, ...],
    piece: StagePieceId,
    target: chess.Square,
) -> tuple[int, int] | None:
    """The (dx, dy) from the piece's current square to ``target``, identical
    across every accepted endpoint where the piece is on the board."""
    offset: tuple[int, int] | None = None
    for context in contexts:
        square = context.piece_square.get(piece)
        if square is None:
            continue
        candidate = (
            chess.square_file(target) - chess.square_file(square),
            chess.square_rank(target) - chess.square_rank(square),
        )
        if offset is None:
            offset = candidate
        elif offset != candidate:
            return None
    return offset


def _pick_tray_piece(
    context: EndpointContext,
    colour: chess.Color,
    piece_type: chess.PieceType,
    contexts: tuple[EndpointContext, ...],
) -> StagePieceId | None:
    """A deterministic off-board piece matching ``(colour, piece_type)`` that is
    off the board in *every* accepted endpoint (smallest start square wins)."""
    candidates = [
        piece
        for piece, square in context.piece_square.items()
        if square is None and piece.colour == colour and piece.piece_type == piece_type
    ]
    candidates = [
        piece
        for piece in candidates
        if all(other.piece_square.get(piece) is None for other in contexts)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda piece: piece.start_square)


def _piece_at(
    squares: dict[StagePieceId, chess.Square], square: chess.Square
) -> StagePieceId | None:
    for piece, piece_square in squares.items():
        if piece_square == square:
            return piece
    return None


def _square_of(context: EndpointContext, piece: StagePieceId) -> chess.Square | None:
    return context.piece_square.get(piece)


def _material_counts(board: chess.Board) -> tuple[int, ...]:
    return tuple(len(board.pieces(piece_type, colour)) for colour, piece_type in PIECE_ORDER)
