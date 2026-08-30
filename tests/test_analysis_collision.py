"""Tests for the lazy final-only occupancy-collision analyser."""

from __future__ import annotations

import chess

from chess_chain.analysis.collision import (
    CollisionRequest,
    CollisionStatus,
    FinalOccupancyCollisionAnalyzer,
)
from chess_chain.domain.position import PositionFingerprint


def _request(
    puzzle_id: str = "P1",
    *,
    start_fen: str,
    target: int,
    scope: str = "up_to_plies",
    max_depth: int = 4,
    max_nodes: int = 50_000,
    accepted_policy: str = "intended_only",
    stage_end_policy: str = "checkmate@v1",
    analyser_version: int = 1,
) -> CollisionRequest:
    return CollisionRequest(
        puzzle_id=puzzle_id,
        source_version=1,
        presented_fingerprint=PositionFingerprint.from_board(chess.Board(start_fen)).to_key(),
        target_occupancy=target,
        accepted_line_policy=accepted_policy,
        stage_end_policy=stage_end_policy,
        scope=scope,
        max_depth=max_depth,
        max_nodes=max_nodes,
        pruning_config="transposition",
        analyser_version=analyser_version,
    )


def _target_after(fen: str, line: tuple[str, ...]) -> int:
    board = chess.Board(fen)
    for uci in line:
        board.push(chess.Move.from_uci(uci))
    return int(board.occupied)


def test_unintended_line_reaches_target_occupancy() -> None:
    # Intended: Nd2-c4. A non-accepted knight zig-zag (e.g. d2e4, ...d6c4, ...)
    # reaches the exact same occupancy mask within the declared scope.
    fen = "r3k3/8/8/8/8/8/3N4/7K w - - 0 1"
    target = _target_after(fen, ("d2c4",))
    request = _request(start_fen=fen, target=target, max_depth=6)
    analyzer = FinalOccupancyCollisionAnalyzer()
    result = analyzer.analyze(
        request=request,
        board=chess.Board(fen),
        accepted_lines=(("d2c4",),),
    )
    assert result.status == CollisionStatus.COLLISION_FOUND
    assert result.verified_acceptance_occupancy is True
    # The witness is a non-accepted line ending with the knight on c4; the exact
    # move order/depth depends on python-chess move generation and DFS order.
    assert result.witness_line is not None
    assert result.witness_depth is not None
    assert result.witness_depth <= request.max_depth
    witness_board = chess.Board(fen)
    for uci in result.witness_line:
        witness_board.push(chess.Move.from_uci(uci))
    assert int(witness_board.occupied) == target


def test_early_prefix_final_occupancy_collision() -> None:
    # The intended 4-ply shuttle returns every piece to its start square, so the
    # intended final mask equals the presented-start mask: the box would open at
    # depth 0, i.e. as soon as the target mask appears.
    fen = "r3k3/8/8/8/8/8/8/2R4K w - - 0 1"  # white Rc1, Kh1; black Ra8, Ke8
    intended = ("c1c7", "a8a1", "c7c1", "a1a8")
    target = _target_after(fen, intended)
    assert target == int(chess.Board(fen).occupied)
    request = _request(start_fen=fen, target=target, max_depth=6)
    analyzer = FinalOccupancyCollisionAnalyzer()
    result = analyzer.analyze(
        request=request,
        board=chess.Board(fen),
        accepted_lines=(intended,),
    )
    assert result.status == CollisionStatus.COLLISION_FOUND
    assert result.witness_depth == 0  # the start occupancy already equals the target


def test_collision_free_within_scope_is_bounded() -> None:
    # MATE fixture: only the intended f1f8 produces the final mask within one ply.
    fen = "6k1/6pp/8/8/8/8/8/5R1K w - - 0 1"
    target = _target_after(fen, ("f1f8",))
    shallow = _request(start_fen=fen, target=target, scope="exact_plies", max_depth=1)
    analyzer = FinalOccupancyCollisionAnalyzer()
    result = analyzer.analyze(
        request=shallow,
        board=chess.Board(fen),
        accepted_lines=(("f1f8",),),
    )
    assert result.status == CollisionStatus.COLLISION_FREE_WITHIN_SCOPE
    # The bounded meaning: a deeper scope finds a shuffle-back collision.
    deep = _request(start_fen=fen, target=target, scope="up_to_plies", max_depth=5)
    deeper = analyzer.analyze(
        request=deep,
        board=chess.Board(fen),
        accepted_lines=(("f1f8",),),
    )
    assert deeper.status == CollisionStatus.COLLISION_FOUND


def test_accepted_line_itself_is_not_a_collision() -> None:
    fen = "r3k3/8/8/8/8/8/3N4/7K w - - 0 1"
    intended = ("d2c4",)
    target = _target_after(fen, intended)
    request = _request(start_fen=fen, target=target, max_depth=1)
    analyzer = FinalOccupancyCollisionAnalyzer()
    result = analyzer.analyze(
        request=request,
        board=chess.Board(fen),
        accepted_lines=(intended,),
    )
    assert result.status == CollisionStatus.COLLISION_FREE_WITHIN_SCOPE
    assert result.verified_acceptance_occupancy is True


def test_accepted_lines_occupancy_mismatch_is_inconclusive() -> None:
    fen = "r3k3/8/8/8/8/8/3N4/7K w - - 0 1"
    intended = ("d2c4",)
    target = _target_after(fen, intended)
    request = _request(start_fen=fen, target=target, max_depth=3)
    analyzer = FinalOccupancyCollisionAnalyzer()
    # An "accepted" line that does not end at the target occupancy.
    result = analyzer.analyze(
        request=request,
        board=chess.Board(fen),
        accepted_lines=(("d2b3",),),
    )
    assert result.status == CollisionStatus.INCONCLUSIVE
    assert result.verified_acceptance_occupancy is False
    assert "occupancy" in (result.note or "")


def test_plausible_horizon_scope_restricts_the_search(tmp_path) -> None:
    from chess_chain.analysis.cache import DiskKeyValueCache

    fen = "r3k3/8/8/8/8/8/3N4/7K w - - 0 1"
    intended = ("d2c4",)
    target = _target_after(fen, intended)
    request = _request(start_fen=fen, target=target, scope="plausible_horizon", max_depth=6)

    # A "plausible" filter that only accepts the intended move prunes the whole
    # alternative tree, so no non-accepted line can reach the target.
    def only_intended(board: chess.Board, move: chess.Move) -> bool:
        return move.uci() == "d2c4"

    restricted_cache = DiskKeyValueCache(tmp_path / "collision_cache_restricted")
    restricted = FinalOccupancyCollisionAnalyzer(cache=restricted_cache).analyze(
        request=request,
        board=chess.Board(fen),
        accepted_lines=(intended,),
        move_filter=only_intended,
    )
    assert restricted.status == CollisionStatus.COLLISION_FREE_WITHIN_SCOPE

    unrestricted_cache = DiskKeyValueCache(tmp_path / "collision_cache_unrestricted")
    unrestricted = FinalOccupancyCollisionAnalyzer(cache=unrestricted_cache).analyze(
        request=request,
        board=chess.Board(fen),
        accepted_lines=(intended,),
        move_filter=None,
    )
    assert unrestricted.status == CollisionStatus.COLLISION_FOUND


def test_node_limit_produces_inconclusive(tmp_path) -> None:
    from chess_chain.analysis.cache import DiskKeyValueCache

    fen = "r3k3/8/8/8/8/8/3N4/7K w - - 0 1"
    target = _target_after(fen, ("d2c4",))
    request = _request(start_fen=fen, target=target, max_depth=3, max_nodes=2)
    cache = DiskKeyValueCache(tmp_path / "collision_cache")
    analyzer = FinalOccupancyCollisionAnalyzer(cache=cache)
    result = analyzer.analyze(
        request=request,
        board=chess.Board(fen),
        accepted_lines=(("d2c4",),),
    )
    assert result.status == CollisionStatus.INCONCLUSIVE


def test_cache_keys_differ_when_configuration_changes(tmp_path) -> None:
    from chess_chain.analysis.cache import DiskKeyValueCache

    fen = "r3k3/8/8/8/8/8/3N4/7K w - - 0 1"
    target = _target_after(fen, ("d2c4",))
    cache = DiskKeyValueCache(tmp_path / "collision_cache")
    shallow = _request(start_fen=fen, target=target, scope="up_to_plies", max_depth=1)
    deep = _request(start_fen=fen, target=target, scope="up_to_plies", max_depth=6)
    assert shallow.cache_key(cache) != deep.cache_key(cache)


def test_cached_collision_result_is_reused(tmp_path) -> None:
    from chess_chain.analysis.cache import DiskKeyValueCache

    fen = "r3k3/8/8/8/8/8/3N4/7K w - - 0 1"
    target = _target_after(fen, ("d2c4",))
    request = _request(start_fen=fen, target=target, max_depth=6)
    cache = DiskKeyValueCache(tmp_path / "collision_cache")
    analyzer = FinalOccupancyCollisionAnalyzer(cache=cache)
    first = analyzer.analyze(
        request=request,
        board=chess.Board(fen),
        accepted_lines=(("d2c4",),),
    )
    assert first.cached is False
    second = analyzer.analyze(
        request=request,
        board=chess.Board(fen),
        accepted_lines=(("d2c4",),),
    )
    assert second.cached is True
    assert second.status == first.status
    assert second.witness_line == first.witness_line
