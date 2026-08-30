"""Tests for stage-6 sequence generation (beam search, lazy finalists, registry)."""

from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest
from typer.testing import CliRunner

from chess_chain.cli import app
from chess_chain.config import Settings, get_settings
from chess_chain.sequence import (
    SequenceGenerationError,
    SequenceRegistry,
    SequenceRequest,
    generate_sequence,
    validate_sequence,
)
from chess_chain.storage.catalog import ensure_catalog
from chess_chain.storage.repository import PuzzleRepository
from tests.conftest import write_csv_zst

runner = CliRunner()

# P1 -> P2 (one king move) -> P3 (exact journey, mate-in-1 Ra8#).
CHAIN_ROWS = [
    (
        "P1,4k3/8/6K1/8/8/8/R7/8 w - - 0 1,a2a1 e8e7 a1a3 e7d6,"
        "1500,80,50,100,advantage,https://lichess.org/p1,\n"
    ),
    (
        "P2,8/8/3k2K1/8/8/R7/8/8 b - - 0 1,d6e7 a3a2 e7f8 a2a1 f8g8,"
        "1500,80,50,100,advantage,https://lichess.org/p2,\n"
    ),
    ("P3,7k/8/6K1/8/8/8/8/R7 b - - 0 1,h8g8 a1a8,1500,80,50,100,mate,https://lichess.org/p3,\n"),
]

# A second mate reachable from P2 (rook b1 -> b8#) so finalists can be ranked.
RANKING_ROWS = CHAIN_ROWS + [
    ("P4,7k/8/6K1/8/8/8/8/1R6 b - - 0 1,h8g8 b1b8,1500,80,50,100,mate,https://lichess.org/p4,\n"),
]

# Two identical intermediate puzzles (P2 and S2y) create a genuine tie for the
# beam's seeded tie-break (their seeded order flips between seeds 1 and 2).
SEED_ROWS = [
    CHAIN_ROWS[0],
    CHAIN_ROWS[1],
    CHAIN_ROWS[1].replace("P2,", "S2y,", 1),
    CHAIN_ROWS[2],
]

# Backtracking: the exact edge P1->P2 looks best (width 1), but P2 dead-ends;
# the search must backtrack to P1->PX which reaches the mate.
BACKTRACK_ROWS = [
    (
        "BT1,4k3/8/6K1/8/8/8/R7/8 w - - 0 1,a2a1 e8e7 a1a3 e7d6,"
        "1500,80,50,100,advantage,https://lichess.org/bt1,\n"
    ),
    (
        "BT2,8/8/6K1/2k5/8/R7/8/8 b - - 0 1,c5d6 a3a2 d6d5 a2a1 d5c5,"
        "1500,80,50,100,advantage,https://lichess.org/bt2,\n"
    ),
    (
        "BTX,8/7K/8/3k4/8/R7/8/8 b - - 0 1,d5e6 a3a2 e6e7 a2a1 e7f8 h7g6 f8g8,"
        "1500,80,50,100,advantage,https://lichess.org/btx,\n"
    ),
    ("BT3,7k/8/6K1/8/8/8/8/R7 b - - 0 1,h8g8 a1a8,1500,80,50,100,mate,https://lichess.org/bt3,\n"),
]

# PS: black to move with exactly two legal evasions (Kg7 captures the g7 pawn,
# Kf7 plain) -> two accepted lines with different occupancy that one generic
# transition must map to P3's presented start.
MULTIBRANCH_ROWS = [
    (
        "PS,6k1/6Pp/8/8/8/8/8/R5K1 w - - 0 1,a1a8 g8f7,"
        "1500,80,50,100,advantage,https://lichess.org/ps,\n"
    ),
    CHAIN_ROWS[2],
]


def _ingest(project_root: Path, monkeypatch, rows: list[str]) -> None:
    monkeypatch.chdir(project_root)
    get_settings.cache_clear()
    source = write_csv_zst(project_root / "in.csv.zst", rows)
    processed = project_root / "data" / "processed"
    result = runner.invoke(
        app,
        [
            "db",
            "ingest",
            "--input",
            str(source),
            "--output",
            str(processed),
            "--batch-size",
            str(len(rows)),
        ],
    )
    assert result.exit_code == 0, result.output


def _open(project_root: Path) -> tuple[Settings, PuzzleRepository]:
    settings = Settings(
        project_root=project_root,
        data_processed_dir=project_root / "data" / "processed",
    )
    connection = ensure_catalog(settings.data_processed_dir)
    return settings, PuzzleRepository(connection)


def _graph_for(settings: Settings, repo: PuzzleRepository, request: SequenceRequest):
    import dataclasses

    from chess_chain.sequence.graph import NeighbourGraph

    return NeighbourGraph(
        settings,
        repo,
        transition_config=dataclasses.replace(
            settings.transition_config,
            default_max_actions=request.max_transition_actions,
        ),
    )


def _request(**overrides) -> SequenceRequest:
    base = dict(
        product_id="SEQ-TEST",
        num_stages=3,
        start_puzzle_id="P1",
        acceptance="intended_only",
        max_plies=4,
        max_nodes=2000,
        max_accepted_lines=8,
        collision_scope="up_to_plies",
        collision_max_depth=1,
        collision_max_nodes=5000,
        rating_min=1400,
        rating_max=1600,
        random_seed=7,
        max_transition_actions=4,
        beam_width=4,
        per_node_candidates=4,
        backtrack_budget=30,
        validated_finalist_alternatives=2,
        max_finalists_checked=6,
    )
    base.update(overrides)
    return SequenceRequest(**base)


def test_end_to_end_three_stage_fixture(project_root: Path, monkeypatch) -> None:
    _ingest(project_root, monkeypatch, CHAIN_ROWS)
    settings, repo = _open(project_root)
    registry = SequenceRegistry.load(settings.sequence_dir / "registry.json")
    result = generate_sequence(_request(), settings, repo, registry=registry)
    assert result.puzzle_ids() == ("P1", "P2", "P3")
    assert result.fingerprint
    assert result.validation.valid is True, result.validation.issues
    assert result.stages[-1].final_selection is not None
    assert result.stages[-1].final_selection.final_ends_in_checkmate is True
    # P1 -> P2 is a one-move transition; P2 -> P3 is an exact journey.
    assert result.stages[1].edge is not None
    assert result.stages[1].edge.action_count == 1
    assert result.stages[2].edge is not None
    assert result.stages[2].edge.exact_journey is True
    # Player-facing output: first setup + visible transitions, no hidden codes.
    player = result.to_dict()["player_facing"]
    assert player["first_setup_fen"] == "4k3/8/6K1/8/8/8/8/R7 b - - 1 1"
    assert "Move the piece that began the stage on e8" in player["stages"][1]["transition"]
    assert "No changes needed" in player["stages"][2]["transition"]
    # The registry recorded the manufactured product.
    assert registry.is_registered("SEQ-TEST")
    assert registry.entries["SEQ-TEST"]["fingerprint"] == result.fingerprint


def test_deterministic_same_seed_output(project_root: Path, monkeypatch) -> None:
    _ingest(project_root, monkeypatch, CHAIN_ROWS)
    settings, repo = _open(project_root)
    first = generate_sequence(_request(), settings, repo, register=False)
    second = generate_sequence(_request(), settings, repo, register=False)
    assert first.puzzle_ids() == second.puzzle_ids()
    assert first.fingerprint == second.fingerprint


def test_different_seeds_produce_different_valid_sequences(project_root: Path, monkeypatch) -> None:
    _ingest(project_root, monkeypatch, SEED_ROWS)
    settings, repo = _open(project_root)
    sequences = []
    for seed in (1, 2):
        result = generate_sequence(
            _request(product_id=f"SEED-{seed}", random_seed=seed),
            settings,
            repo,
            register=False,
        )
        assert result.validation.valid is True, result.validation.issues
        sequences.append(result)
    assert sequences[0].puzzle_ids() != sequences[1].puzzle_ids()


def test_backtracking_from_a_dead_end(project_root: Path, monkeypatch) -> None:
    _ingest(project_root, monkeypatch, BACKTRACK_ROWS)
    settings, repo = _open(project_root)
    result = generate_sequence(
        _request(
            product_id="BT",
            start_puzzle_id="BT1",
            beam_width=2,  # keep the tempting exact BT1->BT2 edge and BTX
            per_node_candidates=2,
            backtrack_budget=20,
        ),
        settings,
        repo,
        register=False,
    )
    # The search had to backtrack past the tempting dead-end BT2 to BTX.
    assert result.search_report.backtrack_steps > 0
    assert result.puzzle_ids() == ("BT1", "BTX", "BT3")
    assert result.validation.valid is True, result.validation.issues


def test_exact_journey_preferred_over_a_more_expensive_transition(
    project_root: Path, monkeypatch
) -> None:
    _ingest(project_root, monkeypatch, RANKING_ROWS)
    settings, repo = _open(project_root)
    result = generate_sequence(_request(), settings, repo, register=False)
    final = result.stages[-1].final_selection
    assert final is not None
    # From P2 the finalists are P3 (exact journey) and P4 (rook move); the
    # exact journey must be preferred.
    assert final.rank.puzzle_id == "P3"
    assert final.edge.exact_journey is True


def test_repeated_or_already_used_source_puzzle_rejected(project_root: Path, monkeypatch) -> None:
    _ingest(project_root, monkeypatch, CHAIN_ROWS)
    settings, repo = _open(project_root)
    # Excluding P2 leaves only P3 as an intermediate candidate, which has no
    # outgoing edges: the search must fail with a clear report.
    with pytest.raises(SequenceGenerationError) as excinfo:
        generate_sequence(
            _request(already_used_puzzle_ids=("P2",)),
            settings,
            repo,
            register=False,
        )
    assert excinfo.value.report is not None
    assert excinfo.value.report.no_path is True


def test_final_non_mating_path_rejected(project_root: Path, monkeypatch) -> None:
    _ingest(project_root, monkeypatch, CHAIN_ROWS)
    settings, repo = _open(project_root)
    with pytest.raises(SequenceGenerationError) as excinfo:
        generate_sequence(
            _request(excluded_puzzle_ids=("P3",)),
            settings,
            repo,
            register=False,
        )
    assert excinfo.value.report is not None
    assert excinfo.value.report.no_path is True


def test_multiple_accepted_intermediate_branches_with_different_occupancy(
    project_root: Path, monkeypatch
) -> None:
    _ingest(project_root, monkeypatch, MULTIBRANCH_ROWS)
    settings, repo = _open(project_root)
    result = generate_sequence(
        _request(
            product_id="MB",
            start_puzzle_id="PS",
            num_stages=2,
            acceptance="all_legal",
            max_plies=1,
            max_transition_actions=6,
        ),
        settings,
        repo,
        register=False,
    )
    assert result.puzzle_ids() == ("PS", "P3")
    ps_stage = result.stages[0]
    assert len(ps_stage.accepted_lines) == 2
    endpoint_fens = ps_stage.endpoint_fens()
    assert len({int(chess.Board(fen).occupied) for fen in endpoint_fens}) == 2
    # The one generic transition (with a conditional branch) maps both
    # different-occupancy endpoints to the same next start.
    final_edge = result.stages[1].edge
    assert final_edge is not None
    assert final_edge.ok is True
    assert result.validation.valid is True, result.validation.issues


def test_no_collision_analysis_invoked_for_intermediate_nodes(
    project_root: Path, monkeypatch
) -> None:
    _ingest(project_root, monkeypatch, CHAIN_ROWS)
    settings, repo = _open(project_root)
    run_log: list[str] = []
    result = generate_sequence(
        _request(),
        settings,
        repo,
        register=False,
        collision_run_log=run_log,
    )
    # The only collision analyses that may run are finalists (P3).
    assert run_log, "the finalist was checked"
    for entry in run_log:
        assert entry.startswith("P3:"), f"non-final puzzle got collision analysis: {entry}"
    assert result.puzzle_ids() == ("P1", "P2", "P3")


def test_cached_final_collision_result_reused(project_root: Path, monkeypatch) -> None:
    _ingest(project_root, monkeypatch, CHAIN_ROWS)
    settings, repo = _open(project_root)
    first = generate_sequence(_request(), settings, repo, register=False)
    assert first.stages[-1].final_selection.collision.cached is False
    second = generate_sequence(
        _request(product_id="SEQ-TEST-2"),
        settings,
        repo,
        register=False,
    )
    assert second.stages[-1].final_selection.collision.cached is True


def test_finalists_ranked_before_lazy_collision_checking(project_root: Path, monkeypatch) -> None:
    _ingest(project_root, monkeypatch, RANKING_ROWS)
    settings, repo = _open(project_root)
    run_log: list[str] = []
    result = generate_sequence(
        _request(),
        settings,
        repo,
        register=False,
        collision_run_log=run_log,
    )
    # P3 (exact journey from P2) is ranked first and must be checked first.
    assert run_log[0].startswith("P3:0:")
    # P4 (a one-move transition) is the second-ranked finalist.
    assert any(entry.startswith("P4:1:") for entry in run_log)
    assert result.stages[-1].final_selection.rank.puzzle_id == "P3"


def test_inconclusive_final_collision_rejected_in_strict_mode(
    project_root: Path, monkeypatch
) -> None:
    _ingest(project_root, monkeypatch, CHAIN_ROWS)
    settings, repo = _open(project_root)
    with pytest.raises(SequenceGenerationError):
        generate_sequence(
            _request(collision_max_depth=2, collision_max_nodes=2),
            settings,
            repo,
            register=False,
        )
    # Non-strict mode accepts the inconclusive (bounded) result.
    result = generate_sequence(
        _request(collision_max_depth=2, collision_max_nodes=2, collision_strict=False),
        settings,
        repo,
        register=False,
    )
    assert result.stages[-1].final_selection.collision.status.value == "inconclusive"
    assert result.validation.valid is True, result.validation.issues


def test_clear_failure_report_when_no_path_satisfies_constraints(
    project_root: Path, monkeypatch
) -> None:
    _ingest(project_root, monkeypatch, CHAIN_ROWS)
    settings, repo = _open(project_root)
    with pytest.raises(SequenceGenerationError) as excinfo:
        generate_sequence(
            _request(excluded_puzzle_ids=("P2", "P3")),
            settings,
            repo,
            register=False,
        )
    report = excinfo.value.report
    assert report is not None
    assert report.no_path is True
    assert report.dead_end is True


def test_final_occupancy_variance_rejected(project_root: Path, monkeypatch) -> None:
    _ingest(project_root, monkeypatch, CHAIN_ROWS)
    settings, repo = _open(project_root)
    graph = _graph_for(settings, repo, _request())
    row = graph._row("P3")  # type: ignore[attr-defined]
    # Two crafted accepted lines that end on different occupancies.
    varied_lines = (("a1a8",), ("a1a7",))
    from chess_chain.sequence.finalise import _occupancy_invariant

    assert _occupancy_invariant(row, varied_lines) is False
    # The validator reports the violation on a crafted sequence document.
    from chess_chain.analysis.policies import get_policy
    from chess_chain.sequence.records import SequenceResult, SequenceStageRecord
    from chess_chain.sequence.request import request_fingerprint
    from chess_chain.sequence.search import SearchReport

    policy = get_policy("checkmate")
    final_stage = SequenceStageRecord(
        index=1,
        puzzle_id="P3",
        is_final=True,
        end_policy_key=policy.key(),
        accepted_lines=varied_lines,
        truncated=False,
        row=row,
    )
    start_stage = SequenceStageRecord(
        index=0,
        puzzle_id="P1",
        is_final=False,
        end_policy_key=get_policy("transition_ready").key(),
        accepted_lines=(("e8e7", "a1a3", "e7d6"),),
        truncated=False,
        row=graph._row("P1"),  # type: ignore[attr-defined]
    )
    result = SequenceResult(
        request=_request(),
        fingerprint=request_fingerprint(_request()),
        stages=(start_stage, final_stage),
        validation=None,  # type: ignore[arg-type]
        search_report=SearchReport(),
        product_id="VAR",
    )
    report = validate_sequence(result, graph, _request())
    assert any("do not share one occupancy" in issue for issue in report.issues)


def _exception_detail(exc: BaseException | None) -> str:
    import traceback

    if exc is None:
        return "(no exception)"
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def test_cli_generate_validate_explain_reproduce_round_trip(
    project_root: Path, monkeypatch
) -> None:
    _ingest(project_root, monkeypatch, CHAIN_ROWS)
    settings = Settings(
        project_root=project_root,
        data_processed_dir=project_root / "data" / "processed",
    )
    request_yaml = project_root / "request.yaml"
    request_yaml.write_text(
        "product_id: CLI-1\n"
        "num_stages: 3\n"
        "start_puzzle_id: P1\n"
        "acceptance: intended_only\n"
        "max_plies: 4\n"
        "collision_scope: up_to_plies\n"
        "collision_max_depth: 1\n"
        "rating_min: 1400\n"
        "rating_max: 1600\n"
        "random_seed: 3\n",
        encoding="utf-8",
    )
    sequence_path = project_root / "sequence.json"
    monkeypatch.chdir(project_root)
    # generate
    result = runner.invoke(
        app,
        [
            "sequence",
            "generate",
            "--config",
            str(request_yaml),
            "--output",
            str(sequence_path),
        ],
    )
    assert result.exit_code == 0, result.output + "\n" + _exception_detail(result.exception)
    assert "CLI-1" in result.output and "P1 -> P2 -> P3" in result.output
    data = json.loads(sequence_path.read_text(encoding="utf-8"))
    assert data["schema"] == "sequence@1"
    assert data["validation"]["valid"] is True
    # validate re-executes from the persisted artifact.
    result = runner.invoke(app, ["sequence", "validate", str(sequence_path)])
    assert result.exit_code == 0, result.output
    # explain shows only the player-facing representation.
    result = runner.invoke(app, ["sequence", "explain", str(sequence_path)])
    assert result.exit_code == 0, result.output
    assert "first_setup_fen" in result.output
    assert "manufacturing_qa" not in result.output
    # reproduce regenerates from the registered request and matches.
    result = runner.invoke(app, ["sequence", "reproduce", "--product-id", "CLI-1"])
    assert result.exit_code == 0, result.output
    assert "match=True" in result.output
    # The registry now contains the fingerprint.
    registry_path = settings.sequence_dir / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert "CLI-1" in registry
