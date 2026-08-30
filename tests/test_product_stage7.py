"""Stage-7 tests: provisional rating, product rendering, independent validation,
config layering and strict player/private artifact separation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from chess_chain.cli import app
from chess_chain.config import Settings, get_settings
from chess_chain.product.validate import validate_product
from chess_chain.sequence import generate_sequence, load_sequence
from chess_chain.sequence.rating import (
    DifficultyEstimate,
    MissingRatingError,
    SimpleAverageRatingEstimator,
    estimate_from_dict,
    estimate_key,
)
from tests.test_sequence_generation import CHAIN_ROWS, _ingest, _open, _request

runner = CliRunner()


def _generate(project_root: Path, monkeypatch, product_id="S7-TEST") -> Path:
    _ingest(project_root, monkeypatch, CHAIN_ROWS)
    settings, repo = _open(project_root)
    result = generate_sequence(
        _request(product_id=product_id),
        settings,
        repo,
        register=False,
    )
    path = project_root / f"{product_id}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path


# --- Provisional rating boundary --------------------------------------------------


def test_rating_exact_arithmetic_mean_and_stage_order() -> None:
    estimator = SimpleAverageRatingEstimator()
    estimate = estimator.estimate((1500, 1600, 1700))
    assert estimate.value == 1600
    assert estimate.stage_ratings == (1500, 1600, 1700)
    assert estimate.count == 3
    assert estimate.minimum == 1500
    assert estimate.maximum == 1700
    assert estimate.method == "mean_source_puzzle_rating"
    assert estimate.scale == "source_lichess_puzzle_rating"
    assert estimate.provisional is True
    assert estimate.estimator_version == 1
    assert isinstance(estimate.value, int)
    assert SimpleAverageRatingEstimator().estimate((1500, 1501)).value == 1500


def test_rating_explicit_provisional_label_and_no_uncertainty_interval() -> None:
    estimate = SimpleAverageRatingEstimator().estimate((1500, 1500, 1500))
    assert estimate.label == "Provisional source-rating average"
    assert estimate.to_dict()["label"] == "Provisional source-rating average"
    assert "sequence elo" not in estimate.label.lower()
    assert "calibrated" not in estimate.label.lower()
    assert estimate.warnings
    assert "multiplicative difficulty" in estimate.warnings[0]


def test_rating_missing_or_invalid_source_rating_fails() -> None:
    with pytest.raises(MissingRatingError) as excinfo:
        SimpleAverageRatingEstimator().estimate((1500, None, 1700))
    assert "stage 1" in str(excinfo.value)
    assert "never imputes" in str(excinfo.value)
    with pytest.raises(MissingRatingError):
        SimpleAverageRatingEstimator().estimate((1500, "1600"))


def test_rating_forward_compatible_storage_keys() -> None:
    assert estimate_key("mean_source_puzzle_rating", 1) == "mean_source_puzzle_rating@1"
    first = SimpleAverageRatingEstimator().estimate((1500, 1500))
    second = DifficultyEstimate(
        method="calibrated_placeholder",
        estimator_version=2,
        value=1500,
        stage_ratings=(1500, 1500),
    )
    assert first.key != second.key
    assert estimate_from_dict(first.to_dict()) == first


def test_old_record_without_estimate_gets_placeholder_during_render(
    project_root: Path, monkeypatch
) -> None:
    sequence_path = _generate(project_root, monkeypatch)
    sequence = load_sequence(sequence_path)
    assert sequence is not None
    assert sequence.difficulty_estimates == {}  # old record: no estimate
    settings, _ = _open(project_root)
    from chess_chain.product.render import render_product, resolve_estimate

    estimate = resolve_estimate(sequence)
    assert estimate.value == 1500
    artifacts = render_product(
        sequence, settings, estimate=estimate, output_dir=project_root / "out"
    )
    spec = json.loads(artifacts.manufacturing_spec.read_text(encoding="utf-8"))
    assert spec["difficulty"]["provisional_average"]["value"] == 1500
    assert spec["difficulty"]["uncertainty_interval"] is None
    qa = artifacts.qa_report.read_text(encoding="utf-8")
    assert "Provisional source-rating average" in qa
    assert "Uncertainty interval: **none**" in qa


# --- Product rendering and strict separation --------------------------------------


def test_product_render_artifacts_and_player_safety(project_root: Path, monkeypatch) -> None:
    sequence_path = _generate(project_root, monkeypatch)
    settings, _ = _open(project_root)
    from chess_chain.product.render import render_product, resolve_estimate

    sequence = load_sequence(sequence_path)
    assert sequence is not None
    estimate = resolve_estimate(sequence)
    artifacts = render_product(
        sequence, settings, estimate=estimate, output_dir=project_root / "out"
    )
    assert artifacts.player_instructions.is_file()
    assert artifacts.manufacturing_spec.is_file()
    assert artifacts.qa_report.is_file()
    assert artifacts.review_html.is_file()
    assert artifacts.review_notes.is_file()

    player = artifacts.player_instructions.read_text(encoding="utf-8")
    # Player-safe: no solution moves, no private artifact names.
    assert "e8e7" not in player
    assert "a1a3" not in player
    assert "manufacturing_spec.json" not in player
    assert "sequence_review.html" not in player
    assert "qa_report.md" not in player
    assert "board orientation" in player.lower()
    assert "Continue choosing the strongest move" in player
    assert "Stage 1" in player and "ends" in player

    spec = json.loads(artifacts.manufacturing_spec.read_text(encoding="utf-8"))
    assert spec["schema"] == "manufacturing_spec@1"
    assert spec["product_id"] == "S7-TEST"
    assert spec["sequence_fingerprint"] == sequence.fingerprint
    assert spec["random_seed"] == sequence.request.random_seed
    final_stage = spec["stages"][-1]
    assert final_stage["is_final"] is True
    assert final_stage["final"]["checkmate"] is True
    assert final_stage["final"]["mechanical_occupancy_mask"].startswith("0x")
    assert "collision" in final_stage["final"]
    assert spec["difficulty"]["uncertainty_interval"] is None

    review = artifacts.review_html.read_text(encoding="utf-8")
    assert "PRIVATE developer/manual-review material" in review
    assert "Intended solution replay" in review
    assert "sequence_review.html" not in player  # strict separation


def test_review_html_contains_intended_and_branched_lines_and_transition_boards(
    project_root: Path, monkeypatch
) -> None:
    _ingest(project_root, monkeypatch, CHAIN_ROWS)
    settings, repo = _open(project_root)
    result = generate_sequence(
        _request(product_id="HTML-1", acceptance="intended_only", max_plies=4),
        settings,
        repo,
        register=False,
    )
    from chess_chain.product.render import render_product, resolve_estimate

    artifacts = render_product(
        result, settings, estimate=resolve_estimate(result), output_dir=project_root / "html"
    )
    html = artifacts.review_html.read_text(encoding="utf-8")
    assert "Intended solution replay" in html
    assert "e8e7" in html  # intended UCI shown in the private review
    assert "Board after applying the transition" in html
    assert "Next stage start" in html
    assert "Exact match" in html and "PASS" in html
    assert "stage-0" in html and "stage-1" in html and "stage-2" in html
    # Final-only collision info must NOT appear on intermediate stages.
    final_section = html.split('id="stage-2"')[1]
    assert "Final mechanical analysis" in final_section
    intermediate_section = html.split('id="stage-0"')[1].split('id="stage-1"')[0]
    assert "Final mechanical analysis" not in intermediate_section
    assert "collision status" not in intermediate_section


def test_review_notes_template_with_exclusion_example(project_root: Path, monkeypatch) -> None:
    sequence_path = _generate(project_root, monkeypatch)
    settings, _ = _open(project_root)
    from chess_chain.product.render import render_product, resolve_estimate

    sequence = load_sequence(sequence_path)
    assert sequence is not None
    artifacts = render_product(
        sequence, settings, estimate=resolve_estimate(sequence), output_dir=project_root / "out2"
    )
    notes = artifacts.review_notes.read_text(encoding="utf-8")
    assert "sequence_fingerprint:" in notes
    assert "status: open" in notes
    assert "rejected_stages: []" in notes
    assert "excluded_puzzle_ids:" in notes  # copy-ready real field name


def test_artifacts_deterministic_except_timestamps(project_root: Path, monkeypatch) -> None:
    sequence_path = _generate(project_root, monkeypatch)
    settings, _ = _open(project_root)
    from chess_chain.product.render import render_product, resolve_estimate

    sequence = load_sequence(sequence_path)
    assert sequence is not None
    estimate = resolve_estimate(sequence)
    first = render_product(
        sequence,
        settings,
        estimate=estimate,
        output_dir=project_root / "d1",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    second = render_product(
        sequence,
        settings,
        estimate=estimate,
        output_dir=project_root / "d2",
        generated_at="2026-01-02T00:00:00+00:00",
    )
    for name in (
        "player_instructions.md",
        "qa_report.md",
        "sequence_review.html",
        "review_notes.yaml",
    ):
        assert (first.output_dir / name).read_text(encoding="utf-8") == (
            second.output_dir / name
        ).read_text(encoding="utf-8")
    first_spec = json.loads(first.manufacturing_spec.read_text(encoding="utf-8"))
    second_spec = json.loads(second.manufacturing_spec.read_text(encoding="utf-8"))
    assert first_spec["generated_at"] == "2026-01-01T00:00:00+00:00"
    assert second_spec["generated_at"] == "2026-01-02T00:00:00+00:00"
    del first_spec["generated_at"], second_spec["generated_at"]
    assert first_spec == second_spec


# --- Independent product validation -----------------------------------------------


def test_product_validate_passes_for_generated_product(project_root: Path, monkeypatch) -> None:
    sequence_path = _generate(project_root, monkeypatch)
    settings, repo = _open(project_root)
    sequence = load_sequence(sequence_path)
    assert sequence is not None
    report = validate_product(sequence, settings, repo=repo)
    assert report.valid is True, report.explain_failure()
    for check, passed in report.checks.items():
        assert passed, check


def test_product_validate_explain_failure_identifies_exact_location(
    project_root: Path, monkeypatch
) -> None:
    sequence_path = _generate(project_root, monkeypatch)
    settings, repo = _open(project_root)
    sequence = load_sequence(sequence_path)
    assert sequence is not None
    import dataclasses

    final = sequence.stages[-1]
    tampered = dataclasses.replace(final, row={**final.row, "solution_uci": ["a1a7"]})
    tampered_sequence = dataclasses.replace(sequence, stages=sequence.stages[:-1] + (tampered,))
    report = validate_product(tampered_sequence, settings, repo=repo)
    assert report.valid is False
    explanation = report.explain_failure()
    assert "FAILED" in explanation
    assert "final_checkmate" in explanation
    assert "stage 3" in explanation


def test_product_validate_rating_consistency(project_root: Path, monkeypatch) -> None:
    sequence_path = _generate(project_root, monkeypatch)
    settings, repo = _open(project_root)
    sequence = load_sequence(sequence_path)
    assert sequence is not None
    from chess_chain.product.render import write_sequence_with_estimate
    from chess_chain.sequence.records import write_sequence

    estimate = SimpleAverageRatingEstimator().estimate(sequence.stage_ratings())
    stored = write_sequence_with_estimate(sequence, estimate)
    write_sequence(stored, sequence_path)
    report = validate_product(stored, settings, repo=repo)
    assert report.checks["rating_consistency"] is True
    import dataclasses

    tampered_estimates = dict(stored.difficulty_estimates)
    tampered_estimates[estimate.key] = {
        **tampered_estimates[estimate.key],
        "value": 9999,
    }
    tampered = dataclasses.replace(stored, difficulty_estimates=tampered_estimates)
    report = validate_product(tampered, settings, repo=repo)
    assert report.checks["rating_consistency"] is False


# --- Config layering --------------------------------------------------------------


def test_config_precedence_cli_over_env_over_yaml_over_default(
    project_root: Path, monkeypatch
) -> None:
    settings = Settings(project_root=project_root)
    assert settings.engine_threads == 1  # default
    (project_root / "config").mkdir(exist_ok=True)
    (project_root / "config" / "default.yaml").write_text(
        "analysis:\n  engine:\n    threads: 2\n", encoding="utf-8"
    )
    monkeypatch.delenv("CHESS_CHAIN_ENGINE_THREADS", raising=False)
    get_settings.cache_clear()
    yaml_settings = Settings.from_yaml(project_root / "config" / "default.yaml")
    assert yaml_settings.engine_threads == 2  # user YAML beats default
    monkeypatch.setenv("CHESS_CHAIN_ENGINE_THREADS", "3")
    get_settings.cache_clear()
    env_settings = Settings.from_yaml(project_root / "config" / "default.yaml")
    assert env_settings.engine_threads == 3  # env beats YAML
    cli_settings = env_settings.model_copy(
        update={"engine_threads": 4, "engine_path": "/cli/stockfish"}
    )
    assert cli_settings.engine_threads == 4  # CLI override beats everything
    assert cli_settings.engine_path == "/cli/stockfish"


def test_engine_path_required_for_plausible_acceptance(project_root: Path, monkeypatch) -> None:
    _ingest(project_root, monkeypatch, CHAIN_ROWS)
    result = runner.invoke(
        app,
        [
            "analyze",
            "variations",
            "--puzzle-id",
            "P1",
            "--acceptance",
            "plausible",
            "--config",
            str(project_root / "config" / "default.yaml"),
        ],
    )
    assert result.exit_code != 0
    assert "needs an engine" in result.output


# --- CLI workflow -----------------------------------------------------------------


def test_cli_rating_and_product_workflow(project_root: Path, monkeypatch) -> None:
    sequence_path = _generate(project_root, monkeypatch)
    result = runner.invoke(app, ["rating", "estimate-simple", str(sequence_path)])
    assert result.exit_code == 0, result.output
    assert "Provisional source-rating average" in result.output
    assert '"value": 1500' in result.output

    result = runner.invoke(app, ["product", "render", str(sequence_path)])
    assert result.exit_code == 0, result.output
    assert "Player-safe" in result.output
    assert "Private manufacturing/developer material" in result.output

    result = runner.invoke(app, ["product", "validate", str(sequence_path), "--explain-failure"])
    assert result.exit_code == 0, result.output
    assert '"valid": true' in result.output

    result = runner.invoke(app, ["product", "review", str(sequence_path)])
    assert result.exit_code == 0, result.output
    assert "Private review report" in result.output
