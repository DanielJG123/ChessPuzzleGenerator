"""CLI smoke tests for the `analyze` command group."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from chess_chain.cli import app
from chess_chain.config import get_settings
from tests.conftest import MATE_ROWS, SAMPLE_ROWS, write_csv_zst

runner = CliRunner()


def _ingest_dataset(project_root: Path, monkeypatch) -> Path:
    monkeypatch.chdir(project_root)
    get_settings.cache_clear()
    source = write_csv_zst(project_root / "in.csv.zst", MATE_ROWS + SAMPLE_ROWS[:1])
    processed = project_root / "data" / "processed"
    result = runner.invoke(
        app,
        ["db", "ingest", "--input", str(source), "--output", str(processed), "--batch-size", "2"],
    )
    assert result.exit_code == 0, result.output
    return processed


def test_analyze_cheap_smoke(project_root: Path, monkeypatch) -> None:
    _ingest_dataset(project_root, monkeypatch)
    result = runner.invoke(app, ["analyze", "cheap"])
    assert result.exit_code == 0, result.output
    assert "Estimated scope:" in result.output
    assert "Wrote" in result.output


def test_analyze_variations_smoke(project_root: Path, monkeypatch) -> None:
    _ingest_dataset(project_root, monkeypatch)
    result = runner.invoke(
        app,
        [
            "analyze",
            "variations",
            "--puzzle-id",
            "MATE1",
            "--end-policy",
            "checkmate",
            "--acceptance",
            "intended_only",
            "--max-plies",
            "2",
            "--max-nodes",
            "100",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Estimated scope:" in result.output
    assert "lines=1" in result.output


def test_analyze_final_collision_smoke(project_root: Path, monkeypatch) -> None:
    _ingest_dataset(project_root, monkeypatch)
    result = runner.invoke(
        app,
        [
            "analyze",
            "final-collision",
            "--puzzle-id",
            "MATE1",
            "--scope",
            "up_to_plies",
            "--max-depth",
            "3",
            "--max-nodes",
            "10000",
            "--end-policy",
            "checkmate",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Search scope:" in result.output
    assert "status=" in result.output
    assert "Persisted" in result.output


def test_analyze_summarize_smoke(project_root: Path, monkeypatch) -> None:
    _ingest_dataset(project_root, monkeypatch)
    result = runner.invoke(app, ["analyze", "summarize", "--puzzle-id", "MATE1"])
    assert result.exit_code == 0, result.output
    assert '"puzzle_id": "MATE1"' in result.output
    assert "possible_stage_end_policies" in result.output


def test_analyze_deep_without_engine_fails_clearly(project_root: Path, monkeypatch) -> None:
    processed = _ingest_dataset(project_root, monkeypatch)
    result = runner.invoke(
        app, ["analyze", "deep", "--puzzle-id", "MATE1", "--output", str(processed)]
    )
    assert result.exit_code == 1
    assert "Engine unavailable" in result.output
