"""CLI smoke tests for `db ingest` and `db stats`."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from chess_chain.cli import app
from chess_chain.config import get_settings
from tests.conftest import BAD_ROWS, MATE_ROWS, SAMPLE_ROWS, write_csv_zst

runner = CliRunner()


def test_cli_ingest_and_stats_smoke(project_root: Path, monkeypatch) -> None:
    monkeypatch.chdir(project_root)
    get_settings.cache_clear()

    source = write_csv_zst(
        project_root / "in.csv.zst",
        SAMPLE_ROWS + MATE_ROWS + [row for row, _ in BAD_ROWS],
    )
    output = project_root / "data" / "processed"

    result = runner.invoke(
        app,
        [
            "db",
            "ingest",
            "--input",
            str(source),
            "--output",
            str(output),
            "--batch-size",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Ingested:" in result.output
    assert output.joinpath("catalog.duckdb").is_file()

    stats = runner.invoke(app, ["db", "stats", "--output", str(output)])
    assert stats.exit_code == 0, stats.output
    assert "Puzzles:" in stats.output
    assert "Quarantined rows:" in stats.output
