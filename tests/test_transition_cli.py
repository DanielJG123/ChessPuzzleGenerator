"""CLI smoke tests for the `transition` command group (stage 5)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from chess_chain.cli import app
from chess_chain.config import get_settings
from tests.conftest import write_csv_zst

runner = CliRunner()

# TRANS-A's accepted endpoint (black king d7, white rook a3) maps to
# TRANS-B's presented start (black king e7, white rook a2) with two short
# moves; both positions are white-to-move so the side never changes.
TRANSITION_ROWS = [
    (
        "TRANS-A,4k3/8/8/8/8/8/8/R3K3 w - - 0 1,a1a2 e8e7 a2a3 e7d7,"
        "1500,80,10,50,advantage,https://lichess.org/a,\n"
    ),
    (
        "TRANS-B,4k3/8/8/8/8/8/R7/4K3 b - - 0 1,e8e7 a2a3 e7d7,"
        "1500,80,10,50,advantage,https://lichess.org/b,\n"
    ),
]


def _ingest_dataset(project_root: Path, monkeypatch) -> None:
    monkeypatch.chdir(project_root)
    get_settings.cache_clear()
    source = write_csv_zst(project_root / "in.csv.zst", TRANSITION_ROWS)
    processed = project_root / "data" / "processed"
    result = runner.invoke(
        app,
        ["db", "ingest", "--input", str(source), "--output", str(processed), "--batch-size", "2"],
    )
    assert result.exit_code == 0, result.output


def test_transition_synthesize_verify_and_explain_leakage(project_root: Path, monkeypatch) -> None:
    _ingest_dataset(project_root, monkeypatch)
    output = project_root / "out" / "transition.json"
    result = runner.invoke(
        app,
        [
            "transition",
            "synthesize",
            "--from",
            "TRANS-A",
            "--to",
            "TRANS-B",
            "--end-policy",
            "transition_ready",
            "--max-actions",
            "2",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Candidate 1" in result.output
    assert "Wrote" in result.output
    assert output.is_file()
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema"] == "transition@1"
    assert document["source_puzzle_id"] == "TRANS-A"
    assert document["next_puzzle_id"] == "TRANS-B"
    assert document["verification"]["passed"] is True
    # The persisted artifact stores the AST, never the rendered text alone.
    assert "actions" in document["program"]
    assert "Move" in document["rendered"]

    verify_result = runner.invoke(app, ["transition", "verify", str(output)])
    assert verify_result.exit_code == 0, verify_result.output
    assert "Verified: True" in verify_result.output

    explain_result = runner.invoke(app, ["transition", "explain-leakage", str(output)])
    assert explain_result.exit_code == 0, explain_result.output
    assert "clue_report" in explain_result.output
    assert "vector_leakage_bits" in explain_result.output


def test_transition_verify_rejects_corrupt_artifact(project_root: Path) -> None:
    path = project_root / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    result = runner.invoke(app, ["transition", "verify", str(path)])
    assert result.exit_code == 2
    assert "cannot load transition document" in result.output
