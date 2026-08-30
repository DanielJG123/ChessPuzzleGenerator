"""CLI smoke tests for the `similarity` command group (stage 4)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from chess_chain.cli import app
from chess_chain.config import get_settings
from tests.conftest import write_csv_zst

runner = CliRunner()

# A deliberately small dataset with a materially feasible pair: puzzle SIM-A's
# endpoint (white rook a2, both kings, black king d8) can reach SIM-B's
# presented start (white rook a2, black king e8) with a short king move.
SIMILARITY_ROWS = [
    (
        "SIM-A,4k3/8/8/8/8/8/8/R3K3 b - - 0 1,e8d8 a1a2,"
        "1500,80,10,50,advantage,https://lichess.org/a,\n"
    ),
    (
        "SIM-B,4k3/8/8/8/8/8/8/R3K3 w - - 0 1,a1a2 e8d8,"
        "1500,80,10,50,advantage,https://lichess.org/b,\n"
    ),
]


def _ingest_dataset(project_root: Path, monkeypatch) -> None:
    monkeypatch.chdir(project_root)
    get_settings.cache_clear()
    source = write_csv_zst(project_root / "in.csv.zst", SIMILARITY_ROWS)
    processed = project_root / "data" / "processed"
    result = runner.invoke(
        app,
        ["db", "ingest", "--input", str(source), "--output", str(processed), "--batch-size", "2"],
    )
    assert result.exit_code == 0, result.output


def test_similarity_build_index_smoke(project_root: Path, monkeypatch) -> None:
    _ingest_dataset(project_root, monkeypatch)
    result = runner.invoke(app, ["similarity", "build-index"])
    assert result.exit_code == 0, result.output
    assert "Indexed" in result.output
    assert "Fingerprint" in result.output
    index_dir = project_root / "data" / "processed" / "analysis" / "similarity"
    assert (index_dir / "index.parquet").is_file()
    assert (index_dir / "index.manifest.json").is_file()


def test_similarity_query_smoke(project_root: Path, monkeypatch) -> None:
    _ingest_dataset(project_root, monkeypatch)
    result = runner.invoke(
        app,
        [
            "similarity",
            "query",
            "--puzzle-id",
            "SIM-A",
            "--top",
            "2",
            "--end-policy",
            "transition_ready",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Index:" in result.output
    assert "Scan:" in result.output
    assert "No candidates" not in result.output


def test_similarity_query_persists_candidates(project_root: Path, monkeypatch) -> None:
    _ingest_dataset(project_root, monkeypatch)
    output = project_root / "out" / "candidates.parquet"
    result = runner.invoke(
        app,
        [
            "similarity",
            "query",
            "--puzzle-id",
            "SIM-A",
            "--top",
            "2",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Persisted" in result.output
    assert output.is_file()


def test_similarity_explain_smoke(project_root: Path, monkeypatch) -> None:
    _ingest_dataset(project_root, monkeypatch)
    result = runner.invoke(
        app,
        ["similarity", "explain", "--from", "SIM-A", "--to", "SIM-B"],
    )
    assert result.exit_code == 0, result.output
    assert '"transition_complexity_score"' in result.output
    assert '"d_occ"' in result.output


def test_similarity_explain_unknown_target(project_root: Path, monkeypatch) -> None:
    _ingest_dataset(project_root, monkeypatch)
    result = runner.invoke(
        app,
        ["similarity", "explain", "--from", "SIM-A", "--to", "MISSING"],
    )
    assert result.exit_code == 2  # typer.BadParameter renders as a usage error
    assert "not present" in result.output


def test_similarity_query_unknown_source(project_root: Path, monkeypatch) -> None:
    _ingest_dataset(project_root, monkeypatch)
    result = runner.invoke(app, ["similarity", "query", "--puzzle-id", "NOPE"])
    assert result.exit_code == 2  # typer.BadParameter renders as a usage error
    assert "not found" in result.output
