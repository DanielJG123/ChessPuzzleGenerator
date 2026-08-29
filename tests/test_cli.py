"""CLI smoke tests."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from chess_chain.cli import app
from chess_chain.config import get_settings
from chess_chain.storage.manifest import (
    DownloadManifest,
    manifest_path_for,
    sha256_file,
    write_manifest,
)
from tests.conftest import MockHttpClient

runner = CliRunner()


def test_cli_download_smoke(settings, project_root: Path, monkeypatch) -> None:
    monkeypatch.chdir(project_root)
    get_settings.cache_clear()
    payload = b"cli-smoke-payload"
    client = MockHttpClient(url="https://example.test/puzzles.zst", payload=payload)

    def _fake_download(**kwargs):
        kwargs["client"] = client
        from chess_chain.ingest.download import download_file as real_download

        return real_download(**kwargs)

    monkeypatch.setattr("chess_chain.cli.download_file", _fake_download)

    result = runner.invoke(app, ["db", "download"])
    assert result.exit_code == 0, result.output
    assert "Downloaded:" in result.output

    target = project_root / "data" / "raw" / "lichess_db_puzzle.csv.zst"
    assert target.read_bytes() == payload


def test_cli_sample_smoke(tiny_puzzle_zst: Path, project_root: Path, monkeypatch) -> None:
    monkeypatch.chdir(project_root)
    manifest = DownloadManifest(
        source_url="fixture",
        retrieved_at="2025-01-01T00:00:00+00:00",
        content_length=tiny_puzzle_zst.stat().st_size,
        etag=None,
        last_modified=None,
        local_filename=tiny_puzzle_zst.name,
        sha256=sha256_file(tiny_puzzle_zst),
    )
    write_manifest(manifest_path_for(tiny_puzzle_zst), manifest)

    result = runner.invoke(
        app,
        [
            "db",
            "sample",
            "--rows",
            "2",
            "--source",
            str(tiny_puzzle_zst),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Sample written:" in result.output
