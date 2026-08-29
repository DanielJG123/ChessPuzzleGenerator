"""Tests for download manifest handling."""

from __future__ import annotations

from pathlib import Path

from chess_chain.storage.manifest import (
    DownloadManifest,
    is_download_complete,
    load_manifest,
    manifest_path_for,
    sha256_file,
    write_manifest,
)


def test_manifest_round_trip_and_hash_validation(tmp_path: Path) -> None:
    target = tmp_path / "lichess_db_puzzle.csv.zst"
    target.write_bytes(b"compressed-puzzle-data")
    digest = sha256_file(target)

    manifest = DownloadManifest(
        source_url="https://example.test/puzzles.zst",
        retrieved_at="2025-01-01T00:00:00+00:00",
        content_length=len(target.read_bytes()),
        etag='"abc"',
        last_modified="Sat, 01 Jan 2025 00:00:00 GMT",
        local_filename=target.name,
        sha256=digest,
    )
    manifest_path = manifest_path_for(target)
    write_manifest(manifest_path, manifest)

    loaded = load_manifest(manifest_path)
    assert loaded == manifest
    assert is_download_complete(target, manifest_path)

    target.write_bytes(b"tampered")
    assert not is_download_complete(target, manifest_path)


def test_incomplete_without_manifest(tmp_path: Path) -> None:
    target = tmp_path / "lichess_db_puzzle.csv.zst"
    target.write_bytes(b"partial")
    assert not is_download_complete(target)
