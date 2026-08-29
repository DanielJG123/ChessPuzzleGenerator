"""Offline HTTP download tests."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from chess_chain.ingest.download import download_file
from chess_chain.storage.manifest import is_download_complete, load_manifest
from tests.conftest import MockHttpClient


def test_fresh_download_writes_manifest_and_is_idempotent(settings, project_root: Path) -> None:
    payload = b"lichess-export-bytes"
    client = MockHttpClient(url="https://example.test/puzzles.zst", payload=payload)
    target = project_root / "data" / "raw" / "lichess_db_puzzle.csv.zst"

    first = download_file(
        url=client.url,
        output_path=target,
        settings=settings,
        client=client,
    )
    assert not first.skipped
    assert target.read_bytes() == payload
    assert is_download_complete(target)
    manifest = load_manifest(first.manifest_path)
    assert manifest is not None
    assert manifest.sha256 == first.manifest.sha256

    second = download_file(
        url=client.url,
        output_path=target,
        settings=settings,
        client=client,
    )
    assert second.skipped
    assert client.head_calls == 1


def test_resume_from_partial_part_file(settings, project_root: Path) -> None:
    payload = b"0123456789abcdef"
    client = MockHttpClient(url="https://example.test/puzzles.zst", payload=payload)
    target = project_root / "data" / "raw" / "lichess_db_puzzle.csv.zst"
    part = target.with_suffix(target.suffix + ".part")
    part.write_bytes(payload[:10])

    download_file(
        url=client.url,
        output_path=target,
        settings=settings,
        client=client,
    )
    assert target.read_bytes() == payload
    assert not part.exists()
    assert is_download_complete(target)
    assert client.get_calls[0]["Range"] == "bytes=10-"


def test_server_ignoring_range_restarts_safely(settings, project_root: Path) -> None:
    payload = b"full-payload-data!"
    client = MockHttpClient(
        url="https://example.test/puzzles.zst",
        payload=payload,
        ignore_range=True,
    )
    target = project_root / "data" / "raw" / "lichess_db_puzzle.csv.zst"
    part = target.with_suffix(target.suffix + ".part")
    part.write_bytes(b"stale-part")

    download_file(
        url=client.url,
        output_path=target,
        settings=settings,
        client=client,
    )
    assert target.read_bytes() == payload


def test_interruption_leaves_part_and_resume_completes(settings, project_root: Path) -> None:
    payload = b"abcdefghijklmnop"
    client = MockHttpClient(
        url="https://example.test/puzzles.zst",
        payload=payload,
        fail_after_bytes=6,
    )
    target = project_root / "data" / "raw" / "lichess_db_puzzle.csv.zst"
    part = target.with_suffix(target.suffix + ".part")

    with pytest.raises(ConnectionError):
        download_file(
            url=client.url,
            output_path=target,
            settings=settings,
            client=client,
        )

    assert part.exists()
    assert not target.exists()

    client.fail_after_bytes = None
    download_file(
        url=client.url,
        output_path=target,
        settings=settings,
        client=client,
    )
    assert target.read_bytes() == payload


def test_atomic_completion_only_after_full_payload(settings, project_root: Path) -> None:
    payload = b"atomic-final-bytes"
    client = MockHttpClient(url="https://example.test/puzzles.zst", payload=payload)
    target = project_root / "data" / "raw" / "lichess_db_puzzle.csv.zst"
    part = target.with_suffix(target.suffix + ".part")

    download_file(
        url=client.url,
        output_path=target,
        settings=settings,
        client=client,
    )
    assert target.exists()
    assert not part.exists()
    assert not is_download_complete(part)


def test_force_redownloads(settings, project_root: Path) -> None:
    payload = b"version-one"
    client = MockHttpClient(url="https://example.test/puzzles.zst", payload=payload)
    target = project_root / "data" / "raw" / "lichess_db_puzzle.csv.zst"

    download_file(url=client.url, output_path=target, settings=settings, client=client)

    client.payload = b"version-two"
    result = download_file(
        url=client.url,
        output_path=target,
        settings=settings,
        client=client,
        force=True,
    )
    assert not result.skipped
    assert target.read_bytes() == b"version-two"


def test_raw_file_is_readonly_and_force_still_replaces_it(settings, project_root: Path) -> None:
    payload = b"immutable-bytes"
    client = MockHttpClient(url="https://example.test/puzzles.zst", payload=payload)
    target = project_root / "data" / "raw" / "lichess_db_puzzle.csv.zst"

    download_file(url=client.url, output_path=target, settings=settings, client=client)
    # Raw files are immutable (read-only) after a successful download.
    assert not target.stat().st_mode & stat.S_IWRITE

    client.payload = b"replacement"
    download_file(
        url=client.url,
        output_path=target,
        settings=settings,
        client=client,
        force=True,
    )
    assert target.read_bytes() == b"replacement"
    assert not target.stat().st_mode & stat.S_IWRITE


def test_stale_target_without_manifest_is_replaced(settings, project_root: Path) -> None:
    payload = b"clean-download"
    client = MockHttpClient(url="https://example.test/puzzles.zst", payload=payload)
    target = project_root / "data" / "raw" / "lichess_db_puzzle.csv.zst"
    # A leftover file with no (or invalid) manifest must not block re-download.
    target.write_bytes(b"stale-incomplete")

    result = download_file(
        url=client.url,
        output_path=target,
        settings=settings,
        client=client,
    )
    assert not result.skipped
    assert target.read_bytes() == payload
    assert is_download_complete(target)


def test_insufficient_disk_space_fails_clearly(settings, project_root: Path, monkeypatch) -> None:
    payload = b"x" * 100
    client = MockHttpClient(url="https://example.test/puzzles.zst", payload=payload)
    target = project_root / "data" / "raw" / "lichess_db_puzzle.csv.zst"

    monkeypatch.setattr(
        "chess_chain.ingest.download.available_disk_bytes",
        lambda _path: 10,
    )

    with pytest.raises(OSError, match="Insufficient disk space"):
        download_file(
            url=client.url,
            output_path=target,
            settings=settings,
            client=client,
        )
