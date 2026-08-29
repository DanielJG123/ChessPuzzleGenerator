"""Resumable HTTP download utilities for immutable raw datasets."""

from __future__ import annotations

import logging
import os
import shutil
import stat
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from chess_chain.config import Settings
from chess_chain.storage.manifest import (
    DownloadManifest,
    IncompleteDownloadError,
    is_download_complete,
    load_manifest,
    manifest_path_for,
    sha256_file,
    utc_now_iso,
    write_manifest,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemoteResourceInfo:
    url: str
    content_length: int | None
    etag: str | None
    last_modified: str | None
    accept_ranges: bool


@dataclass(frozen=True)
class DownloadResult:
    target_path: Path
    manifest_path: Path
    manifest: DownloadManifest
    skipped: bool


class HttpClient(Protocol):
    def head(self, url: str) -> dict[str, str]: ...

    def stream_get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        chunk_size: int = 1024 * 1024,
    ) -> tuple[int, dict[str, str], Iterator[bytes]]: ...


class UrllibHttpClient:
    def head(self, url: str) -> dict[str, str]:
        request = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return {key.lower(): value for key, value in response.headers.items()}
        except urllib.error.HTTPError:
            # Servers may reject HEAD; the download itself will still work.
            return {}
        except urllib.error.URLError:
            return {}

    def stream_get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        chunk_size: int = 1024 * 1024,
    ) -> tuple[int, dict[str, str], Iterator[bytes]]:
        request = urllib.request.Request(url, headers=headers or {})

        def _iter_body(response: Any) -> Iterator[bytes]:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                yield chunk

        try:
            response = urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as exc:
            headers_map = {key.lower(): value for key, value in exc.headers.items()}
            body = exc.read()

            def _error_body() -> Iterator[bytes]:
                if body:
                    yield body

            return exc.code, headers_map, _error_body()

        status = getattr(response, "status", response.getcode())
        response_headers = {key.lower(): value for key, value in response.headers.items()}
        return status, response_headers, _iter_body(response)


def _parse_int_header(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def inspect_remote(url: str, client: HttpClient) -> RemoteResourceInfo:
    headers = client.head(url)
    accept_ranges = headers.get("accept-ranges", "").lower() == "bytes"
    return RemoteResourceInfo(
        url=url,
        content_length=_parse_int_header(headers.get("content-length")),
        etag=headers.get("etag"),
        last_modified=headers.get("last-modified"),
        accept_ranges=accept_ranges,
    )


def available_disk_bytes(path: Path) -> int:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(resolved.parent)
    return usage.free


def ensure_sufficient_space(
    destination: Path,
    required_bytes: int | None,
    margin_bytes: int,
) -> None:
    if required_bytes is None:
        logger.warning(
            "Remote content length unavailable; skipping free-space pre-check for %s",
            destination,
        )
        return
    needed = required_bytes + margin_bytes
    free = available_disk_bytes(destination)
    if free < needed:
        raise OSError(
            f"Insufficient disk space for download to {destination}: "
            f"need at least {needed} bytes (file {required_bytes} + margin {margin_bytes}), "
            f"but only {free} bytes are available."
        )


def _part_path(target: Path) -> Path:
    return target.with_suffix(target.suffix + ".part")


def _atomic_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
    os.replace(source, destination)


def _make_readonly(path: Path) -> None:
    """Make a raw file immutable (Windows: sets the read-only attribute)."""
    path.chmod(stat.S_IREAD)


def _make_writable(path: Path) -> None:
    path.chmod(stat.S_IREAD | stat.S_IWRITE)


def _write_stream_to_part(
    body: Iterator[bytes],
    part_path: Path,
    *,
    mode: str,
) -> int:
    part_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with part_path.open(mode) as handle:
        for chunk in body:
            handle.write(chunk)
            written += len(chunk)
    return written


def _finalise_part(
    *,
    part_path: Path,
    target: Path,
    manifest_path: Path,
    url: str,
    remote: RemoteResourceInfo,
    chunk_size: int,
) -> DownloadManifest:
    final_size = part_path.stat().st_size
    if remote.content_length is not None and final_size != remote.content_length:
        raise IncompleteDownloadError(
            f"Downloaded size {final_size} does not match expected {remote.content_length}"
        )

    digest = sha256_file(part_path, chunk_size=chunk_size)
    _atomic_replace(part_path, target)
    _make_readonly(target)

    manifest = DownloadManifest(
        source_url=url,
        retrieved_at=utc_now_iso(),
        content_length=remote.content_length if remote.content_length is not None else final_size,
        etag=remote.etag,
        last_modified=remote.last_modified,
        local_filename=target.name,
        sha256=digest,
    )
    write_manifest(manifest_path, manifest)
    logger.info("Download complete: %s (sha256=%s)", target, digest)
    return manifest


def download_file(
    *,
    url: str,
    output_path: Path,
    settings: Settings,
    force: bool = False,
    resume: bool = True,
    client: HttpClient | None = None,
) -> DownloadResult:
    """Download a remote file with resume support and manifest recording."""
    http = client or UrllibHttpClient()
    target = output_path.resolve()
    part_path = _part_path(target)
    manifest_path = manifest_path_for(target)

    if not force and is_download_complete(target, manifest_path):
        manifest = load_manifest(manifest_path)
        if manifest is None:
            raise IncompleteDownloadError(f"Manifest missing for complete-looking file: {target}")
        logger.info("Download already complete: %s", target)
        return DownloadResult(
            target_path=target,
            manifest_path=manifest_path,
            manifest=manifest,
            skipped=True,
        )

    remote = inspect_remote(url, http)
    if not force and target.is_file() and not is_download_complete(target, manifest_path):
        logger.warning(
            "Existing file at %s lacks a valid manifest; removing it and re-downloading",
            target,
        )
        _make_writable(target)
        target.unlink()

    if force:
        part_path.unlink(missing_ok=True)
        if target.exists():
            _make_writable(target)
            target.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

    starting_offset = 0
    if resume and part_path.is_file():
        starting_offset = part_path.stat().st_size
        logger.info("Resuming partial download at byte %s", starting_offset)

    if remote.content_length is not None:
        remaining = max(0, remote.content_length - starting_offset)
        ensure_sufficient_space(
            target,
            remaining,
            settings.download_space_margin_bytes,
        )

    request_headers: dict[str, str] = {}
    if resume and starting_offset > 0:
        request_headers["Range"] = f"bytes={starting_offset}-"

    status, response_headers, body = http.stream_get(
        url,
        headers=request_headers,
        chunk_size=settings.download_chunk_size,
    )

    if status == 416:
        if remote.content_length is not None and starting_offset == remote.content_length:
            logger.info("Partial file already matches remote size; finalising %s", target)
            manifest = _finalise_part(
                part_path=part_path,
                target=target,
                manifest_path=manifest_path,
                url=url,
                remote=remote,
                chunk_size=settings.download_chunk_size,
            )
            return DownloadResult(
                target_path=target,
                manifest_path=manifest_path,
                manifest=manifest,
                skipped=False,
            )
        raise IncompleteDownloadError(
            f"Server rejected range resume at offset {starting_offset} for {url}"
        )

    if status == 206:
        _write_stream_to_part(body, part_path, mode="ab" if starting_offset > 0 else "wb")
    elif status == 200:
        if starting_offset > 0 and resume:
            logger.warning("Server ignored Range request; restarting download from the beginning")
            part_path.unlink(missing_ok=True)
        _write_stream_to_part(body, part_path, mode="wb")
    else:
        raise OSError(f"Download failed: HTTP status {status} for {url}")

    manifest = _finalise_part(
        part_path=part_path,
        target=target,
        manifest_path=manifest_path,
        url=url,
        remote=remote,
        chunk_size=settings.download_chunk_size,
    )
    return DownloadResult(
        target_path=target,
        manifest_path=manifest_path,
        manifest=manifest,
        skipped=False,
    )
