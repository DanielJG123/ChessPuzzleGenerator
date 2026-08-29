"""Dataset ingestion: downloader, streamed sampling and the Parquet pipeline."""

from __future__ import annotations

from chess_chain.domain.source import PUZZLE_HEADER
from chess_chain.ingest.download import (
    DownloadResult,
    HttpClient,
    RemoteResourceInfo,
    UrllibHttpClient,
    download_file,
    ensure_sufficient_space,
    inspect_remote,
)
from chess_chain.ingest.pipeline import DEFAULT_DATASET_NAME, ingest_puzzles
from chess_chain.ingest.reader import iter_csv_fields, open_text_stream
from chess_chain.ingest.records import IngestSummary, core_to_record, source_fingerprint
from chess_chain.ingest.sample import stream_sample_rows

__all__ = [
    "DEFAULT_DATASET_NAME",
    "DownloadResult",
    "HttpClient",
    "IngestSummary",
    "PUZZLE_HEADER",
    "RemoteResourceInfo",
    "UrllibHttpClient",
    "core_to_record",
    "download_file",
    "ensure_sufficient_space",
    "ingest_puzzles",
    "inspect_remote",
    "iter_csv_fields",
    "open_text_stream",
    "source_fingerprint",
    "stream_sample_rows",
]
