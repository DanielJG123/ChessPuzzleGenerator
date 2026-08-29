"""Domain-level errors for the puzzle ingestion pipeline."""

from __future__ import annotations


class IngestRejectError(ValueError):
    """Raised when a source row cannot be processed.

    The streaming pipeline catches this error and writes the row to the
    quarantine dataset instead of aborting the run.
    """
