"""Stable sequence fingerprints (stage 6).

The fingerprint covers the request plus every stage's source puzzle id, the
applied symmetry, the accepted lines and the transition ASTs, so two sequences
generated from the same request are comparable without re-execution.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["sequence_fingerprint", "stage_fingerprint"]


def stage_fingerprint(
    puzzle_id: str,
    *,
    symmetry: str | None,
    accepted_lines: tuple[tuple[str, ...], ...],
    transition_ast: dict[str, Any] | None,
) -> str:
    payload = {
        "puzzle_id": puzzle_id,
        "symmetry": symmetry,
        "accepted_lines": [list(line) for line in accepted_lines],
        "transition_ast": transition_ast,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sequence_fingerprint(
    request_fingerprint_key: str,
    stages: tuple[dict[str, Any], ...],
) -> str:
    """Fingerprint over the request fingerprint and every stage summary."""
    payload = {
        "request": request_fingerprint_key,
        "stages": list(stages),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
