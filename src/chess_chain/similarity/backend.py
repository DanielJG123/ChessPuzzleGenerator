"""Nearest-occupancy retrieval backends behind a ``SimilarityBackend`` protocol.

Backends compute, for a query occupancy mask, the per-group occupancy Hamming
distances ``popcount(query XOR candidate_mask)`` without materialising an
all-pairs matrix: each group is compared against the single query vector.

- ``ExactChunkedBackend``: a pure NumPy vectorised ``bitwise_count`` kernel.
- ``ScipyAcceleratedBackend``: the same distances through SciPy's compiled
  ``cdist`` ``hamming`` metric over unpacked bits. SciPy is an existing,
  cross-platform dependency, so no awkward native package (e.g. FAISS) is ever
  required. Both backends return identical results; the accelerated one is
  selected via configuration and stays behind the protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.spatial.distance import cdist

__all__ = [
    "ExactChunkedBackend",
    "NearestResult",
    "ScipyAcceleratedBackend",
    "ScanStats",
    "SimilarityBackend",
    "get_backend",
]

_BACKEND_CHUNK = 4096


@dataclass(frozen=True)
class ScanStats:
    """Bounded-scan accounting proving no all-pairs materialisation occurred."""

    groups_scanned: int = 0
    rows_scanned: int = 0
    pairs_materialised: int = 0
    exact_lookup_hits: int = 0
    truncated: bool = False

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "groups_scanned": self.groups_scanned,
            "rows_scanned": self.rows_scanned,
            "pairs_materialised": self.pairs_materialised,
            "exact_lookup_hits": self.exact_lookup_hits,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class NearestResult:
    """Ranked ``(row_index, d_occ)`` hits and the scan accounting."""

    hits: tuple[tuple[int, int], ...]
    stats: ScanStats


@runtime_checkable
class SimilarityBackend(Protocol):
    """Distance kernel used by the retriever. ``groups`` is a sequence of
    ``(row_indices, masks)`` pairs; every mask is a uint64 occupancy bitboard."""

    name: str

    def nearest(
        self,
        query_mask: int,
        groups: Sequence[tuple[np.ndarray, np.ndarray]],
        *,
        top_k: int,
    ) -> NearestResult:
        """Return the top ``top_k`` (row_index, d_occ) across all groups."""
        ...


class ExactChunkedBackend:
    """NumPy vectorised ``bitwise_count`` baseline (no extra memory)."""

    name = "exact"

    def nearest(
        self,
        query_mask: int,
        groups: Sequence[tuple[np.ndarray, np.ndarray]],
        *,
        top_k: int,
    ) -> NearestResult:
        query = np.uint64(query_mask)
        hits: list[tuple[int, int]] = []
        stats = ScanStats()
        for row_indices, masks in groups:
            masks = np.asarray(masks, dtype=np.uint64)
            distances = np.bitwise_count(query ^ masks)
            stats = ScanStats(
                groups_scanned=stats.groups_scanned + 1,
                rows_scanned=stats.rows_scanned + int(len(masks)),
                pairs_materialised=stats.pairs_materialised + int(len(masks)),
                exact_lookup_hits=stats.exact_lookup_hits,
                truncated=stats.truncated,
            )
            for index, distance in enumerate(distances.tolist()):
                hits.append((int(row_indices[index]), int(distance)))
        hits.sort(key=lambda item: (item[1], item[0]))
        return NearestResult(hits=tuple(hits[:top_k]), stats=stats)


class ScipyAcceleratedBackend:
    """Optional compiled-kernel backend using SciPy's ``cdist`` hamming metric.

    The 64-bit occupancy mask is unpacked to 64 bit-columns per row and each
    group is chunked so the working set stays small. ``hamming * 64`` equals
    ``popcount(query XOR mask)`` exactly.
    """

    name = "accelerated"

    def nearest(
        self,
        query_mask: int,
        groups: Sequence[tuple[np.ndarray, np.ndarray]],
        *,
        top_k: int,
    ) -> NearestResult:
        query_bits = _unpack_bits(np.asarray([np.uint64(query_mask)], dtype=np.uint64))
        hits: list[tuple[int, int]] = []
        stats = ScanStats()
        for row_indices, masks in groups:
            masks = np.asarray(masks, dtype=np.uint64)
            bits = _unpack_bits(masks)
            distances: list[int] = []
            for start in range(0, len(bits), _BACKEND_CHUNK):
                chunk = bits[start : start + _BACKEND_CHUNK]
                hamming = cdist(query_bits, chunk, metric="hamming")
                distances.extend(
                    int(round(float(distance) * 64)) for distance in hamming[0].tolist()
                )
            stats = ScanStats(
                groups_scanned=stats.groups_scanned + 1,
                rows_scanned=stats.rows_scanned + int(len(masks)),
                pairs_materialised=stats.pairs_materialised + int(len(masks)),
                exact_lookup_hits=stats.exact_lookup_hits,
                truncated=stats.truncated,
            )
            for index, distance in enumerate(distances):
                hits.append((int(row_indices[index]), distance))
        hits.sort(key=lambda item: (item[1], item[0]))
        return NearestResult(hits=tuple(hits[:top_k]), stats=stats)


def _unpack_bits(masks: np.ndarray) -> np.ndarray:
    """Convert an (n,) uint64 array into an (n, 64) uint8 bit-column array."""
    bytes_view = masks.reshape(-1, 1).view(np.uint8).reshape(-1, 8)
    return np.unpackbits(bytes_view, axis=1, bitorder="little")


def get_backend(name: str) -> SimilarityBackend:
    """Return the named backend (``exact`` or ``accelerated``)."""
    if name == "exact":
        return ExactChunkedBackend()
    if name == "accelerated":
        return ScipyAcceleratedBackend()
    raise ValueError(f"unknown similarity backend {name!r}; expected 'exact' or 'accelerated'")
