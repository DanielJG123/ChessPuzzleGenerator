"""Tests for similarity backends (exact and accelerated parity, bounded scans)."""

from __future__ import annotations

import numpy as np
import pytest

from chess_chain.similarity.backend import (
    ExactChunkedBackend,
    ScanStats,
    ScipyAcceleratedBackend,
    get_backend,
)


def _groups(rows: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    masks = rng.integers(0, 1 << 60, size=rows, dtype=np.uint64)
    indices = np.arange(rows, dtype=np.int64)
    return indices, masks


def test_exact_backend_known_distances() -> None:
    backend = ExactChunkedBackend()
    # Two masks differing in exactly one bit.
    masks = np.asarray([0b0001, 0b0011, 0b1111], dtype=np.uint64)
    indices = np.asarray([0, 1, 2], dtype=np.int64)
    result = backend.nearest(0b0001, [(indices, masks)], top_k=3)
    assert result.hits == ((0, 0), (1, 1), (2, 3))


def test_backends_return_compatible_top_results_on_fixtures() -> None:
    indices, masks = _groups(5000, seed=42)
    query = int(masks[17])
    exact = ExactChunkedBackend().nearest(query, [(indices, masks)], top_k=10)
    accelerated = ScipyAcceleratedBackend().nearest(query, [(indices, masks)], top_k=10)
    assert exact.hits == accelerated.hits
    assert [distance for _, distance in exact.hits] == sorted(
        distance for _, distance in exact.hits
    )


def test_backends_identical_distance_for_all_masks() -> None:
    indices, masks = _groups(200, seed=3)
    for query_index in (0, 51, 199):
        query = int(masks[query_index])
        exact = ExactChunkedBackend().nearest(query, [(indices, masks)], top_k=200)
        accelerated = ScipyAcceleratedBackend().nearest(query, [(indices, masks)], top_k=200)
        assert exact.hits == accelerated.hits
        for row, distance in exact.hits:
            expected = int(np.bitwise_count(np.uint64(query) ^ masks[row]))
            assert distance == expected


def test_accelerated_backend_matches_brute_popcount() -> None:
    rng = np.random.default_rng(11)
    masks = rng.integers(0, 1 << 62, size=500, dtype=np.uint64)
    indices = np.arange(500, dtype=np.int64)
    query = int(masks[123])
    result = ScipyAcceleratedBackend().nearest(query, [(indices, masks)], top_k=500)
    for row, distance in result.hits:
        expected = int(np.bitwise_count(np.uint64(query) ^ masks[row]))
        assert distance == expected


def test_get_backend_names_and_unknown() -> None:
    assert get_backend("exact").name == "exact"
    assert get_backend("accelerated").name == "accelerated"
    with pytest.raises(ValueError):
        get_backend("faiss")


def test_scan_stats_accounting() -> None:
    backend = ExactChunkedBackend()
    group_a = _groups(10, seed=1)
    group_b = _groups(20, seed=2)
    result = backend.nearest(0, [group_a, group_b], top_k=5)
    assert result.stats.groups_scanned == 2
    assert result.stats.rows_scanned == 30
    assert result.stats.pairs_materialised == 30  # 30 x 1, never 30 x 30
    assert result.stats.exact_lookup_hits == 0
    assert result.stats.truncated is False


def test_empty_groups_yield_no_hits() -> None:
    backend = ExactChunkedBackend()
    result = backend.nearest(0, [], top_k=5)
    assert result.hits == ()
    assert result.stats == ScanStats()
