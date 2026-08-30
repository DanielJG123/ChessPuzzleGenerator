"""Similarity retrieval orchestration.

The pipeline never materialises an all-pairs comparison:

1. filter candidate rows by rating band and product-quality constraints first;
2. group or partition the remaining rows by piece count and material
   signature (after a cheap componentwise material-feasibility prefilter);
3. query nearest occupancy masks within those groups (one query vector per
   group, bounded by ``max_candidates_scanned``), with an exact occupancy-hash
   lookup for distance zero;
4. retain only a configurable top-K shortlist;
5. apply the exact per-endpoint piece-assignment costs to that shortlist and
   rank by the decomposable transition-complexity score.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import chess
import numpy as np
import pyarrow.parquet as pq

from chess_chain.analysis.policies import get_policy
from chess_chain.analysis.variations import TerminalVariationSummary
from chess_chain.config import Settings
from chess_chain.domain.pieces import COUNT_NAMES, MaterialSignature
from chess_chain.domain.position import CanonicalPosition
from chess_chain.similarity.backend import (
    ScanStats,
    SimilarityBackend,
)
from chess_chain.similarity.candidate import PairCandidate, build_candidate
from chess_chain.similarity.index import (
    SimilarityIndex,
    index_is_fresh,
    load_index,
)
from chess_chain.similarity.scoring import SimilarityScoreWeights
from chess_chain.similarity.symmetry import (
    SYMMETRIES,
    apply_occupancy_symmetry,
)

__all__ = [
    "RetrievalStats",
    "explain_pair",
    "resolve_accepted_endpoints",
    "retrieve",
    "retrieve_with_index",
]


@dataclass(frozen=True)
class RetrievalStats:
    """Accounting for one retrieval run."""

    index_rows: int
    filtered_rows: int
    shortlisted_ids: int
    scored_candidates: int
    scan: ScanStats
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_rows": self.index_rows,
            "filtered_rows": self.filtered_rows,
            "shortlisted_ids": self.shortlisted_ids,
            "scored_candidates": self.scored_candidates,
            "scan": self.scan.to_dict(),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }


def resolve_accepted_endpoints(
    settings: Settings,
    puzzle_row: dict[str, Any],
    end_policy_name: str,
) -> tuple[tuple[CanonicalPosition, ...], str]:
    """Return the accepted endpoints for a puzzle under a stage-end policy.

    Uses the persisted terminal-variation analysis for the exact policy when
    available; otherwise falls back to the puzzle's principal-line final
    position (a single accepted endpoint). Endpoints are reconstructed and
    re-validated with python-chess.
    """
    policy = get_policy(end_policy_name)
    puzzle_id = str(puzzle_row["puzzle_id"])
    variation_path = settings.analysis_dir / "variations" / f"{puzzle_id}_{policy.name}.parquet"
    if variation_path.is_file():
        summary = TerminalVariationSummary.from_dict(pq.read_table(variation_path).to_pylist()[0])
        endpoints = tuple(
            CanonicalPosition.from_board(chess.Board(fen)) for fen in summary.terminal_fens
        )
        return endpoints, f"variation_analysis:{policy.key()}"
    final_fen = str(puzzle_row["final_fen"])
    endpoint = CanonicalPosition.from_board(chess.Board(final_fen))
    return (endpoint,), f"final_position:{policy.key()}"


def retrieve_with_index(
    *,
    settings: Settings,
    index: SimilarityIndex,
    source_puzzle_id: str,
    source_row: dict[str, Any],
    end_policy_name: str,
    backend: SimilarityBackend,
    score_weights: SimilarityScoreWeights | None = None,
    cost_weights: Any | None = None,
    symmetries: tuple[str, ...] = SYMMETRIES,
    rating_min: int | None = None,
    rating_max: int | None = None,
    max_rating_deviation: int | None = None,
    min_popularity: int = 0,
    piece_count_tolerance: int = 4,
    max_candidates_scanned: int = 200_000,
    shortlist_multiplier: int = 4,
    top_k: int = 50,
    reject_non_visible_rights: bool = False,
    exclude_unusual_material: bool = False,
) -> tuple[list[PairCandidate], RetrievalStats]:
    """Retrieve and rank candidate next puzzles for ``source_puzzle_id``."""
    started = time.monotonic()
    endpoints, endpoint_source = resolve_accepted_endpoints(settings, source_row, end_policy_name)
    policy = get_policy(end_policy_name)
    source_start_counts = tuple(int(source_row[f"start_{name}"]) for name in COUNT_NAMES)

    candidates, stats = retrieve(
        source_puzzle_id=source_puzzle_id,
        source_start_counts=source_start_counts,
        endpoints=endpoints,
        index=index,
        backend=backend,
        end_policy_key=policy.key(),
        endpoint_source=endpoint_source,
        score_weights=score_weights,
        cost_weights=cost_weights,
        symmetries=symmetries,
        rating_min=rating_min,
        rating_max=rating_max,
        max_rating_deviation=max_rating_deviation,
        min_popularity=min_popularity,
        piece_count_tolerance=piece_count_tolerance,
        max_candidates_scanned=max_candidates_scanned,
        shortlist_multiplier=shortlist_multiplier,
        top_k=top_k,
        reject_non_visible_rights=reject_non_visible_rights,
        exclude_unusual_material=exclude_unusual_material,
    )
    elapsed = time.monotonic() - started
    stats = RetrievalStats(
        index_rows=stats.index_rows,
        filtered_rows=stats.filtered_rows,
        shortlisted_ids=stats.shortlisted_ids,
        scored_candidates=stats.scored_candidates,
        scan=stats.scan,
        elapsed_seconds=elapsed,
    )
    return candidates, stats


def retrieve(
    *,
    source_puzzle_id: str,
    source_start_counts: tuple[int, ...],
    endpoints: tuple[CanonicalPosition, ...],
    index: SimilarityIndex,
    backend: SimilarityBackend,
    end_policy_key: str,
    endpoint_source: str,
    score_weights: SimilarityScoreWeights | None = None,
    cost_weights: Any | None = None,
    symmetries: tuple[str, ...] = SYMMETRIES,
    rating_min: int | None = None,
    rating_max: int | None = None,
    max_rating_deviation: int | None = None,
    min_popularity: int = 0,
    piece_count_tolerance: int = 4,
    max_candidates_scanned: int = 200_000,
    shortlist_multiplier: int = 4,
    top_k: int = 50,
    reject_non_visible_rights: bool = False,
    exclude_unusual_material: bool = False,
) -> tuple[list[PairCandidate], RetrievalStats]:
    """Core retrieval over an already-loaded index (see module docstring)."""
    rows = index.rows
    if not rows:
        empty_stats = RetrievalStats(
            index_rows=0,
            filtered_rows=0,
            shortlisted_ids=0,
            scored_candidates=0,
            scan=ScanStats(),
            elapsed_seconds=0.0,
        )
        return [], empty_stats

    ratings = np.asarray([row.rating for row in rows], dtype=np.int64)
    deviations = np.asarray([row.rating_deviation for row in rows], dtype=np.int64)
    popularities = np.asarray([row.popularity for row in rows], dtype=np.int64)
    piece_counts = np.asarray([row.presented_piece_count for row in rows], dtype=np.int64)
    start_counts_matrix = np.asarray([list(row.start_counts) for row in rows], dtype=np.int64)
    start_occupancies = np.asarray([row.start_occupancy for row in rows], dtype=np.uint64)
    puzzle_ids = np.asarray([row.puzzle_id for row in rows], dtype=object)

    selected = np.ones(len(rows), dtype=bool)
    if rating_min is not None:
        selected &= ratings >= rating_min
    if rating_max is not None:
        selected &= ratings <= rating_max
    if max_rating_deviation is not None:
        selected &= deviations <= max_rating_deviation
    selected &= popularities >= min_popularity
    selected &= puzzle_ids != source_puzzle_id
    if exclude_unusual_material:
        selected &= np.asarray(
            [not row.unusual_material and not row.duplicate_queens for row in rows],
            dtype=bool,
        )

    available = _union_available(source_start_counts, endpoints)
    available_row = np.asarray(list(available), dtype=np.int64)
    feasible = (start_counts_matrix <= available_row[None, :]).all(axis=1)

    available_count = int(np.asarray(list(available), dtype=np.int64).sum())
    count_window = (piece_counts >= available_count - piece_count_tolerance) & (
        piece_counts <= available_count + piece_count_tolerance
    )

    selected = selected & feasible & count_window
    filtered_rows = int(selected.sum())

    # --- Group the selected rows by piece count + material signature once ---
    group_map: dict[tuple[int, str], list[int]] = {}
    selected_indices = [int(i) for i in np.flatnonzero(selected)]
    for row_index in selected_indices:
        row = rows[row_index]
        key = (row.presented_piece_count, _signature_key(row.start_counts))
        group_map.setdefault(key, []).append(row_index)

    best_distance: dict[int, int] = {}
    merged_scan = ScanStats()
    query_masks = sorted({int(endpoint.fingerprint.occupied) for endpoint in endpoints})
    for query_mask in query_masks:
        for symmetry in symmetries:
            transformed_query = apply_occupancy_symmetry(query_mask, symmetry)

            # Distance-zero exact lookup via the occupancy hash map.
            exact_rows = [i for i in index.occupancy_map.get(transformed_query, ()) if selected[i]]
            for row_index in exact_rows:
                best_distance[row_index] = min(best_distance.get(row_index, 1 << 30), 0)
            merged_scan = ScanStats(
                groups_scanned=merged_scan.groups_scanned,
                rows_scanned=merged_scan.rows_scanned,
                pairs_materialised=merged_scan.pairs_materialised,
                exact_lookup_hits=merged_scan.exact_lookup_hits + len(exact_rows),
                truncated=merged_scan.truncated,
            )

            # Bounded nearest-occupancy query within the groups. The scan
            # budget is a hard cap: groups are truncated so the total rows
            # scanned per query never exceeds ``max_candidates_scanned``.
            groups = _build_groups(group_map, start_occupancies)
            total_rows_in_groups = sum(int(len(masks)) for _, masks in groups)
            rows_before = merged_scan.rows_scanned
            remaining = max(0, max_candidates_scanned - merged_scan.rows_scanned)
            truncated = False
            for row_indices, masks in groups:
                if remaining <= 0:
                    truncated = True
                    break
                take = min(int(len(masks)), remaining)
                if take <= 0:
                    truncated = True
                    break
                result = backend.nearest(
                    int(transformed_query),
                    [(row_indices[:take], masks[:take])],
                    top_k=max(1, top_k),
                )
                remaining -= take
                merged_scan = ScanStats(
                    groups_scanned=merged_scan.groups_scanned + result.stats.groups_scanned,
                    rows_scanned=merged_scan.rows_scanned + result.stats.rows_scanned,
                    pairs_materialised=(
                        merged_scan.pairs_materialised + result.stats.pairs_materialised
                    ),
                    exact_lookup_hits=merged_scan.exact_lookup_hits,
                    truncated=merged_scan.truncated or truncated,
                )
                for row_index, distance in result.hits:
                    current = best_distance.get(int(row_index))
                    if current is None or distance < current:
                        best_distance[int(row_index)] = distance
            if merged_scan.rows_scanned - rows_before < total_rows_in_groups:
                truncated = True
            merged_scan = ScanStats(
                groups_scanned=merged_scan.groups_scanned,
                rows_scanned=merged_scan.rows_scanned,
                pairs_materialised=merged_scan.pairs_materialised,
                exact_lookup_hits=merged_scan.exact_lookup_hits,
                truncated=merged_scan.truncated or truncated,
            )

    shortlist_k = max(top_k, min(len(best_distance), top_k * shortlist_multiplier))
    shortlisted = sorted(best_distance.items(), key=lambda item: (item[1], item[0]))[:shortlist_k]

    candidates: list[PairCandidate] = []
    for row_index, _ in shortlisted:
        row = rows[row_index]
        candidate = build_candidate(
            source_puzzle_id=source_puzzle_id,
            next_puzzle_id=row.puzzle_id,
            source_stage_end_policy=end_policy_key,
            endpoint_source=endpoint_source,
            endpoints=endpoints,
            source_start_counts=source_start_counts,
            next_start=CanonicalPosition.from_board(row.board()),
            symmetries=symmetries,
            cost_weights=cost_weights,
            score_weights=score_weights,
            reject_non_visible_rights=reject_non_visible_rights,
        )
        if candidate.transition_complexity_score == float("inf"):
            continue
        candidates.append(candidate)

    candidates.sort(key=lambda c: (c.transition_complexity_score, c.next_puzzle_id))
    top_candidates = candidates[:top_k]
    stats = RetrievalStats(
        index_rows=len(rows),
        filtered_rows=filtered_rows,
        shortlisted_ids=len(shortlisted),
        scored_candidates=len(candidates),
        scan=merged_scan,
        elapsed_seconds=0.0,
    )
    return top_candidates, stats


def explain_pair(
    *,
    source_puzzle_id: str,
    source_start_counts: tuple[int, ...],
    endpoints: tuple[CanonicalPosition, ...],
    index: SimilarityIndex,
    next_puzzle_id: str,
    end_policy_key: str,
    endpoint_source: str,
    cost_weights: Any | None = None,
    score_weights: SimilarityScoreWeights | None = None,
    symmetries: tuple[str, ...] = SYMMETRIES,
    reject_non_visible_rights: bool = False,
) -> PairCandidate | None:
    """Score the exact pair (source -> next) with full per-endpoint detail.

    Returns None when ``next_puzzle_id`` is not present in the index or when no
    symmetry produces a legal reconstructed position.
    """
    row_index = index.row_index.get(next_puzzle_id)
    if row_index is None:
        return None
    row = index.rows[row_index]
    return build_candidate(
        source_puzzle_id=source_puzzle_id,
        next_puzzle_id=next_puzzle_id,
        source_stage_end_policy=end_policy_key,
        endpoint_source=endpoint_source,
        endpoints=endpoints,
        source_start_counts=source_start_counts,
        next_start=CanonicalPosition.from_board(row.board()),
        symmetries=symmetries,
        cost_weights=cost_weights,
        score_weights=score_weights,
        reject_non_visible_rights=reject_non_visible_rights,
    )


def load_or_build_index(
    settings: Settings,
    repo: Any,
    *,
    version: int,
    where: str = "",
) -> SimilarityIndex:
    """Load the persisted index when fresh, otherwise rebuild and persist it.

    Stale detection uses the version fingerprint (similarity version, dataset
    transform version, source fingerprint, row count); a stale index is
    transparently rebuilt and reported.
    """
    from chess_chain.similarity.index import build_index, save_index

    directory = settings.similarity_dir
    candidate_index = build_index(repo, version=version, where=where)
    if directory.is_dir():
        persisted = load_index(directory)
        if persisted is not None and index_is_fresh(directory, candidate_index):
            return persisted
    save_index(directory, candidate_index)
    return candidate_index


def _union_available(
    source_start_counts: tuple[int, ...],
    endpoints: tuple[CanonicalPosition, ...],
) -> tuple[int, ...]:
    """The union (per type) of the physical pieces available at any accepted
    endpoint: for each endpoint that is ``max(start_counts, endpoint_counts)``;
    the union is the element-wise maximum over all endpoints."""
    available = list(source_start_counts)
    for endpoint in endpoints:
        endpoint_counts = MaterialSignature.from_board(endpoint.board).counts
        for index in range(len(source_start_counts)):
            available[index] = max(
                available[index],
                max(source_start_counts[index], endpoint_counts[index]),
            )
    return tuple(available)


def _signature_key(counts: tuple[int, ...]) -> str:
    return ",".join(str(count) for count in counts)


def _build_groups(
    group_map: dict[tuple[int, str], list[int]],
    start_occupancies: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convert the grouped row indices into (row_indices, masks) pairs for the
    backend. Each group is compared against the single query vector, never an
    N x N matrix."""
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    for row_indices in group_map.values():
        indices = np.asarray(row_indices, dtype=np.int64)
        masks = start_occupancies[indices]
        groups.append((indices, masks))
    return groups
