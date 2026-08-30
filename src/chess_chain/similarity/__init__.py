"""Stage 4 puzzle similarity organiser.

Retrieves promising puzzle pairs (i, j) where the solved endpoints of puzzle i
can be transformed into the presented start of puzzle j with short,
low-leakage physical instructions. Natural-language transitions and full
sequences are *not* produced here: this package ranks candidate pairs and
explains their structural costs (layered distances, minimum-cost physical piece
assignment, symmetry handling and non-visible-rights annotations).
"""

from __future__ import annotations

from chess_chain.similarity.backend import (
    ExactChunkedBackend,
    NearestResult,
    ScanStats,
    ScipyAcceleratedBackend,
    SimilarityBackend,
    get_backend,
)
from chess_chain.similarity.candidate import PairCandidate, build_candidate
from chess_chain.similarity.distance import (
    EndpointComparison,
    available_counts,
    compare_endpoint,
)
from chess_chain.similarity.index import (
    IndexedPuzzle,
    SimilarityIndex,
    build_index,
    load_index,
    save_index,
)
from chess_chain.similarity.retrieval import (
    RetrievalStats,
    explain_pair,
    load_or_build_index,
    resolve_accepted_endpoints,
    retrieve,
    retrieve_with_index,
)
from chess_chain.similarity.scoring import (
    DEFAULT_SCORE_WEIGHTS,
    SimilarityScoreWeights,
    compute_candidate_score,
)
from chess_chain.similarity.symmetry import (
    SYMMETRIES,
    SYMMETRY_FILE_REFLECTION,
    SYMMETRY_IDENTITY,
    SYMMETRY_RANK_REFLECTION,
    SYMMETRY_ROT180,
    apply_occupancy_symmetry,
    apply_symmetry,
)

__all__ = [
    "DEFAULT_SCORE_WEIGHTS",
    "EndpointComparison",
    "ExactChunkedBackend",
    "IndexedPuzzle",
    "NearestResult",
    "PairCandidate",
    "RetrievalStats",
    "SYMMETRIES",
    "SYMMETRY_FILE_REFLECTION",
    "SYMMETRY_IDENTITY",
    "SYMMETRY_RANK_REFLECTION",
    "SYMMETRY_ROT180",
    "ScipyAcceleratedBackend",
    "ScanStats",
    "SimilarityBackend",
    "SimilarityIndex",
    "SimilarityScoreWeights",
    "apply_occupancy_symmetry",
    "apply_symmetry",
    "available_counts",
    "build_candidate",
    "build_index",
    "compare_endpoint",
    "compute_candidate_score",
    "explain_pair",
    "get_backend",
    "load_index",
    "load_or_build_index",
    "resolve_accepted_endpoints",
    "retrieve",
    "retrieve_with_index",
    "save_index",
]
