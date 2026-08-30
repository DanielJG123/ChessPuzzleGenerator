"""Stage 6: complete puzzle sequence generation.

Puzzles are nodes and synthesised transitions are directed edges, retrieved on
demand (no global all-pairs graph). The search is a bounded, reproducible beam
search with backtracking behind a ``SequenceSearcher`` protocol; the final
stage is selected lazily with collision analysis run only for ranked finalists
against the persistent final-collision cache. Outputs are the canonical
``sequence@1`` document, a machine validation report, a search report, a
player-facing representation and private manufacturing/QA data.
"""

from __future__ import annotations

from chess_chain.sequence.finalise import FinalistRank, FinalSelection, select_final
from chess_chain.sequence.generate import (
    SequenceGenerationError,
    generate_sequence,
)
from chess_chain.sequence.graph import EdgeAttempt, EdgeCache, NeighbourGraph
from chess_chain.sequence.rating import (
    DifficultyEstimate,
    MissingRatingError,
    RatingEstimator,
    SimpleAverageRatingEstimator,
)
from chess_chain.sequence.records import (
    SCHEMA as SEQUENCE_SCHEMA,
)
from chess_chain.sequence.records import (
    SequenceResult,
    SequenceStageRecord,
    load_sequence,
)
from chess_chain.sequence.registry import SequenceRegistry, load_registry
from chess_chain.sequence.request import (
    SequenceRequest,
    request_fingerprint,
    request_from_dict,
)
from chess_chain.sequence.search import (
    BeamSearchSearcher,
    PathState,
    SearchReport,
    SearchStage,
    SequenceSearcher,
)
from chess_chain.sequence.validate import (
    ValidationReport,
    validate_sequence,
    validation_from_dict,
)

__all__ = [
    "BeamSearchSearcher",
    "DifficultyEstimate",
    "EdgeAttempt",
    "EdgeCache",
    "FinalSelection",
    "FinalistRank",
    "MissingRatingError",
    "NeighbourGraph",
    "PathState",
    "RatingEstimator",
    "SEQUENCE_SCHEMA",
    "SearchReport",
    "SearchStage",
    "SequenceGenerationError",
    "SequenceRegistry",
    "SequenceRequest",
    "SequenceResult",
    "SequenceSearcher",
    "SequenceStageRecord",
    "SimpleAverageRatingEstimator",
    "ValidationReport",
    "generate_sequence",
    "load_registry",
    "load_sequence",
    "request_fingerprint",
    "request_from_dict",
    "select_final",
    "validate_sequence",
    "validation_from_dict",
]
