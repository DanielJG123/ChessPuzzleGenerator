"""Stage 3 chess analysis tiers (cheap, deep, variation, final-collision)."""

from __future__ import annotations

from chess_chain.analysis.cache import DiskKeyValueCache
from chess_chain.analysis.cheap import analyze_cheap_puzzle, stream_cheap_analysis
from chess_chain.analysis.collision import (
    CollisionRequest,
    CollisionResult,
    CollisionStatus,
    FinalOccupancyCollisionAnalyzer,
)
from chess_chain.analysis.deep import analyze_deep_puzzle, estimate_engine_calls
from chess_chain.analysis.engine import (
    EngineAdapter,
    EngineAnalysisResult,
    EngineConfig,
    EngineLike,
    EngineMoveInfo,
    EngineUnavailableError,
    engine_fen,
)
from chess_chain.analysis.forcedness import (
    ForcednessConfig,
    classify_forcedness,
    forcedness_config_from_settings,
)
from chess_chain.analysis.policies import (
    CHECKMATE_POLICY,
    EXPLICIT_PREDICATE_QUEEN_CAPTURED,
    FIXED_PLIES_POLICY_4,
    STAGE_END_POLICIES,
    TRANSITION_READY_POLICY,
    StageEndPolicy,
    evaluate_predicate,
    get_policy,
)
from chess_chain.analysis.summarize import summarize_puzzle
from chess_chain.analysis.variations import (
    TerminalVariationSummary,
    analyze_variations,
    estimate_scope_along_reference,
)
from chess_chain.domain.analysis import ForcednessCategory, PlyAnalysis

__all__ = [
    "CHECKMATE_POLICY",
    "CollisionRequest",
    "CollisionResult",
    "CollisionStatus",
    "DiskKeyValueCache",
    "EXPLICIT_PREDICATE_QUEEN_CAPTURED",
    "EngineAdapter",
    "EngineAnalysisResult",
    "EngineConfig",
    "EngineLike",
    "EngineMoveInfo",
    "EngineUnavailableError",
    "FIXED_PLIES_POLICY_4",
    "FinalOccupancyCollisionAnalyzer",
    "ForcednessCategory",
    "ForcednessConfig",
    "PlyAnalysis",
    "STAGE_END_POLICIES",
    "StageEndPolicy",
    "TRANSITION_READY_POLICY",
    "TerminalVariationSummary",
    "analyze_cheap_puzzle",
    "analyze_deep_puzzle",
    "analyze_variations",
    "classify_forcedness",
    "engine_fen",
    "estimate_engine_calls",
    "estimate_scope_along_reference",
    "evaluate_predicate",
    "forcedness_config_from_settings",
    "get_policy",
    "stream_cheap_analysis",
    "summarize_puzzle",
]
