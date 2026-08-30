"""Stage 5: deterministic transition language and synthesiser.

A typed transition AST is executed and verified by the executor (the only
semantic authority), rendered deterministically from templates, and persisted
with machine-verifiable before/after FENs. The stage-boundary guard is a
separate typed object. No LLM ever invents executable transition instructions.
"""

from __future__ import annotations

from chess_chain.transitions.config import TransitionConfig
from chess_chain.transitions.context import (
    EndpointContext,
    build_context_from_boards,
    build_context_from_core,
    build_context_from_row,
    build_prefix_contexts,
    replay_solution,
)
from chess_chain.transitions.executor import (
    STATUS_AMBIGUOUS,
    STATUS_COLLISION,
    STATUS_DUPLICATE_USE,
    STATUS_INVALID_DESTINATION,
    STATUS_INVALID_PROGRAM,
    STATUS_INVALID_TRANSFORM,
    STATUS_INVENTORY,
    STATUS_MISSING,
    STATUS_NO_BRANCH,
    STATUS_OFF_BOARD,
    STATUS_OK,
    STATUS_RIGHTS_CONFLICT,
    ProgramResult,
    execute_program,
)
from chess_chain.transitions.guards import (
    GuardDisclosures,
    GuardEvaluation,
    StageBoundaryGuard,
    evaluate_guard,
    guard_disclosures,
    verify_guard,
)
from chess_chain.transitions.leakage import (
    ClueReport,
    PlausibleEndpointSet,
    PlausibleLines,
    anchor_uncertainty_bits,
    compute_clue_report,
    conditional_leakage,
    derive_plausible_lines,
    plausible_endpoint_set,
    relative_vector_leak_bits,
    vector_origin_count,
)
from chess_chain.transitions.records import (
    SCHEMA as TRANSITION_SCHEMA,
)
from chess_chain.transitions.records import (
    TransitionArtifact,
    build_transition_document,
    load_transition_document,
    write_transition_document,
)
from chess_chain.transitions.render import render_program
from chess_chain.transitions.synth import SynthesisCandidate, synthesize
from chess_chain.transitions.verify import (
    StateOutcome,
    VerificationResult,
    generate_plausible_wrong_contexts,
    verify_program,
)

__all__ = [
    "ClueReport",
    "EndpointContext",
    "GuardDisclosures",
    "GuardEvaluation",
    "PlausibleEndpointSet",
    "PlausibleLines",
    "ProgramResult",
    "StageBoundaryGuard",
    "StateOutcome",
    "STATUS_AMBIGUOUS",
    "STATUS_COLLISION",
    "STATUS_DUPLICATE_USE",
    "STATUS_INVALID_DESTINATION",
    "STATUS_INVALID_PROGRAM",
    "STATUS_INVALID_TRANSFORM",
    "STATUS_INVENTORY",
    "STATUS_MISSING",
    "STATUS_NO_BRANCH",
    "STATUS_OFF_BOARD",
    "STATUS_OK",
    "STATUS_RIGHTS_CONFLICT",
    "SynthesisCandidate",
    "TRANSITION_SCHEMA",
    "TransitionArtifact",
    "TransitionConfig",
    "VerificationResult",
    "anchor_uncertainty_bits",
    "build_context_from_boards",
    "build_context_from_core",
    "build_context_from_row",
    "build_prefix_contexts",
    "build_transition_document",
    "compute_clue_report",
    "conditional_leakage",
    "derive_plausible_lines",
    "evaluate_guard",
    "execute_program",
    "generate_plausible_wrong_contexts",
    "guard_disclosures",
    "load_transition_document",
    "plausible_endpoint_set",
    "relative_vector_leak_bits",
    "render_program",
    "replay_solution",
    "synthesize",
    "vector_origin_count",
    "verify_guard",
    "verify_program",
    "write_transition_document",
]
