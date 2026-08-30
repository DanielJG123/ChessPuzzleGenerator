"""Versioned sequence-generation request (stage 6).

Every field participates in the request fingerprint so deterministic
regeneration from the same seed and configuration is guaranteed and cached
edge attempts key on the complete configuration.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = ["SequenceRequest", "request_fingerprint", "request_from_dict"]

#: Accepted values for the staged request knobs.
SOLVER_UNIQUENESS_OPTIONS = ("strict", "lenient")
OPPONENT_FORCEDNESS_OPTIONS = ("required", "preferred", "any")
EXACT_JOURNEY_OPTIONS = ("required", "preferred", "optional")
INVENTORY_PROFILES = ("standard", "any")
ACCEPTANCE_OPTIONS = ("intended_only", "all_legal", "plausible")


@dataclass(frozen=True)
class SequenceRequest:
    """The full, versioned request for one generated sequence."""

    version: int = 1
    product_id: str = "seq-0000"
    num_stages: int = 3

    # Difficulty.
    rating_min: int | None = None
    rating_max: int | None = None
    target_rating: int | None = None
    rating_spread: int = 100

    # Quality gates.
    min_popularity: int = 0
    min_nb_plays: int = 0
    max_rating_deviation: int = 200

    # Themes.
    allowed_themes: tuple[str, ...] = ()
    forbidden_themes: tuple[str, ...] = ()

    # Solver / opponent strength.
    solver_uniqueness: str = "lenient"  # strict | lenient
    opponent_forcedness: str = "preferred"  # required | preferred | any

    # Intermediate stage-end policies.
    allowed_intermediate_policies: tuple[str, ...] = ("transition_ready",)
    preferred_intermediate_policies: tuple[str, ...] = ("transition_ready",)

    # Final-only occupancy-collision requirement.
    final_collision_status: str = "collision_free_within_scope"
    collision_scope: str = "up_to_plies"
    collision_max_depth: int = 12
    collision_max_nodes: int = 100_000
    collision_strict: bool = True  # inconclusive is rejected when True

    # Transition budget.
    max_transition_actions: int = 4
    max_leakage_bits: float = 64.0
    max_absolute_square_disclosures: int = 8

    # Physical inventory.
    inventory_profile: str = "standard"  # standard | any

    # Journey preference.
    exact_journey: str = "preferred"  # required | preferred | optional
    final_checkmate: bool = True

    # Accepted-line enumeration for intermediate and final stages.
    acceptance: str = "all_legal"  # intended_only | all_legal | plausible
    max_plies: int = 2
    max_nodes: int = 2000
    max_accepted_lines: int = 64

    # Search knobs.
    random_seed: int = 0
    beam_width: int = 8
    per_node_candidates: int = 6
    backtrack_budget: int = 40
    validated_finalist_alternatives: int = 3
    max_finalists_checked: int = 12

    # Exclusions.
    start_puzzle_id: str | None = None
    excluded_puzzle_ids: tuple[str, ...] = ()
    already_used_puzzle_ids: tuple[str, ...] = ()
    forbid_puzzle_reuse: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("request version must be >= 1")
        if self.num_stages < 2:
            raise ValueError("a sequence needs at least 2 stages")
        if self.solver_uniqueness not in SOLVER_UNIQUENESS_OPTIONS:
            raise ValueError(f"solver_uniqueness must be one of {SOLVER_UNIQUENESS_OPTIONS}")
        if self.opponent_forcedness not in OPPONENT_FORCEDNESS_OPTIONS:
            raise ValueError(f"opponent_forcedness must be one of {OPPONENT_FORCEDNESS_OPTIONS}")
        if self.exact_journey not in EXACT_JOURNEY_OPTIONS:
            raise ValueError(f"exact_journey must be one of {EXACT_JOURNEY_OPTIONS}")
        if self.inventory_profile not in INVENTORY_PROFILES:
            raise ValueError(f"inventory_profile must be one of {INVENTORY_PROFILES}")
        if self.acceptance not in ACCEPTANCE_OPTIONS:
            raise ValueError(f"acceptance must be one of {ACCEPTANCE_OPTIONS}")
        if not self.allowed_intermediate_policies:
            raise ValueError("at least one intermediate stage-end policy is required")


def request_fingerprint(request: SequenceRequest) -> str:
    """Stable fingerprint over the complete request (seed included)."""
    canonical = json.dumps(request.to_dict(), sort_keys=True, separators=(",", ":"))
    import hashlib

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def request_from_dict(data: dict[str, Any]) -> SequenceRequest:
    """Rebuild a request from a JSON document (tuples round-tripped as lists)."""
    allowed = set(SequenceRequest.__dataclass_fields__)
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in allowed:
            continue
        field_def = SequenceRequest.__dataclass_fields__[key]
        if isinstance(value, list) and field_def.type.startswith("tuple"):
            value = tuple(value)
        kwargs[key] = value
    return SequenceRequest(**kwargs)


def request_from_yaml(path: Path) -> SequenceRequest:
    """Load a ``SequenceRequest`` from a YAML request file (top-level keys)."""
    import yaml

    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return request_from_dict(dict(data))
