"""Named forcedness categories and their configurable thresholds."""

from __future__ import annotations

import json
from dataclasses import dataclass

from chess_chain.config import Settings
from chess_chain.domain.analysis import ForcednessCategory

__all__ = [
    "ForcednessConfig",
    "classify_forcedness",
    "forcedness_config_from_settings",
]


@dataclass(frozen=True)
class ForcednessConfig:
    """Versioned centipawn thresholds; never hard-coded in analysis code."""

    version: int
    unique_best_gap_cp: int
    practically_forced_gap_cp: int
    mate_value_cp: int
    plausible_gap_cp: int

    def fingerprint(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "unique_best_gap_cp": self.unique_best_gap_cp,
                "practically_forced_gap_cp": self.practically_forced_gap_cp,
                "mate_value_cp": self.mate_value_cp,
                "plausible_gap_cp": self.plausible_gap_cp,
            },
            sort_keys=True,
        )


def forcedness_config_from_settings(settings: Settings) -> ForcednessConfig:
    return ForcednessConfig(
        version=settings.forcedness_version,
        unique_best_gap_cp=settings.unique_best_gap_cp,
        practically_forced_gap_cp=settings.practically_forced_gap_cp,
        mate_value_cp=settings.mate_value_cp,
        plausible_gap_cp=settings.plausible_gap_cp,
    )


def classify_forcedness(
    *,
    legal_move_count: int,
    intended_rank: int | None,
    intended_gap_cp: int | None,
    second_gap_cp: int | None,
    engine_analysed: bool,
    config: ForcednessConfig,
) -> ForcednessCategory | None:
    """Classify the intended move at one ply.

    ``intended_gap_cp`` is ``best - intended`` (>= 0); ``second_gap_cp`` is
    ``best - second_best`` (>= 0). ``engine_analysed`` distinguishes "no engine
    information yet" (returns None) from "engine analysed but the intended move
    was not among the scored candidates" (returns ``invalid_source_line``).

    Rules (documented; thresholds come from ``config``):
      1. exactly one legal move               -> literal_forced
      2. not engine-analysed                  -> None (defer to deep tier)
      3. intended not scored or decisively
         worse than the best move             -> invalid_source_line
      4. intended == best and every
         alternative is decisively worse      -> practically_forced
      5. intended == best with a clear gap
         to the second-best move              -> unique_best
      6. otherwise (intended == best with a
         close second, or intended != best
         but not decisively worse)            -> ambiguous
    """
    if legal_move_count == 1:
        return ForcednessCategory.LITERAL_FORCED
    if not engine_analysed:
        return None
    if intended_rank is None or intended_gap_cp is None:
        return ForcednessCategory.INVALID_SOURCE_LINE
    if intended_gap_cp > config.practically_forced_gap_cp:
        return ForcednessCategory.INVALID_SOURCE_LINE
    if intended_gap_cp <= config.unique_best_gap_cp:
        if intended_gap_cp != 0:
            # Intended is comparable but not the best move.
            return ForcednessCategory.AMBIGUOUS
        if second_gap_cp is None:
            return ForcednessCategory.UNIQUE_BEST
        if second_gap_cp >= config.practically_forced_gap_cp:
            return ForcednessCategory.PRACTICALLY_FORCED
        if second_gap_cp >= config.unique_best_gap_cp:
            return ForcednessCategory.UNIQUE_BEST
        return ForcednessCategory.AMBIGUOUS
    return ForcednessCategory.AMBIGUOUS
