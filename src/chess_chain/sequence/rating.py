"""Provisional difficulty estimation (stage 7).

This module defines a small, replaceable difficulty-estimation boundary so the
later optional calibration stage can upgrade it without refactoring sequence
generation or rendering. The default estimator deliberately does **not** do any
calibration: it only averages the source Lichess puzzle rating of every stage.

``SimpleAverageRatingEstimator`` ignores the both-sides protocol, visible
transition clues, stage-end information and the multiplicative difficulty of
completing several stages. That limitation is intentional: the purpose is
manual product review, not a commercial difficulty claim. The estimate is
``provisional=true`` and labelled **"Provisional source-rating average"** - never
"sequence Elo", "physical puzzle rating" or "calibrated rating". No uncertainty
interval is invented.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "DifficultyEstimate",
    "MissingRatingError",
    "RatingEstimator",
    "SimpleAverageRatingEstimator",
    "estimate_key",
    "estimate_from_dict",
]

METHOD_MEAN_SOURCE = "mean_source_puzzle_rating"
SCALE_SOURCE_LICHESS = "source_lichess_puzzle_rating"
LABEL_PROVISIONAL_AVERAGE = "Provisional source-rating average"

#: Why the average is only a placeholder, never a calibrated difficulty claim.
SIMPLISTIC_WARNING = (
    "Intentionally simplistic: ignores the both-sides protocol, visible "
    "transition clues, stage-end information and the multiplicative difficulty "
    "of completing several stages. Manual product review only, not a "
    "commercial difficulty claim."
)


def estimate_key(method: str, estimator_version: int) -> str:
    """The forward-compatible storage key for one named/versioned estimator."""
    return f"{method}@{estimator_version}"


class MissingRatingError(ValueError):
    """Raised when a stage source rating is missing or invalid.

    The default policy is strict: missing ratings fail with an actionable
    message instead of being silently imputed. A future explicit missing-data
    policy may relax this.
    """


@dataclass(frozen=True)
class DifficultyEstimate:
    """Versioned output of one ``RatingEstimator``.

    ``value`` is the rounded integer summary on ``scale``; ``stage_ratings``
    preserve the individual source ratings in stage order; ``count``,
    ``minimum`` and ``maximum`` are descriptive context. No uncertainty
    interval is fabricated.
    """

    method: str = METHOD_MEAN_SOURCE
    scale: str = SCALE_SOURCE_LICHESS
    provisional: bool = True
    label: str = LABEL_PROVISIONAL_AVERAGE
    value: int = 0
    count: int = 0
    minimum: int | None = None
    maximum: int | None = None
    stage_ratings: tuple[int, ...] = ()
    estimator_version: int = 1
    warnings: tuple[str, ...] = (SIMPLISTIC_WARNING,)

    @property
    def key(self) -> str:
        return estimate_key(self.method, self.estimator_version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "scale": self.scale,
            "provisional": self.provisional,
            "label": self.label,
            "value": self.value,
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "stage_ratings": list(self.stage_ratings),
            "estimator_version": self.estimator_version,
            "warnings": list(self.warnings),
        }


def estimate_from_dict(data: dict[str, Any]) -> DifficultyEstimate:
    return DifficultyEstimate(
        method=str(data.get("method", METHOD_MEAN_SOURCE)),
        scale=str(data.get("scale", SCALE_SOURCE_LICHESS)),
        provisional=bool(data.get("provisional", True)),
        label=str(data.get("label", LABEL_PROVISIONAL_AVERAGE)),
        value=int(data["value"]),
        count=int(data.get("count", 0)),
        minimum=int(data["minimum"]) if data.get("minimum") is not None else None,
        maximum=int(data["maximum"]) if data.get("maximum") is not None else None,
        stage_ratings=tuple(int(r) for r in data.get("stage_ratings", [])),
        estimator_version=int(data.get("estimator_version", 1)),
        warnings=tuple(str(w) for w in data.get("warnings", [])),
    )


class RatingEstimator(Protocol):
    """Boundary for difficulty estimators.

    A future calibrated estimator implements the same protocol and stores its
    result under its own ``method@estimator_version`` key, leaving existing
    records valid.
    """

    estimator_version: int

    def estimate(self, stage_ratings: Sequence[int | None]) -> DifficultyEstimate: ...


def _validate_rating(rating: int | None, stage_index: int) -> int:
    if rating is None:
        raise MissingRatingError(
            f"stage {stage_index}: source rating is missing. Add the source "
            "Lichess puzzle rating before estimating; the placeholder never "
            "imputes a value."
        )
    if isinstance(rating, bool) or not isinstance(rating, int):
        raise MissingRatingError(
            f"stage {stage_index}: source rating is invalid ({rating!r}). "
            "Expected an integer Lichess puzzle rating."
        )
    return rating


class SimpleAverageRatingEstimator:
    """Arithmetic mean of the source ratings of every stage, in order.

    Strict by default: a missing or invalid source rating raises
    ``MissingRatingError`` rather than being imputed. Descriptive ``count``,
    ``minimum`` and ``maximum`` are reported; no uncertainty interval is
    invented, and ``value`` is rounded to an integer to avoid false precision.
    """

    estimator_version: int = 1

    def estimate(self, stage_ratings: Sequence[int | None]) -> DifficultyEstimate:
        parsed = [_validate_rating(rating, index) for index, rating in enumerate(stage_ratings)]
        if not parsed:
            raise MissingRatingError("cannot estimate difficulty for an empty sequence")
        value = round(sum(parsed) / len(parsed))
        return DifficultyEstimate(
            method=METHOD_MEAN_SOURCE,
            scale=SCALE_SOURCE_LICHESS,
            provisional=True,
            label=LABEL_PROVISIONAL_AVERAGE,
            value=value,
            count=len(parsed),
            minimum=min(parsed),
            maximum=max(parsed),
            stage_ratings=tuple(parsed),
            estimator_version=self.estimator_version,
            warnings=(SIMPLISTIC_WARNING,),
        )
