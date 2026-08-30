"""Product rendering (stage 7).

``render_product`` materialises the canonical ``sequence@1`` document into five
artifacts:

1. ``player_instructions.md`` - the only player-safe artifact: board
   orientation, initial setup, rights statement, global rule, configured stage
   boundaries and every transition instruction. Never solution moves, hidden
   codes or references to unavailable information.
2. ``manufacturing_spec.json`` - the complete machine-readable product
   specification (puzzle ids, lines, endpoint FENs, transition ASTs, final
   occupancy target, provisional rating metadata, fingerprints).
3. ``qa_report.md`` - pass/fail per hard constraint, leakage decompositions,
   per-ply forcedness, source ratings and the provisional average, and the
   reproduce commands.
4. ``sequence_review.html`` - a self-contained private developer/manual-review
   report (python-chess SVG boards + a small Jinja2 template, embedded
   CSS/SVG so it opens locally without a server). May reveal every solution.
5. ``review_notes.yaml`` - a human review aid template (never read back).

Public and private artifacts are strictly separated: the player instructions
never link to or name the review/QA artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from chess_chain.config import Settings
from chess_chain.sequence.rating import (
    DifficultyEstimate,
    SimpleAverageRatingEstimator,
    estimate_key,
)
from chess_chain.sequence.records import SequenceResult

__all__ = ["ProductArtifacts", "render_product", "write_sequence_with_estimate"]

ARTIFACT_PLAYER_INSTRUCTIONS = "player_instructions.md"
ARTIFACT_MANUFACTURING_SPEC = "manufacturing_spec.json"
ARTIFACT_QA_REPORT = "qa_report.md"
ARTIFACT_REVIEW_HTML = "sequence_review.html"
ARTIFACT_REVIEW_NOTES = "review_notes.yaml"

#: Lichess puzzle export licence (CC0); per-stage puzzle ids are referenced
#: from the canonical record; this is provenance, not a claim about solutions.
DATA_LICENCE = "CC0 1.0 (Lichess puzzle database export, https://database.lichess.org/#puzzles)"

BOARD_ORIENTATION = (
    "All diagrams and positions are given from White's perspective: rank 8 is "
    "the far side and the a-file is on the left as you face the board."
)


@dataclass(frozen=True)
class ProductArtifacts:
    """Paths to the five rendered artifacts."""

    output_dir: Path
    player_instructions: Path
    manufacturing_spec: Path
    qa_report: Path
    review_html: Path
    review_notes: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "output_dir": str(self.output_dir),
            "player_instructions": str(self.player_instructions),
            "manufacturing_spec": str(self.manufacturing_spec),
            "qa_report": str(self.qa_report),
            "review_html": str(self.review_html),
            "review_notes": str(self.review_notes),
        }


def write_sequence_with_estimate(
    sequence: SequenceResult, estimate: DifficultyEstimate
) -> SequenceResult:
    """Return ``sequence`` with ``estimate`` stored under its forward-compatible
    ``method@estimator_version`` key (old records keep other keys intact)."""
    import dataclasses

    estimates = dict(sequence.difficulty_estimates)
    estimates[estimate.key] = estimate.to_dict()
    return dataclasses.replace(sequence, difficulty_estimates=estimates)


def resolve_estimate(
    sequence: SequenceResult,
    *,
    estimator: SimpleAverageRatingEstimator | None = None,
) -> DifficultyEstimate:
    """Return the stored provisional estimate or compute the placeholder.

    A missing or invalid source rating raises ``MissingRatingError`` with an
    actionable message (the default strict policy; no silent imputation).
    """
    from chess_chain.sequence.rating import estimate_from_dict

    estimator = estimator or SimpleAverageRatingEstimator()
    key = estimate_key("mean_source_puzzle_rating", estimator.estimator_version)
    stored = sequence.difficulty_estimates.get(key)
    ratings = sequence.stage_ratings()
    if stored is not None:
        estimate = estimate_from_dict(stored)
        if estimate.estimator_version == estimator.estimator_version and tuple(
            estimate.stage_ratings
        ) == tuple(ratings):
            return estimate
    return estimator.estimate(ratings)


def render_product(
    sequence: SequenceResult,
    settings: Settings,
    *,
    output_dir: Path | None = None,
    estimate: DifficultyEstimate | None = None,
    overwrite: bool = True,
    generated_at: str | None = None,
) -> ProductArtifacts:
    """Render all five artifacts for ``sequence``.

    ``estimate`` defaults to the stored placeholder, computed fresh when absent
    or stale. The canonical product specification (``manufacturing_spec.json``)
    is independent of the rendering formats used for the other artifacts.
    """
    from chess_chain.product.player import build_player_instructions
    from chess_chain.product.qa import build_qa_model
    from chess_chain.product.review import render_review_html, render_review_notes
    from chess_chain.product.spec import build_manufacturing_spec

    if estimate is None:
        estimate = resolve_estimate(sequence)

    directory = (output_dir or _default_output_dir(settings, sequence.product_id)).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = generated_at or datetime.now(UTC).isoformat()

    if not overwrite:
        _reject_existing(directory)

    artifacts = ProductArtifacts(
        output_dir=directory,
        player_instructions=directory / ARTIFACT_PLAYER_INSTRUCTIONS,
        manufacturing_spec=directory / ARTIFACT_MANUFACTURING_SPEC,
        qa_report=directory / ARTIFACT_QA_REPORT,
        review_html=directory / ARTIFACT_REVIEW_HTML,
        review_notes=directory / ARTIFACT_REVIEW_NOTES,
    )

    (directory / ARTIFACT_PLAYER_INSTRUCTIONS).write_text(
        build_player_instructions(sequence, settings), encoding="utf-8"
    )
    (directory / ARTIFACT_MANUFACTURING_SPEC).write_text(
        json.dumps(
            build_manufacturing_spec(sequence, settings, estimate, generated_at=timestamp),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (directory / ARTIFACT_QA_REPORT).write_text(
        build_qa_model(sequence, settings, estimate), encoding="utf-8"
    )
    (directory / ARTIFACT_REVIEW_HTML).write_text(
        render_review_html(sequence, settings, estimate), encoding="utf-8"
    )
    (directory / ARTIFACT_REVIEW_NOTES).write_text(render_review_notes(sequence), encoding="utf-8")
    return artifacts


def _default_output_dir(settings: Settings, product_id: str) -> Path:
    return (settings.analysis_dir / "products" / product_id).resolve()


def _reject_existing(directory: Path) -> None:
    conflicts = (
        [path for path in directory.iterdir() if path.is_file()] if directory.is_dir() else []
    )
    if conflicts:
        raise FileExistsError(
            f"output directory {directory} is not empty "
            f"({len(conflicts)} file(s)); use --overwrite to replace"
        )
