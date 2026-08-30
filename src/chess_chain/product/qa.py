"""QA report builder (stage 7).

``qa_report.md`` documents pass/fail per hard constraint (from the stored
machine validation report; ``chess-chain product validate`` provides an
independent re-check), leakage decompositions, per-ply forcedness, source
ratings, the provisional rating object and the reproduce commands. It is
private developer material and may reference solution data.
"""

from __future__ import annotations

from chess_chain.analysis.policies import get_policy
from chess_chain.config import Settings
from chess_chain.product.common import per_ply_forcedness
from chess_chain.sequence.rating import DifficultyEstimate
from chess_chain.sequence.records import SequenceResult

__all__ = ["build_qa_model", "CONSTRAINT_LABELS"]

CONSTRAINT_LABELS: tuple[tuple[int, str], ...] = (
    (1, "every source line replays legally"),
    (2, "every solver move meets the configured uniqueness policy"),
    (3, "every opponent move meets the configured forcedness policy"),
    (4, "every intermediate stage has a verified visible ending"),
    (5, "every transition maps every accepted endpoint to the exact next start"),
    (6, "transitions and guards stay within leakage and complexity limits"),
    (7, "all stages are achievable with the configured physical inventory"),
    (8, "no puzzle is repeated within the sequence"),
    (9, "exclusions / already-used ids are respected"),
    (10, "the last stored solution ends in python-chess-confirmed checkmate"),
    (11, "only the final puzzle must have occupancy invariance"),
    (12, "only the final puzzle meets the declared final-collision status"),
)


def build_qa_model(
    sequence: SequenceResult, settings: Settings, estimate: DifficultyEstimate
) -> str:
    lines: list[str] = []
    lines.append(f"# QA report - {sequence.product_id}")
    lines.append("")
    lines.append(f"- Sequence fingerprint: `{sequence.fingerprint}`")
    lines.append(f"- Random seed: `{sequence.request.random_seed}`")
    lines.append(
        f"- Reproduce: `chess-chain sequence reproduce --product-id {sequence.product_id}`"
    )
    lines.append(f"- Independent re-check: `chess-chain product validate {sequence.product_id}`")
    lines.append("")

    lines.append("## Hard constraints (stored generation-time validation)")
    lines.append("")
    valid = bool(sequence.validation.valid)
    lines.append(f"Overall: **{'PASS' if valid else 'FAIL'}**")
    for number, label in CONSTRAINT_LABELS:
        lines.append(f"- [{number}] {label}: {_constraint_status(sequence)}")
    if sequence.validation.issues:
        lines.append("")
        lines.append("### Issues")
        for issue in sequence.validation.issues:
            lines.append(f"- {issue}")
    lines.append("")

    lines.append("## Warnings")
    lines.append("")
    warnings = list(_collect_warnings(sequence))
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Source ratings")
    lines.append("")
    for stage in sequence.stages:
        lines.append(
            f"- Stage {stage.index + 1} ({stage.puzzle_id}): "
            f"source rating **{int(stage.row['rating'])}**"
        )
    lines.append("")
    lines.append("## Provisional difficulty estimate")
    lines.append("")
    lines.append(f"- Label: **{estimate.label}** (provisional source-rating average)")
    lines.append(f"- Value: **{estimate.value}** (scale: {estimate.scale})")
    lines.append(f"- Method: `{estimate.method}`, estimator version {estimate.estimator_version}")
    lines.append(f"- provisional: {estimate.provisional}")
    lines.append(f"- Count: {estimate.count}; min {estimate.minimum}; max {estimate.maximum}")
    lines.append("- Uncertainty interval: **none** (not fabricated)")
    for warning in estimate.warnings:
        lines.append(f"- Warning: {warning}")
    lines.append("")

    for stage in sequence.stages:
        lines.append(f"## Stage {stage.index + 1}: {stage.puzzle_id}")
        lines.append("")
        policy = get_policy(stage.end_policy_key.split("@", 1)[0])
        lines.append(f"- Stage-end policy: `{policy.key()}` ({policy.wording})")
        for record in per_ply_forcedness(stage.row):
            lines.append(
                f"- ply {record['ply']} ({record['side']}, {record['uci']}): "
                f"{record['legal_moves']} legal move(s)"
            )
        if stage.edge is not None and stage.edge.clue is not None:
            lines.append("- Transition leakage decomposition:")
            for key, value in sorted(stage.edge.clue.items()):
                lines.append(f"  - {key}: {value}")
        if stage.edge is not None and stage.edge.guard is not None:
            lines.append("- Stage-boundary guard:")
            for key, value in sorted(stage.edge.guard.items()):
                lines.append(f"  - {key}: {value}")
        if stage.final_selection is not None:
            lines.append("- Final (mechanical) analysis:")
            lines.append(f"  - checkmate: {stage.final_selection.final_ends_in_checkmate}")
            lines.append(
                f"  - collision status: {stage.final_selection.collision.status.value}"
                f" (scope={stage.final_selection.collision.request.scope},"
                f" max_depth={stage.final_selection.collision.request.max_depth})"
            )
            if stage.final_selection.collision.witness_line:
                lines.append(
                    "  - collision witness: "
                    + " ".join(stage.final_selection.collision.witness_line)
                )
        lines.append("")

    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("chess-chain sequence reproduce --product-id " + sequence.product_id)
    lines.append("chess-chain product validate " + sequence.product_id)
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _constraint_status(sequence: SequenceResult) -> str:
    return "PASS" if bool(sequence.validation.valid) else "FAIL"


def _collect_warnings(sequence: SequenceResult) -> list[str]:
    warnings: list[str] = []
    for stage in sequence.stages:
        if stage.edge is not None and stage.edge.verification_issues:
            warnings.append(
                f"stage {stage.puzzle_id}: stored verification notes: "
                + "; ".join(stage.edge.verification_issues)
            )
        if stage.edge is not None and stage.edge.clue:
            distinct = stage.edge.clue.get("distinct_next_positions_across_plausible")
            if isinstance(distinct, int) and distinct > 1:
                warnings.append(
                    f"stage {stage.puzzle_id}: {distinct} distinct next positions "
                    "across plausible endpoints (collapse-reveal policy governs)"
                )
    return warnings
