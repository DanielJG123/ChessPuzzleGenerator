"""Record flattening and ingest manifest helpers."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import chess

from chess_chain import __version__
from chess_chain.domain.pieces import COUNT_NAMES, MaterialSignature
from chess_chain.domain.puzzle import PuzzleCore
from chess_chain.storage.manifest import load_manifest, manifest_path_for

__all__ = [
    "IngestSummary",
    "core_to_record",
    "load_ingest_manifest",
    "quarantine_record",
    "source_fingerprint",
    "write_ingest_manifest",
]


@dataclass(frozen=True)
class IngestSummary:
    """Result of one ingestion run."""

    dataset_dir: Path
    dataset_name: str
    transform_version: int
    source_path: str
    source_fingerprint: str
    puzzle_rows: int
    quarantine_rows: int
    created_at: str
    elapsed_seconds: float
    skipped: bool

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "transform_version": self.transform_version,
            "source_path": self.source_path,
            "source_fingerprint": self.source_fingerprint,
            "puzzle_rows": self.puzzle_rows,
            "quarantine_rows": self.quarantine_rows,
            "partition_columns": ["rating_bucket"],
            "created_at": self.created_at,
            "code_version": __version__,
        }


def load_ingest_manifest(dataset_dir: Path) -> dict[str, object] | None:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def write_ingest_manifest(dataset_dir: Path, manifest: dict[str, object]) -> None:
    path = dataset_dir / "manifest.json"
    temp_path = path.with_suffix(path.suffix + ".part")
    temp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def source_fingerprint(path: Path) -> str:
    """Return a stable fingerprint of the source file.

    Prefers the SHA-256 recorded by the stage-1 downloader manifest; otherwise
    computes the SHA-256 of the file in a streamed fashion.
    """
    download_manifest = load_manifest(manifest_path_for(path))
    if download_manifest is not None:
        return f"sha256:{download_manifest.sha256}"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def core_to_record(
    core: PuzzleCore,
    *,
    source_name: str,
    transform_version: int,
) -> dict[str, object]:
    """Flatten a PuzzleCore into a dict matching PUZZLE_SCHEMA."""
    start = core.presented_start
    final = core.final_position
    start_material = MaterialSignature.from_board(start.board)
    final_material = MaterialSignature.from_board(final.board)
    start_bitboards = start.fingerprint.bitboards

    record: dict[str, object] = {
        "puzzle_id": core.source.puzzle_id,
        "source": source_name,
        "transform_version": transform_version,
        "rating": core.source.rating,
        "rating_deviation": core.source.rating_deviation,
        "popularity": core.source.popularity,
        "nb_plays": core.source.nb_plays,
        "themes": list(core.source.themes),
        "opening_tags": list(core.source.opening_tags),
        "game_url": core.source.game_url,
        "daily_date": core.source.daily_date,
        "start_fen": start.fen,
        "start_fingerprint": start.fingerprint.to_key(),
        "start_occupancy": start.fingerprint.occupied,
        "start_castling": start.fingerprint.castling,
        "start_ep_square": start.fingerprint.ep_square,
        "start_material_w": start_material.material_score(chess.WHITE),
        "start_material_b": start_material.material_score(chess.BLACK),
        "presented_piece_count": start_material.piece_count(),
        "solution_uci": list(core.solution_uci),
        "solution_san": list(core.solution_san),
        "solution_plies": len(core.solution_uci),
        "final_fen": final.fen,
        "final_fingerprint": final.fingerprint.to_key(),
        "final_occupancy": final.fingerprint.occupied,
        "final_material_w": final_material.material_score(chess.WHITE),
        "final_material_b": final_material.material_score(chess.BLACK),
        "outcome": core.outcome,
        "is_check": core.is_check,
        "is_checkmate": core.is_checkmate,
        "is_stalemate": core.is_stalemate,
        "solver_colour": "white" if core.solver_colour == chess.WHITE else "black",
        "moved_piece_ids": sorted(identity.key() for identity in core.moved_ids),
        "unmoved_piece_ids": sorted(identity.key() for identity in core.unmoved_ids),
        "captured_piece_ids": [identity.key() for identity in core.captured_ids],
        "last_moved_piece_id": core.last_moved_id.key(),
        "final_checking_piece_ids": sorted(identity.key() for identity in core.final_checking_ids),
        "promotion_events": list(core.promotion_events),
        "has_promotion": core.has_promotion,
        "duplicate_queens": core.duplicate_queens,
        "unusual_material": core.unusual_material,
    }
    for index, name in enumerate(COUNT_NAMES):
        record[f"start_bb_{name}"] = start_bitboards[index]
        record[f"start_{name}"] = start_material.counts[index]
        record[f"final_{name}"] = final_material.counts[index]
    return record


def quarantine_record(
    *,
    source_name: str,
    line_number: int,
    reason: str,
    fields: list[str],
) -> dict[str, object]:
    puzzle_id = fields[0] if fields else ""
    return {
        "source": source_name,
        "line_number": line_number,
        "puzzle_id": puzzle_id,
        "reason": reason,
        "raw_fields": list(fields),
    }
