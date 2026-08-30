"""Application configuration loaded from YAML and environment variables."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_LICHESS_PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
DEFAULT_PUZZLE_FILENAME = "lichess_db_puzzle.csv.zst"

DEFAULT_GLOBAL_RULE = (
    "Continue choosing the strongest move for each side until checkmate occurs "
    "or the stated transition condition becomes uniquely applicable."
)


@dataclass(frozen=True)
class TransitionConfig:
    """Tunable knobs for transition synthesis / verification.

    Defined here (not inside the ``transitions`` package) so ``Settings`` can
    build it without triggering the package's heavier imports. The global rule
    wording is configurable rather than assumed final.
    """

    version: int = 1
    default_max_actions: int = 4
    plausible_depth: int = 2
    plausible_node_cap: int = 2000
    min_anchor_uncertainty_bits: float = 0.0
    min_independent_anchors: int = 1
    allow_absolute_square_fallback: bool = True
    allow_establish_rights: bool = True
    reject_collapsing_reveal: bool = True
    global_rule_wording: str = DEFAULT_GLOBAL_RULE

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "default_max_actions": self.default_max_actions,
            "plausible_depth": self.plausible_depth,
            "plausible_node_cap": self.plausible_node_cap,
            "min_anchor_uncertainty_bits": self.min_anchor_uncertainty_bits,
            "min_independent_anchors": self.min_independent_anchors,
            "allow_absolute_square_fallback": self.allow_absolute_square_fallback,
            "allow_establish_rights": self.allow_establish_rights,
            "reject_collapsing_reveal": self.reject_collapsing_reveal,
            "global_rule_wording": self.global_rule_wording,
        }


def _default_project_root() -> Path:
    return Path.cwd().resolve()


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _drop_env_override_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Remove YAML-derived kwargs that have a ``CHESS_CHAIN_<FIELD>`` env var.

    Precedence (highest wins): CLI overrides > environment variables > user
    YAML request file > checked-in defaults. Explicit init kwargs would
    otherwise beat environment variables in pydantic-settings.
    """
    import os

    overridden = {key for key in kwargs if f"CHESS_CHAIN_{key.upper()}" in os.environ}
    return {key: value for key, value in kwargs.items() if key not in overridden}


class Settings(BaseSettings):
    """Runtime settings for Chess Chain."""

    model_config = SettingsConfigDict(
        env_prefix="CHESS_CHAIN_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=_default_project_root)
    config_path: Path | None = None

    data_raw_dir: Path | None = None
    data_interim_dir: Path | None = None
    data_processed_dir: Path | None = None
    data_samples_dir: Path | None = None

    lichess_puzzle_url: str = DEFAULT_LICHESS_PUZZLE_URL
    lichess_puzzle_filename: str = DEFAULT_PUZZLE_FILENAME

    download_chunk_size: int = 1024 * 1024  # 1 MiB
    download_space_margin_bytes: int = 64 * 1024 * 1024  # 64 MiB safety margin
    default_sample_rows: int = 10_000

    ingest_batch_size: int = 50_000
    parquet_row_group_rows: int = 65_536
    rating_bucket_width: int = 100
    transform_version: int = 1

    # Stage 3 analysis settings (versioned thresholds in config/default.yaml).
    engine_path: str = ""
    engine_threads: int = 1
    engine_hash_mb: int = 64
    engine_movetime_ms: int | None = None
    engine_depth: int | None = None
    engine_nodes: int | None = None
    engine_multipv: int = 4

    forcedness_version: int = 1
    unique_best_gap_cp: int = 50
    practically_forced_gap_cp: int = 200
    mate_value_cp: int = 10_000
    plausible_gap_cp: int = 30

    variation_max_plies: int = 8
    variation_max_nodes: int = 20_000
    variation_acceptance: str = "intended_only"

    collision_analyser_version: int = 1
    collision_scope: str = "up_to_plies"
    collision_max_depth: int = 12
    collision_max_nodes: int = 100_000

    # Stage 4 similarity settings (versioned thresholds in config/default.yaml).
    similarity_version: int = 1
    similarity_backend: str = "exact"
    similarity_default_top_k: int = 50
    similarity_max_rating_deviation: int = 200
    similarity_min_popularity: int = 0
    similarity_piece_count_tolerance: int = 4
    similarity_max_candidates_scanned: int = 200_000
    similarity_shortlist_multiplier: int = 4
    similarity_reject_non_visible_rights: bool = False
    similarity_exclude_unusual_material: bool = False
    similarity_symmetries: tuple[str, ...] = (
        "identity",
        "file_reflection",
        "rank_reflection",
        "rot180",
    )
    similarity_score_weights: dict[str, float] = {
        "occupancy_best": 3.0,
        "occupancy_mean": 1.5,
        "occupancy_worst": 0.5,
        "cost_best": 1.0,
        "cost_mean": 0.5,
        "cost_worst": 0.25,
        "swap": 1.0,
        "replace": 4.0,
        "branch": 2.0,
        "symmetry_non_identity": 3.0,
        "rights_annotation": 5.0,
        "inventory_invalid": 20.0,
    }
    similarity_cost_weights: dict[str, float] = {
        "move": 1.0,
        "add": 3.0,
        "remove": 2.0,
    }

    # Stage 5 transition settings (versioned thresholds in config/default.yaml).
    transition_version: int = 1
    transition_default_max_actions: int = 4
    transition_plausible_depth: int = 2
    transition_plausible_node_cap: int = 2000
    transition_min_anchor_uncertainty_bits: float = 0.0
    transition_min_independent_anchors: int = 1
    transition_allow_absolute_square_fallback: bool = True
    transition_allow_establish_rights: bool = True
    transition_reject_collapsing_reveal: bool = True
    transition_global_rule_wording: str = (
        "Continue choosing the strongest move for each side until checkmate "
        "occurs or the stated transition condition becomes uniquely applicable."
    )

    @field_validator("project_root", mode="before")
    @classmethod
    def _resolve_project_root(cls, value: Any) -> Path:
        return Path(value).expanduser().resolve()

    @classmethod
    def from_yaml(cls, config_path: Path | None = None) -> Settings:
        root = _default_project_root()
        resolved_config = (config_path or root / "config" / "default.yaml").resolve()
        yaml_data = _load_yaml_config(resolved_config)
        paths = yaml_data.get("paths", {})
        ingest = yaml_data.get("ingest", {})
        analysis = yaml_data.get("analysis", {})

        kwargs: dict[str, Any] = {"project_root": root, "config_path": resolved_config}
        if paths:
            for key in ("raw", "interim", "processed", "samples"):
                if key in paths:
                    path_value = Path(paths[key])
                    if not path_value.is_absolute():
                        path_value = root / path_value
                    kwargs[f"data_{key}_dir"] = path_value.resolve()
        if ingest:
            if "lichess_puzzle_url" in ingest:
                kwargs["lichess_puzzle_url"] = ingest["lichess_puzzle_url"]
            if "lichess_puzzle_filename" in ingest:
                kwargs["lichess_puzzle_filename"] = ingest["lichess_puzzle_filename"]
            if "download_chunk_size" in ingest:
                kwargs["download_chunk_size"] = ingest["download_chunk_size"]
            if "download_space_margin_bytes" in ingest:
                kwargs["download_space_margin_bytes"] = ingest["download_space_margin_bytes"]
            if "ingest_batch_size" in ingest:
                kwargs["ingest_batch_size"] = int(ingest["ingest_batch_size"])
            if "parquet_row_group_rows" in ingest:
                kwargs["parquet_row_group_rows"] = int(ingest["parquet_row_group_rows"])
            if "rating_bucket_width" in ingest:
                kwargs["rating_bucket_width"] = int(ingest["rating_bucket_width"])
            if "transform_version" in ingest:
                kwargs["transform_version"] = int(ingest["transform_version"])
        if analysis:
            if "default_sample_rows" in analysis:
                kwargs["default_sample_rows"] = int(analysis["default_sample_rows"])
            engine = analysis.get("engine", {})
            if "path" in engine:
                kwargs["engine_path"] = str(engine["path"])
            if "threads" in engine:
                kwargs["engine_threads"] = int(engine["threads"])
            if "hash_mb" in engine:
                kwargs["engine_hash_mb"] = int(engine["hash_mb"])
            if "movetime_ms" in engine:
                kwargs["engine_movetime_ms"] = (
                    int(engine["movetime_ms"]) if engine["movetime_ms"] is not None else None
                )
            if "depth" in engine:
                kwargs["engine_depth"] = (
                    int(engine["depth"]) if engine["depth"] is not None else None
                )
            if "nodes" in engine:
                kwargs["engine_nodes"] = (
                    int(engine["nodes"]) if engine["nodes"] is not None else None
                )
            if "multipv" in engine:
                kwargs["engine_multipv"] = int(engine["multipv"])
            forcedness = analysis.get("forcedness", {})
            if "version" in forcedness:
                kwargs["forcedness_version"] = int(forcedness["version"])
            if "unique_best_gap_cp" in forcedness:
                kwargs["unique_best_gap_cp"] = int(forcedness["unique_best_gap_cp"])
            if "practically_forced_gap_cp" in forcedness:
                kwargs["practically_forced_gap_cp"] = int(forcedness["practically_forced_gap_cp"])
            if "mate_value_cp" in forcedness:
                kwargs["mate_value_cp"] = int(forcedness["mate_value_cp"])
            if "plausible_gap_cp" in forcedness:
                kwargs["plausible_gap_cp"] = int(forcedness["plausible_gap_cp"])
            variations = analysis.get("variations", {})
            if "default_max_plies" in variations:
                kwargs["variation_max_plies"] = int(variations["default_max_plies"])
            if "default_max_nodes" in variations:
                kwargs["variation_max_nodes"] = int(variations["default_max_nodes"])
            if "default_acceptance" in variations:
                kwargs["variation_acceptance"] = str(variations["default_acceptance"])
            collision = analysis.get("collision", {})
            if "analyser_version" in collision:
                kwargs["collision_analyser_version"] = int(collision["analyser_version"])
            if "default_scope" in collision:
                kwargs["collision_scope"] = str(collision["default_scope"])
            if "default_max_depth" in collision:
                kwargs["collision_max_depth"] = int(collision["default_max_depth"])
            if "default_max_nodes" in collision:
                kwargs["collision_max_nodes"] = int(collision["default_max_nodes"])
            kwargs["_analysis"] = analysis
        similarity = yaml_data.get("similarity", {})
        if similarity:
            if "version" in similarity:
                kwargs["similarity_version"] = int(similarity["version"])
            if "backend" in similarity:
                kwargs["similarity_backend"] = str(similarity["backend"])
            if "default_top_k" in similarity:
                kwargs["similarity_default_top_k"] = int(similarity["default_top_k"])
            if "max_rating_deviation" in similarity:
                kwargs["similarity_max_rating_deviation"] = int(similarity["max_rating_deviation"])
            if "min_popularity" in similarity:
                kwargs["similarity_min_popularity"] = int(similarity["min_popularity"])
            if "piece_count_tolerance" in similarity:
                kwargs["similarity_piece_count_tolerance"] = int(
                    similarity["piece_count_tolerance"]
                )
            if "max_candidates_scanned" in similarity:
                kwargs["similarity_max_candidates_scanned"] = int(
                    similarity["max_candidates_scanned"]
                )
            if "shortlist_multiplier" in similarity:
                kwargs["similarity_shortlist_multiplier"] = int(similarity["shortlist_multiplier"])
            if "reject_non_visible_rights" in similarity:
                kwargs["similarity_reject_non_visible_rights"] = bool(
                    similarity["reject_non_visible_rights"]
                )
            if "exclude_unusual_material" in similarity:
                kwargs["similarity_exclude_unusual_material"] = bool(
                    similarity["exclude_unusual_material"]
                )
            if "symmetries" in similarity:
                kwargs["similarity_symmetries"] = tuple(
                    str(item) for item in similarity["symmetries"]
                )
            if "score_weights" in similarity:
                kwargs["similarity_score_weights"] = {
                    str(key): float(value) for key, value in similarity["score_weights"].items()
                }
            if "cost_weights" in similarity:
                kwargs["similarity_cost_weights"] = {
                    str(key): float(value) for key, value in similarity["cost_weights"].items()
                }
        transition = yaml_data.get("transition", {})
        if transition:
            if "version" in transition:
                kwargs["transition_version"] = int(transition["version"])
            if "default_max_actions" in transition:
                kwargs["transition_default_max_actions"] = int(transition["default_max_actions"])
            if "plausible_depth" in transition:
                kwargs["transition_plausible_depth"] = int(transition["plausible_depth"])
            if "plausible_node_cap" in transition:
                kwargs["transition_plausible_node_cap"] = int(transition["plausible_node_cap"])
            if "min_anchor_uncertainty_bits" in transition:
                kwargs["transition_min_anchor_uncertainty_bits"] = float(
                    transition["min_anchor_uncertainty_bits"]
                )
            if "min_independent_anchors" in transition:
                kwargs["transition_min_independent_anchors"] = int(
                    transition["min_independent_anchors"]
                )
            if "allow_absolute_square_fallback" in transition:
                kwargs["transition_allow_absolute_square_fallback"] = bool(
                    transition["allow_absolute_square_fallback"]
                )
            if "allow_establish_rights" in transition:
                kwargs["transition_allow_establish_rights"] = bool(
                    transition["allow_establish_rights"]
                )
            if "reject_collapsing_reveal" in transition:
                kwargs["transition_reject_collapsing_reveal"] = bool(
                    transition["reject_collapsing_reveal"]
                )
            if "global_rule_wording" in transition:
                kwargs["transition_global_rule_wording"] = str(transition["global_rule_wording"])
        # Explicit precedence: environment variables (machine-specific paths
        # such as Stockfish) beat user YAML and checked-in defaults. CLI
        # overrides are applied on top by the commands that support them.
        kwargs = _drop_env_override_kwargs(kwargs)
        return cls(**kwargs)

    def model_post_init(self, __context: Any) -> None:
        defaults = {
            "data_raw_dir": self.project_root / "data" / "raw",
            "data_interim_dir": self.project_root / "data" / "interim",
            "data_processed_dir": self.project_root / "data" / "processed",
            "data_samples_dir": self.project_root / "data" / "samples",
        }
        for field_name, default_path in defaults.items():
            if getattr(self, field_name) is None:
                object.__setattr__(self, field_name, default_path.resolve())

    @property
    def default_puzzle_path(self) -> Path:
        raw = self.data_raw_dir
        assert raw is not None
        return (raw / self.lichess_puzzle_filename).resolve()

    @property
    def default_manifest_path(self) -> Path:
        return self.default_puzzle_path.with_suffix(
            self.default_puzzle_path.suffix + ".manifest.json"
        )

    @property
    def analysis_dir(self) -> Path:
        processed = self.data_processed_dir
        assert processed is not None
        return (processed / "analysis").resolve()

    @property
    def similarity_dir(self) -> Path:
        return (self.analysis_dir / "similarity").resolve()

    @property
    def sequence_dir(self) -> Path:
        return (self.analysis_dir / "sequence").resolve()

    @property
    def transition_config(self) -> TransitionConfig:
        return TransitionConfig(
            version=self.transition_version,
            default_max_actions=self.transition_default_max_actions,
            plausible_depth=self.transition_plausible_depth,
            plausible_node_cap=self.transition_plausible_node_cap,
            min_anchor_uncertainty_bits=self.transition_min_anchor_uncertainty_bits,
            min_independent_anchors=self.transition_min_independent_anchors,
            allow_absolute_square_fallback=self.transition_allow_absolute_square_fallback,
            allow_establish_rights=self.transition_allow_establish_rights,
            reject_collapsing_reveal=self.transition_reject_collapsing_reveal,
            global_rule_wording=self.transition_global_rule_wording,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_yaml()
