"""Application configuration loaded from YAML and environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_LICHESS_PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
DEFAULT_PUZZLE_FILENAME = "lichess_db_puzzle.csv.zst"


def _default_project_root() -> Path:
    return Path.cwd().resolve()


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded if isinstance(loaded, dict) else {}


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


@lru_cache
def get_settings() -> Settings:
    return Settings.from_yaml()
