"""UCI engine adapter around python-chess.

Stockfish (or any UCI engine) is an *external configured executable*; it is
never vendored into the repository or commercial artifacts. The executable
path, threads, hash, nodes/depth/movetime and MultiPV are configuration. Every
analysis result records the engine name/version and the budget used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import chess
import chess.engine

__all__ = [
    "EngineAdapter",
    "EngineAnalysisResult",
    "EngineConfig",
    "EngineLike",
    "EngineMoveInfo",
    "EngineUnavailableError",
    "engine_fen",
    "engine_fingerprint_from_identity",
]


def engine_fen(board: chess.Board) -> str:
    """Normalise a board to the four engine-relevant FEN fields."""
    return " ".join(board.fen().split()[:4])


@dataclass(frozen=True)
class EngineConfig:
    path: str
    threads: int = 1
    hash_mb: int = 64
    movetime_ms: int | None = None
    depth: int | None = None
    nodes: int | None = None
    multipv: int = 4

    def budget_key(self) -> str:
        return json.dumps(
            {
                "path": self.path,
                "threads": self.threads,
                "hash_mb": self.hash_mb,
                "movetime_ms": self.movetime_ms,
                "depth": self.depth,
                "nodes": self.nodes,
                "multipv": self.multipv,
            },
            sort_keys=True,
        )


@dataclass(frozen=True)
class EngineMoveInfo:
    uci: str
    score_cp: int | None
    score_mate: int | None
    rank: int


@dataclass(frozen=True)
class EngineAnalysisResult:
    fen: str
    moves: tuple[EngineMoveInfo, ...]
    engine_name: str
    engine_version: str
    budget: str

    def engine_fingerprint(self) -> str:
        return engine_fingerprint_from_identity(
            budget=self.budget, name=self.engine_name, version=self.engine_version
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "fen": self.fen,
            "moves": [
                {
                    "uci": move.uci,
                    "score_cp": move.score_cp,
                    "score_mate": move.score_mate,
                    "rank": move.rank,
                }
                for move in self.moves
            ],
            "engine_name": self.engine_name,
            "engine_version": self.engine_version,
            "budget": self.budget,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineAnalysisResult:
        moves = tuple(
            EngineMoveInfo(
                uci=str(item["uci"]),
                score_cp=(int(item["score_cp"]) if item["score_cp"] is not None else None),
                score_mate=(int(item["score_mate"]) if item["score_mate"] is not None else None),
                rank=int(item["rank"]),
            )
            for item in data.get("moves", [])
        )
        return cls(
            fen=str(data["fen"]),
            moves=moves,
            engine_name=str(data["engine_name"]),
            engine_version=str(data["engine_version"]),
            budget=str(data["budget"]),
        )


def engine_fingerprint_from_identity(budget: str, name: str, version: str) -> str:
    digest = hashlib.sha256()
    digest.update(budget.encode("utf-8"))
    digest.update(f"|{name}|{version}".encode())
    return digest.hexdigest()


class EngineUnavailableError(RuntimeError):
    """Raised when the configured engine cannot be started or queried."""


class EngineLike(Protocol):
    def engine_identity(self) -> tuple[str, str]: ...

    def analyze(
        self,
        board: chess.Board,
        *,
        multipv: int | None = None,
        mate_value_cp: int = 10_000,
    ) -> EngineAnalysisResult: ...

    def close(self) -> None: ...


class EngineAdapter:
    """Thin wrapper around python-chess's UCI support."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._engine: chess.engine.SimpleEngine | None = None
        self._identity: tuple[str, str] | None = None

    def _ensure_engine(self) -> chess.engine.SimpleEngine:
        if self._engine is not None:
            return self._engine
        if not self.config.path:
            raise EngineUnavailableError(
                "No engine configured (analysis.engine.path is empty). "
                "Cheap analysis works without an engine."
            )
        if not Path(self.config.path).is_file():
            raise EngineUnavailableError(f"Engine executable not found: {self.config.path}")
        try:
            engine = chess.engine.SimpleEngine.popen_uci(self.config.path)
        except (OSError, chess.engine.EngineError) as exc:
            raise EngineUnavailableError(
                f"Could not start engine {self.config.path!r}: {exc}"
            ) from exc
        try:
            engine.configure({"Threads": self.config.threads})
            engine.configure({"Hash": self.config.hash_mb})
        except chess.engine.EngineError as exc:
            engine.quit()
            raise EngineUnavailableError(f"Could not configure engine: {exc}") from exc
        self._engine = engine
        self._identity = (str(engine.id.get("name", "unknown")), str(engine.id.get("author", "")))
        return engine

    def engine_identity(self) -> tuple[str, str]:
        self._ensure_engine()
        assert self._identity is not None
        return self._identity

    def analyze(
        self,
        board: chess.Board,
        *,
        multipv: int | None = None,
        mate_value_cp: int = 10_000,
    ) -> EngineAnalysisResult:
        engine = self._ensure_engine()
        multipv = multipv or self.config.multipv
        limit = chess.engine.Limit(
            time=self.config.movetime_ms / 1000.0 if self.config.movetime_ms else None,
            depth=self.config.depth,
            nodes=self.config.nodes,
        )
        if not any((limit.time, limit.depth, limit.nodes)):
            raise EngineUnavailableError(
                "Engine budget is empty; set movetime_ms, depth or nodes in config"
            )
        infos = engine.analyse(board, limit, multipv=multipv)
        moves: list[EngineMoveInfo] = []
        for rank, info in enumerate(infos):
            pv = info.get("pv", [])
            if not pv:
                continue
            score = info.get("score")
            score_cp: int | None = None
            score_mate: int | None = None
            if score is not None:
                score_obj: Any = score  # PovScore annotations are partial
                mate = score_obj.mate()
                if mate is not None:
                    score_mate = int(mate)
                converted = score_obj.score(mate_score=mate_value_cp)
                if converted is not None:
                    score_cp = int(converted)
            moves.append(
                EngineMoveInfo(uci=pv[0].uci(), score_cp=score_cp, score_mate=score_mate, rank=rank)
            )
        name, version = self.engine_identity()
        return EngineAnalysisResult(
            fen=engine_fen(board),
            moves=tuple(moves),
            engine_name=name,
            engine_version=version,
            budget=self.config.budget_key(),
        )

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None
