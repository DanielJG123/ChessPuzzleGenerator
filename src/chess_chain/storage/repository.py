"""Query/repository abstraction over the processed puzzle dataset.

All DuckDB SQL lives here; later modules (similarity, transitions, sequence)
use the typed query methods instead of writing their own SQL.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import duckdb
import polars as pl

from chess_chain.domain.pieces import COUNT_NAMES, MaterialSignature

__all__ = ["PuzzleRepository"]


class PuzzleRepository:
    """Typed queries over the active ``puzzles`` DuckDB view."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._con = connection

    def query(self, sql: str, params: list[object] | tuple[object, ...] = ()) -> pl.DataFrame:
        """Run arbitrary SQL against the catalog connection."""
        return self._con.execute(sql, list(params)).pl()

    def _select(
        self,
        conditions: list[str],
        params: list[object],
        *,
        limit: int | None,
        columns: str = "*",
    ) -> pl.DataFrame:
        sql = f"SELECT {columns} FROM puzzles"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return self.query(sql, params)

    def count(self) -> int:
        row = self._con.execute("SELECT count(*) FROM puzzles").fetchone()
        return int(row[0]) if row is not None else 0

    def rating_range(
        self,
        min_rating: int | None = None,
        max_rating: int | None = None,
        *,
        limit: int | None = None,
    ) -> pl.DataFrame:
        conditions: list[str] = []
        params: list[object] = []
        if min_rating is not None:
            conditions.append("rating >= ?")
            params.append(min_rating)
        if max_rating is not None:
            conditions.append("rating <= ?")
            params.append(max_rating)
        return self._select(conditions, params, limit=limit)

    def rating_deviation(self, max_deviation: int, *, limit: int | None = None) -> pl.DataFrame:
        return self._select(["rating_deviation <= ?"], [max_deviation], limit=limit)

    def popularity(self, min_popularity: int, *, limit: int | None = None) -> pl.DataFrame:
        return self._select(["popularity >= ?"], [min_popularity], limit=limit)

    def play_count(self, min_plays: int, *, limit: int | None = None) -> pl.DataFrame:
        return self._select(["nb_plays >= ?"], [min_plays], limit=limit)

    def theme(self, theme: str, *, limit: int | None = None) -> pl.DataFrame:
        return self._select(["list_contains(themes, ?)"], [theme], limit=limit)

    def line_length(
        self,
        min_plies: int | None = None,
        max_plies: int | None = None,
        *,
        limit: int | None = None,
    ) -> pl.DataFrame:
        conditions: list[str] = []
        params: list[object] = []
        if min_plies is not None:
            conditions.append("solution_plies >= ?")
            params.append(min_plies)
        if max_plies is not None:
            conditions.append("solution_plies <= ?")
            params.append(max_plies)
        return self._select(conditions, params, limit=limit)

    def checkmate_status(self, outcome: str, *, limit: int | None = None) -> pl.DataFrame:
        return self._select(["outcome = ?"], [outcome], limit=limit)

    def piece_count(
        self,
        min_count: int | None = None,
        max_count: int | None = None,
        *,
        limit: int | None = None,
    ) -> pl.DataFrame:
        conditions: list[str] = []
        params: list[object] = []
        if min_count is not None:
            conditions.append("presented_piece_count >= ?")
            params.append(min_count)
        if max_count is not None:
            conditions.append("presented_piece_count <= ?")
            params.append(max_count)
        return self._select(conditions, params, limit=limit)

    def material(
        self,
        signature: MaterialSignature,
        *,
        position: str = "start",
        mode: str = "equal",
        limit: int | None = None,
    ) -> pl.DataFrame:
        """Filter rows by material compatibility against ``signature``.

        ``position`` selects the start or final material; ``mode`` is one of
        ``equal``, ``superset`` (the stored material contains at least the
        requested counts) or ``subset``.
        """
        if position not in ("start", "final"):
            raise ValueError(f"position must be 'start' or 'final', got {position!r}")
        if mode not in ("equal", "superset", "subset"):
            raise ValueError(f"mode must be 'equal', 'superset' or 'subset', got {mode!r}")
        operator = {"equal": "=", "superset": ">=", "subset": "<="}[mode]
        prefix = f"{position}_"
        conditions = [f"{prefix}{name} {operator} ?" for name in COUNT_NAMES]
        return self._select(conditions, list(signature.counts), limit=limit)

    def count_where(self, where: str = "") -> int:
        sql = "SELECT count(*) FROM puzzles"
        if where:
            sql += f" WHERE {where}"
        row = self._con.execute(sql).fetchone()
        return int(row[0]) if row is not None else 0

    def stream_analysis_rows(
        self,
        where: str = "",
        batch_size: int = 1000,
    ) -> Iterator[tuple[Any, ...]]:
        """Yield analysis rows (puzzle_id, source, transform_version, start_fen,
        solution_uci, solver_colour, outcome) in bounded batches."""
        sql = (
            "SELECT puzzle_id, source, transform_version, start_fen, "
            "solution_uci, solver_colour, outcome FROM puzzles"
        )
        if where:
            sql += f" WHERE {where}"
        cursor = self._con.execute(sql)
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            yield from rows

    def fetch_puzzle(self, puzzle_id: str) -> dict[str, Any] | None:
        """Return one puzzle row as a dict (or None)."""
        rows = self._con.execute(
            "SELECT * FROM puzzles WHERE puzzle_id = ?", [puzzle_id]
        ).fetchmany(1)
        if not rows:
            return None
        columns = [column[0] for column in self._con.description]
        return dict(zip(columns, rows[0], strict=True))

    def close(self) -> None:
        self._con.close()
