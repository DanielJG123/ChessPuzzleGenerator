"""Data-quality statistics computed in SQL (no full dataset in Python memory)."""

from __future__ import annotations

import duckdb
import polars as pl

__all__ = ["compute_stats"]

_STATS_QUERIES: dict[str, str] = {
    "counts": "SELECT count(*) AS puzzle_rows FROM puzzles",
    "quarantined": "SELECT count(*) AS quarantine_rows FROM quarantined",
    "by_rating_bucket": (
        "SELECT rating_bucket, count(*) AS n FROM puzzles "
        "GROUP BY rating_bucket ORDER BY rating_bucket"
    ),
    "by_piece_count": (
        "SELECT presented_piece_count, count(*) AS n FROM puzzles "
        "GROUP BY presented_piece_count ORDER BY presented_piece_count"
    ),
    "by_line_length": (
        "SELECT solution_plies, count(*) AS n FROM puzzles "
        "GROUP BY solution_plies ORDER BY solution_plies"
    ),
    "by_outcome": ("SELECT outcome, count(*) AS n FROM puzzles GROUP BY outcome ORDER BY n DESC"),
    "by_solver_colour": (
        "SELECT solver_colour, count(*) AS n FROM puzzles "
        "GROUP BY solver_colour ORDER BY solver_colour"
    ),
    "by_theme": (
        "SELECT theme, count(*) AS n FROM "
        "(SELECT unnest(themes) AS theme FROM puzzles) "
        "GROUP BY theme ORDER BY n DESC LIMIT 20"
    ),
    "by_opening_tag": (
        "SELECT tag, count(*) AS n FROM "
        "(SELECT unnest(opening_tags) AS tag FROM puzzles) "
        "GROUP BY tag ORDER BY n DESC LIMIT 20"
    ),
    "flags": (
        "SELECT sum(has_promotion::INT) AS has_promotion, "
        "sum(duplicate_queens::INT) AS duplicate_queens, "
        "sum(unusual_material::INT) AS unusual_material, "
        "sum(is_checkmate::INT) AS checkmates FROM puzzles"
    ),
    "quarantine_reasons": (
        "SELECT reason, count(*) AS n FROM quarantined GROUP BY reason ORDER BY n DESC"
    ),
}


def compute_stats(connection: duckdb.DuckDBPyConnection) -> dict[str, pl.DataFrame]:
    """Return a mapping of stat name to a small aggregated DataFrame."""
    return {name: connection.execute(sql).pl() for name, sql in _STATS_QUERIES.items()}
